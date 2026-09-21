"""Router observability: per-layer expert usage, entropy, confidence, drops and specialization.

``RoutingMonitor`` reads the statistics the model collects when ``model.set_collect_stats(True)``,
accumulates them over a window, appends a JSON snapshot to ``routing_log.jsonl`` and renders
dependency-free SVG heatmaps/charts plus an ``index.html`` that shows them together.
"""

import html
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor


class RoutingMonitor:
    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        top_k: int,
        domains: Sequence[str] = ("default",),
        token_classes: Optional[tuple[Tensor, list[str]]] = None,
        out_dir: Optional[Path] = None,
    ) -> None:
        self.num_layers, self.num_experts, self.top_k = num_layers, num_experts, top_k
        self.domains = list(domains)
        self.token_classes = token_classes
        self.out_dir = out_dir
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[dict] = []
        self._reset_window()
        num_classes = len(token_classes[1]) if token_classes else 0
        # cumulative over the whole run
        self.class_usage = torch.zeros(num_layers, num_classes, num_experts, dtype=torch.long)
        self.domain_usage = torch.zeros(num_layers, len(self.domains), num_experts, dtype=torch.long)
        self.total_load = torch.zeros(num_layers, num_experts, dtype=torch.long)

    def _reset_window(self) -> None:
        L, E = self.num_layers, self.num_experts
        self.window_assigned = torch.zeros(L, E, dtype=torch.long)
        self.window_load = torch.zeros(L, E, dtype=torch.long)
        self.window_entropy = torch.zeros(L)
        self.window_confidence = torch.zeros(L)
        self.window_confidence_hist = torch.zeros(L, 10)
        self.window_updates = 0
        self.window_drops_by_expert = torch.zeros(L, E, dtype=torch.long)
        self.window_drops_by_class: dict[str, int] = {}

    @torch.no_grad()
    def update(self, model, input_ids: Tensor, domain_ids: Optional[Tensor] = None) -> None:
        """Fold in the routing stats of the forward pass that just ran on ``input_ids``."""
        stats_per_layer = model.routing_stats()
        if not all(stats_per_layer):
            raise RuntimeError("model has no routing stats; call model.set_collect_stats(True) before the forward pass")
        flat_tokens = input_ids.reshape(-1).cpu()
        sequence = input_ids.size(-1)
        classes = None
        if self.token_classes is not None:
            classes = self.token_classes[0][flat_tokens]
        token_domains = None
        if domain_ids is not None:
            token_domains = domain_ids.repeat_interleave(sequence).cpu()
        indices_per_layer = [moe.last_top_indices for moe in model.moes()]
        for layer, (stats, top_indices) in enumerate(zip(stats_per_layer, indices_per_layer)):
            self.window_assigned[layer] += stats["expert_assigned"].cpu()
            self.window_load[layer] += stats["expert_load"].cpu()
            self.total_load[layer] += stats["expert_load"].cpu()
            self.window_entropy[layer] += float(stats["entropy_mean"])
            self.window_confidence[layer] += float(stats["confidence_mean"])
            self.window_confidence_hist[layer] += stats["confidence_hist"].cpu()
            dropped_tokens = stats["dropped_tokens"].cpu()
            dropped_experts = stats["dropped_experts"].cpu()
            if dropped_tokens.numel():
                self.window_drops_by_expert[layer] += torch.bincount(dropped_experts, minlength=self.num_experts)
                if classes is not None:
                    names = self.token_classes[1]
                    counts = torch.bincount(classes[dropped_tokens], minlength=len(names))
                    for name, count in zip(names, counts.tolist()):
                        if count:
                            self.window_drops_by_class[name] = self.window_drops_by_class.get(name, 0) + count
            experts = top_indices.reshape(-1).cpu()
            if classes is not None:
                key = classes.repeat_interleave(self.top_k) * self.num_experts + experts
                self.class_usage[layer] += torch.bincount(key, minlength=self.class_usage[layer].numel()).reshape(
                    self.class_usage[layer].shape
                )
            if token_domains is not None:
                key = token_domains.repeat_interleave(self.top_k) * self.num_experts + experts
                self.domain_usage[layer] += torch.bincount(key, minlength=self.domain_usage[layer].numel()).reshape(
                    self.domain_usage[layer].shape
                )
        self.window_updates += 1

    # ---- window metrics -----------------------------------------------------
    def imbalance(self) -> Optional[float]:
        """Mean over layers of the coefficient of variation of routed load (0 = perfectly even)."""
        if not self.window_updates:
            return None
        counts = self.window_assigned.double()
        return float((counts.std(dim=1, unbiased=False) / counts.mean(dim=1).clamp(min=1e-9)).mean())

    def active_fraction(self, floor: float = 0.1) -> Optional[float]:
        """Share of experts that receive at least ``floor`` of an even share of the routes."""
        if not self.window_updates:
            return None
        counts = self.window_assigned.double()
        even = counts.sum(dim=1, keepdim=True) / self.num_experts
        return float((counts >= floor * even).double().mean())

    def snapshot(self, step: int, reset: bool = True) -> dict:
        if not self.window_updates:
            return {}
        n = self.window_updates
        assigned = self.window_assigned.double()
        load = self.window_load.double()
        total_routes = assigned.sum(dim=1).clamp(min=1)
        record = {
            "step": step,
            "layers": [
                {
                    "utilization": (load[layer] / load[layer].sum().clamp(min=1)).tolist(),
                    "requested_share": (assigned[layer] / total_routes[layer]).tolist(),
                    "entropy": float(self.window_entropy[layer] / n),
                    "entropy_max": math.log(self.num_experts),
                    "confidence": float(self.window_confidence[layer] / n),
                    "confidence_hist": (self.window_confidence_hist[layer] / n).tolist(),
                    "drop_rate": float(1 - load[layer].sum() / total_routes[layer]),
                    "drops_by_expert": self.window_drops_by_expert[layer].tolist(),
                    "dead_experts": int((assigned[layer] == 0).sum()),
                }
                for layer in range(self.num_layers)
            ],
            "imbalance": self.imbalance(),
            "active_fraction": self.active_fraction(),
            "drops_by_token_class": dict(self.window_drops_by_class),
        }
        self.snapshots.append(record)
        if self.out_dir is not None:
            with (self.out_dir / "routing_log.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        if reset:
            self._reset_window()
        return record

    # ---- reports ----------------------------------------------------------
    def write_report(self, out_dir: Optional[Path] = None) -> Path:
        """Render heatmaps and time series as SVG plus an index.html; returns the index path."""
        out_dir = out_dir or self.out_dir
        if out_dir is None:
            raise ValueError("no output directory configured")
        out_dir.mkdir(parents=True, exist_ok=True)
        pages: list[tuple[str, str]] = []

        def save(name: str, svg: str) -> None:
            (out_dir / name).write_text(svg)
            pages.append((name, name))

        experts = [str(e) for e in range(self.num_experts)]
        layers = [f"layer {i}" for i in range(self.num_layers)]
        if self.total_load.sum() > 0:
            share = self.total_load.double() / self.total_load.sum(dim=1, keepdim=True).clamp(min=1)
            save("expert_usage_by_layer.svg", heatmap_svg(share.tolist(), layers, experts, "Expert usage share by layer (whole run)"))
        if self.token_classes is not None and self.class_usage.sum() > 0:
            names = self.token_classes[1]
            for layer in range(self.num_layers):
                usage = self.class_usage[layer].double()
                usage = usage / usage.sum(dim=1, keepdim=True).clamp(min=1)
                save(f"specialization_layer{layer}.svg", heatmap_svg(usage.tolist(), names, experts, f"Layer {layer}: expert share per token class"))
        if len(self.domains) > 1 and self.domain_usage.sum() > 0:
            for layer in range(self.num_layers):
                usage = self.domain_usage[layer].double()
                usage = usage / usage.sum(dim=1, keepdim=True).clamp(min=1)
                save(f"domain_usage_layer{layer}.svg", heatmap_svg(usage.tolist(), self.domains, experts, f"Layer {layer}: expert share per domain"))
        if self.snapshots:
            steps = [s["step"] for s in self.snapshots]
            save("drop_rate_over_time.svg", line_chart_svg(
                steps, {f"layer {i}": [s["layers"][i]["drop_rate"] for s in self.snapshots] for i in range(self.num_layers)},
                "Token drop rate over time"))
            save("router_entropy_over_time.svg", line_chart_svg(
                steps, {f"layer {i}": [s["layers"][i]["entropy"] for s in self.snapshots] for i in range(self.num_layers)},
                f"Router entropy over time (max {math.log(self.num_experts):.2f} nats)"))
            for layer in range(self.num_layers):
                save(f"utilization_layer{layer}_over_time.svg", heatmap_svg(
                    [s["layers"][layer]["utilization"] for s in self.snapshots], [f"step {s}" for s in steps], experts,
                    f"Layer {layer}: per-expert utilization over time"))
        body = "".join(
            f'<h2>{html.escape(name)}</h2><img src="{html.escape(src)}" alt="{html.escape(name)}">' for name, src in pages
        )
        index = out_dir / "index.html"
        index.write_text(
            "<!doctype html><meta charset='utf-8'><title>Routing diagnostics</title>"
            "<style>body{font:14px sans-serif;margin:24px;max-width:1100px}img{max-width:100%}</style>"
            f"<h1>Routing diagnostics</h1>{body or '<p>No routing data recorded.</p>'}"
        )
        return index


def _ramp(value: float) -> str:
    """White-to-blue sequential colour for value in [0, 1]."""
    value = min(1.0, max(0.0, value))
    r = round(247 + (8 - 247) * value)
    g = round(251 + (48 - 251) * value)
    b = round(255 + (107 - 255) * value)
    return f"rgb({r},{g},{b})"


def heatmap_svg(matrix: Sequence[Sequence[float]], row_labels: Sequence[str], col_labels: Sequence[str], title: str) -> str:
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    cell = max(4, min(28, 900 // max(cols, 1)))
    left, top = 90, 40
    width, height = left + cols * cell + 20, top + rows * cell + 30
    peak = max((max(row) for row in matrix), default=0) or 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
        f'<text x="{left}" y="20" font-size="14" font-weight="bold">{html.escape(title)}</text>',
    ]
    for r, row in enumerate(matrix):
        parts.append(f'<text x="{left - 6}" y="{top + r * cell + cell * 0.7:.1f}" text-anchor="end">{html.escape(str(row_labels[r]))}</text>')
        for c, value in enumerate(row):
            parts.append(
                f'<rect x="{left + c * cell}" y="{top + r * cell}" width="{cell}" height="{cell}" fill="{_ramp(value / peak)}">'
                f"<title>{html.escape(str(row_labels[r]))} / expert {html.escape(str(col_labels[c]))}: {value:.4f}</title></rect>"
            )
    step = max(1, cols // 16)
    for c in range(0, cols, step):
        parts.append(f'<text x="{left + c * cell + cell / 2}" y="{top + rows * cell + 14}" text-anchor="middle">{html.escape(str(col_labels[c]))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def line_chart_svg(x: Sequence[float], series: dict[str, Sequence[float]], title: str) -> str:
    width, height, left, top, bottom = 720, 300, 60, 40, 30
    values = [v for line in series.values() for v in line]
    low, high = (min(values), max(values)) if values else (0.0, 1.0)
    if high == low:
        high = low + 1.0
    x_low, x_high = (min(x), max(x)) if x else (0, 1)
    if x_high == x_low:
        x_high = x_low + 1
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
        f'<text x="{left}" y="20" font-size="14" font-weight="bold">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#888"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - 10}" y2="{height - bottom}" stroke="#888"/>',
        f'<text x="{left - 6}" y="{top + 4}" text-anchor="end">{high:.3g}</text>',
        f'<text x="{left - 6}" y="{height - bottom}" text-anchor="end">{low:.3g}</text>',
        f'<text x="{left}" y="{height - 8}">{x_low}</text><text x="{width - 10}" y="{height - 8}" text-anchor="end">{x_high}</text>',
    ]
    for index, (name, line) in enumerate(series.items()):
        points = " ".join(
            f"{left + (xv - x_low) / (x_high - x_low) * (width - left - 10):.1f},"
            f"{height - bottom - (yv - low) / (high - low) * (height - bottom - top):.1f}"
            for xv, yv in zip(x, line)
        )
        colour = palette[index % len(palette)]
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{points}"/>')
        parts.append(f'<text x="{width - 100}" y="{top + 12 * index}" fill="{colour}">{html.escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)
