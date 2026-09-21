"""Benchmark a saved QuantaWeave sparse-MoE checkpoint.

Writes a JSON report (and a Markdown twin) covering quality (loss, perplexity), speed (tokens/s, time per
batch, per-expert compute), memory, exact parameter counts, and routing health (per-layer utilization,
entropy, confidence, drops, dead experts). Pass --training-metrics to fold in training stability.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch

from data_pipeline import WindowDataset, build_corpus, load_tokenizer
from hardware import choose_precision, detect
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import autocast_context, select_device
from training_metrics import detect_plateau, first_stable_step


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def process_rss_bytes() -> Optional[int]:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss
    except ImportError:
        return None


def training_stability(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    steps = [r["step"] for r in records if "loss" in r]
    losses = [r["loss"] for r in records if "loss" in r]
    if not losses:
        return {}
    tail = losses[-min(len(losses), 20):]
    return {
        "logged_steps": len(losses),
        "final_loss": sum(tail) / len(tail),
        "best_loss": min(losses),
        "plateaued": detect_plateau(losses),
        "first_stable_step": first_stable_step(steps, losses),
        "mean_tokens_per_second": sum(r.get("tokens_per_second", 0) for r in records) / len(records),
        "total_overflow_routes": sum(r.get("overflow_routes", 0) for r in records),
    }


def histogram(values: list[float], bins: int = 10) -> dict:
    top = max(values) if values and max(values) > 0 else 1.0
    counts = torch.histc(torch.tensor(values, dtype=torch.float32), bins=bins, min=0.0, max=top).tolist()
    return {"edges": [top * i / bins for i in range(bins + 1)], "counts": [int(c) for c in counts]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--data", type=Path, default=Path("data/smoke/tinystories.jsonl"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--output", type=Path, help="JSON report path; a .md report is written beside it")
    parser.add_argument("--training-metrics", type=Path, help="metrics.jsonl from training, for stability metrics")
    parser.add_argument("--no-routing-stats", action="store_true", help="skip the routing/compute diagnostic pass")
    return parser


def run_benchmark(args) -> dict:
    config_path, model_path = args.checkpoint / "config.json", args.checkpoint / "model.pt"
    for path in (config_path, model_path, args.data):
        if not path.exists():
            raise FileNotFoundError(f"missing required path: {path}")

    device = select_device(args.device)
    config = QuantaWeaveConfig(**json.loads(config_path.read_text()))
    tokenizer = load_tokenizer(args.checkpoint)
    dataset = WindowDataset(build_corpus([args.data], tokenizer, args.examples), config.max_sequence_length - 1)
    if len(dataset) == 0:
        raise ValueError("benchmark dataset contains no usable sequences")
    batches = [
        dataset.batch(range(start, min(len(dataset), start + args.batch_size)))
        for start in range(0, len(dataset), args.batch_size)
    ]

    model = QuantaWeaveMoEForCausalLM(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    precision = choose_precision(detect(device.type), args.precision)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    per_expert_parameters = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    expert_parameters = config.layers * config.num_experts * per_expert_parameters
    shared_parameters = total_parameters - expert_parameters
    active_parameters = shared_parameters + config.layers * config.top_k * per_expert_parameters
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    total_loss = total_router_aux = 0.0
    total_tokens = total_dropped = total_overflow = routed_tokens = 0
    expert_counts = [0] * config.num_experts
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            batch = batch.to(device)
            with autocast_context(device, precision):
                outputs = model(batch, labels=batch)
            token_count = batch.size(0) * (batch.size(1) - 1)
            total_loss += float(outputs["loss"].item()) * token_count
            total_router_aux += float(outputs["router_aux_loss"].item())
            total_dropped += int(outputs["dropped_routes"].item())
            total_overflow += int(outputs["overflow_routes"].item())
            total_tokens += token_count
            routed_tokens += batch.numel()
            for indices in outputs["expert_indices"]:
                counts = torch.bincount(indices.reshape(-1), minlength=config.num_experts).cpu().tolist()
                expert_counts = [left + right for left, right in zip(expert_counts, counts)]
    synchronize(device)
    elapsed = time.perf_counter() - started
    memory = {
        "parameter_bytes": parameter_bytes,
        "bytes_per_expert": per_expert_parameters * config.layers * next(model.parameters()).element_size(),
        "peak_device_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "process_rss_bytes": process_rss_bytes(),
    }

    mean_loss = total_loss / total_tokens
    routes = routed_tokens * config.layers * config.top_k
    utilization = [count / routes for count in expert_counts]
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else device.type,
        "precision": precision,
        "tokenizer": tokenizer.kind,
        "examples": len(dataset),
        "tokens": total_tokens,
        "batches": len(batches),
        "elapsed_seconds": elapsed,
        "seconds_per_batch": elapsed / len(batches),
        "tokens_per_second": total_tokens / elapsed,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "router_aux_loss": total_router_aux / len(batches),
        "dropped_routes": total_dropped,
        "dropped_route_fraction": total_dropped / max(1, routes),
        "overflow_routes": total_overflow,
        "overflow_route_fraction": total_overflow / max(1, routes),
        "capacity_factor": config.capacity_factor,
        "min_expert_capacity": config.min_expert_capacity,
        "drop_overflow_tokens": config.drop_overflow_tokens,
        "overflow_policy": config.overflow_policy,
        "router_temperature": config.router_temperature,
        "total_parameters": total_parameters,
        "shared_parameters_estimate": shared_parameters,
        "expert_parameters_estimate": expert_parameters,
        "active_parameters_per_token_estimate": active_parameters,
        "active_parameter_ratio": active_parameters / total_parameters,
        "num_experts": config.num_experts,
        "active_experts_per_token": config.top_k,
        "inactive_experts_per_token": config.num_experts - config.top_k,
        "expert_utilization_min": min(utilization),
        "expert_utilization_max": max(utilization),
        "expert_utilization_mean": sum(utilization) / len(utilization),
        "expert_utilization_histogram": histogram(utilization),
        "memory": memory,
        "training_step_in_checkpoint": checkpoint.get("step"),
    }
    if not args.no_routing_stats:
        report["routing"], report["expert_compute"] = routing_and_compute(model, batches, device, precision, config)
    if args.training_metrics is not None and args.training_metrics.exists():
        report["training_stability"] = training_stability(args.training_metrics)
    return report


@torch.inference_mode()
def routing_and_compute(model, batches, device, precision, config) -> tuple[dict, dict]:
    """Second pass with router statistics and per-expert timing switched on (slower than the timed pass)."""
    model.set_collect_stats(True)
    for moe in model.moes():
        moe.profile_experts = True
        moe.expert_seconds = [0.0] * moe.num_local_experts
    layers = config.layers
    entropy = [0.0] * layers
    confidence = [0.0] * layers
    confidence_hist = [torch.zeros(10) for _ in range(layers)]
    assigned = [torch.zeros(config.num_experts, dtype=torch.long) for _ in range(layers)]
    load = [torch.zeros(config.num_experts, dtype=torch.long) for _ in range(layers)]
    synchronize(device)
    started = time.perf_counter()
    for batch in batches:
        with autocast_context(device, precision):
            model(batch.to(device))
        for layer, stats in enumerate(model.routing_stats()):
            entropy[layer] += float(stats["entropy_mean"])
            confidence[layer] += float(stats["confidence_mean"])
            confidence_hist[layer] += stats["confidence_hist"].cpu()
            assigned[layer] += stats["expert_assigned"].cpu()
            load[layer] += stats["expert_load"].cpu()
    synchronize(device)
    elapsed = time.perf_counter() - started
    n = len(batches)
    per_layer = []
    for layer in range(layers):
        share = (load[layer].double() / load[layer].sum().clamp(min=1)).tolist()
        even = assigned[layer].double().sum() / config.num_experts
        counts = assigned[layer].double()
        per_layer.append({
            "utilization": share,
            "utilization_histogram": histogram(share),
            "entropy": entropy[layer] / n,
            "entropy_normalized": entropy[layer] / n / math.log(config.num_experts) if config.num_experts > 1 else 0.0,
            "confidence_mean": confidence[layer] / n,
            "confidence_histogram": (confidence_hist[layer] / n).tolist(),
            "dead_experts": int((counts == 0).sum()),
            "underused_experts": int((counts < 0.1 * even).sum()),
            "load_cv": float(counts.std(unbiased=False) / counts.mean().clamp(min=1e-9)),
            "drop_rate": float(1 - load[layer].sum() / assigned[layer].sum().clamp(min=1)),
        })
    seconds = [sum(moe.expert_seconds[e] for moe in model.moes()) for e in range(model.moes()[0].num_local_experts)]
    moe_total = sum(seconds)
    for moe in model.moes():
        moe.profile_experts = False
    model.set_collect_stats(False)
    routing = {
        "per_layer": per_layer,
        "entropy_mean": sum(l["entropy"] for l in per_layer) / layers,
        "confidence_mean": sum(l["confidence_mean"] for l in per_layer) / layers,
        "dead_experts_total": sum(l["dead_experts"] for l in per_layer),
    }
    compute = {
        "seconds_per_expert": seconds,
        "moe_seconds_total": moe_total,
        "moe_share_of_forward": moe_total / elapsed if elapsed else None,
        "note": "measured on a separate profiled pass; timing hooks add overhead",
    }
    return routing, compute


def render_markdown(report: dict) -> str:
    mb = lambda b: "n/a" if b is None else f"{b / 1e6:,.1f} MB"  # noqa: E731
    lines = [
        f"# Benchmark: {report['checkpoint']}", "",
        f"- device: {report['device_name']} ({report['device']}), precision {report['precision']}, tokenizer {report['tokenizer']}",
        f"- checkpoint step: {report['training_step_in_checkpoint']}", "",
        "## Quality", "", "| metric | value |", "|---|---|",
        f"| loss | {report['loss']:.4f} |", f"| perplexity | {report['perplexity']:.2f} |",
        f"| tokens evaluated | {report['tokens']:,} |", "",
        "## Speed and memory", "", "| metric | value |", "|---|---|",
        f"| tokens / second | {report['tokens_per_second']:,.0f} |",
        f"| seconds / batch | {report['seconds_per_batch']:.4f} |",
        f"| parameter memory | {mb(report['memory']['parameter_bytes'])} |",
        f"| memory per expert (all layers) | {mb(report['memory']['bytes_per_expert'])} |",
        f"| peak device memory | {mb(report['memory']['peak_device_bytes'])} |",
        f"| process RSS | {mb(report['memory']['process_rss_bytes'])} |", "",
        "## Parameters", "", "| metric | value |", "|---|---|",
        f"| total | {report['total_parameters']:,} |", f"| active per token | {report['active_parameters_per_token_estimate']:,} |",
        f"| active / total | {report['active_parameter_ratio']:.1%} |",
        f"| experts (total / active) | {report['num_experts']} / {report['active_experts_per_token']} |", "",
        "## Routing", "", "| metric | value |", "|---|---|",
        f"| aux (balance) loss | {report['router_aux_loss']:.4f} |",
        f"| overflow routes | {report['overflow_routes']:,} ({report['overflow_route_fraction']:.2%}) |",
        f"| dropped routes | {report['dropped_routes']:,} ({report['dropped_route_fraction']:.2%}) |",
        f"| capacity factor | {report['capacity_factor']} |",
        f"| expert utilization min / mean / max | {report['expert_utilization_min']:.4f} / "
        f"{report['expert_utilization_mean']:.4f} / {report['expert_utilization_max']:.4f} |", "",
    ]
    if "routing" in report:
        lines += ["### Per layer", "", "| layer | entropy (norm.) | confidence | load CV | dead | underused | drop rate |", "|---|---|---|---|---|---|---|"]
        for index, layer in enumerate(report["routing"]["per_layer"]):
            lines.append(
                f"| {index} | {layer['entropy']:.3f} ({layer['entropy_normalized']:.0%}) | {layer['confidence_mean']:.3f} | "
                f"{layer['load_cv']:.3f} | {layer['dead_experts']} | {layer['underused_experts']} | {layer['drop_rate']:.2%} |"
            )
        compute = report["expert_compute"]
        lines += ["", f"MoE layers spent {compute['moe_seconds_total']:.3f}s across experts "
                      f"({(compute['moe_share_of_forward'] or 0):.0%} of the profiled pass).", ""]
    if "training_stability" in report:
        stability = report["training_stability"]
        lines += ["## Training stability", "", "| metric | value |", "|---|---|"]
        lines += [f"| {key} | {value} |" for key, value in stability.items()]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        markdown_path = args.output.with_suffix(".md")
        markdown_path.write_text(render_markdown(report))
        print(f"wrote benchmark report to {args.output} and {markdown_path}")


if __name__ == "__main__":
    main()
