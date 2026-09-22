"""Experiment manager: reproducible, comparable training runs.

  python src/experiment_manager.py run --config configs/exp_moe_balanced.yaml
  python src/experiment_manager.py list    [--runs-dir artifacts/experiments]
  python src/experiment_manager.py compare artifacts/experiments/*  [--markdown report.md]

A config bundle (YAML or JSON) looks like:

  name: moe-balanced
  description: adaptive capacity + aux weight
  tags: [balanced]
  train:                      # any option of train_quantweave_moe.py, using its python names
    steps: 300
    total_experts: 8
    capacity_adapt: true
  benchmark:                  # optional; evaluates the final model (default: the training data)
    examples: 500
  objective: benchmark.loss   # what "best" means (lower is better); default benchmark.loss, else final_loss

Each run gets a unique id and a directory holding config.json, metrics.jsonl, train.log, routing diagnostics,
checkpoints, the final model, summary.json and report.md. best.json in the runs directory tracks the best run.
"""

import argparse
import contextlib
import hashlib
import html
import io
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

DEFAULT_RUNS_DIR = Path("artifacts/experiments")


def load_config(path: Path) -> dict:
    text = Path(path).read_text()
    if Path(path).suffix in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "run"


def git_revision() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


class _Tee(io.TextIOBase):
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def dig(record: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(record, dict) or part not in record:
            return None
        record = record[part]
    return record


def objective_of(summary: dict, objective: Optional[str]) -> tuple[str, Optional[float]]:
    name = objective or ("benchmark.loss" if summary.get("benchmark") else "train.final_loss")
    value = dig(summary, name)
    return name, float(value) if isinstance(value, (int, float)) else None


def update_best(runs_dir: Path, summary: dict, objective: Optional[str]) -> bool:
    """Record ``summary`` as the best run if it beats the current best on the same objective."""
    name, value = objective_of(summary, objective)
    if value is None:
        return False
    best_path = runs_dir / "best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
        if best.get("objective") == name and best["value"] <= value:
            return False
    best_path.write_text(json.dumps({"run_id": summary["run_id"], "objective": name, "value": value, "path": summary["run_dir"]}, indent=2) + "\n")
    return True


def run_experiment(config: dict, runs_dir: Path = DEFAULT_RUNS_DIR, name: Optional[str] = None, run_id: Optional[str] = None) -> dict:
    import train_quantweave_moe as trainer

    name = name or config.get("name", "experiment")
    overrides = dict(config.get("train", {}))
    for key in ("output", "checkpoint_dir", "metrics_file", "diagnostics_dir", "resume"):
        overrides.pop(key, None)  # the manager owns run locations
    trainer.default_args(**overrides)  # rejects unknown option names before anything is created
    fingerprint = hashlib.sha256(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:6]
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(name)}-{fingerprint}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    steps = overrides.get("steps", trainer.default_args().steps)
    resolved = {"run_id": run_id, "name": name, "description": config.get("description"), "tags": config.get("tags", []),
                "train": overrides, "benchmark": config.get("benchmark", {}), "objective": config.get("objective"),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_revision": git_revision(), "argv": sys.argv}
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2, default=str) + "\n")

    args = trainer.default_args(
        **{"diagnostics_interval": max(1, steps // 20), **overrides},
        output=run_dir / "model", checkpoint_dir=run_dir / "checkpoints", metrics_file=run_dir / "metrics.jsonl",
        diagnostics_dir=run_dir / "diagnostics", resume=False,
    )
    summary: dict = {"run_id": run_id, "name": name, "run_dir": str(run_dir), "tags": resolved["tags"]}
    started = time.perf_counter()
    with (run_dir / "train.log").open("w") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        try:
            result = trainer.run_training(args)
            summary["status"] = "completed"
            routing = result.get("last_routing") or {}
            summary["train"] = {key: result.get(key) for key in (
                "steps", "final_loss", "best_loss", "tokens_per_second", "plateaued", "first_stable_step",
                "active_parameter_ratio", "total_parameters", "active_parameters", "precision")}
            summary["train"]["overflow_fraction"] = (result.get("last_log") or {}).get("overflow_fraction")
            summary["train"]["routing_imbalance"] = routing.get("imbalance")
            summary["train"]["active_expert_fraction"] = routing.get("active_fraction")
            summary["train"]["controller_changes"] = len(result.get("controller_history", []))
            if config.get("benchmark") is not None:
                summary["benchmark"] = run_final_benchmark(run_dir, args, config.get("benchmark") or {})
        except Exception as error:
            summary["status"] = "failed"
            summary["error"] = f"{type(error).__name__}: {error}"
            (run_dir / "error.txt").write_text(traceback.format_exc())
    summary["seconds"] = time.perf_counter() - started
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if summary["status"] == "completed":
        summary["is_best"] = update_best(runs_dir, summary, config.get("objective"))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (run_dir / "report.md").write_text(render_run_report(resolved, summary))
    return summary


def run_final_benchmark(run_dir: Path, args, options: dict) -> dict:
    from benchmark_quantweave_moe import build_parser, render_markdown, run_benchmark

    argv = ["--checkpoint", str(run_dir / "model"), "--data", str(options.get("data", args.data[0])),
            "--examples", str(options.get("examples", 500)), "--batch-size", str(options.get("batch_size", args.batch_size)),
            "--device", args.device, "--training-metrics", str(run_dir / "metrics.jsonl")]
    if options.get("no_routing_stats"):
        argv.append("--no-routing-stats")
    report = run_benchmark(build_parser().parse_args(argv))
    (run_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    (run_dir / "benchmark.md").write_text(render_markdown(report))
    keep = ("loss", "perplexity", "tokens_per_second", "dropped_route_fraction", "overflow_route_fraction",
            "expert_utilization_min", "expert_utilization_max", "active_parameter_ratio", "total_parameters")
    result = {key: report[key] for key in keep}
    if "routing" in report:
        result["dead_experts"] = report["routing"]["dead_experts_total"]
        result["routing_entropy"] = report["routing"]["entropy_mean"]
    return result


def render_run_report(config: dict, summary: dict) -> str:
    lines = [f"# {summary['name']} ({summary['run_id']})", "", f"- status: **{summary['status']}**",
             f"- wall time: {summary.get('seconds', 0):.1f}s", f"- git revision: {config.get('git_revision')}", ""]
    if config.get("description"):
        lines += [config["description"], ""]
    lines += ["## Configuration", "", "| option | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in config["train"].items()] + [""]
    if summary["status"] == "failed":
        return "\n".join(lines + ["## Failure", "", f"`{summary['error']}` (traceback in error.txt)", ""])
    for section in ("train", "benchmark"):
        if summary.get(section):
            lines += [f"## {section.capitalize()} results", "", "| metric | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in summary[section].items()] + [""]
    return "\n".join(lines)


# ---- comparison ------------------------------------------------------------------------------
def load_summaries(paths: list[Path]) -> list[dict]:
    summaries = []
    for path in paths:
        summary_path = path / "summary.json" if path.is_dir() else path
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            config_path = summary_path.parent / "config.json"
            summary["config"] = json.loads(config_path.read_text()) if config_path.exists() else {}
            summaries.append(summary)
    return summaries


def compare_runs(summaries: list[dict], metric: str = "benchmark.loss") -> str:
    """Markdown table of runs ranked by ``metric`` (lower is better), with the options that differ."""
    completed = [s for s in summaries if s.get("status") == "completed"]
    keyed = sorted(completed, key=lambda s: (dig(s, metric) is None, dig(s, metric) or 0))
    differing = []
    trains = [s["config"].get("train", {}) for s in keyed]
    for option in sorted({k for t in trains for k in t}):
        if len({json.dumps(t.get(option), default=str) for t in trains}) > 1:
            differing.append(option)
    columns = ["run", metric, "train.final_loss", "benchmark.perplexity", "train.tokens_per_second", "train.active_parameter_ratio",
               "train.overflow_fraction", "benchmark.dead_experts"]
    columns = list(dict.fromkeys(columns))
    fmt = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(columns + differing) + " |", "|" + "---|" * (len(columns) + len(differing))]
    for summary in keyed:
        train = summary["config"].get("train", {})
        cells = [summary["run_id"]] + [fmt(dig(summary, c)) for c in columns[1:]] + [fmt(train.get(o)) for o in differing]
        lines.append("| " + " | ".join(cells) + " |")
    failed = [s for s in summaries if s.get("status") == "failed"]
    if failed:
        lines += ["", "Failed runs: " + ", ".join(f"{s['run_id']} ({s.get('error')})" for s in failed)]
    return "\n".join(lines)


def parse_objectives(spec) -> list[tuple[str, str]]:
    """["benchmark.loss:min", "train.tokens_per_second:max"] -> [(path, direction), ...]. A bare path defaults to min."""
    if isinstance(spec, str):
        spec = [spec]
    objectives = []
    for item in spec:
        path, _, direction = item.partition(":")
        direction = direction or "min"
        if direction not in {"min", "max"}:
            raise ValueError(f"objective '{item}': direction must be 'min' or 'max'")
        objectives.append((path.strip(), direction))
    if len(objectives) < 2:
        raise ValueError("multi-objective needs at least two objectives")
    return objectives


def dominates(a: dict, b: dict, objectives: list[tuple[str, str]]) -> bool:
    """True if summary ``a`` is at least as good as ``b`` on every objective, and strictly better on one."""
    at_least_as_good, strictly_better = True, False
    for path, direction in objectives:
        va, vb = dig(a, path), dig(b, path)
        if va is None or vb is None:
            return False
        better = va < vb if direction == "min" else va > vb
        worse = va > vb if direction == "min" else va < vb
        at_least_as_good = at_least_as_good and not worse
        strictly_better = strictly_better or better
    return at_least_as_good and strictly_better


def pareto_front(summaries: list[dict], objectives) -> list[dict]:
    """The non-dominated summaries: no other completed summary is at least as good on every objective and
    strictly better on one. ``objectives`` is parsed with ``parse_objectives`` if given as raw strings."""
    if objectives and isinstance(objectives[0], str):
        objectives = parse_objectives(objectives)
    completed = [s for s in summaries if s.get("status") == "completed" and all(dig(s, path) is not None for path, _ in objectives)]
    return [s for s in completed if not any(dominates(other, s, objectives) for other in completed if other is not s)]


def compare_pareto(summaries: list[dict], objectives) -> str:
    """Markdown table of every completed run, its objective values, and whether it is Pareto-optimal."""
    parsed = parse_objectives(objectives) if objectives and isinstance(objectives[0], str) else objectives
    completed = [s for s in summaries if s.get("status") == "completed"]
    front = {id(s) for s in pareto_front(completed, parsed)}
    fmt = lambda v: "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
    columns = [f"{path} ({direction})" for path, direction in parsed]
    lines = ["| run | " + " | ".join(columns) + " | pareto-optimal |", "|" + "---|" * (len(columns) + 2)]
    ranked = sorted(completed, key=lambda s: id(s) not in front)  # Pareto-optimal rows first
    for summary in ranked:
        values = [fmt(dig(summary, path)) for path, _ in parsed]
        lines.append(f"| {summary['run_id']} | " + " | ".join(values) + f" | {'yes' if id(summary) in front else ''} |")
    return "\n".join(lines)


def bar_chart_svg(values: dict[str, float], title: str) -> str:
    width, row, left = 720, 22, 260
    height = 50 + row * len(values)
    peak = max(values.values(), default=1.0) or 1.0
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
             f'<text x="10" y="20" font-size="14" font-weight="bold">{html.escape(title)}</text>']
    for index, (name, value) in enumerate(values.items()):
        y = 36 + index * row
        bar = max(1.0, (value / peak) * (width - left - 80))
        parts.append(f'<text x="{left - 6}" y="{y + 12}" text-anchor="end">{html.escape(name[-40:])}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="{row - 6}" fill="#1f77b4"/>')
        parts.append(f'<text x="{left + bar + 4:.1f}" y="{y + 12}">{value:.4g}</text>')
    return "".join(parts) + "</svg>"


def loss_curves_svg(summaries: list[dict]) -> str:
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]
    series = {}
    for summary in summaries:
        path = Path(summary["run_dir"]) / "metrics.jsonl"
        if path.exists():
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            series[summary["run_id"]] = [(r["step"], r["loss"]) for r in records if "loss" in r]
    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs:
        return ""
    width, height, left, top, bottom = 760, 320, 60, 40, 30
    x0, x1, y0, y1 = min(xs), max(xs) or 1, min(ys), max(ys)
    x1 = x1 if x1 != x0 else x0 + 1
    y1 = y1 if y1 != y0 else y0 + 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 14 * len(series)}" font-family="sans-serif" font-size="11">',
             f'<text x="{left}" y="20" font-size="14" font-weight="bold">Training loss</text>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#888"/>',
             f'<line x1="{left}" y1="{height - bottom}" x2="{width - 10}" y2="{height - bottom}" stroke="#888"/>',
             f'<text x="{left - 6}" y="{top + 4}" text-anchor="end">{y1:.3g}</text><text x="{left - 6}" y="{height - bottom}" text-anchor="end">{y0:.3g}</text>']
    for index, (name, points) in enumerate(series.items()):
        colour = palette[index % len(palette)]
        coords = " ".join(f"{left + (x - x0) / (x1 - x0) * (width - left - 10):.1f},{height - bottom - (y - y0) / (y1 - y0) * (height - bottom - top):.1f}" for x, y in points)
        parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" points="{coords}"/>')
        parts.append(f'<text x="{left}" y="{height + 14 * index}" fill="{colour}">{html.escape(name)}</text>')
    return "".join(parts) + "</svg>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute a config bundle as a tracked run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--name")
    run.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    listing = commands.add_parser("list", help="list runs and the best one")
    listing.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    compare = commands.add_parser("compare", help="rank runs and draw comparison charts")
    compare.add_argument("runs", type=Path, nargs="+")
    compare.add_argument("--metric", default="benchmark.loss")
    compare.add_argument("--objectives", nargs="+", help="2+ dotted.path:min|max entries; ranks by Pareto-optimality instead of --metric")
    compare.add_argument("--markdown", type=Path)
    compare.add_argument("--charts-dir", type=Path, help="write loss_curves.svg and metric_bars.svg here")
    args = parser.parse_args()

    if args.command == "run":
        summary = run_experiment(load_config(args.config), args.runs_dir, args.name)
        print(f"\n{summary['run_id']}: {summary['status']} in {summary['seconds']:.1f}s -> {summary['run_dir']}")
        if summary["status"] == "failed":
            sys.exit(f"run failed: {summary['error']}")
    elif args.command == "list":
        dirs = sorted(p for p in args.runs_dir.iterdir() if p.is_dir()) if args.runs_dir.exists() else []
        print(compare_runs(load_summaries(dirs)))
        best = args.runs_dir / "best.json"
        if best.exists():
            print("\nbest:", json.loads(best.read_text()))
    else:
        summaries = load_summaries(args.runs)
        table = compare_pareto(summaries, args.objectives) if args.objectives else compare_runs(summaries, args.metric)
        print(table)
        if args.markdown:
            args.markdown.write_text(table + "\n")
        if args.charts_dir:
            args.charts_dir.mkdir(parents=True, exist_ok=True)
            (args.charts_dir / "loss_curves.svg").write_text(loss_curves_svg(summaries))
            values = {s["run_id"]: dig(s, args.metric) for s in summaries if isinstance(dig(s, args.metric), (int, float))}
            (args.charts_dir / "metric_bars.svg").write_text(bar_chart_svg(values, args.metric))


if __name__ == "__main__":
    main()
