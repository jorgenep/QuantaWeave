"""Archive a finished run into its own folder named by epoch time.

    artifacts/runs/1789939200/
        README.md            what is here, headline results, commands to reproduce and reuse
        summary.json         machine-readable results, timings, warnings
        config.json          every option the run used (+ reproduce.sh, the command line that recreates it)
        environment.json     python/torch versions, git revision, device, platform
        model/               weights, config, tokenizer, checkpoint metadata (the run's output directory)
        data/                the training data (copied when small) and manifest.json (paths, sizes, hashes, rows used)
        benchmarks/          benchmark on rows the model trained on, and on rows it never saw (+ the sample used)
        training/            metrics.jsonl and routing diagnostics
        samples.txt          a few generations from the finished model

The folder name is the epoch second the run finished. A second run finishing in the same second gets a "-1"
suffix rather than overwriting. Archiving never fails a run: anything that cannot be produced (a benchmark that
errors, data that is too large to copy) is recorded under "warnings" in summary.json.

Held-out rows: training reads the first ``examples`` rows of each file, so the rows after them were never seen. The
benchmark uses those, which gives an honest generalisation number without a separate validation split.
"""

import argparse
import hashlib
import json
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch

DEFAULT_PROMPTS = ("Once upon a time", "The ")
PARTIAL_HASH_BYTES = 16 << 20
FULL_HASH_LIMIT = 512 << 20


def add_archive_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("run archive")
    group.add_argument("--archive-dir", type=Path, help="save weights, data, benchmarks and more in ARCHIVE_DIR/<epoch seconds>/ "
                                                        "(the command line uses artifacts/runs unless --no-archive)")
    group.add_argument("--no-archive", action="store_true", help="do not create a run archive")
    group.add_argument("--archive-data-limit-mb", type=float, default=200.0, help="copy the data into the archive if it is at most this big (0 = never)")
    group.add_argument("--archive-benchmark-examples", type=int, default=500, help="rows per benchmark in the archive")


def resolve_archive_dir(args: argparse.Namespace, default: Path = Path("artifacts/runs")) -> None:
    """For command-line entry points: archive by default, unless --no-archive."""
    if args.no_archive:
        args.archive_dir = None
    elif args.archive_dir is None:
        args.archive_dir = default


def epoch_folder(root: Path, now: Optional[float] = None) -> Path:
    """Create root/<epoch seconds>, adding -1, -2, ... if that second is already taken."""
    root.mkdir(parents=True, exist_ok=True)
    epoch = int(time.time() if now is None else now)
    candidate = root / str(epoch)
    suffix = 0
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = root / f"{epoch}-{suffix}"


def _sha256(path: Path) -> tuple[str, str]:
    """(hex digest, "full" | "partial"). Files over 512 MB hash their first and last 16 MB plus the size."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    if size <= FULL_HASH_LIMIT:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest(), "full"
    with path.open("rb") as handle:
        digest.update(handle.read(PARTIAL_HASH_BYTES))
        handle.seek(max(0, size - PARTIAL_HASH_BYTES))
        digest.update(handle.read(PARTIAL_HASH_BYTES))
    digest.update(str(size).encode())
    return digest.hexdigest(), "partial"


def _count_lines(path: Path) -> Optional[int]:
    if path.stat().st_size > FULL_HASH_LIMIT:
        return None                                        # not worth reading gigabytes just to count rows
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def archive_data(data_paths: Sequence[Path], bundle: Path, limit_mb: float, rows_used: Optional[int]) -> dict:
    """Copy the training files when their total size is within ``limit_mb``; always write a manifest."""
    files = [Path(p) for p in data_paths if Path(p).is_file()]
    total = sum(p.stat().st_size for p in files)
    copy = bool(files) and limit_mb > 0 and total <= limit_mb * (1 << 20)
    entries = []
    target = bundle / "data"
    target.mkdir(exist_ok=True)
    for index, path in enumerate(files):
        digest, coverage = _sha256(path)
        entry = {
            "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest, "hash_coverage": coverage,
            "rows": _count_lines(path), "rows_used_for_training": rows_used,
        }
        if copy:
            copy_name = path.name if len({p.name for p in files}) == len(files) else f"{index}-{path.name}"
            shutil.copy2(path, target / copy_name)
            entry["copied_as"] = f"data/{copy_name}"
        entries.append(entry)
    manifest = {
        "copied": copy, "total_bytes": total, "limit_mb": limit_mb, "files": entries,
        "note": None if copy else (
            "data was not copied: " + ("no data files were found" if not files else f"{total / (1 << 20):.0f} MB exceeds the {limit_mb:g} MB limit")
            + "; the manifest records the sources and hashes so they can be matched to this run"),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_held_out_sample(data_paths: Sequence[Path], skip_rows: int, count: int, destination: Path) -> Optional[dict]:
    """Rows that come after the ones training read (the first ``skip_rows`` of each file). None if there are none."""
    files = [Path(p) for p in data_paths if Path(p).is_file()]
    if not files or skip_rows is None:
        return None
    share = max(1, count // len(files))
    rows: list[str] = []
    for path in files:
        taken = 0
        with path.open(encoding="utf-8") as lines:
            for number, line in enumerate(lines):
                if number < skip_rows:
                    continue
                if line.strip():
                    rows.append(line.rstrip("\n"))
                    taken += 1
                    if taken >= share:
                        break
    if not rows:
        return None
    destination.write_text("\n".join(rows) + "\n")
    return {"rows": len(rows), "skipped_rows_per_file": skip_rows, "file": destination.name}


def reproduce_command(script: str, parser: argparse.ArgumentParser, args: argparse.Namespace, skip: Sequence[str] = (), explicit: bool = False) -> str:
    """The command line that recreates ``args``.

    By default only options that differ from the parser's defaults are listed (readable). ``explicit`` lists every option,
    so the command keeps meaning the same thing even if a default changes later."""
    defaults = None if explicit or any(a.required for a in parser._actions) else parser.parse_args([])
    parts = [f"python src/{script}"]
    for action in parser._actions:
        if not action.option_strings or action.dest in ("help", *skip):
            continue
        value = getattr(args, action.dest, None)
        if defaults is not None and value == getattr(defaults, action.dest, None):
            continue
        if value is None:
            continue
        if isinstance(action, argparse.BooleanOptionalAction):
            parts.append(action.option_strings[0] if value else action.option_strings[1])
        elif action.nargs == 0:                                     # store_true / store_false
            if value != action.default:
                parts.append(action.option_strings[-1])
        elif isinstance(value, (list, tuple)):
            parts.append(action.option_strings[-1] + " " + " ".join(shlex.quote(str(v)) for v in value))
        else:
            parts.append(f"{action.option_strings[-1]} {shlex.quote(str(value))}")
    return " \\\n  ".join(parts)


def environment_info(device_arg: str) -> dict:
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(), "argv": sys.argv}
    try:
        info["git_revision"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
                                              cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        info["git_revision"] = None
    try:
        from dataclasses import asdict

        from hardware import detect

        info["device"] = asdict(detect(device_arg if device_arg in {"auto", "cuda", "xpu", "cpu", "rocm"} else "auto"))
    except Exception as error:
        info["device"] = f"unavailable: {error}"
    return info


def write_samples(model_dir: Path, destination: Path, prompts: Iterable[str], device_arg: str, tokens: int = 200, temperature: float = 0.8) -> None:
    from data_pipeline import load_tokenizer
    from generate_quantweave_moe import sample_text
    from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
    from train_quantweave_moe import select_device

    device = select_device(device_arg)
    config = QuantaWeaveConfig(**json.loads((model_dir / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    tokenizer = load_tokenizer(model_dir)
    torch.manual_seed(0)
    blocks = [f"temperature {temperature}, up to {tokens} tokens; sampling seed 0\n"]
    for prompt in prompts:
        blocks.append(f"=== prompt: {prompt!r}\n{sample_text(model, tokenizer, prompt, tokens, temperature, device)}\n")
    destination.write_text("\n".join(blocks))


def run_benchmarks(model_dir: Path, bundle: Path, data_paths: Sequence[Path], held_out: Optional[dict], metrics_file: Optional[Path],
                   examples: int, device: str, precision: str, warnings: list[str]) -> dict:
    from benchmark_quantweave_moe import build_parser, render_markdown, run_benchmark

    out = bundle / "benchmarks"
    results: dict = {}
    jobs = []
    if data_paths and Path(data_paths[0]).is_file():
        jobs.append(("train-slice", Path(data_paths[0]), "rows the model trained on"))
    if held_out is not None:
        jobs.append(("held-out", out / held_out["file"], "rows the model never saw"))
    else:
        warnings.append("no unseen rows were available for a held-out benchmark (the training run read every row)")
    for name, data, description in jobs:
        try:
            argv = ["--checkpoint", str(model_dir), "--data", str(data), "--examples", str(examples), "--batch-size", "16",
                    "--device", device, "--precision", precision]
            if name == "train-slice" and metrics_file is not None and metrics_file.exists():
                argv += ["--training-metrics", str(metrics_file)]
            report = run_benchmark(build_parser().parse_args(argv))
            (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
            (out / f"{name}.md").write_text(f"_{description}_\n\n" + render_markdown(report))
            results[name] = {key: report[key] for key in ("loss", "perplexity", "tokens_per_second", "tokens", "dropped_route_fraction",
                                                          "overflow_route_fraction", "expert_utilization_min", "expert_utilization_max")}
            results[name]["what"] = description
            if "routing" in report:
                results[name]["dead_experts"] = report["routing"]["dead_experts_total"]
        except Exception as error:
            warnings.append(f"benchmark '{name}' failed: {type(error).__name__}: {error}")
            (out / f"{name}.error.txt").write_text(traceback.format_exc())
    if "train-slice" in results and "held-out" in results:
        results["generalisation_gap"] = results["held-out"]["loss"] - results["train-slice"]["loss"]
    return results


def _render_readme(bundle: Path, kind: str, epoch: int, summary: dict, benchmarks: dict, manifest: Optional[dict], reproduce: Optional[str],
                   warnings: list[str], model_names: Sequence[str]) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(epoch))
    model = bundle.name
    lines = [f"# {kind} run {model}", "", f"Archived {when} (epoch {epoch}). The run itself took {summary.get('duration_seconds', 0):.0f} s.", ""]
    rows = []
    for key, label in (("final_loss", "final training loss"), ("best_loss", "best training loss"), ("steps", "steps"), ("tokens_per_second", "training tokens/s"),
                       ("active_parameter_ratio", "active / total parameters")):
        value = summary.get("train", {}).get(key)
        if value is not None:
            rows.append(f"| {label} | {value:.4g} |" if isinstance(value, float) else f"| {label} | {value} |")
    for name in ("train-slice", "held-out"):
        if name in benchmarks:
            rows.append(f"| {name} loss (perplexity) | {benchmarks[name]['loss']:.4f} ({benchmarks[name]['perplexity']:.2f}) — {benchmarks[name]['what']} |")
    if "generalisation_gap" in benchmarks:
        rows.append(f"| generalisation gap (held-out − train) | {benchmarks['generalisation_gap']:+.4f} |")
    if rows:
        lines += ["| result | value |", "|---|---|"] + rows + [""]
    if warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in warnings] + [""]
    lines += ["## Contents", ""]
    lines += [f"- `{name}/` — weights, config, tokenizer" for name in model_names]
    lines += ["- `data/` — the training data" + (" (copied)" if manifest and manifest["copied"] else " (not copied; `manifest.json` records the sources and hashes)"),
              "- `benchmarks/` — `train-slice` (rows it trained on) and `held-out` (rows it never saw), each as JSON and Markdown",
              "- `training/` — `metrics.jsonl` and routing diagnostics (`diagnostics/index.html`)",
              "- `config.json`, `reproduce.sh`, `environment.json`, `summary.json`, `samples.txt`", ""]
    if kind == "train":
        lines += ["## Use this model", "", "```bash", f"cd {Path.cwd()}",
                  f"# text completion (a base model continues text; it does not answer questions)",
                  f".axolotl-venv/bin/python src/generate_quantweave_moe.py --checkpoint {bundle}/model --prompt \"Once upon a time\"",
                  f".axolotl-venv/bin/python src/benchmark_quantweave_moe.py --checkpoint {bundle}/model --data <jsonl> --output /tmp/bench.json",
                  f".axolotl-venv/bin/python src/export_quantweave.py --checkpoint {bundle}/model --output {bundle}/export --quantize 8",
                  f".axolotl-venv/bin/python src/finetune_quantweave_moe.py --checkpoint {bundle}/model --data <jsonl> --output {bundle}/qlora",
                  "```", ""]
    if reproduce:
        lines += ["## Reproduce", "", "```bash", reproduce, "```", ""]
    return "\n".join(lines)


def create_run_bundle(
    root: Path,
    *,
    kind: str,
    model_dirs: dict[str, Path],
    benchmark_model: Optional[Path],
    data_paths: Sequence[Path],
    options: dict,
    summary: dict,
    started: float,
    reproduce: Optional[str] = None,
    reproduce_full: Optional[str] = None,
    finished: Optional[float] = None,
    metrics_file: Optional[Path] = None,
    diagnostics_dir: Optional[Path] = None,
    skip_rows: Optional[int] = None,
    benchmark_examples: int = 500,
    device: str = "auto",
    precision: str = "auto",
    data_limit_mb: float = 200.0,
    sample_prompts: Optional[Sequence[str]] = None,
    now: Optional[float] = None,
) -> Path:
    """Write the archive folder and return its path. ``model_dirs`` maps folder name -> source directory to copy."""
    warnings: list[str] = []
    bundle = epoch_folder(Path(root), now)
    epoch = int(bundle.name.split("-")[0])
    for sub in ("benchmarks", "training"):
        (bundle / sub).mkdir(exist_ok=True)

    def attempt(label: str, action):
        try:
            return action()
        except Exception as error:
            warnings.append(f"{label} failed: {type(error).__name__}: {error}")
            (bundle / f"{label.replace(' ', '_')}.error.txt").write_text(traceback.format_exc())
            return None

    for name, source in model_dirs.items():
        attempt(f"copy {name}", lambda name=name, source=source: shutil.copytree(
            source, bundle / name, ignore=shutil.ignore_patterns("shards", ".*.tmp", ".*.previous")))
    manifest = attempt("archive data", lambda: archive_data(data_paths, bundle, data_limit_mb, skip_rows))
    if manifest and not manifest["copied"] and manifest["note"]:
        warnings.append(manifest["note"])

    held_out = attempt("held-out sample", lambda: write_held_out_sample(data_paths, skip_rows, benchmark_examples, bundle / "benchmarks" / "heldout_sample.jsonl"))
    benchmarks: dict = {}
    if benchmark_model is not None:
        benchmarks = attempt("benchmarks", lambda: run_benchmarks(benchmark_model, bundle, data_paths, held_out, metrics_file, benchmark_examples,
                                                                  device, precision, warnings)) or {}
    else:
        warnings.append("no benchmark was run for this output (it is not a standalone checkpoint)")

    if metrics_file is not None and Path(metrics_file).exists():
        shutil.copy2(metrics_file, bundle / "training" / "metrics.jsonl")
    if diagnostics_dir is not None and Path(diagnostics_dir).exists():
        shutil.copytree(diagnostics_dir, bundle / "training" / "diagnostics", dirs_exist_ok=True)

    if benchmark_model is not None and (benchmark_model / "model.pt").exists():
        prompts = list(sample_prompts or DEFAULT_PROMPTS)
        attempt("samples", lambda: write_samples(benchmark_model, bundle / "samples.txt", prompts, device))

    finished = time.time() if finished is None else finished           # when training ended, before archiving work
    summary = {**summary, "duration_seconds": finished - started}
    (bundle / "config.json").write_text(json.dumps(options, indent=2, default=str) + "\n")
    if reproduce or reproduce_full:
        (bundle / "reproduce.sh").write_text("#!/usr/bin/env bash\n# every option is listed, so this stays correct if defaults change\nset -euo pipefail\n"
                                              + (reproduce_full or reproduce) + "\n")
    (bundle / "environment.json").write_text(json.dumps(environment_info(device), indent=2, default=str) + "\n")
    record = {
        "kind": kind, "epoch": epoch, "folder": bundle.name, "started_epoch": int(started), "training_finished_epoch": int(finished), "archived_epoch": epoch,
        "duration_seconds": summary["duration_seconds"], "results": summary, "benchmarks": benchmarks,
        "data": {"copied": bool(manifest and manifest["copied"]), "total_bytes": manifest["total_bytes"] if manifest else None},
        "held_out": held_out, "warnings": warnings,
    }
    (bundle / "summary.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
    readme_summary = {"duration_seconds": summary["duration_seconds"], "train": summary}
    (bundle / "README.md").write_text(_render_readme(bundle, kind, epoch, readme_summary, benchmarks, manifest, reproduce, warnings, list(model_dirs)))
    return bundle
