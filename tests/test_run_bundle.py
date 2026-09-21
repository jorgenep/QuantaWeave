import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import distill_quantweave_moe as distill
import finetune_quantweave_moe as ft
import run_bundle as rb
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from training_metrics import MetricsLogger

REPO = Path(__file__).parents[1]


def corpus(path: Path, rows: int = 60) -> Path:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(rows)))
    return path


def train(tmp_path: Path, data: Path, **overrides):
    values = dict(data=[data], steps=6, batch_size=2, sequence_length=16, examples=30, hidden_size=16, layers=2, ffn_size=24, total_experts=4,
                  active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, output=tmp_path / "out", checkpoint_dir=tmp_path / "ck")
    values.update(overrides)
    args = trainer.default_args(**values)
    return args, trainer.run_training(args)


def only_bundle(archive: Path) -> Path:
    bundles = [p for p in archive.iterdir() if p.is_dir()]
    assert len(bundles) == 1, bundles
    return bundles[0]


# ---- folder naming --------------------------------------------------------------------------------------------
def test_folder_is_named_by_epoch_seconds_and_never_overwrites(tmp_path):
    a = rb.epoch_folder(tmp_path / "runs", now=1789939200.9)
    assert a.name == "1789939200" and a.is_dir()
    assert rb.epoch_folder(tmp_path / "runs", now=1789939200).name == "1789939200-1"
    assert rb.epoch_folder(tmp_path / "runs", now=1789939200).name == "1789939200-2"
    assert rb.epoch_folder(tmp_path / "runs", now=1789939201).name == "1789939201"
    before = int(time.time())
    assert before <= int(rb.epoch_folder(tmp_path / "runs").name) <= before + 2


# ---- a training run produces the full archive -----------------------------------------------------------------
def test_training_run_is_archived_with_weights_data_benchmarks_and_more(tmp_path):
    data = corpus(tmp_path / "d.jsonl")
    before = int(time.time())
    args, summary = train(tmp_path, data, archive_dir=tmp_path / "archive")
    bundle = only_bundle(tmp_path / "archive")
    assert re.fullmatch(r"\d{10}", bundle.name) and before <= int(bundle.name) <= int(time.time()) + 1
    assert summary["archive"] == str(bundle)

    for name in ("README.md", "summary.json", "config.json", "reproduce.sh", "environment.json", "samples.txt", "model/model.pt", "model/config.json",
                 "model/vocab.json", "data/manifest.json", "data/d.jsonl", "benchmarks/train-slice.json", "benchmarks/held-out.json",
                 "benchmarks/train-slice.md", "benchmarks/heldout_sample.jsonl", "training/metrics.jsonl", "training/diagnostics/index.html"):
        assert (bundle / name).exists(), name
    assert not (bundle / "model" / "shards").exists()

    # the weights in the archive are the trained model
    config = QuantaWeaveConfig(**json.loads((bundle / "model" / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(torch.load(bundle / "model" / "model.pt", weights_only=False)["model"])
    original = torch.load(args.output / "model.pt", weights_only=False)["model"]
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())

    # the data is a faithful copy, and the manifest matches it
    assert (bundle / "data" / "d.jsonl").read_bytes() == data.read_bytes()
    manifest = json.loads((bundle / "data" / "manifest.json").read_text())
    entry = manifest["files"][0]
    assert manifest["copied"] and entry["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest() and entry["hash_coverage"] == "full"
    assert entry["rows"] == 60 and entry["rows_used_for_training"] == 30 and entry["copied_as"] == "data/d.jsonl"

    # the held-out benchmark uses rows training never read
    seen = {json.loads(l)["text"] for l in data.read_text().splitlines()[:30]}
    held = [json.loads(l)["text"] for l in (bundle / "benchmarks" / "heldout_sample.jsonl").read_text().splitlines()]
    assert len(held) == 30 and not seen & set(held)
    record = json.loads((bundle / "summary.json").read_text())
    assert record["warnings"] == [] and record["kind"] == "train" and record["archived_epoch"] == int(bundle.name)
    assert {"train-slice", "held-out", "generalisation_gap"} <= set(record["benchmarks"])
    assert record["benchmarks"]["held-out"]["loss"] > 0 and record["held_out"]["rows"] == 30
    assert record["training_finished_epoch"] <= record["archived_epoch"] + 1 and record["duration_seconds"] > 0

    readme = (bundle / "README.md").read_text()
    for needle in ("held-out loss", "generalisation gap", "generate_quantweave_moe.py", str(bundle / "model"), "Reproduce", "not answer questions"):
        assert needle in readme, needle
    assert json.loads((bundle / "environment.json").read_text())["torch"] == torch.__version__
    assert json.loads((bundle / "config.json").read_text())["hidden_size"] == 16
    assert "=== prompt: 'Once upon a time'" in (bundle / "samples.txt").read_text()
    lines = [json.loads(l) for l in (bundle / "training" / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 6


def test_final_loss_is_the_loss_at_the_end_even_with_sparse_logging(tmp_path):
    logger = MetricsLogger(None)
    for step, loss in ((20, 5.0), (40, 4.0), (60, 3.0)):
        logger.log(step, 1, loss=loss)
    assert logger.summary()["final_loss"] == 3.0                                 # not the mean of 5, 4 and 3
    dense = MetricsLogger(None)
    for step in range(1, 41):
        dense.log(step, 1, loss=float(41 - step))
    assert dense.summary()["final_loss"] == pytest.approx(sum(range(1, 21)) / 20)  # the last 20 steps


def test_archive_is_off_by_default_in_the_library_and_leaves_the_callers_args_untouched(tmp_path):
    data = corpus(tmp_path / "d.jsonl")
    train(tmp_path, data)                                                        # no archive_dir
    train(tmp_path / "b", data, archive_dir=tmp_path / "x", no_archive=True) if (tmp_path / "b").mkdir() is None else None
    assert not (tmp_path / "x").exists() and not (tmp_path / "archive").exists()
    args, _ = train(tmp_path / "c", data, archive_dir=tmp_path / "archive") if (tmp_path / "c").mkdir() is None else (None, None)
    assert args.metrics_file is None and args.diagnostics_dir is None and args.diagnostics_interval == 0
    leftovers = [p for p in Path(os.environ.get("TMPDIR", "/tmp")).glob("diagnostics-*")]
    assert not leftovers, "temporary diagnostics must be cleaned up"


# ---- the reproduce script really reproduces the run -------------------------------------------------------------
def test_reproduce_script_recreates_the_exact_weights(tmp_path):
    data = corpus(tmp_path / "d.jsonl")
    args, _ = train(tmp_path, data, archive_dir=tmp_path / "archive", lr=2e-3, curriculum="entropy", curriculum_steps=6, capacity_factor=0.7)
    bundle = only_bundle(tmp_path / "archive")
    script = (bundle / "reproduce.sh").read_text()
    assert "--lr 0.002" in script and "--seed 0" in script and "--curriculum entropy" in script      # every option, defaults included
    assert "--archive-dir" not in script and "--no-archive" not in script      # running it archives the re-run by default

    rerun_out = tmp_path / "rerun-out"
    command = script.replace(str(args.output), str(rerun_out)).replace(str(args.checkpoint_dir), str(tmp_path / "rerun-ck"))
    command = command.replace("python src/", f"{sys.executable} src/").rstrip("\n") + " --no-archive\n"
    (tmp_path / "reproduce.sh").write_text(command)
    subprocess.run(["bash", str(tmp_path / "reproduce.sh")], cwd=REPO, check=True, capture_output=True, env={**os.environ, "PYTHONHASHSEED": "0"})
    a = torch.load(bundle / "model" / "model.pt", weights_only=False)["model"]
    b = torch.load(rerun_out / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_reproduce_command_formatting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="a")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--flag", action="store_true")
    parser.add_argument("--maybe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paths", nargs="+", default=["x"])
    parser.add_argument("--skipme", default="s")
    args = parser.parse_args(["--name", "has space", "--flag", "--no-maybe", "--paths", "a b", "c", "--skipme", "zzz"])
    short = rb.reproduce_command("t.py", parser, args, skip=("skipme",))
    assert "--name 'has space'" in short and "--flag" in short and "--no-maybe" in short and "--paths 'a b' c" in short
    assert "--count" not in short and "--skipme" not in short                       # defaults and skipped options are left out
    full = rb.reproduce_command("t.py", parser, args, skip=("skipme",), explicit=True)
    assert "--count 1" in full and "--no-maybe" in full and "--skipme" not in full
    required = argparse.ArgumentParser()
    required.add_argument("--must", required=True)
    assert "--must v" in rb.reproduce_command("r.py", required, required.parse_args(["--must", "v"]))


# ---- archiving never fails the run ----------------------------------------------------------------------------------
def test_failures_while_archiving_become_warnings_not_errors(tmp_path, monkeypatch):
    import benchmark_quantweave_moe as bench
    data = corpus(tmp_path / "d.jsonl")
    monkeypatch.setattr(bench, "run_benchmark", lambda args: (_ for _ in ()).throw(RuntimeError("benchmark exploded")))
    monkeypatch.setattr(rb, "write_samples", lambda *a, **k: (_ for _ in ()).throw(ValueError("no samples today")))
    _, summary = train(tmp_path, data, archive_dir=tmp_path / "archive")
    assert summary["steps"] == 6                                                     # training result is intact
    bundle = only_bundle(tmp_path / "archive")
    record = json.loads((bundle / "summary.json").read_text())
    assert any("benchmark 'train-slice' failed" in w and "exploded" in w for w in record["warnings"])
    assert any("samples failed" in w for w in record["warnings"])
    assert (bundle / "model" / "model.pt").exists() and (bundle / "data" / "d.jsonl").exists() and (bundle / "README.md").exists()
    assert (bundle / "benchmarks" / "train-slice.error.txt").exists()
    assert "## Warnings" in (bundle / "README.md").read_text()


def test_large_data_is_recorded_not_copied_and_huge_files_get_a_partial_hash(tmp_path, monkeypatch):
    data = corpus(tmp_path / "d.jsonl")
    train(tmp_path, data, archive_dir=tmp_path / "a1", archive_data_limit_mb=0)
    b1 = only_bundle(tmp_path / "a1")
    manifest = json.loads((b1 / "data" / "manifest.json").read_text())
    assert not manifest["copied"] and not (b1 / "data" / "d.jsonl").exists() and manifest["files"][0]["sha256"]
    assert any("exceeds the 0 MB limit" in w for w in json.loads((b1 / "summary.json").read_text())["warnings"])

    monkeypatch.setattr(rb, "FULL_HASH_LIMIT", 100)
    monkeypatch.setattr(rb, "PARTIAL_HASH_BYTES", 16)
    train(tmp_path / "p", data, archive_dir=tmp_path / "a2") if (tmp_path / "p").mkdir() is None else None
    entry = json.loads((only_bundle(tmp_path / "a2") / "data" / "manifest.json").read_text())["files"][0]
    assert entry["hash_coverage"] == "partial" and entry["rows"] is None


def test_without_unseen_rows_only_the_training_benchmark_runs_and_it_says_so(tmp_path):
    data = corpus(tmp_path / "d.jsonl", rows=20)
    train(tmp_path, data, examples=100, archive_dir=tmp_path / "archive")
    bundle = only_bundle(tmp_path / "archive")
    record = json.loads((bundle / "summary.json").read_text())
    assert "train-slice" in record["benchmarks"] and "held-out" not in record["benchmarks"] and record["held_out"] is None
    assert any("no unseen rows" in w for w in record["warnings"])


def test_packed_token_corpus_runs_are_archived_without_a_benchmark(tmp_path):
    pytest.importorskip("numpy")
    import data_pipeline as dp
    data = corpus(tmp_path / "d.jsonl")
    tokenizer = dp.CharTokenizer.build([json.loads(l)["text"] for l in data.read_text().splitlines()], 64)
    dp.write_token_bin([data], tokenizer, tmp_path / "packed")
    train(tmp_path, data, token_data=tmp_path / "packed", archive_dir=tmp_path / "archive")
    bundle = only_bundle(tmp_path / "archive")
    record = json.loads((bundle / "summary.json").read_text())
    assert (bundle / "model" / "model.pt").exists() and record["benchmarks"] == {}
    assert any("no benchmark" in w or "no data files" in w for w in record["warnings"])


# ---- command-line defaults ----------------------------------------------------------------------------------------------
def test_command_line_archives_into_artifacts_runs_unless_told_not_to(tmp_path, monkeypatch):
    data = corpus(tmp_path / "d.jsonl")
    monkeypatch.chdir(tmp_path)
    base = ["train", "--data", str(data), "--steps", "3", "--batch-size", "2", "--sequence-length", "16", "--examples", "30", "--hidden-size", "16",
            "--layers", "1", "--ffn-size", "24", "--total-experts", "4", "--active-experts", "2", "--vocab-size", "64", "--device", "cpu",
            "--checkpoint-interval", "0", "--output", str(tmp_path / "o1"), "--checkpoint-dir", str(tmp_path / "c1")]
    monkeypatch.setattr(sys, "argv", base)
    trainer.main()
    assert re.fullmatch(r"\d{10}(-\d+)?", only_bundle(tmp_path / "artifacts" / "runs").name)
    monkeypatch.setattr(sys, "argv", [*base[:-4], "--output", str(tmp_path / "o2"), "--checkpoint-dir", str(tmp_path / "c2"), "--no-archive"])
    trainer.main()
    assert len(list((tmp_path / "artifacts" / "runs").iterdir())) == 1


# ---- distillation and fine-tuning ------------------------------------------------------------------------------------------
def test_distillation_and_finetuning_runs_are_archived_too(tmp_path):
    data = corpus(tmp_path / "d.jsonl")
    args, _ = train(tmp_path, data, steps=8, capacity_factor=0)
    distilled = distill.run_distillation(distill.build_parser().parse_args([
        "--student", str(args.output), "--teacher-data", str(data), "--output", str(tmp_path / "dout"), "--checkpoint-dir", str(tmp_path / "dck"),
        "--steps", "3", "--sequence-length", "16", "--batch-size", "2", "--device", "cpu", "--checkpoint-interval", "0", "--examples", "30",
        "--archive-dir", str(tmp_path / "d-archive")]))
    bundle = only_bundle(tmp_path / "d-archive")
    record = json.loads((bundle / "summary.json").read_text())
    assert distilled["archive"] == str(bundle) and record["kind"] == "distill" and "held-out" in record["benchmarks"]
    assert "--teacher-data" in (bundle / "reproduce.sh").read_text()

    tuned = ft.run_finetune(ft.build_parser().parse_args([
        "--checkpoint", str(args.output), "--data", str(data), "--output", str(tmp_path / "fout"), "--merge-output", str(tmp_path / "fmerged"),
        "--steps", "4", "--batch-size", "4", "--sequence-length", "16", "--examples", "30", "--device", "cpu", "--bits", "8",
        "--archive-dir", str(tmp_path / "f-archive")]))
    bundle = only_bundle(tmp_path / "f-archive")
    record = json.loads((bundle / "summary.json").read_text())
    assert tuned["archive"] == str(bundle) and record["kind"] == "finetune"
    assert (bundle / "adapter" / "adapter.pt").exists() and (bundle / "merged" / "model.pt").exists() and "held-out" in record["benchmarks"]

    adapter_only = ft.run_finetune(ft.build_parser().parse_args([
        "--checkpoint", str(args.output), "--data", str(data), "--output", str(tmp_path / "gout"), "--steps", "2", "--batch-size", "4",
        "--sequence-length", "16", "--examples", "30", "--device", "cpu", "--archive-dir", str(tmp_path / "g-archive")]))
    warnings = json.loads((only_bundle(tmp_path / "g-archive") / "summary.json").read_text())["warnings"]
    assert adapter_only["archive"] and any("not a standalone checkpoint" in w for w in warnings)


# ---- multi-process runs archive once ------------------------------------------------------------------------------------------
def test_multi_process_run_creates_exactly_one_archive_with_the_consolidated_model(tmp_path):
    import test_parallelism as par
    par.write_corpus(tmp_path / "d.jsonl")
    par.launch(par.training_worker, 2, str(tmp_path), "run", dict(expert_parallel=True, steps=3, examples=30, archive_dir=tmp_path / "archive"))
    bundle = only_bundle(tmp_path / "archive")
    assert (bundle / "model" / "model.pt").exists() and not (bundle / "model" / "shards").exists()
    record = json.loads((bundle / "summary.json").read_text())
    assert "held-out" in record["benchmarks"] and (bundle / "training" / "metrics.jsonl").exists()
