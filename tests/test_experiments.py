import json
import sys
import xml.dom.minidom
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import experiment_manager as em
import sweep


def write_corpus(path: Path) -> Path:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(50)))
    return path


def base_config(data: Path, **train) -> dict:
    values = dict(data=[str(data)], steps=4, batch_size=2, sequence_length=16, examples=50, hidden_size=16, layers=1, ffn_size=24,
                  total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0)
    values.update(train)
    return {"name": "Tiny Run", "description": "unit test run", "train": values, "benchmark": {"examples": 20}}


def test_run_creates_a_complete_tracked_directory_and_best_pointer(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    summary = em.run_experiment(base_config(data), tmp_path / "runs")
    run_dir = Path(summary["run_dir"])
    assert summary["status"] == "completed" and summary["run_id"].endswith(run_dir.name[-6:])
    for name in ("config.json", "metrics.jsonl", "train.log", "summary.json", "report.md", "benchmark.json", "benchmark.md", "diagnostics/index.html", "model/model.pt"):
        assert (run_dir / name).exists(), name
    assert "step=4/4" in (run_dir / "train.log").read_text()
    config = json.loads((run_dir / "config.json").read_text())
    assert config["train"]["steps"] == 4 and config["name"] == "Tiny Run" and "created" in config
    assert summary["benchmark"]["loss"] > 0 and summary["train"]["final_loss"] > 0 and summary["is_best"] is True
    best = json.loads((tmp_path / "runs" / "best.json").read_text())
    assert best["run_id"] == summary["run_id"] and best["objective"] == "benchmark.loss"
    assert "unit test run" in (run_dir / "report.md").read_text()


def test_best_pointer_only_moves_to_a_better_run(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    runs = tmp_path / "runs"
    good = em.run_experiment(base_config(data, steps=30, lr=3e-3), runs, run_id="good")
    worse = em.run_experiment(base_config(data, steps=1), runs, run_id="worse")
    assert good["benchmark"]["loss"] < worse["benchmark"]["loss"] and worse["is_best"] is False
    assert json.loads((runs / "best.json").read_text())["run_id"] == "good"


def test_failed_runs_are_recorded_not_raised_and_bad_options_are_rejected_early(tmp_path):
    summary = em.run_experiment(base_config(tmp_path / "missing.jsonl"), tmp_path / "runs")
    assert summary["status"] == "failed" and "FileNotFoundError" in summary["error"]
    assert (Path(summary["run_dir"]) / "error.txt").exists() and not (tmp_path / "runs" / "best.json").exists()
    with pytest.raises(TypeError, match="unknown training option"):
        em.run_experiment(base_config(tmp_path / "d.jsonl", not_an_option=1), tmp_path / "runs2")
    assert not (tmp_path / "runs2").exists()


def test_yaml_and_json_configs_load(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "a.json").write_text(json.dumps({"name": "j", "train": {"steps": 3}}))
    (tmp_path / "a.yaml").write_text("name: y\ntrain:\n  steps: 3\n  capacity_adapt: true\n")
    assert em.load_config(tmp_path / "a.json")["name"] == "j"
    assert em.load_config(tmp_path / "a.yaml")["train"] == {"steps": 3, "capacity_adapt": True}


def test_compare_ranks_runs_shows_only_differing_options_and_draws_charts(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    runs = tmp_path / "runs"
    a = em.run_experiment(base_config(data, steps=20, lr=3e-3), runs, run_id="a")
    b = em.run_experiment(base_config(data, steps=2, lr=3e-3), runs, run_id="b")
    summaries = em.load_summaries([Path(a["run_dir"]), Path(b["run_dir"])])
    table = em.compare_runs(summaries)
    rows = [line for line in table.splitlines()[2:]]
    assert rows[0].startswith("| a ") and rows[1].startswith("| b ")          # ranked best first
    header = table.splitlines()[0]
    assert "steps" in header and "lr" not in header and "hidden_size" not in header   # only options that differ
    for svg in (em.loss_curves_svg(summaries), em.bar_chart_svg({"a": 1.0, "b": 2.0}, "loss")):
        xml.dom.minidom.parseString(svg)


def test_cli_run_list_and_compare(tmp_path, capsys, monkeypatch):
    data = write_corpus(tmp_path / "d.jsonl")
    config_path = tmp_path / "exp.json"
    config_path.write_text(json.dumps(base_config(data)))
    runs = tmp_path / "runs"
    for argv in (["em", "run", "--config", str(config_path), "--runs-dir", str(runs)],):
        monkeypatch.setattr(sys, "argv", argv)
        em.main()
    run_dir = next(p for p in runs.iterdir() if p.is_dir())
    monkeypatch.setattr(sys, "argv", ["em", "list", "--runs-dir", str(runs)])
    em.main()
    assert "best:" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["em", "compare", str(run_dir), "--charts-dir", str(tmp_path / "charts"), "--markdown", str(tmp_path / "c.md")])
    em.main()
    assert (tmp_path / "charts" / "loss_curves.svg").exists() and (tmp_path / "c.md").read_text().startswith("| run")


# ---- sweeps ---------------------------------------------------------------------------------
def test_grid_expansion_and_random_sampling():
    grid = sweep.expand_grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(grid) == 6 and {"a": 2, "b": 5} in grid
    with pytest.raises(ValueError, match="list"):
        sweep.expand_grid({"a": {"uniform": [0, 1]}})

    space = {"lr": {"loguniform": [1e-4, 1e-3]}, "k": [1, 2], "x": {"uniform": [0, 1]}, "n": {"int": [2, 5]}}
    first, again = sweep.sample_random(space, 6, seed=1), sweep.sample_random(space, 6, seed=1)
    assert first == again and len(first) == 6 and first != sweep.sample_random(space, 6, seed=2)
    assert all(1e-4 <= t["lr"] <= 1e-3 and t["k"] in (1, 2) and 0 <= t["x"] <= 1 and 2 <= t["n"] <= 5 for t in first)
    assert len(sweep.sample_random({"k": [1, 2]}, 10, seed=0)) == 2            # small space: no duplicates
    with pytest.raises(ValueError, match="unsupported"):
        sweep.sample_value({"weird": [1, 2]}, __import__("random").Random(0))


def test_invalid_combinations_are_named():
    assert "exceeds" in sweep.invalid_reason({"total_experts": 2, "active_experts": 4})
    assert "divisible" in sweep.invalid_reason({"hidden_size": 30, "attention_heads": 4})
    assert sweep.invalid_reason({"hidden_size": 32, "total_experts": 8, "active_experts": 2}) is None


def sweep_config(data: Path, **extra) -> dict:
    return {"name": "tiny-sweep", "base": base_config(data)["train"], "benchmark": {"examples": 20, "no_routing_stats": True},
            "metric": "benchmark.loss", **extra}


def test_dry_run_plans_without_training(tmp_path):
    config = sweep_config(write_corpus(tmp_path / "d.jsonl"), space={"total_experts": [2, 4], "active_experts": [1, 4]})
    plan = sweep.run_sweep(config, tmp_path / "runs", dry_run=True)
    assert plan["dry_run"] and plan["candidates"] == 4 and plan["runnable"] == 3
    assert plan["skipped"][0]["trial"] == {"total_experts": 2, "active_experts": 4}
    assert not (tmp_path / "runs").exists()


def test_grid_sweep_runs_ranks_and_skips_invalid_trials(tmp_path):
    config = sweep_config(write_corpus(tmp_path / "d.jsonl"), space={"lr": [3e-3, 1e-6], "total_experts": [2, 4]})
    outcome = sweep.run_sweep(config, tmp_path / "runs")
    assert len(outcome["ranking"]) == 4 and outcome["best"]["value"] == outcome["ranking"][0]["value"]
    assert outcome["best"]["trial"]["lr"] == 3e-3                                # a real learning rate beats 1e-6
    values = [r["value"] for r in outcome["ranking"]]
    assert values == sorted(values)
    sweep_dir = Path(outcome["sweep_dir"])
    assert (sweep_dir / "sweep_results.json").exists() and "Sweep tiny-sweep" in (sweep_dir / "sweep_results.md").read_text()
    assert (sweep_dir / "best.json").exists()


def test_successive_halving_keeps_the_best_and_grows_the_budget(tmp_path):
    config = sweep_config(write_corpus(tmp_path / "d.jsonl"), strategy="halving", n_trials=4, seed=3,
                          space={"lr": {"loguniform": [1e-6, 1e-2]}}, halving={"eta": 2, "rungs": 3})
    config["base"]["steps"] = 2
    outcome = sweep.run_sweep(config, tmp_path / "runs")
    assert [r["trials"] for r in outcome["rungs"]] == [4, 2, 1]
    assert [r["steps"] for r in outcome["rungs"]] == [2, 4, 8]
    assert len(outcome["ranking"]) == 1 and outcome["best"]["run_id"].startswith("trial-r2")
    with pytest.raises(ValueError, match="strategy"):
        sweep.run_sweep({**config, "strategy": "annealing"}, tmp_path / "runs2")
