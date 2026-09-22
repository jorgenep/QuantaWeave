"""Hyperparameter sweeps over MoE settings, run through the experiment manager.

  python src/sweep.py --config configs/sweep_example.yaml [--dry-run]

  name: moe-sweep
  strategy: grid            # grid | random | halving | bayes | bohb
  base: {steps: 200, examples: 2000, sequence_length: 64}     # train options shared by every trial
  benchmark: {examples: 300}
  metric: benchmark.loss    # lower is better
  space:
    hidden_size: [64, 128]
    total_experts: [8, 16]
    active_experts: [1, 2]
    capacity_factor: [1.0, 1.5]
    lr: {loguniform: [1.0e-4, 1.0e-3]}      # random/halving also accept uniform, int, and lists (choice)
  n_trials: 8               # random / halving
  halving: {eta: 2, rungs: 3}    # halving: trials that finish in the top 1/eta continue with eta x the steps
  bayes: {n_init: 5}             # bayes: random trials first, then Gaussian-process expected improvement
  seed: 0
# bohb: like halving, but a Gaussian process (not random sampling) suggests the base-rung configs, so search effort
# concentrates on promising regions even before any config reaches a higher, more expensive rung.

Invalid combinations (more active than total experts, a hidden size the attention heads do not divide) are skipped and
listed (bayes/bohb never propose them). `bayes` fits a Gaussian process to the trials so far (bayes_opt.py) and runs
the point with the highest expected improvement next, so it is sequential; n_trials is the total budget.

`bohb` is a real but scoped multi-fidelity search: it is `halving`'s rung/promotion structure (successive halving
survivors advance to eta x the steps, same as `halving`), but the *base rung's* configs are chosen by the same
Gaussian-process expected-improvement rule as `bayes` (fit on base-rung outcomes only) instead of drawn at random.
This is not full BOHB (Falkner et al. 2018): that fits a separate density model per fidelity level and mixes them;
here a single GP guides only the cheapest rung, and successive halving (not the model) decides who is promoted.
"""

import argparse
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Optional

from bayes_opt import BayesianOptimizer, SearchSpace
from experiment_manager import DEFAULT_RUNS_DIR, compare_pareto, compare_runs, dig, load_config, load_summaries, pareto_front, parse_objectives, run_experiment


def expand_grid(space: dict) -> list[dict]:
    names = list(space)
    for name, options in space.items():
        if not isinstance(options, list):
            raise ValueError(f"grid search needs a list of values for '{name}'")
    return [dict(zip(names, values)) for values in itertools.product(*(space[n] for n in names))]


def sample_value(spec, rng: random.Random):
    if isinstance(spec, list):
        return rng.choice(spec)
    if isinstance(spec, dict) and len(spec) == 1:
        kind, (low, high) = next(iter(spec.items()))
        if kind == "loguniform":
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        if kind == "uniform":
            return rng.uniform(low, high)
        if kind == "int":
            return rng.randint(int(low), int(high))
    raise ValueError(f"unsupported search-space entry {spec!r}; use a list, loguniform, uniform or int")


def sample_random(space: dict, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    trials, seen = [], set()
    for _ in range(count * 20):                     # avoid returning duplicates when the space is small
        trial = {name: sample_value(spec, rng) for name, spec in space.items()}
        key = json.dumps(trial, sort_keys=True)
        if key not in seen:
            seen.add(key)
            trials.append(trial)
        if len(trials) == count:
            break
    return trials


def invalid_reason(train: dict) -> Optional[str]:
    experts, active = train.get("total_experts", 184), train.get("active_experts", 1)
    if active > experts:
        return f"active_experts {active} exceeds total_experts {experts}"
    hidden, heads = train.get("hidden_size", 64), train.get("attention_heads", 4)
    if hidden % heads:
        return f"hidden_size {hidden} is not divisible by attention_heads {heads}"
    return None


def make_experiment(config: dict, sweep_name: str, trial: dict, steps: Optional[int] = None, label: str = "") -> dict:
    train = {**config.get("base", {}), **trial}
    if steps is not None:
        train["steps"] = steps
    return {
        "name": f"{sweep_name}{label}", "train": train, "benchmark": config.get("benchmark", {"examples": 300}),
        "objective": config.get("metric", "benchmark.loss"), "tags": ["sweep", sweep_name],
    }


def run_sweep(config: dict, runs_dir: Path = DEFAULT_RUNS_DIR, dry_run: bool = False) -> dict:
    name = config.get("name", "sweep")
    metric = config.get("metric", "benchmark.loss")
    strategy = config.get("strategy", "grid")
    if strategy not in {"grid", "random", "halving", "bayes", "bohb"}:
        raise ValueError("strategy must be grid, random, halving, bayes or bohb")
    space, seed = config["space"], config.get("seed", 0)
    if strategy == "bayes":
        return run_bayes_sweep(config, runs_dir, dry_run)
    if strategy == "bohb":
        return run_bohb_sweep(config, runs_dir, dry_run)
    candidates = expand_grid(space) if strategy == "grid" else sample_random(space, config.get("n_trials", 8), seed)
    skipped = []
    valid = []
    for trial in candidates:
        reason = invalid_reason({**config.get("base", {}), **trial})
        (skipped.append({"trial": trial, "reason": reason}) if reason else valid.append(trial))
    sweep_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    plan = {"name": name, "strategy": strategy, "candidates": len(candidates), "runnable": len(valid), "skipped": skipped}
    if dry_run:
        return {**plan, "dry_run": True, "first_trials": valid[:5]}
    sweep_dir.mkdir(parents=True)

    results: list[dict] = []
    rungs: list[dict] = []
    survivors = valid
    if strategy == "halving":
        halving = config.get("halving", {})
        eta, rung_count = halving.get("eta", 2), halving.get("rungs", 3)
        base_steps = config.get("base", {}).get("steps", 100)
    else:
        eta, rung_count, base_steps = 2, 1, None
    for rung in range(rung_count):
        steps = base_steps * eta**rung if strategy == "halving" else None
        rung_results = []
        for index, trial in enumerate(survivors):
            label = f"-r{rung}-t{index:02d}"
            summary = run_experiment(make_experiment(config, name, trial, steps, label), sweep_dir, run_id=f"trial{label}")
            summary["trial"], summary["rung"] = trial, rung
            rung_results.append(summary)
            print(f"[{name}] rung {rung} trial {index + 1}/{len(survivors)} {summary['status']}: {trial}")
        results += rung_results
        scored = sorted((s for s in rung_results if s["status"] == "completed" and dig(s, metric) is not None), key=lambda s: dig(s, metric))
        rungs.append({"rung": rung, "steps": steps, "trials": len(rung_results), "completed": len(scored)})
        if strategy == "halving" and rung < rung_count - 1:
            survivors = [s["trial"] for s in scored[: max(1, math.ceil(len(scored) / eta))]]
            if not survivors:
                break

    final_rung = max(s["rung"] for s in results) if results else 0
    finals = [s for s in results if s["rung"] == final_rung]
    ranked = sorted((s for s in finals if s["status"] == "completed" and dig(s, metric) is not None), key=lambda s: dig(s, metric))
    outcome = {
        **plan, "metric": metric, "rungs": rungs, "sweep_dir": str(sweep_dir),
        "best": {"run_id": ranked[0]["run_id"], "trial": ranked[0]["trial"], "value": dig(ranked[0], metric)} if ranked else None,
        "ranking": [{"run_id": s["run_id"], "trial": s["trial"], "value": dig(s, metric)} for s in ranked],
        "failed": [{"run_id": s["run_id"], "trial": s["trial"], "error": s.get("error")} for s in finals if s["status"] == "failed"],
    }
    pareto_md = ""
    if config.get("objectives"):
        # multi-objective reporting: computed over the final rung's completed trials. For "halving", which rung a
        # trial survives to is still decided by the single scalar ``metric`` above; this is a report, not a selector.
        objectives = parse_objectives(config["objectives"])
        front = pareto_front(finals, objectives)
        outcome["pareto_front"] = [{"run_id": s["run_id"], "trial": s["trial"],
                                   "values": {path: dig(s, path) for path, _ in objectives}} for s in front]
        pareto_md = "\n## Pareto front\n\n" + compare_pareto(finals, objectives) + "\n"
    (sweep_dir / "sweep_results.json").write_text(json.dumps(outcome, indent=2, default=str) + "\n")
    lines = [f"# Sweep {name} ({strategy})", "", f"metric: `{metric}` (lower is better)", ""]
    lines.append(compare_runs(load_summaries([sweep_dir / s["run_id"] for s in finals]), metric))
    lines.append(pareto_md)
    if skipped:
        lines += ["", "## Skipped combinations", ""] + [f"- {s['trial']}: {s['reason']}" for s in skipped]
    (sweep_dir / "sweep_results.md").write_text("\n".join(lines) + "\n")
    return outcome


def run_bayes_sweep(config: dict, runs_dir: Path, dry_run: bool) -> dict:
    name, metric = config.get("name", "sweep"), config.get("metric", "benchmark.loss")
    budget = config.get("n_trials", 12)
    base = config.get("base", {})
    optimizer = BayesianOptimizer(
        SearchSpace(config["space"]), seed=config.get("seed", 0), n_init=config.get("bayes", {}).get("n_init", 5),
        n_candidates=config.get("bayes", {}).get("candidates", 2000), validator=lambda trial: invalid_reason({**base, **trial}),
    )
    plan = {"name": name, "strategy": "bayes", "candidates": budget, "runnable": budget, "skipped": []}
    if dry_run:
        return {**plan, "dry_run": True, "first_trials": [optimizer.suggest() for _ in range(min(3, budget))]}
    sweep_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    sweep_dir.mkdir(parents=True)
    results = []
    for index in range(budget):
        trial = optimizer.suggest()
        summary = run_experiment(make_experiment(config, name, trial, None, f"-t{index:02d}"), sweep_dir, run_id=f"trial-t{index:02d}")
        summary["trial"], summary["rung"] = trial, 0
        value = dig(summary, metric) if summary["status"] == "completed" else None
        optimizer.observe(trial, value) if value is not None else optimizer.mark_failed(trial)
        results.append(summary)
        print(f"[{name}] trial {index + 1}/{budget} {summary['status']} {metric}={value}: {trial}")
    ranked = sorted((s for s in results if s["status"] == "completed" and dig(s, metric) is not None), key=lambda s: dig(s, metric))
    outcome = {
        **plan, "metric": metric, "rungs": [{"rung": 0, "steps": None, "trials": len(results), "completed": len(ranked)}],
        "sweep_dir": str(sweep_dir),
        "best": {"run_id": ranked[0]["run_id"], "trial": ranked[0]["trial"], "value": dig(ranked[0], metric)} if ranked else None,
        "ranking": [{"run_id": s["run_id"], "trial": s["trial"], "value": dig(s, metric)} for s in ranked],
        "failed": [{"run_id": s["run_id"], "trial": s["trial"], "error": s.get("error")} for s in results if s["status"] == "failed"],
        "trajectory": [dig(s, metric) for s in results],
    }
    (sweep_dir / "sweep_results.json").write_text(json.dumps(outcome, indent=2, default=str) + "\n")
    lines = [f"# Sweep {name} (bayes)", "", f"metric: `{metric}` (lower is better)", "",
             compare_runs(load_summaries([sweep_dir / s["run_id"] for s in results]), metric)]
    (sweep_dir / "sweep_results.md").write_text("\n".join(lines) + "\n")
    return outcome


def run_bohb_sweep(config: dict, runs_dir: Path, dry_run: bool) -> dict:
    """halving's rung/promotion structure, with the base rung's configs chosen by Bayesian expected improvement
    instead of at random. See the module docstring for exactly what this does and does not do."""
    name, metric = config.get("name", "sweep"), config.get("metric", "benchmark.loss")
    base = config.get("base", {})
    halving, bayes_cfg = config.get("halving", {}), config.get("bayes", {})
    eta, rung_count = halving.get("eta", 2), halving.get("rungs", 3)
    base_steps = base.get("steps", 100)
    budget = config.get("n_trials", 12)
    optimizer = BayesianOptimizer(
        SearchSpace(config["space"]), seed=config.get("seed", 0), n_init=bayes_cfg.get("n_init", 5),
        n_candidates=bayes_cfg.get("candidates", 2000), validator=lambda trial: invalid_reason({**base, **trial}),
    )
    plan = {"name": name, "strategy": "bohb", "candidates": budget, "runnable": budget, "skipped": []}
    if dry_run:
        return {**plan, "dry_run": True, "first_trials": [optimizer.suggest() for _ in range(min(3, budget))]}
    sweep_dir = runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"
    sweep_dir.mkdir(parents=True)

    results: list[dict] = []
    rungs: list[dict] = []
    trajectory: list[Optional[float]] = []
    survivors: list[dict] = []
    for rung in range(rung_count):
        steps = base_steps * eta**rung
        rung_results = []
        if rung == 0:
            for index in range(budget):
                trial = optimizer.suggest()
                summary = run_experiment(make_experiment(config, name, trial, steps, f"-r0-t{index:02d}"), sweep_dir, run_id=f"trial-r0-t{index:02d}")
                summary["trial"], summary["rung"] = trial, 0
                value = dig(summary, metric) if summary["status"] == "completed" else None
                optimizer.observe(trial, value) if value is not None else optimizer.mark_failed(trial)
                trajectory.append(value)
                rung_results.append(summary)
                print(f"[{name}] rung 0 trial {index + 1}/{budget} {summary['status']} {metric}={value}: {trial}")
        else:
            for index, trial in enumerate(survivors):
                label = f"-r{rung}-t{index:02d}"
                summary = run_experiment(make_experiment(config, name, trial, steps, label), sweep_dir, run_id=f"trial{label}")
                summary["trial"], summary["rung"] = trial, rung
                rung_results.append(summary)
                print(f"[{name}] rung {rung} trial {index + 1}/{len(survivors)} {summary['status']}: {trial}")
        results += rung_results
        scored = sorted((s for s in rung_results if s["status"] == "completed" and dig(s, metric) is not None), key=lambda s: dig(s, metric))
        rungs.append({"rung": rung, "steps": steps, "trials": len(rung_results), "completed": len(scored)})
        if rung < rung_count - 1:
            survivors = [s["trial"] for s in scored[: max(1, math.ceil(len(scored) / eta))]]
            if not survivors:
                break

    final_rung = max(s["rung"] for s in results) if results else 0
    finals = [s for s in results if s["rung"] == final_rung]
    ranked = sorted((s for s in finals if s["status"] == "completed" and dig(s, metric) is not None), key=lambda s: dig(s, metric))
    outcome = {
        **plan, "metric": metric, "rungs": rungs, "sweep_dir": str(sweep_dir), "trajectory": trajectory,
        "best": {"run_id": ranked[0]["run_id"], "trial": ranked[0]["trial"], "value": dig(ranked[0], metric)} if ranked else None,
        "ranking": [{"run_id": s["run_id"], "trial": s["trial"], "value": dig(s, metric)} for s in ranked],
        "failed": [{"run_id": s["run_id"], "trial": s["trial"], "error": s.get("error")} for s in finals if s["status"] == "failed"],
    }
    pareto_md = ""
    if config.get("objectives"):
        objectives = parse_objectives(config["objectives"])
        front = pareto_front(finals, objectives)
        outcome["pareto_front"] = [{"run_id": s["run_id"], "trial": s["trial"],
                                   "values": {path: dig(s, path) for path, _ in objectives}} for s in front]
        pareto_md = "\n## Pareto front\n\n" + compare_pareto(finals, objectives) + "\n"
    (sweep_dir / "sweep_results.json").write_text(json.dumps(outcome, indent=2, default=str) + "\n")
    lines = [f"# Sweep {name} (bohb)", "", f"metric: `{metric}` (lower is better)", "",
             compare_runs(load_summaries([sweep_dir / s["run_id"] for s in finals]), metric), pareto_md]
    (sweep_dir / "sweep_results.md").write_text("\n".join(lines) + "\n")
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--dry-run", action="store_true", help="list what would run without training")
    args = parser.parse_args()
    outcome = run_sweep(load_config(args.config), args.runs_dir, args.dry_run)
    print(json.dumps({k: outcome[k] for k in outcome if k not in {"ranking", "skipped"}}, indent=2, default=str))
    if outcome.get("best"):
        print(f"best: {outcome['best']}")


if __name__ == "__main__":
    main()
