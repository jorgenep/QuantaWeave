import math
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import bayes_opt as bo
import sweep


def branin(x1: float, x2: float) -> float:
    a, b, c = 1.0, 5.1 / (4 * math.pi**2), 5.0 / math.pi
    return a * (x2 - b * x1**2 + c * x1 - 6) ** 2 + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1) + 10


BRANIN_SPACE = {"x1": {"uniform": [-5, 10]}, "x2": {"uniform": [0, 15]}}


def optimise(strategy: str, seed: int, budget: int = 25) -> float:
    space = bo.SearchSpace(BRANIN_SPACE)
    if strategy == "random":
        rng = random.Random(seed)
        return min(branin(**space.sample(rng)) for _ in range(budget))
    optimizer = bo.BayesianOptimizer(space, seed=seed, n_init=6)
    for _ in range(budget):
        params = optimizer.suggest()
        optimizer.observe(params, branin(**params))
    return optimizer.best()[1]


def test_bayesian_search_beats_random_search_on_a_smooth_objective():
    bayes = [optimise("bayes", seed) for seed in range(4)]
    rand = [optimise("random", seed) for seed in range(4)]
    assert sum(bayes) / 4 < sum(rand) / 4
    assert min(bayes) < 0.6                       # Branin's global minimum is 0.398


def test_space_encodes_every_parameter_kind_into_the_unit_cube():
    space = bo.SearchSpace({"lr": {"loguniform": [1e-4, 1e-2]}, "n": {"int": [2, 10]}, "k": [1, 2, 4], "mode": ["a", "b", "c"]})
    assert space.dimension == 1 + 1 + 1 + 3
    vector = space.encode({"lr": 1e-3, "n": 6, "k": 4, "mode": "b"})
    assert vector.tolist() == pytest.approx([0.5, 0.5, 1.0, 0.0, 1.0, 0.0])
    rng = random.Random(0)
    for _ in range(50):
        params = space.sample(rng)
        assert 1e-4 <= params["lr"] <= 1e-2 and 2 <= params["n"] <= 10 and params["k"] in (1, 2, 4)
        assert ((space.encode(params) >= 0) & (space.encode(params) <= 1)).all()
        near = space.perturb(params, rng, 0.1)
        assert 1e-4 <= near["lr"] <= 1e-2 and 2 <= near["n"] <= 10


@pytest.mark.parametrize("bad", [{}, {"a": []}, {"a": {"uniform": [3, 1]}}, {"a": {"loguniform": [0, 1]}}, {"a": {"weird": [0, 1]}}, {"a": 5}])
def test_bad_spaces_are_rejected(bad):
    with pytest.raises(ValueError):
        bo.SearchSpace(bad)


def test_gp_interpolates_and_is_uncertain_away_from_data():
    torch.manual_seed(0)
    x = torch.linspace(0, 1, 8, dtype=torch.float64)[:, None]
    y = torch.sin(3 * x[:, 0])
    gp = bo.GaussianProcess().fit(x, y)
    mean, std = gp.predict(x)
    assert torch.allclose(mean, y, atol=1e-2) and std.max() < 0.05
    far_mean, far_std = gp.predict(torch.tensor([[3.0]], dtype=torch.float64))
    assert far_std.item() > 10 * std.max().item()
    between, between_std = gp.predict(torch.tensor([[0.5]], dtype=torch.float64))
    assert abs(between.item() - math.sin(1.5)) < 0.05


def test_expected_improvement_prefers_low_mean_and_high_uncertainty():
    mean = torch.tensor([1.0, 0.0, 1.0, 2.0])
    std = torch.tensor([0.1, 0.1, 1.0, 0.1])
    ei = bo.expected_improvement(mean, std, best=0.5)
    assert ei[1] > ei[0] and ei[2] > ei[0] and ei[3] < 1e-6 and (ei >= 0).all()


def test_suggestions_are_valid_never_repeat_and_failures_are_remembered():
    space = bo.SearchSpace({"total": [2, 4, 8], "active": [1, 2, 4]})
    validator = lambda p: "too many" if p["active"] > p["total"] else None      # noqa: E731
    optimizer = bo.BayesianOptimizer(space, seed=1, n_init=3, validator=validator)
    seen = set()
    for _ in range(8):                                                            # 8 of the 9 combinations are valid
        params = optimizer.suggest()
        assert validator(params) is None and space.key(params) not in seen
        seen.add(space.key(params))
        optimizer.observe(params, float(params["total"] - params["active"])) if len(seen) % 3 else optimizer.mark_failed(params)
    with pytest.raises(RuntimeError, match="exhausted"):
        optimizer.suggest()
    assert optimizer.best()[1] == min(optimizer.values)


def test_non_finite_observations_are_ignored():
    optimizer = bo.BayesianOptimizer(bo.SearchSpace({"a": {"uniform": [0, 1]}}), n_init=2)
    optimizer.observe({"a": 0.5}, float("nan"))
    assert optimizer.values == [] and optimizer.best() is None


def test_bayes_strategy_in_the_sweep_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run(experiment, runs_dir, name=None, run_id=None):
        train = experiment["train"]
        calls.append(train)
        loss = (math.log10(train["lr"]) + 3) ** 2 + 0.1 * train["total_experts"] / 8
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        return {"run_id": run_id, "status": "completed", "run_dir": str(run_dir), "name": experiment["name"], "benchmark": {"loss": loss}}

    monkeypatch.setattr(sweep, "run_experiment", fake_run)
    config = {"name": "b", "strategy": "bayes", "n_trials": 14, "seed": 0, "bayes": {"n_init": 4}, "base": {"hidden_size": 32},
              "space": {"lr": {"loguniform": [1e-5, 1e-1]}, "total_experts": [2, 4, 8], "active_experts": [1, 2, 4]}}
    plan = sweep.run_sweep(config, tmp_path / "runs", dry_run=True)
    assert plan["dry_run"] and len(plan["first_trials"]) == 3
    outcome = sweep.run_sweep(config, tmp_path / "runs")
    assert len(calls) == 14 and all(c["active_experts"] <= c["total_experts"] for c in calls)
    trajectory = outcome["trajectory"]
    assert min(trajectory[4:]) < min(trajectory[:4])                              # the model-guided trials improved on the random ones
    assert 1e-4 < outcome["best"]["trial"]["lr"] < 1e-2                           # optimum is at lr = 1e-3
    assert (Path(outcome["sweep_dir"]) / "sweep_results.md").read_text().startswith("# Sweep b (bayes)")
