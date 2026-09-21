"""Bayesian optimisation over a mixed search space (Gaussian process + expected improvement).

Self-contained (torch only). The space uses the same entries as sweep.py: a list (a choice), or one of
{"loguniform": [lo, hi]}, {"uniform": [lo, hi]}, {"int": [lo, hi]}.

    space = SearchSpace({"lr": {"loguniform": [1e-4, 1e-2]}, "experts": [4, 8, 16], "act": ["a", "b"]})
    optimizer = BayesianOptimizer(space, seed=0, n_init=5)
    for _ in range(30):
        params = optimizer.suggest()
        optimizer.observe(params, evaluate(params))     # lower is better
    optimizer.best()

Numeric parameters are scaled to [0, 1] (log-scaled for loguniform); numeric choice lists are encoded by their
position, non-numeric choices are one-hot encoded. The GP uses a Matern-5/2 kernel whose length scale and noise
are chosen by marginal likelihood each time it is refit; the next point maximises expected improvement over a
pool of random candidates plus perturbations of the best points so far.
"""

import json
import math
import random
from typing import Callable, Optional

import torch


class SearchSpace:
    def __init__(self, spec: dict) -> None:
        if not spec:
            raise ValueError("the search space is empty")
        self.spec = spec
        self.names = list(spec)
        self._columns: list[tuple[str, int]] = []  # (name, width)
        for name, entry in spec.items():
            self._validate(name, entry)
            self._columns.append((name, self._width(entry)))
        self.dimension = sum(width for _, width in self._columns)

    @staticmethod
    def _validate(name: str, entry) -> None:
        if isinstance(entry, list):
            if not entry:
                raise ValueError(f"'{name}' has no choices")
            return
        if isinstance(entry, dict) and len(entry) == 1:
            kind, bounds = next(iter(entry.items()))
            if kind in {"loguniform", "uniform", "int"} and len(bounds) == 2 and bounds[0] < bounds[1]:
                if kind == "loguniform" and bounds[0] <= 0:
                    raise ValueError(f"'{name}': loguniform bounds must be positive")
                return
        raise ValueError(f"unsupported search-space entry for '{name}': {entry!r}")

    @staticmethod
    def _numeric_choices(entry: list) -> bool:
        return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in entry)

    def _width(self, entry) -> int:
        if isinstance(entry, list) and not self._numeric_choices(entry):
            return len(entry)
        return 1

    def sample(self, rng: random.Random) -> dict:
        params = {}
        for name, entry in self.spec.items():
            if isinstance(entry, list):
                params[name] = rng.choice(entry)
            else:
                kind, (low, high) = next(iter(entry.items()))
                if kind == "loguniform":
                    params[name] = min(high, max(low, math.exp(rng.uniform(math.log(low), math.log(high)))))
                elif kind == "uniform":
                    params[name] = rng.uniform(low, high)
                else:
                    params[name] = rng.randint(int(low), int(high))
        return params

    def encode(self, params: dict) -> torch.Tensor:
        parts: list[float] = []
        for name, entry in self.spec.items():
            value = params[name]
            if isinstance(entry, list):
                index = entry.index(value)
                if self._numeric_choices(entry):
                    parts.append(index / max(1, len(entry) - 1))
                else:
                    parts.extend(1.0 if i == index else 0.0 for i in range(len(entry)))
            else:
                kind, (low, high) = next(iter(entry.items()))
                if kind == "loguniform":
                    parts.append((math.log(value) - math.log(low)) / (math.log(high) - math.log(low)))
                else:
                    parts.append((value - low) / (high - low))
        return torch.tensor(parts, dtype=torch.float64)

    def perturb(self, params: dict, rng: random.Random, scale: float = 0.1) -> dict:
        """A nearby point: continuous parameters jitter in encoded space, choices resample with small probability."""
        encoded = self.encode(params)
        out = {}
        for name, entry in self.spec.items():
            if isinstance(entry, list):
                out[name] = rng.choice(entry) if rng.random() < 0.2 else params[name]
                continue
            kind, (low, high) = next(iter(entry.items()))
            unit = min(1.0, max(0.0, float(encoded[self._offset(name)]) + rng.gauss(0, scale)))
            if kind == "loguniform":
                out[name] = min(high, max(low, math.exp(math.log(low) + unit * (math.log(high) - math.log(low)))))
            elif kind == "uniform":
                out[name] = low + unit * (high - low)
            else:
                out[name] = int(round(low + unit * (high - low)))
        return out

    def _offset(self, name: str) -> int:
        offset = 0
        for column, width in self._columns:
            if column == name:
                return offset
            offset += width
        raise KeyError(name)

    def key(self, params: dict) -> str:
        return json.dumps(params, sort_keys=True, default=str)


def matern52(a: torch.Tensor, b: torch.Tensor, length: float) -> torch.Tensor:
    distance = torch.cdist(a, b) / length
    root5 = math.sqrt(5.0) * distance
    return (1 + root5 + 5.0 / 3.0 * distance**2) * torch.exp(-root5)


class GaussianProcess:
    """GP regression on standardised targets; hyperparameters picked by marginal likelihood."""

    LENGTHS = (0.1, 0.2, 0.35, 0.6, 1.0)
    NOISES = (1e-6, 1e-4, 1e-2, 1e-1)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianProcess":
        self.x = x
        self.mean, self.scale = y.mean(), y.std(unbiased=False).clamp(min=1e-9)
        self.y = (y - self.mean) / self.scale
        best = None
        n = x.size(0)
        for length in self.LENGTHS:
            base = matern52(x, x, length)
            for noise in self.NOISES:
                try:
                    chol = torch.linalg.cholesky(base + (noise + 1e-8) * torch.eye(n, dtype=x.dtype))
                except RuntimeError:
                    continue
                alpha = torch.cholesky_solve(self.y[:, None], chol)
                log_likelihood = -0.5 * (self.y[:, None] * alpha).sum() - torch.log(torch.diagonal(chol)).sum()
                if best is None or log_likelihood > best[0]:
                    best = (log_likelihood, length, noise, chol, alpha)
        if best is None:
            raise RuntimeError("could not fit the Gaussian process")
        _, self.length, self.noise, self.chol, self.alpha = best
        return self

    def predict(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cross = matern52(points, self.x, self.length)
        mean = (cross @ self.alpha).squeeze(-1)
        solved = torch.cholesky_solve(cross.t(), self.chol)
        variance = (1.0 - (cross * solved.t()).sum(dim=1)).clamp(min=1e-12)
        return mean * self.scale + self.mean, variance.sqrt() * self.scale


def expected_improvement(mean: torch.Tensor, std: torch.Tensor, best: float) -> torch.Tensor:
    """EI for minimisation."""
    z = (best - mean) / std.clamp(min=1e-12)
    cdf = torch.special.ndtr(z)
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    return ((best - mean) * cdf + std * pdf).clamp(min=0.0)


class BayesianOptimizer:
    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        n_init: int = 5,
        n_candidates: int = 2000,
        validator: Optional[Callable[[dict], Optional[str]]] = None,
    ) -> None:
        self.space = space
        self.rng = random.Random(seed)
        self.n_init = n_init
        self.n_candidates = n_candidates
        self.validator = validator
        self.params: list[dict] = []
        self.values: list[float] = []
        self.seen: set[str] = set()

    def _valid(self, candidate: dict) -> bool:
        return self.space.key(candidate) not in self.seen and (self.validator is None or self.validator(candidate) is None)

    def _random_valid(self) -> dict:
        for _ in range(1000):
            candidate = self.space.sample(self.rng)
            if self._valid(candidate):
                return candidate
        raise RuntimeError("could not find an unseen valid configuration; the space may be exhausted")

    def observe(self, params: dict, value: float) -> None:
        self.seen.add(self.space.key(params))
        if math.isfinite(value):
            self.params.append(params)
            self.values.append(float(value))

    def mark_failed(self, params: dict) -> None:
        """Never suggest this point again, without teaching the model a fake value."""
        self.seen.add(self.space.key(params))

    def suggest(self) -> dict:
        if len(self.values) < max(2, self.n_init):
            return self._random_valid()
        x = torch.stack([self.space.encode(p) for p in self.params])
        gp = GaussianProcess().fit(x, torch.tensor(self.values, dtype=torch.float64))
        candidates = [self.space.sample(self.rng) for _ in range(self.n_candidates)]
        ranked = sorted(range(len(self.values)), key=lambda i: self.values[i])[:3]
        for index in ranked:
            for scale in (0.03, 0.1, 0.2):
                candidates += [self.space.perturb(self.params[index], self.rng, scale) for _ in range(60)]
        candidates = [c for c in candidates if self._valid(c)]
        if not candidates:
            return self._random_valid()
        encoded = torch.stack([self.space.encode(c) for c in candidates])
        mean, std = gp.predict(encoded)
        score = expected_improvement(mean, std, min(self.values))
        if float(score.max()) <= 1e-12:               # nothing promises improvement: explore the most uncertain point
            return candidates[int(std.argmax())]
        return candidates[int(score.argmax())]

    def best(self) -> Optional[tuple[dict, float]]:
        if not self.values:
            return None
        index = min(range(len(self.values)), key=lambda i: self.values[i])
        return self.params[index], self.values[index]
