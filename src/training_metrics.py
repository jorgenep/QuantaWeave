"""Structured training metrics: JSONL log, throughput, stability and plateau detection."""

import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence


def detect_plateau(losses: Sequence[float], window: int = 20, tolerance: float = 0.005) -> bool:
    """True if the mean of the last ``window`` losses improved by less than ``tolerance`` (relative)
    over the mean of the ``window`` before it."""
    if len(losses) < 2 * window:
        return False
    recent = sum(losses[-window:]) / window
    previous = sum(losses[-2 * window : -window]) / window
    return previous - recent < tolerance * abs(previous)


def first_stable_step(steps: Sequence[int], losses: Sequence[float], window: int = 20, tolerance: float = 0.02) -> Optional[int]:
    """First step after which the windowed mean loss never moves by more than ``tolerance`` (relative)
    between consecutive windows. None while training is still moving."""
    if len(losses) < 2 * window:
        return None
    means = [sum(losses[i : i + window]) / window for i in range(0, len(losses) - window + 1, window)]
    for start in range(len(means) - 1):
        tail = means[start:]
        if all(abs(a - b) <= tolerance * max(abs(a), 1e-9) for a, b in zip(tail, tail[1:])):
            return int(steps[start * window])
    return None


class MetricsLogger:
    """Appends one JSON object per logged step to ``metrics.jsonl`` and keeps a summary."""

    def __init__(self, path: Optional[Path], total_parameters: int = 0, active_parameters: int = 0) -> None:
        self.path = path
        self.total_parameters = total_parameters
        self.active_parameters = active_parameters
        self.steps: list[int] = []
        self.losses: list[float] = []
        self.started = time.perf_counter()
        self.tokens = 0
        self.last: dict = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, step: int, tokens: int, **values: float) -> dict:
        self.tokens += tokens
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        record = {"step": step, "tokens_per_second": self.tokens / elapsed, **values}
        self.last = record
        if "loss" in values and math.isfinite(values["loss"]):
            self.steps.append(step)
            self.losses.append(values["loss"])
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        return record

    def summary(self) -> dict:
        losses = self.losses
        # the mean over the last 20 *steps*; with sparse logging that may be one or two entries (never older ones)
        tail = [loss for step, loss in zip(self.steps, losses) if step > self.steps[-1] - 20] if losses else []
        efficiency = self.active_parameters / self.total_parameters if self.total_parameters else None
        return {
            "steps_logged": len(losses),
            "final_loss": sum(tail) / len(tail) if tail else None,
            "best_loss": min(losses) if losses else None,
            "tokens_per_second": self.tokens / max(time.perf_counter() - self.started, 1e-9),
            "plateaued": detect_plateau(losses),
            "first_stable_step": first_stable_step(self.steps, losses),
            "total_parameters": self.total_parameters,
            "active_parameters": self.active_parameters,
            "active_parameter_ratio": efficiency,
        }
