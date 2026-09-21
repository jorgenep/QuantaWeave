"""Training schedules and feedback controllers for MoE routing.

Open-loop schedules depend only on the step: learning-rate warmup/decay, the router aux-loss weight
and the router temperature. Closed-loop controllers adjust from what training reports every
``interval`` steps: plateau LR decay, aux-weight balancing, expert-capacity growth and router
exploration. All controller state is exposed through ``state_dict`` so a resumed run continues the
same trajectory.
"""

import math
from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class ScheduleConfig:
    # learning rate
    lr: float = 3e-4
    total_steps: int = 0
    warmup_steps: int = 0
    lr_decay: str = "constant"  # constant | cosine | linear (needs total_steps)
    min_lr_ratio: float = 0.1
    plateau_patience: int = 0  # controller intervals without improvement before decaying LR; 0 = off
    plateau_factor: float = 0.5
    plateau_min_delta: float = 1e-3
    min_lr_scale: float = 0.05
    # router auxiliary (load-balancing) loss weight
    aux_coef: float = 0.01
    aux_coef_end: Optional[float] = None  # anneal linearly to this over total_steps
    aux_adapt: bool = False
    aux_factor_min: float = 0.25
    aux_factor_max: float = 4.0
    aux_up: float = 1.05
    aux_down: float = 0.98
    balance_low: float = 0.25  # expert-load coefficient of variation below which balancing can relax
    balance_high: float = 1.0  # ... above which it is tightened
    # expert capacity
    capacity_factor: float = 1.25
    capacity_adapt: bool = False
    capacity_min: float = 1.0
    capacity_max: float = 4.0
    capacity_up: float = 1.1
    capacity_down: float = 0.99
    drop_threshold: float = 0.01  # tolerated fraction of routes over capacity
    capacity_release_step: int = 0  # from this step on capacity is not enforced (0 = never)
    # router temperature
    temperature_start: Optional[float] = None
    temperature_end: float = 1.0
    temperature_steps: int = 0
    temperature_adapt: bool = False
    temperature_boost_max: float = 4.0
    dead_expert_fraction: float = 0.5  # boost temperature when fewer than this share of experts is active
    # cadence
    interval: int = 50

    def __post_init__(self) -> None:
        if self.lr_decay not in {"constant", "cosine", "linear"}:
            raise ValueError("lr_decay must be constant, cosine or linear")
        if self.interval < 1:
            raise ValueError("interval must be positive")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if self.capacity_min > self.capacity_max:
            raise ValueError("capacity_min cannot exceed capacity_max")


class TrainingController:
    def __init__(self, config: ScheduleConfig, base_temperature: float = 1.0) -> None:
        self.config = config
        self.base_temperature = base_temperature
        self.lr_scale = 1.0
        self.aux_factor = 1.0
        self.temperature_boost = 1.0
        self.capacity_factor = config.capacity_factor
        self.released = False
        self.best_loss = math.inf
        self.stale_intervals = 0
        self._losses: list[float] = []
        self._drop_fractions: list[float] = []
        self.history: list[dict] = []

    # ---- open-loop schedules ----------------------------------------------
    def lr_at(self, step: int) -> float:
        cfg = self.config
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps * self.lr_scale
        span = cfg.total_steps - cfg.warmup_steps
        if cfg.lr_decay == "constant" or span <= 0:
            return cfg.lr * self.lr_scale
        progress = min(1.0, max(0.0, (step - cfg.warmup_steps) / span))
        if cfg.lr_decay == "cosine":
            factor = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            factor = 1 - (1 - cfg.min_lr_ratio) * progress
        return cfg.lr * factor * self.lr_scale

    def aux_coef_at(self, step: int) -> float:
        cfg = self.config
        if cfg.aux_coef_end is None or cfg.total_steps <= 0:
            base = cfg.aux_coef
        else:
            progress = min(1.0, step / cfg.total_steps)
            base = cfg.aux_coef + (cfg.aux_coef_end - cfg.aux_coef) * progress
        return base * self.aux_factor

    def temperature_at(self, step: int) -> float:
        cfg = self.config
        if cfg.temperature_start is None:
            base = self.base_temperature
        elif cfg.temperature_steps <= 0:
            base = cfg.temperature_start
        else:
            progress = min(1.0, step / cfg.temperature_steps)
            base = cfg.temperature_start + (cfg.temperature_end - cfg.temperature_start) * progress
        return base * self.temperature_boost

    # ---- closed-loop control ----------------------------------------------
    def observe(self, loss: float, dropped_fraction: float) -> None:
        self._losses.append(loss)
        self._drop_fractions.append(dropped_fraction)

    def should_update(self, step: int) -> bool:
        return step % self.config.interval == 0

    def update(self, step: int, imbalance: Optional[float] = None, active_fraction: Optional[float] = None) -> dict:
        """Adjust controllers from the last interval. Returns what changed (for logging)."""
        cfg = self.config
        changes: dict[str, float] = {}
        if self._losses:
            mean_loss = sum(self._losses) / len(self._losses)
            if mean_loss < self.best_loss - cfg.plateau_min_delta:
                self.best_loss, self.stale_intervals = mean_loss, 0
            else:
                self.stale_intervals += 1
            if cfg.plateau_patience and self.stale_intervals >= cfg.plateau_patience:
                new_scale = max(self.lr_scale * cfg.plateau_factor, cfg.min_lr_scale)
                if new_scale != self.lr_scale:
                    self.lr_scale = new_scale
                    changes["lr_scale"] = new_scale
                self.stale_intervals = 0
        if cfg.capacity_adapt and not self.released and self._drop_fractions:
            drops = sum(self._drop_fractions) / len(self._drop_fractions)
            if drops > cfg.drop_threshold:
                target = min(self.capacity_factor * cfg.capacity_up, cfg.capacity_max)
            else:
                target = max(self.capacity_factor * cfg.capacity_down, cfg.capacity_min)
            if target != self.capacity_factor:
                self.capacity_factor = target
                changes["capacity_factor"] = target
        if cfg.aux_adapt and imbalance is not None:
            if imbalance > cfg.balance_high:
                factor = min(self.aux_factor * cfg.aux_up, cfg.aux_factor_max)
            elif imbalance < cfg.balance_low:
                factor = max(self.aux_factor * cfg.aux_down, cfg.aux_factor_min)
            else:
                factor = self.aux_factor
            if factor != self.aux_factor:
                self.aux_factor = factor
                changes["aux_factor"] = factor
        if cfg.temperature_adapt and active_fraction is not None:
            if active_fraction < cfg.dead_expert_fraction:
                boost = min(self.temperature_boost * 1.05, cfg.temperature_boost_max)
            else:
                boost = max(self.temperature_boost * 0.98, 1.0)
            if boost != self.temperature_boost:
                self.temperature_boost = boost
                changes["temperature_boost"] = boost
        if cfg.capacity_release_step and step >= cfg.capacity_release_step and not self.released:
            self.released = True
            changes["capacity_released"] = 1.0
        self._losses.clear()
        self._drop_fractions.clear()
        if changes:
            self.history.append({"step": step, **changes})
        return changes

    def apply(self, model, optimizer, step: int) -> dict[str, float]:
        """Push the current schedule values into the model and optimizer; returns them for logging."""
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        controls = dict(
            capacity_factor=self.capacity_factor,
            drop_overflow_tokens=not self.released,
            router_temperature=self.temperature_at(step),
            router_aux_loss_coef=self.aux_coef_at(step),
        )
        model.set_routing_controls(**controls)
        return {
            "lr": lr,
            "capacity_factor": self.capacity_factor,
            "capacity_enforced": float(not self.released),
            "router_temperature": controls["router_temperature"],
            "router_aux_loss_coef": controls["router_aux_loss_coef"],
        }

    # ---- checkpoint state --------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "lr_scale": self.lr_scale,
            "aux_factor": self.aux_factor,
            "temperature_boost": self.temperature_boost,
            "capacity_factor": self.capacity_factor,
            "released": self.released,
            "best_loss": self.best_loss,
            "stale_intervals": self.stale_intervals,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict) -> None:
        for name in ("lr_scale", "aux_factor", "temperature_boost", "capacity_factor", "released", "best_loss", "stale_intervals"):
            setattr(self, name, state[name])
        self.history = list(state.get("history", []))


def schedule_field_names() -> list[str]:
    return [item.name for item in fields(ScheduleConfig)]
