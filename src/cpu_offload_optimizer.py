"""CPU-offloaded AdamW for the standalone trainer.

PARAMETERS.md documents that the standalone ``train_quantweave_moe.py`` trainer has no CPU-offload path: the
16 bytes/parameter of training state (4 weight + 4 grad + 4 + 4 AdamW moments, all fp32) live entirely in device
memory, unlike the separate Axolotl + DeepSpeed ZeRO-3 path (`configs/deepspeed_zero3.json`), which offloads
optimizer state to pinned host RAM. This module is that missing piece for the standalone trainer: the same idea
DeepSpeed's ZeRO-Offload implements, applied to a single AdamW instance instead of a distributed one.

``CPUOffloadAdamW`` keeps ``exp_avg``/``exp_avg_sq`` (AdamW's two fp32 moments, 8 of the 16 bytes/parameter) in
pinned CPU memory instead of device memory, freeing that much device memory for a bigger model or batch. Weights
and gradients stay on the device as usual (this offloads optimizer *state*, not parameters or gradients — the same
scope as DeepSpeed's ``offload_optimizer`` with ``offload_param: none``). Every step, each parameter's gradient is
copied to a reused pinned staging buffer, the AdamW update is computed on the CPU, and the result is copied back to
the device — a real host<->device transfer per step, not a free lunch: expect this to cost meaningful wall-clock
time, worse with more PCIe traffic (bigger models, smaller batches making the transfer a bigger fraction of a step).

Only plain (non-fused, non-foreach) AdamW math is implemented, matching ``torch.optim.AdamW``'s default: decoupled
weight decay, bias-corrected first/second moments. Verified against ``torch.optim.AdamW`` step-for-step on identical
gradients (see ``tests/test_future_features.py``).
"""

from typing import Iterable

import torch


class CPUOffloadAdamW:
    def __init__(
        self, parameters: Iterable[torch.nn.Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8, weight_decay: float = 0.01,
    ) -> None:
        self.params = [p for p in parameters if p.requires_grad]
        self.lr, self.betas, self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.param_groups = [{"params": self.params, "lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}]
        # id(param) -> {"exp_avg", "exp_avg_sq", "grad_staging": pinned CPU tensors; "step": int}
        self.state: dict[int, dict] = {}

    def _state_for(self, p: torch.nn.Parameter) -> dict:
        entry = self.state.get(id(p))
        if entry is None:
            pinned = torch.cuda.is_available()  # pin_memory needs a CUDA context; harmless (just unpinned) on CPU-only runs
            entry = {
                "exp_avg": torch.zeros(p.shape, dtype=torch.float32, pin_memory=pinned),
                "exp_avg_sq": torch.zeros(p.shape, dtype=torch.float32, pin_memory=pinned),
                "grad_staging": torch.empty(p.shape, dtype=torch.float32, pin_memory=pinned),
                "step": 0,
            }
            self.state[id(p)] = entry
        return entry

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, (beta1, beta2), eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._state_for(p)
                state["grad_staging"].copy_(p.grad, non_blocking=True)
                grad = state["grad_staging"]
                state["step"] += 1
                step = state["step"]
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)          # decoupled weight decay, applied on the device parameter directly
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                update = (exp_avg / denom).mul_(lr / bias_correction1)
                p.add_(update.to(device=p.device, dtype=p.dtype, non_blocking=True), alpha=-1.0)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def state_numel(self) -> int:
        """Elements of optimizer state (AdamW moments) held in pinned host memory, not device memory."""
        return sum(state["exp_avg"].numel() + state["exp_avg_sq"].numel() for state in self.state.values())

    def state_dict(self) -> dict:
        # keyed by position (matches torch.optim's own convention closely enough for our own save/load round-trip;
        # this optimizer is not meant to interoperate with torch.optim.AdamW's checkpoint format)
        order = {id(p): index for index, p in enumerate(self.params)}
        return {
            "lr": self.lr, "betas": self.betas, "eps": self.eps, "weight_decay": self.weight_decay,
            "state": {
                order[key]: {"exp_avg": value["exp_avg"], "exp_avg_sq": value["exp_avg_sq"], "step": value["step"]}
                for key, value in self.state.items()
            },
        }

    def load_state_dict(self, state: dict) -> None:
        self.lr, self.betas, self.eps, self.weight_decay = state["lr"], tuple(state["betas"]), state["eps"], state["weight_decay"]
        for group in self.param_groups:
            group["lr"], group["betas"], group["eps"], group["weight_decay"] = self.lr, self.betas, self.eps, self.weight_decay
        for index, p in enumerate(self.params):
            saved = state["state"].get(index)
            if saved is None:
                continue
            entry = self._state_for(p)
            entry["exp_avg"].copy_(saved["exp_avg"])
            entry["exp_avg_sq"].copy_(saved["exp_avg_sq"])
            entry["step"] = saved["step"]
