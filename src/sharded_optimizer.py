"""ZeRO-style optimizer-state sharding for expert-parallel training.

In expert-parallel training every non-expert parameter (attention, embeddings, routers, norms) is replicated across the
expert-parallel group, so a plain optimizer would keep a full copy of its AdamW moments on every rank. ShardedAdamW
assigns each replicated parameter to one owner rank (balanced by size):

  reduce_gradients()  gradients are averaged onto the owner only (ZeRO-2: non-owners drop them);
  step()              the owner alone holds the moments and updates the parameter;
  ...then every owner broadcasts its updated parameters, so the replicas stay identical.

Experts are already unique to their rank and use an ordinary local AdamW. With N ranks the moment memory for
replicated parameters is about 1/N of the unsharded cost. State is saved per rank (the checkpoint layout must match
on resume). fp16 loss scaling is not supported here (use bf16).
"""

from typing import Iterable

import torch
import torch.distributed as dist

from expert_parallel import ExpertParallelContext, is_expert_parameter


class ShardedAdamW:
    def __init__(self, model, ctx: ExpertParallelContext, lr: float = 1e-3, **adamw_kwargs) -> None:
        self.model, self.ctx = model, ctx
        self.shared = [(name, p) for name, p in model.named_parameters() if not is_expert_parameter(name)]
        self.local = [p for name, p in model.named_parameters() if is_expert_parameter(name)]
        world = ctx.world_size
        loads = [0] * world
        self.owner: list[int] = [0] * len(self.shared)
        for index in sorted(range(len(self.shared)), key=lambda i: -self.shared[i][1].numel()):   # largest first
            target = loads.index(min(loads))
            self.owner[index] = target
            loads[target] += self.shared[index][1].numel()
        self.by_owner: list[list[torch.nn.Parameter]] = [[] for _ in range(world)]
        for (name, parameter), owner in zip(self.shared, self.owner):
            self.by_owner[owner].append(parameter)
        owned = self.by_owner[ctx.rank]
        self.opt_shared = torch.optim.AdamW(owned, lr=lr, **adamw_kwargs) if owned else None
        self.opt_local = torch.optim.AdamW(self.local, lr=lr, **adamw_kwargs) if self.local else None

    # ---- optimizer interface -------------------------------------------------------------
    @property
    def param_groups(self) -> list[dict]:
        return [group for optimizer in (self.opt_shared, self.opt_local) if optimizer is not None for group in optimizer.param_groups]

    def _root(self, ep_rank: int) -> int:
        group = self.ctx.group
        return dist.get_global_rank(group, ep_rank) if group is not None else ep_rank

    def reduce_gradients(self) -> None:
        """Average each owner's gradients onto the owner; other ranks release them."""
        world = self.ctx.world_size
        for owner, parameters in enumerate(self.by_owner):
            if not parameters:
                continue
            for parameter in parameters:
                if parameter.grad is None:      # keep the collectives aligned across ranks
                    parameter.grad = torch.zeros_like(parameter)
            flat = torch.cat([p.grad.flatten() for p in parameters])
            dist.reduce(flat, dst=self._root(owner), group=self.ctx.group)
            if owner == self.ctx.rank:
                flat /= world
                offset = 0
                for parameter in parameters:
                    parameter.grad.copy_(flat[offset : offset + parameter.numel()].view_as(parameter))
                    offset += parameter.numel()
            else:
                for parameter in parameters:
                    parameter.grad = None

    def step(self) -> None:
        if self.opt_shared is not None:
            self.opt_shared.step()
        if self.opt_local is not None:
            self.opt_local.step()
        for owner, parameters in enumerate(self.by_owner):
            if not parameters:
                continue
            flat = torch.cat([p.data.flatten() for p in parameters])
            dist.broadcast(flat, src=self._root(owner), group=self.ctx.group)
            if owner != self.ctx.rank:
                offset = 0
                for parameter in parameters:
                    parameter.data.copy_(flat[offset : offset + parameter.numel()].view_as(parameter))
                    offset += parameter.numel()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.model.parameters():
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    # ---- state ---------------------------------------------------------------------------------
    def state_numel(self) -> int:
        """Elements of optimizer state (AdamW moments) held on this rank."""
        total = 0
        for optimizer in (self.opt_shared, self.opt_local):
            if optimizer is not None:
                total += sum(t.numel() for state in optimizer.state.values() for t in state.values() if torch.is_tensor(t) and t.dim() > 0)
        return total

    def state_dict(self) -> dict:
        return {
            "shared": self.opt_shared.state_dict() if self.opt_shared else None,
            "local": self.opt_local.state_dict() if self.opt_local else None,
            "owners": list(self.owner), "world": self.ctx.world_size,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["owners"] != self.owner or state["world"] != self.ctx.world_size:
            raise ValueError("sharded optimizer state was saved with a different parameter ownership or world size")
        if self.opt_shared is not None:
            self.opt_shared.load_state_dict(state["shared"])
        if self.opt_local is not None:
            self.opt_local.load_state_dict(state["local"])
