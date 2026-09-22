"""Expert parallelism (with optional tensor parallelism): experts are sharded across ranks and tokens travel to the
rank that owns their expert over two differentiable all-to-alls.

Launch:  torchrun --nproc_per_node=N src/train_quantweave_moe.py --expert-parallel [--tensor-parallel T] ...
World size N = expert-parallel size x T. Ranks form a 2-D grid, tensor-parallel rank varying fastest:
    global_rank = ep_rank * T + tp_rank
Ranks with the same ep_rank share a tensor-parallel group (they hold slices of the same layers); ranks with the same
tp_rank share an expert-parallel group (they hold different experts and different data).

Everything except the experts (attention, embeddings, routers, norms) is replicated across the expert-parallel group
and its gradients are averaged; expert gradients are already summed over all ranks' tokens by the return all-to-all, so
they only need the 1/ep batch-mean scaling. Gradient clipping uses the true global norm (see clip_grad_norm_parallel).

Capacity is enforced per source rank on its local batch (grouped capacity). Straggler-aware routing (DeviceLoadTracker)
measures how long each rank spends in its experts and biases the routers away from slow or overloaded ranks. Tested with
CPU (Gloo) processes; NCCL/XCCL use the same collectives but have not been exercised on multi-GPU hardware.
"""

import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor

from quantweave_moe_model import QuantaWeaveConfig, TopKMoE
from tensor_parallel import is_tp_sharded_shared, merge_state_dicts, shard_state_dict

EXPERT_KEY = re.compile(r"(experts\.)(\d+)(\.)")
BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")


@dataclass
class ExpertParallelContext:
    rank: int                       # position along the expert-parallel dimension
    world_size: int                 # expert-parallel size
    device: torch.device
    group: Optional[object] = None  # expert-parallel group (None: the default group, when there is no tensor parallelism)
    owns_process_group: bool = False
    tp_rank: int = 0
    tp_size: int = 1
    tp_group: Optional[object] = None
    global_rank: int = -1
    global_size: int = -1

    def __post_init__(self) -> None:
        if self.global_size < 0:
            self.global_size = self.world_size * self.tp_size
        if self.global_rank < 0:
            self.global_rank = self.rank * self.tp_size + self.tp_rank

    def layout(self) -> dict:
        return {"kind": "ep", "ep_rank": self.rank, "ep_size": self.world_size, "tp_rank": self.tp_rank, "tp_size": self.tp_size}

    @property
    def group_root(self) -> int:
        """Global rank of expert-parallel rank 0 within this rank's expert-parallel group."""
        return dist.get_global_rank(self.group, 0) if self.group is not None else 0


def init_expert_parallel(device_arg: str = "auto", tensor_parallel: int = 1) -> ExpertParallelContext:
    """Join (or create) the process group described by torchrun's environment variables."""
    owns = False
    if not dist.is_initialized():
        missing = [name for name in ("RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError("--expert-parallel needs a process group; launch with torchrun (missing " + ", ".join(missing) + ")")
        local = int(os.environ.get("LOCAL_RANK", 0))
        if device_arg in {"auto", "cuda", "rocm"} and torch.cuda.is_available():
            device, backend = torch.device("cuda", local), "nccl"
            torch.cuda.set_device(device)
        elif device_arg in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
            device, backend = torch.device("xpu", local), "ccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        owns = True
    else:
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    world, rank = dist.get_world_size(), dist.get_rank()
    if tensor_parallel < 1 or world % tensor_parallel:
        raise ValueError(f"world size {world} is not divisible by --tensor-parallel {tensor_parallel}")
    ep_size = world // tensor_parallel
    ep_rank, tp_rank = divmod(rank, tensor_parallel)
    ep_group = tp_group = None
    if tensor_parallel > 1:                       # every rank must create every group, in the same order
        for column in range(tensor_parallel):
            group = dist.new_group([row * tensor_parallel + column for row in range(ep_size)])
            ep_group = group if column == tp_rank else ep_group
        for row in range(ep_size):
            group = dist.new_group([row * tensor_parallel + column for column in range(tensor_parallel)])
            tp_group = group if row == ep_rank else tp_group
    return ExpertParallelContext(ep_rank, ep_size, device, ep_group, owns, tp_rank, tensor_parallel, tp_group, rank, world)


class _AllToAll(torch.autograd.Function):
    """all_to_all_single with uneven splits whose backward is the reverse exchange."""

    @staticmethod
    def forward(ctx, x: Tensor, in_splits: list[int], out_splits: list[int], group):
        ctx.in_splits, ctx.out_splits, ctx.group = in_splits, out_splits, group
        x = x.contiguous()
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x, out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        grad = grad.contiguous()
        out = grad.new_empty((sum(ctx.in_splits),) + tuple(grad.shape[1:]))
        dist.all_to_all_single(out, grad, ctx.in_splits, ctx.out_splits, group=ctx.group)
        return out, None, None, None


class ExpertParallelMoE(TopKMoE):
    def __init__(self, config: QuantaWeaveConfig, context: ExpertParallelContext) -> None:
        if config.num_experts % context.world_size:
            raise ValueError(f"num_experts {config.num_experts} must be divisible by the expert-parallel size {context.world_size}")
        super().__init__(config, num_local_experts=config.num_experts // context.world_size, tensor_parallel=context)
        self.context = context
        # per-rank work accounting for DeviceLoadTracker
        self.measure_load = False
        self.simulated_cost_per_row = 0.0   # test hook: charge this many seconds per processed row instead of wall time
        self.load_rows = 0
        self.load_seconds = 0.0

    def _execute(self, flat_hidden: Tensor, tokens: Tensor, weights: Tensor, kept_counts: Tensor) -> Tensor:
        ctx, group = self.context, self.context.group
        world, local = ctx.world_size, self.num_local_experts
        counts = kept_counts.tolist()
        send_splits = [sum(counts[r * local : (r + 1) * local]) for r in range(world)]
        # every rank learns how many rows each source rank sends for each of its local experts
        recv_counts = torch.empty_like(kept_counts)
        dist.all_to_all_single(recv_counts, kept_counts.contiguous(), group=group)
        per_source = recv_counts.reshape(world, local).tolist()
        recv_splits = [sum(row) for row in per_source]

        received = self._tp_in(_AllToAll.apply(flat_hidden[tokens], send_splits, recv_splits, group))
        # received rows are ordered (source rank, local expert); gather each local expert's rows across sources
        offset, positions = 0, [[] for _ in range(local)]
        for source in range(world):
            for expert_index in range(local):
                count = per_source[source][expert_index]
                positions[expert_index].append(torch.arange(offset, offset + count, device=received.device))
                offset += count
        started = self._start_clock(received) if self.measure_load else None
        processed = received.new_zeros(received.shape)
        for expert_index, expert in enumerate(self.experts):
            rows = torch.cat(positions[expert_index])
            if rows.numel():
                result = expert(received[rows]).to(received.dtype)
                processed = processed.index_copy(0, rows, result)
        if started is not None:
            self.load_rows += received.size(0)
            self.load_seconds += self._stop_clock(received, started) + received.size(0) * self.simulated_cost_per_row
        returned = _AllToAll.apply(self._tp_out(processed), recv_splits, send_splits, group)

        output = torch.zeros_like(flat_hidden)
        output.index_add_(0, tokens, (returned * weights[:, None]).to(output.dtype))
        return output

    @staticmethod
    def _start_clock(reference: Tensor) -> float:
        if reference.device.type in {"cuda", "xpu"}:
            getattr(torch, reference.device.type).synchronize()
        return time.perf_counter()

    @staticmethod
    def _stop_clock(reference: Tensor, started: float) -> float:
        if reference.device.type in {"cuda", "xpu"}:
            getattr(torch, reference.device.type).synchronize()
        return time.perf_counter() - started


# ---- parameter classes ----------------------------------------------------------------------------------
def is_expert_parameter(name: str) -> bool:
    return ".moe.experts." in name


def broadcast_shared_parameters(model, ctx: ExpertParallelContext) -> None:
    """Make replicated tensors identical: attention slices across the expert-parallel group, everything else
    replicated across the whole world. Experts stay rank-local."""
    for name, parameter in model.named_parameters():
        if is_expert_parameter(name):
            continue
        if is_tp_sharded_shared(name):
            if ctx.world_size > 1:
                dist.broadcast(parameter.data, src=ctx.group_root, group=ctx.group)
        else:
            dist.broadcast(parameter.data, src=0)
    for _, buffer in model.named_buffers():
        dist.broadcast(buffer.data, src=0)


def sync_gradients(model, ctx: ExpertParallelContext, sharded_optimizer=None) -> None:
    """Average replicated gradients across the expert-parallel group; scale expert gradients by 1/ep.

    With a sharded optimizer, replicated gradients are reduced to their owner instead of all-reduced."""
    shared = [p for name, p in model.named_parameters() if not is_expert_parameter(name)]
    if ctx.world_size > 1:
        if sharded_optimizer is not None:
            sharded_optimizer.reduce_gradients()
        else:
            for parameter in shared:
                if parameter.grad is None:      # collectives must match across ranks
                    parameter.grad = torch.zeros_like(parameter)
            flat = torch.cat([p.grad.flatten() for p in shared])
            dist.all_reduce(flat, group=ctx.group)
            flat /= ctx.world_size
            offset = 0
            for parameter in shared:
                parameter.grad.copy_(flat[offset : offset + parameter.numel()].view_as(parameter))
                offset += parameter.numel()
    for name, parameter in model.named_parameters():
        if is_expert_parameter(name) and parameter.grad is not None:
            parameter.grad /= ctx.world_size


def clip_grad_norm_parallel(model, ctx: ExpertParallelContext, max_norm: float, sharded: bool = False) -> float:
    """Clip by the true global gradient norm, identically on every rank.

    A per-rank norm (torch's clip_grad_norm_) would mix each rank's own experts into the scale, so ranks would clip
    differently and their replicated weights would drift apart. Here every distinct parameter is counted exactly once:
    replicated tensors once, tensor-parallel slices summed over the tensor group, experts summed over every rank.
    ``sharded``: replicated gradients exist only on their owner rank, so they are summed over the expert group."""
    device = ctx.device
    replicated = tp_slices = experts = torch.zeros((), dtype=torch.float64, device=device)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        square = parameter.grad.detach().double().pow(2).sum()
        if is_expert_parameter(name):
            experts = experts + square
        elif is_tp_sharded_shared(name):
            tp_slices = tp_slices + square
        else:
            replicated = replicated + square
    if sharded and ctx.world_size > 1:
        replicated, tp_slices = replicated.clone(), tp_slices.clone()
        dist.all_reduce(replicated, group=ctx.group)
        dist.all_reduce(tp_slices, group=ctx.group)
    if ctx.tp_size > 1:
        tp_slices = tp_slices.clone()
        dist.all_reduce(tp_slices, group=ctx.tp_group)
    experts = experts.clone()
    dist.all_reduce(experts)                          # experts are distinct on every rank of the whole world
    total = (replicated + tp_slices + experts).sqrt()
    coefficient = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return float(total)


def all_reduce_mean(values: list[float], ctx: ExpertParallelContext) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(tensor, group=ctx.group)
    return (tensor / ctx.world_size).tolist()


def barrier(ctx=None) -> None:
    dist.barrier()


def finish(ctx) -> None:
    barrier(ctx)
    if getattr(ctx, "owns_process_group", False) and dist.is_initialized():
        dist.destroy_process_group()


# ---- straggler-aware routing -------------------------------------------------------------------------------
class DeviceLoadTracker:
    """Measures per-rank expert compute and steers routing (and, optionally, capacity) away from slow or
    overloaded ranks.

    Every ``interval`` steps each rank's accumulated expert time (and processed rows) per layer is gathered. This is
    an integral controller: a rank whose time is above the group mean has its routing bias lowered a little, a fast rank
    raised,  bias_rank -= rate * strength * log(time_rank / mean_time),  clamped to +-max_bias and re-centred. The bias
    is added to the router logits when experts are *chosen* (not to the mixing weights), so a hot rank sheds traffic
    without its experts being trusted less. Because the bias accumulates, it holds its value once the ranks are balanced
    instead of snapping back (which would oscillate). Time grows with rows and with per-row cost, so this balances both
    load and device speed. Biases are computed identically on every rank so replicated routers agree.

    ``adapt_capacity=True`` adds a second, faster-reacting lever on top of the bias: each rank's own expert capacity
    (``TopKMoE.capacity_scale``) is set to ``clamp(mean_time / this_rank_time, 0.5, 2.0)`` every interval, so a rank
    that is still slow *after* the routing bias has taken effect (an imperfectly-balanced bias, or a genuinely
    popular local expert) sheds the excess by dropping instead of falling behind. This is a real capacity response,
    not just less traffic being routed there.

    ``gather_expert_utilization()`` gathers every rank's own per-(global)-expert routed-token counts for the
    data batch it processed (only meaningful for a step where the model's ``collect_stats`` was on) into one
    ``[world_size, layers, num_experts]`` tensor: since each rank sees different data through the same
    fully-replicated router, this shows how the *global* expert distribution differs by device/data-shard, for a
    genuine per-device routing report instead of only the coordinating rank's local view (see
    ``routing_diagnostics.RoutingMonitor.record_device_utilization``).
    """

    def __init__(self, model, ctx: ExpertParallelContext, strength: float = 0.5, max_bias: float = 2.0,
                 rate: float = 0.5, interval: int = 10, adapt_capacity: bool = False,
                 capacity_scale_min: float = 0.5, capacity_scale_max: float = 2.0) -> None:
        self.model, self.ctx = model, ctx
        self.strength, self.max_bias, self.rate, self.interval = strength, max_bias, rate, interval
        self.adapt_capacity = adapt_capacity
        self.capacity_scale_min, self.capacity_scale_max = capacity_scale_min, capacity_scale_max
        self.layers = len(model.moes())
        self.rank_bias = torch.zeros(self.layers, ctx.world_size, dtype=torch.float64)
        self.history: list[dict] = []
        for moe in model.moes():
            moe.measure_load = True

    def _gather(self) -> tuple[Tensor, Tensor]:
        local = torch.tensor([[moe.load_seconds, float(moe.load_rows)] for moe in self.model.moes()], dtype=torch.float64, device=self.ctx.device)
        for moe in self.model.moes():
            moe.load_seconds, moe.load_rows = 0.0, 0
        if self.ctx.tp_size > 1:                        # tensor-parallel replicas must agree on the numbers
            dist.all_reduce(local, group=self.ctx.tp_group)
            local /= self.ctx.tp_size
        gathered = [torch.zeros_like(local) for _ in range(self.ctx.world_size)]
        dist.all_gather(gathered, local, group=self.ctx.group)
        stacked = torch.stack(gathered, dim=1).cpu()    # [layers, ep, 2]
        return stacked[..., 0], stacked[..., 1]

    def update(self, step: int) -> Optional[dict]:
        if step % self.interval:
            return None
        seconds, rows = self._gather()
        per_rank_seconds, per_rank_rows = seconds.sum(dim=0), rows.sum(dim=0)
        metrics = {
            "step": step,
            "device_rows": per_rank_rows.tolist(),
            "device_seconds": per_rank_seconds.tolist(),
            "rows_imbalance": float(per_rank_rows.max() / per_rank_rows.mean().clamp(min=1e-9)),
            "time_imbalance": float(per_rank_seconds.max() / per_rank_seconds.mean().clamp(min=1e-9)),
        }
        if self.strength > 0 and float(seconds.sum()) > 0:
            relative = seconds / seconds.mean(dim=1, keepdim=True).clamp(min=1e-12)
            correction = -self.strength * torch.log(relative.clamp(min=0.05))
            bias = (self.rank_bias + self.rate * correction).clamp(-self.max_bias, self.max_bias)
            self.rank_bias = bias - bias.mean(dim=1, keepdim=True)       # only differences between ranks matter
            self._apply()
        metrics["rank_bias"] = self.rank_bias.mean(dim=0).tolist()
        if self.adapt_capacity and float(seconds.sum()) > 0:
            mean_time = seconds.mean(dim=1, keepdim=True).clamp(min=1e-12)
            scale = (mean_time / seconds.clamp(min=1e-12)).clamp(self.capacity_scale_min, self.capacity_scale_max)
            for layer, moe in enumerate(self.model.moes()):
                moe.capacity_scale = float(scale[layer, self.ctx.rank])
            metrics["capacity_scale"] = scale.mean(dim=0).tolist()
        self.history.append(metrics)
        return metrics

    def _apply(self) -> None:
        per_rank = self.model.moes()[0].num_local_experts
        for layer, moe in enumerate(self.model.moes()):
            bias = self.rank_bias[layer].repeat_interleave(per_rank).to(torch.float32)
            moe.expert_bias = bias.to(self.ctx.device)

    def gather_expert_utilization(self) -> Tensor:
        """Every rank's own ``expert_load`` (from ``TopKMoE.last_stats``, i.e. requires ``collect_stats=True`` for
        this forward pass — a [num_experts] count over ALL global experts, since the router is fully replicated and
        scores every expert regardless of expert-parallel sharding), gathered into ``[world_size, layers,
        num_experts]`` and returned on every rank."""
        local = torch.stack([moe.last_stats["expert_load"].to(torch.float64) for moe in self.model.moes()]).to(self.ctx.device)
        if self.ctx.tp_size > 1:
            dist.all_reduce(local, group=self.ctx.tp_group)
            local /= self.ctx.tp_size
        gathered = [torch.zeros_like(local) for _ in range(self.ctx.world_size)]
        dist.all_gather(gathered, local, group=self.ctx.group)
        return torch.stack(gathered).cpu()  # [world_size, layers, num_experts]

    def state_dict(self) -> dict:
        return {"rank_bias": self.rank_bias.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.rank_bias = torch.tensor(state["rank_bias"], dtype=torch.float64)
        if bool(self.rank_bias.abs().sum() > 0):
            self._apply()


# ---- sharded checkpoints ------------------------------------------------------------------------------------
def shard_path(directory: Path, rank: int) -> Path:
    return directory / f"model.rank{rank}.pt"


def sharded_checkpoint_exists(directory) -> bool:
    return shard_path(directory, 0).exists()


def save_sharded_checkpoint(directory, model, optimizer, step: int, tokenizer, ctx, extra: dict) -> None:
    """Each rank writes its own shard (replaced atomically); rank 0 adds config, tokenizer and metadata."""
    import random

    from train_quantweave_moe import write_checkpoint_metadata

    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".model.rank{ctx.global_rank}.pt.tmp"
    torch.save(
        {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
            "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state()},
            "world_size": ctx.global_size, "rank": ctx.global_rank, "layout": ctx.layout(), "extra": extra,
        },
        temporary,
    )
    os.replace(temporary, shard_path(directory, ctx.global_rank))
    barrier(ctx)
    if ctx.global_rank == 0:
        write_checkpoint_metadata(directory, model.config, tokenizer, step, extra)
    barrier(ctx)


def load_sharded_checkpoint(directory, model, optimizer, ctx, extra_out: Optional[dict] = None) -> int:
    """Load a rank's shard. ``optimizer: null`` (as written by ``reshard_checkpoint``, which has no optimizer
    state to reshard) is valid: the model loads and training resumes with a fresh optimizer, same as loading any
    weights-only checkpoint."""
    from train_quantweave_moe import restore_rng

    checkpoint = torch.load(shard_path(directory, ctx.global_rank), map_location=ctx.device, weights_only=False)
    if checkpoint["world_size"] != ctx.global_size or checkpoint.get("layout", ctx.layout()) != ctx.layout():
        raise ValueError(f"checkpoint was written with layout {checkpoint.get('layout')} on {checkpoint['world_size']} ranks "
                         f"but this run is {ctx.layout()} on {ctx.global_size}")
    model.load_state_dict(checkpoint["model"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint.get("rng_state", {}), ctx.device)
    if extra_out is not None:
        extra_out.update(checkpoint.get("extra", {}))
    return int(checkpoint["step"])


def _merge_expert_parallel(states: list[tuple[int, dict]]) -> dict:
    """states: [(ep_rank, tp-merged state)] -> one state with experts renumbered globally."""
    per_rank = max(int(m.group(2)) for key in states[0][1] for m in [EXPERT_KEY.search(key)] if m) + 1
    merged: dict = {}
    for ep_rank, state in states:
        for key, tensor in state.items():
            match = EXPERT_KEY.search(key)
            if match is None:
                merged.setdefault(key, tensor)        # replicated across the expert group: rank 0's copy
            else:
                index = ep_rank * per_rank + int(match.group(2))
                merged[EXPERT_KEY.sub(lambda m: f"{m.group(1)}{index}{m.group(3)}", key, count=1)] = tensor
    return merged


def _merge_pipeline(shards: list[dict]) -> dict:
    merged: dict = {}
    for shard in sorted(shards, key=lambda s: s["layout"]["stage"]):
        start = shard["layout"]["layer_start"]
        for key, tensor in shard["model"].items():
            match = BLOCK_KEY.match(key)
            if match:
                key = f"blocks.{start + int(match.group(1))}." + key[match.end():]
            merged[key] = tensor
    return merged


def reshard_checkpoint(checkpoint_dir, output_dir, ep_size: int, tp_size: int = 1) -> dict:
    """The inverse of ``consolidate_checkpoint``: split a single-file checkpoint into ``ep_size x tp_size`` shards
    for a *different* expert-parallel / tensor-parallel world size than it was made with (or made from scratch,
    dense or otherwise). The usual path is train -> consolidate -> reshard for more (or fewer) GPUs -> resume.

    Weights only, like ``consolidate_checkpoint``: there is no optimizer state to reshard, since the source is an
    ordinary checkpoint. Training resumed from a reshard starts with a fresh optimizer, same as loading any
    checkpoint into a new run without its own saved optimizer state.
    """
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "config.json").read_text())
    if config["num_experts"] % ep_size:
        raise ValueError(f"num_experts {config['num_experts']} must be divisible by ep_size {ep_size}")
    if tp_size > 1:
        if config["attention_heads"] % tp_size:
            raise ValueError(f"attention_heads {config['attention_heads']} must be divisible by tp_size {tp_size}")
        if config["ffn_size"] % tp_size:
            raise ValueError(f"ffn_size {config['ffn_size']} must be divisible by tp_size {tp_size}")
    checkpoint = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    per_rank = config["num_experts"] // ep_size

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ep_rank in range(ep_size):
        keep = set(range(ep_rank * per_rank, (ep_rank + 1) * per_rank))
        ep_state: dict = {}
        for key, tensor in state.items():
            match = EXPERT_KEY.search(key)
            if match is None:
                ep_state[key] = tensor
                continue
            index = int(match.group(2))
            if index in keep:
                ep_state[EXPERT_KEY.sub(lambda m: f"{m.group(1)}{index - ep_rank * per_rank}{m.group(3)}", key, count=1)] = tensor
        for tp_rank in range(tp_size):
            shard_state = shard_state_dict(ep_state, tp_rank, tp_size) if tp_size > 1 else ep_state
            global_rank = ep_rank * tp_size + tp_rank
            torch.save({
                "model": shard_state, "optimizer": None, "step": checkpoint.get("step", 0),
                "rng_state": {}, "world_size": ep_size * tp_size, "rank": global_rank,
                "layout": {"kind": "ep", "ep_rank": ep_rank, "ep_size": ep_size, "tp_rank": tp_rank, "tp_size": tp_size},
                "extra": checkpoint.get("extra", {}),
            }, shard_path(output_dir, global_rank))
    for name in ("config.json", "vocab.json", "tokenizer.json", "tokenizer_meta.json", "metadata.json"):
        if (checkpoint_dir / name).exists():
            shutil.copy(checkpoint_dir / name, output_dir / name)
    return {"ep_size": ep_size, "tp_size": tp_size, "world_size": ep_size * tp_size, "per_rank_experts": per_rank,
            "shards": [str(shard_path(output_dir, r)) for r in range(ep_size * tp_size)]}


def consolidate_checkpoint(shards_dir, output_dir) -> None:
    """Merge rank shards (expert-, tensor- and/or pipeline-parallel) into an ordinary single-file checkpoint.

    Weights only: optimizer state stays in the shards."""
    shards, rank = [], 0
    while shard_path(shards_dir, rank).exists():
        shards.append(torch.load(shard_path(shards_dir, rank), map_location="cpu", weights_only=False))
        rank += 1
    if not shards:
        raise FileNotFoundError(f"no shards under {shards_dir}")
    world = shards[0]["world_size"]
    if len(shards) != world:
        raise ValueError(f"expected {world} shards under {shards_dir}, found {len(shards)}")
    for shard in shards:                                     # shards written before layouts existed are expert-parallel only
        shard.setdefault("layout", {"kind": "ep", "ep_rank": shard["rank"], "ep_size": world, "tp_rank": 0, "tp_size": 1})
    if shards[0]["layout"]["kind"] == "pp":
        merged = _merge_pipeline(shards)
    else:
        by_ep: dict[int, list[dict]] = {}
        for shard in shards:
            by_ep.setdefault(shard["layout"]["ep_rank"], []).append(shard)
        states = []
        for ep_rank in sorted(by_ep):
            group = sorted(by_ep[ep_rank], key=lambda s: s["layout"]["tp_rank"])
            tp_states = [s["model"] for s in group]
            states.append((ep_rank, merge_state_dicts(tp_states) if len(tp_states) > 1 else tp_states[0]))
        merged = _merge_expert_parallel(states)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged, "optimizer": None, "step": shards[0]["step"], "rng_state": {}, "extra": shards[0]["extra"]}, output_dir / "model.pt")
    for name in ("config.json", "vocab.json", "tokenizer.json", "tokenizer_meta.json", "metadata.json"):
        if (shards_dir / name).exists():
            shutil.copy(shards_dir / name, output_dir / name)
