"""Tensor parallelism (Megatron style) for attention and expert feed-forward layers.

Within a tensor-parallel group every rank holds a slice of each weight matrix and the activations between layers are
replicated:

  attention  q/k/v are column-parallel (each rank owns heads/tp heads), the output projection is row-parallel and its
             partial results are summed with an all-reduce, then the (replicated) bias is added.
  experts    gate/up are column-parallel and down is row-parallel, so one all-reduce per MoE layer restores the
             full expert output.

Two conjugate autograd operators keep gradients right: copy_to_tp is the identity forward and an all-reduce backward
(placed where a replicated activation enters a parallel region); reduce_from_tp is an all-reduce forward and the
identity backward (where partial sums leave it). Embeddings, norms, routers and lm_head stay replicated.

State-dict slicing helpers turn a full checkpoint into per-rank shards and back, so tensor-parallel runs can start
from, and consolidate into, ordinary checkpoints.
"""

import math
import re
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class _CopyToTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, group):
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad, None


def copy_to_tp(x: Tensor, group) -> Tensor:
    return _CopyToTP.apply(x, group) if group is not None else x


def reduce_from_tp(x: Tensor, group) -> Tensor:
    return _ReduceFromTP.apply(x, group) if group is not None else x


class _RowParallelOutput(nn.Module):
    """Holds the output projection: weight [hidden, local_width] (row-parallel slice) and a replicated bias."""

    def __init__(self, local_width: int, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, local_width))
        self.bias = nn.Parameter(torch.zeros(hidden))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class TensorParallelAttention(nn.Module):
    """Causal multi-head self-attention with heads split across the tensor-parallel group.

    Parameter names and layout match nn.MultiheadAttention (in_proj_weight, in_proj_bias, out_proj.weight/bias) so
    full checkpoints can be sliced into and merged out of these shards.
    """

    def __init__(self, hidden_size: int, heads: int, tp_size: int, tp_group) -> None:
        super().__init__()
        if heads % tp_size:
            raise ValueError(f"attention_heads {heads} must be divisible by the tensor-parallel size {tp_size}")
        self.heads_local = heads // tp_size
        self.head_dim = hidden_size // heads
        self.local_width = self.heads_local * self.head_dim
        self.tp_group = tp_group
        self.in_proj_weight = nn.Parameter(torch.empty(3 * self.local_width, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * self.local_width))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj = _RowParallelOutput(self.local_width, hidden_size)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, need_weights: bool = False):
        batch, sequence, _ = query.shape
        x = copy_to_tp(query, self.tp_group)
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        split = lambda t: t.reshape(batch, sequence, self.heads_local, self.head_dim).transpose(1, 2)  # noqa: E731
        attended = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, sequence, self.local_width)
        partial = F.linear(attended, self.out_proj.weight)
        return reduce_from_tp(partial, self.tp_group) + self.out_proj.bias.to(partial.dtype), None


# ---- checkpoint slicing ------------------------------------------------------------------------------
_QKV = re.compile(r"\.attention\.in_proj_(weight|bias)$")
_OUT_WEIGHT = re.compile(r"\.attention\.out_proj\.weight$")
_COLUMN_EXPERT = re.compile(r"\.experts\.\d+\.(gate|up)\.weight$")
_ROW_EXPERT = re.compile(r"\.experts\.\d+\.down\.weight$")


def is_tp_sharded(name: str) -> bool:
    """True for tensors whose value differs across the tensor-parallel group."""
    return bool(_QKV.search(name) or _OUT_WEIGHT.search(name) or _COLUMN_EXPERT.search(name) or _ROW_EXPERT.search(name))


def is_tp_sharded_shared(name: str) -> bool:
    """TP-sharded but not an expert: replicated across the expert-parallel group, distinct across the TP group."""
    return bool(_QKV.search(name) or _OUT_WEIGHT.search(name))


def shard_tensor(name: str, tensor: Tensor, tp_rank: int, tp_size: int) -> Tensor:
    if _QKV.search(name):
        parts = tensor.chunk(3, dim=0)                                     # q, k, v
        return torch.cat([p.chunk(tp_size, dim=0)[tp_rank] for p in parts], dim=0).contiguous()
    if _OUT_WEIGHT.search(name) or _ROW_EXPERT.search(name):
        return tensor.chunk(tp_size, dim=1)[tp_rank].contiguous()
    if _COLUMN_EXPERT.search(name):
        return tensor.chunk(tp_size, dim=0)[tp_rank].contiguous()
    return tensor


def shard_state_dict(state: dict[str, Tensor], tp_rank: int, tp_size: int) -> dict[str, Tensor]:
    """Slice a full model state dict down to one tensor-parallel rank."""
    return {name: shard_tensor(name, tensor, tp_rank, tp_size) for name, tensor in state.items()}


def merge_tensors(name: str, pieces: list[Tensor]) -> Tensor:
    if _QKV.search(name):
        thirds = [p.chunk(3, dim=0) for p in pieces]                      # per rank: (q, k, v)
        return torch.cat([torch.cat([t[i] for t in thirds], dim=0) for i in range(3)], dim=0)
    if _OUT_WEIGHT.search(name) or _ROW_EXPERT.search(name):
        return torch.cat(pieces, dim=1)
    if _COLUMN_EXPERT.search(name):
        return torch.cat(pieces, dim=0)
    return pieces[0]


def merge_state_dicts(shards: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Inverse of shard_state_dict over all tensor-parallel ranks (ordered by rank)."""
    return {name: merge_tensors(name, [shard[name] for shard in shards]) for name in shards[0]}
