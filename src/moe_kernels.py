"""Triton grouped-GEMM kernels for the expert feed-forward networks.

The default expert path launches one small matmul per expert and projection. Here routes are already sorted by expert
(see TopKMoE._plan), so each expert owns a contiguous row block, and one kernel launch computes every
expert's block:  Y[rows of e] = X[rows of e] @ W[e]^T.  A SwiGLU expert needs three such products (gate, up, down);
silu(gate) * up stays in PyTorch autograd.

Backward is also grouped: dX = dY @ W (same kernel, transposed strides) and dW[e] = dY[rows of e]^T @ X[rows of e]
(one program per expert and output tile, looping over that expert's rows). Accumulation is fp32; fp32 inputs use
IEEE dot precision so results match the loop path to ~1e-5.

Requires CUDA and Triton. Experts must be plain nn.Linear projections (quantized/LoRA experts use the loop path).
A weight stack is copied each call (torch.stack), so this pays off when there are many experts per layer.
"""

import time
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only installs
    HAS_TRITON = False

BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32


def kernels_available(device: torch.device) -> bool:
    return HAS_TRITON and device.type == "cuda" and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _grouped_gemm_kernel(
        a_ptr, b_ptr, c_ptr, tile_expert_ptr, tile_start_ptr, tile_end_ptr, K, N,
        stride_am, stride_ak, stride_be, stride_bk, stride_bn, stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr,
    ):
        tile = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.load(tile_expert_ptr + tile).to(tl.int64)
        start = tl.load(tile_start_ptr + tile)
        end = tl.load(tile_end_ptr + tile)
        offs_m = (start + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        mask_m = offs_m < end
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = (k + tl.arange(0, BLOCK_K)).to(tl.int64)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptr + expert * stride_be + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                        mask=(offs_k[:, None] < K) & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, input_precision=PRECISION)
        tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(c_ptr.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _grouped_dw_kernel(
        dy_ptr, x_ptr, dw_ptr, expert_start_ptr, expert_end_ptr, N, K,
        stride_dym, stride_dyn, stride_xm, stride_xk, stride_dwe, stride_dwn, stride_dwk,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, PRECISION: tl.constexpr,
    ):
        expert = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        start = tl.load(expert_start_ptr + expert)
        end = tl.load(expert_end_ptr + expert)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        offs_k = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m in range(start, end, BLOCK_M):
            offs_m = (m + tl.arange(0, BLOCK_M)).to(tl.int64)
            mask_m = offs_m < end
            dy = tl.load(dy_ptr + offs_m[None, :] * stride_dym + offs_n[:, None] * stride_dyn,
                         mask=mask_m[None, :] & (offs_n[:, None] < N), other=0.0)
            x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                        mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
            acc = tl.dot(dy, x, acc, input_precision=PRECISION)
        tl.store(dw_ptr + expert.to(tl.int64) * stride_dwe + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk,
                 acc.to(dw_ptr.dtype.element_ty), mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))


class GroupedPlan:
    """Tile schedule for one set of per-expert row counts (rows are sorted by expert)."""

    def __init__(self, counts: Sequence[int], device: torch.device) -> None:
        experts, starts, ends, offset = [], [], [], 0
        expert_starts, expert_ends = [], []
        for expert, count in enumerate(counts):
            expert_starts.append(offset)
            for first in range(0, count, BLOCK_M):
                experts.append(expert)
                starts.append(offset + first)
                ends.append(offset + min(count, first + BLOCK_M))
            offset += count
            expert_ends.append(offset)
        as_tensor = lambda values: torch.tensor(values, dtype=torch.int32, device=device)  # noqa: E731
        self.tiles = len(experts)
        self.rows = offset
        self.num_experts = len(counts)
        self.tile_expert, self.tile_start, self.tile_end = as_tensor(experts), as_tensor(starts), as_tensor(ends)
        self.expert_start, self.expert_end = as_tensor(expert_starts), as_tensor(expert_ends)


def _precision(dtype: torch.dtype) -> str:
    return "ieee" if dtype == torch.float32 else "tf32"


def _grouped_forward(a: Tensor, weight: Tensor, plan: GroupedPlan, transpose: bool) -> Tensor:
    """out[rows of e] = a[rows of e] @ (W[e]^T if not transpose else W[e]);  weight is [E, N_w, K_w]."""
    rows = a.size(0)
    if transpose:                                # reduce over the weight's first matrix dim
        reduction, out_features = weight.size(1), weight.size(2)
        stride_bk, stride_bn = weight.stride(1), weight.stride(2)
    else:
        reduction, out_features = weight.size(2), weight.size(1)
        stride_bk, stride_bn = weight.stride(2), weight.stride(1)
    out = torch.empty(rows, out_features, dtype=a.dtype, device=a.device)
    if plan.tiles:
        grid = (plan.tiles, triton.cdiv(out_features, BLOCK_N))
        _grouped_gemm_kernel[grid](
            a, weight, out, plan.tile_expert, plan.tile_start, plan.tile_end, reduction, out_features,
            a.stride(0), a.stride(1), weight.stride(0), stride_bk, stride_bn, out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, PRECISION=_precision(a.dtype),
        )
    return out


class _GroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, plan: GroupedPlan):
        x, weight = x.contiguous(), weight.contiguous()
        ctx.save_for_backward(x, weight)
        ctx.plan = plan
        return _grouped_forward(x, weight, plan, transpose=False)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        x, weight = ctx.saved_tensors
        plan = ctx.plan
        grad_output = grad_output.contiguous().to(x.dtype)
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = _grouped_forward(grad_output, weight, plan, transpose=True)
        if ctx.needs_input_grad[1]:
            experts, out_features, in_features = weight.shape
            grad_weight = torch.empty_like(weight)
            grid = (experts, triton.cdiv(out_features, BLOCK_N), triton.cdiv(in_features, BLOCK_K))
            _grouped_dw_kernel[grid](
                grad_output, x, grad_weight, plan.expert_start, plan.expert_end, out_features, in_features,
                grad_output.stride(0), grad_output.stride(1), x.stride(0), x.stride(1),
                grad_weight.stride(0), grad_weight.stride(1), grad_weight.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, PRECISION=_precision(x.dtype),
            )
        return grad_x, grad_weight, None


def grouped_matmul(x: Tensor, weight: Tensor, plan: GroupedPlan) -> Tensor:
    """Per-expert x @ W[e]^T over expert-sorted rows; weight is [E, out, in]."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    return _GroupedMatmul.apply(x, weight, plan)


def experts_support_grouped(experts: nn.ModuleList) -> bool:
    return all(isinstance(getattr(e, name, None), nn.Linear) and getattr(e, name).bias is None
               for e in experts for name in ("gate", "up", "down"))


def compute_dtype(inputs: Tensor) -> torch.dtype:
    device_type = inputs.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return inputs.dtype


def grouped_swiglu(inputs: Tensor, experts: nn.ModuleList, counts: Sequence[int], plan: Optional[GroupedPlan] = None) -> Tensor:
    """SwiGLU over expert-sorted rows: down(silu(gate(x)) * up(x)) with each expert's own weights."""
    plan = plan or GroupedPlan(counts, inputs.device)
    dtype = compute_dtype(inputs)
    x = inputs.to(dtype)
    gate = torch.stack([e.gate.weight for e in experts]).to(dtype)
    up = torch.stack([e.up.weight for e in experts]).to(dtype)
    down = torch.stack([e.down.weight for e in experts]).to(dtype)
    hidden = F.silu(grouped_matmul(x, gate, plan)) * grouped_matmul(x, up, plan)
    return grouped_matmul(hidden, down, plan)


def benchmark(experts: int = 32, tokens: int = 8192, hidden: int = 256, ffn: int = 512, top_k: int = 2, repeats: int = 20,
              bf16: bool = False) -> dict:
    """Wall-clock the loop and grouped expert paths on the current GPU (forward + backward), optionally under bf16 autocast."""
    from quantweave_moe_model import QuantaWeaveConfig, TopKMoE

    if not kernels_available(torch.device("cuda")):
        raise RuntimeError("benchmark needs CUDA and Triton")
    config = QuantaWeaveConfig(vocab_size=8, hidden_size=hidden, layers=1, ffn_size=ffn, num_experts=experts, top_k=top_k,
                               attention_heads=1, max_sequence_length=tokens, capacity_factor=0)
    moe = TopKMoE(config).cuda()
    x = torch.randn(1, tokens, hidden, device="cuda", requires_grad=True)
    result = {}
    for name in ("loop", "triton"):
        moe.use_triton_kernels = name == "triton"
        context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
        for _ in range(3):
            with context:
                out = moe(x)[0]
            out.float().sum().backward()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            with context:
                out = moe(x)[0]
            out.float().sum().backward()
        torch.cuda.synchronize()
        result[name + "_ms"] = (time.perf_counter() - started) / repeats * 1000
    result["speedup"] = result["loop_ms"] / result["triton_ms"]
    result.update(experts=experts, tokens=tokens, hidden=hidden, ffn=ffn, top_k=top_k, bf16=bf16)
    return result


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    print(json.dumps(benchmark(), indent=2))
