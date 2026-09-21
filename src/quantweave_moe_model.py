"""Configurable sparse-MoE causal language model for QuantaWeave."""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tensor_parallel import TensorParallelAttention, copy_to_tp, reduce_from_tp


@dataclass
class QuantaWeaveConfig:
    vocab_size: int = 7168
    hidden_size: int = 64
    layers: int = 2
    ffn_size: int = 128
    num_experts: int = 184
    top_k: int = 1
    attention_heads: int = 4
    max_sequence_length: int = 256
    router_aux_loss_coef: float = 0.01
    capacity_factor: float = 1.25
    min_expert_capacity: int = 4
    drop_overflow_tokens: bool = True
    overflow_policy: str = "drop"
    router_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2 or self.hidden_size < 1 or self.layers < 1:
            raise ValueError("vocab_size, hidden_size, and layers must be positive")
        if self.ffn_size < 1 or self.attention_heads < 1:
            raise ValueError("ffn_size and attention_heads must be positive")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.num_experts < 1 or not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        if self.capacity_factor < 0 or self.min_expert_capacity < 1:
            raise ValueError("capacity_factor must be >= 0 and min_expert_capacity must be positive")
        if self.overflow_policy not in {"drop", "residual"}:
            raise ValueError("overflow_policy must be 'drop' or 'residual'")
        if self.router_temperature <= 0:
            raise ValueError("router_temperature must be positive")


class SparseSwiGLU(nn.Module):
    def __init__(self, config: QuantaWeaveConfig, ffn_size: Optional[int] = None) -> None:
        super().__init__()
        ffn_size = ffn_size or config.ffn_size
        self.gate = nn.Linear(config.hidden_size, ffn_size, bias=False)
        self.up = nn.Linear(config.hidden_size, ffn_size, bias=False)
        self.down = nn.Linear(ffn_size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class TopKMoE(nn.Module):
    """Top-k routed mixture of SwiGLU experts.

    Routing runs in fp32 even under autocast. Tokens are dispatched with one
    vectorised sort (no per-expert masks): routes are ordered by expert and,
    within an expert, by router weight, so capacity keeps the strongest routes.
    """

    def __init__(self, config: QuantaWeaveConfig, num_local_experts: Optional[int] = None, tensor_parallel=None) -> None:
        super().__init__()
        if config.top_k < 1 or config.top_k > config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.tp_size = getattr(tensor_parallel, "tp_size", 1)
        self.tp_group = getattr(tensor_parallel, "tp_group", None) if self.tp_size > 1 else None
        if config.ffn_size % self.tp_size:
            raise ValueError(f"ffn_size {config.ffn_size} must be divisible by the tensor-parallel size {self.tp_size}")
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.min_expert_capacity = config.min_expert_capacity
        self.drop_overflow_tokens = config.drop_overflow_tokens
        self.overflow_policy = config.overflow_policy
        self.router_temperature = config.router_temperature
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.num_local_experts = num_local_experts or config.num_experts
        self.experts = nn.ModuleList(SparseSwiGLU(config, config.ffn_size // self.tp_size) for _ in range(self.num_local_experts))
        # runtime switches; none of these are parameters or saved state
        self.collect_stats = False
        self.static_dispatch = False
        self.profile_experts = False
        self.use_triton_kernels = False
        self.expert_bias: Optional[Tensor] = None   # straggler-aware routing: added to logits when choosing experts
        self.expert_seconds = [0.0] * self.num_local_experts
        self.last_top_indices: Optional[Tensor] = None
        self.last_stats: dict[str, Tensor] = {}

    def capacity(self, num_tokens: int) -> int:
        """Route slots per expert; everything fits when capacity is disabled."""
        if self.capacity_factor > 0 and self.drop_overflow_tokens:
            return max(
                self.min_expert_capacity,
                math.ceil(num_tokens * self.top_k / self.num_experts * self.capacity_factor),
            )
        return num_tokens * self.top_k

    def forward(
        self, hidden: Tensor, token_domains: Optional[Tensor] = None, num_domains: int = 0
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Route tokens to experts.

        Returns (output, balance_loss, dropped_routes, overflow_routes, domain_loss).
        A route that does not fit its expert's capacity is skipped and the token
        keeps its residual stream unchanged for that route. Under the "drop"
        policy skipped routes are reported as dropped_routes; under "residual"
        they are only reported in overflow_routes.
        """
        batch, sequence, width = hidden.shape
        flat_hidden = hidden.reshape(-1, width)
        num_tokens = flat_hidden.size(0)
        with torch.autocast(hidden.device.type, enabled=False):
            router_logits = F.linear(flat_hidden.float(), self.router.weight.float()) / self.router_temperature
            router_probs = router_logits.softmax(dim=-1)
        if self.expert_bias is None:
            top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)
        else:
            # the bias steers which experts are chosen but not how much they are trusted: weights stay unbiased
            top_indices = (router_logits + self.expert_bias.to(router_logits)).topk(self.top_k, dim=-1).indices
            top_weights = router_probs.gather(1, top_indices)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        self.last_top_indices = top_indices.detach()

        zero = torch.zeros((), device=hidden.device, dtype=torch.long)
        overflow_routes = zero
        if self.static_dispatch:
            output = self._static_output(flat_hidden, router_probs, top_indices, top_weights)
        else:
            tokens, weights, kept_counts, overflow_routes, dropped = self._plan(
                top_indices, top_weights, self.capacity(num_tokens)
            )
            output = self._execute(flat_hidden, tokens, weights, kept_counts)
            if self.collect_stats:
                self._record_stats(router_probs, top_indices, kept_counts, dropped)
        dropped_routes = overflow_routes if self.overflow_policy == "drop" else zero

        importance = router_probs.mean(dim=0)
        assignments = F.one_hot(top_indices, self.num_experts).float().mean(dim=(0, 1))
        balance_loss = self.num_experts * (importance * assignments).sum()
        domain_loss = router_probs.new_zeros(())
        if token_domains is not None and num_domains > 1:
            domain_loss = self._domain_loss(router_probs, token_domains, num_domains)
        return (
            output.reshape(batch, sequence, width),
            balance_loss,
            dropped_routes,
            overflow_routes,
            domain_loss,
        )

    def _plan(self, top_indices: Tensor, top_weights: Tensor, capacity: int):
        """Order routes by (expert, -weight) and drop those past `capacity`.

        Returns (token index per kept route, weight per kept route, kept routes
        per expert, overflow count, (dropped token index, dropped expert)).
        """
        route_experts = top_indices.reshape(-1)
        route_weights = top_weights.reshape(-1)
        by_weight = torch.argsort(route_weights, descending=True, stable=True)
        order = by_weight[torch.argsort(route_experts[by_weight], stable=True)]
        counts = torch.bincount(route_experts, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0) - counts
        position = torch.arange(order.numel(), device=order.device) - offsets[route_experts[order]]
        keep = position < capacity
        kept = order[keep]
        dropped = order[~keep]
        return (
            kept // self.top_k,
            route_weights[kept],
            counts.clamp(max=capacity),
            (~keep).sum(),
            (dropped // self.top_k, route_experts[dropped]),
        )

    def _execute(self, flat_hidden: Tensor, tokens: Tensor, weights: Tensor, kept_counts: Tensor) -> Tensor:
        """Run local experts over their contiguous slice of the sorted routes."""
        output = torch.zeros_like(flat_hidden)
        if tokens.numel() == 0:
            return output
        inputs = self._tp_in(flat_hidden[tokens])
        if self.use_triton_kernels:
            from moe_kernels import experts_support_grouped, grouped_swiglu, kernels_available

            if kernels_available(inputs.device) and experts_support_grouped(self.experts) and not self.profile_experts:
                expert_output = self._tp_out(grouped_swiglu(inputs, self.experts, kept_counts.tolist())) * weights[:, None]
                output.index_add_(0, tokens, expert_output.to(output.dtype))
                return output
        results = []
        start = 0
        for index, (expert, count) in enumerate(zip(self.experts, kept_counts.tolist())):
            if not count:
                continue
            results.append(self._run_expert(index, expert, inputs[start : start + count]))
            start += count
        expert_output = self._tp_out(torch.cat(results)) * weights[:, None]
        output.index_add_(0, tokens, expert_output.to(output.dtype))
        return output

    def _tp_in(self, x: Tensor) -> Tensor:
        return copy_to_tp(x, self.tp_group)

    def _tp_out(self, x: Tensor) -> Tensor:
        return reduce_from_tp(x, self.tp_group)

    def _run_expert(self, index: int, expert: nn.Module, inputs: Tensor) -> Tensor:
        if not self.profile_experts:
            return expert(inputs)
        import time

        device_type = inputs.device.type
        if device_type in {"cuda", "xpu"}:
            getattr(torch, device_type).synchronize()
        started = time.perf_counter()
        result = expert(inputs)
        if device_type in {"cuda", "xpu"}:
            getattr(torch, device_type).synchronize()
        self.expert_seconds[index] += time.perf_counter() - started
        return result

    def _static_output(
        self, flat_hidden: Tensor, router_probs: Tensor, top_indices: Tensor, top_weights: Tensor
    ) -> Tensor:
        """Capacity-free dense evaluation with no data-dependent control flow (for tracing/export)."""
        if self.tp_size > 1:
            raise RuntimeError("static (export) dispatch needs a consolidated, non-tensor-parallel model")
        dense_weights = torch.zeros_like(router_probs).scatter(1, top_indices, top_weights)
        output = torch.zeros_like(flat_hidden)
        for index, expert in enumerate(self.experts):
            output = output + expert(flat_hidden) * dense_weights[:, index : index + 1].to(flat_hidden.dtype)
        return output

    def _domain_loss(self, router_probs: Tensor, token_domains: Tensor, num_domains: int) -> Tensor:
        """Mean pairwise cosine similarity between the routing profiles of the domains in the batch."""
        membership = F.one_hot(token_domains, num_domains).to(router_probs.dtype)
        counts = membership.sum(dim=0)
        profiles = (membership.t() @ router_probs) / counts.clamp(min=1)[:, None]
        profiles = F.normalize(profiles[counts > 0], dim=-1)
        present = profiles.size(0)
        if present < 2:
            return router_probs.new_zeros(())
        similarity = profiles @ profiles.t()
        return (similarity.sum() - similarity.diagonal().sum()) / (present * (present - 1))

    @torch.no_grad()
    def _record_stats(self, router_probs: Tensor, top_indices: Tensor, kept_counts: Tensor, dropped) -> None:
        entropy = -(router_probs * router_probs.clamp_min(1e-12).log()).sum(dim=-1)
        top_probability = router_probs.max(dim=-1).values
        self.last_stats = {
            "expert_load": kept_counts.detach(),
            "expert_assigned": torch.bincount(top_indices.reshape(-1), minlength=self.num_experts),
            "importance": router_probs.mean(dim=0).detach(),
            "entropy_mean": entropy.mean(),
            "confidence_mean": top_probability.mean(),
            "confidence_hist": torch.histc(top_probability.float(), bins=10, min=0.0, max=1.0),
            "dropped_tokens": dropped[0],
            "dropped_experts": dropped[1],
        }


class QuantaWeaveBlock(nn.Module):
    def __init__(self, config: QuantaWeaveConfig, expert_parallel=None) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        tp_size = getattr(expert_parallel, "tp_size", 1)
        if tp_size > 1:
            self.attention = TensorParallelAttention(config.hidden_size, config.attention_heads, tp_size, expert_parallel.tp_group)
        else:
            self.attention = nn.MultiheadAttention(
                config.hidden_size,
                config.attention_heads,
                batch_first=True,
            )
        self.moe_norm = nn.LayerNorm(config.hidden_size)
        if expert_parallel is None:
            self.moe = TopKMoE(config)
        else:
            from expert_parallel import ExpertParallelMoE

            self.moe = ExpertParallelMoE(config, expert_parallel)

    def forward(
        self,
        hidden: Tensor,
        causal_mask: Tensor,
        token_domains: Optional[Tensor] = None,
        num_domains: int = 0,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
        )
        hidden = hidden + attended
        moe_output, balance_loss, dropped, overflow, domain_loss = self.moe(
            self.moe_norm(hidden), token_domains, num_domains
        )
        return hidden + moe_output, balance_loss, dropped, overflow, domain_loss


class QuantaWeaveMoEForCausalLM(nn.Module):
    def __init__(self, config: QuantaWeaveConfig, expert_parallel=None, pipeline=None) -> None:
        """``expert_parallel`` shards experts (and tensor-parallel slices); ``pipeline`` keeps only this stage's layers."""
        super().__init__()
        if expert_parallel is not None and pipeline is not None:
            raise ValueError("pipeline parallelism cannot be combined with expert/tensor parallelism")
        self.config = config
        self.expert_parallel = expert_parallel
        self.pipeline = pipeline
        first = pipeline is None or pipeline.stage == 0
        last = pipeline is None or pipeline.stage == pipeline.num_stages - 1
        layer_count = config.layers if pipeline is None else pipeline.layer_end - pipeline.layer_start
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.domain_specialization_coef = 0.0
        self.activation_checkpointing = False
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size) if first else None
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size) if first else None
        self.blocks = nn.ModuleList(
            QuantaWeaveBlock(config, expert_parallel) for _ in range(layer_count)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size) if last else None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) if last else None
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    # ---- runtime controls -------------------------------------------------
    def moes(self) -> list[TopKMoE]:
        return [block.moe for block in self.blocks]

    def set_routing_controls(
        self,
        capacity_factor: Optional[float] = None,
        drop_overflow_tokens: Optional[bool] = None,
        overflow_policy: Optional[str] = None,
        router_temperature: Optional[float] = None,
        router_aux_loss_coef: Optional[float] = None,
    ) -> None:
        """Change routing settings mid-training. The saved config always reflects the current values."""
        if overflow_policy is not None and overflow_policy not in {"drop", "residual"}:
            raise ValueError("overflow_policy must be 'drop' or 'residual'")
        if router_temperature is not None and router_temperature <= 0:
            raise ValueError("router_temperature must be positive")
        if capacity_factor is not None and capacity_factor < 0:
            raise ValueError("capacity_factor must be >= 0")
        updates = {
            "capacity_factor": capacity_factor,
            "drop_overflow_tokens": drop_overflow_tokens,
            "overflow_policy": overflow_policy,
            "router_temperature": router_temperature,
        }
        for name, value in updates.items():
            if value is None:
                continue
            for moe in self.moes():
                setattr(moe, name, value)
            setattr(self.config, name, value)
        if router_aux_loss_coef is not None:
            self.router_aux_loss_coef = router_aux_loss_coef
            self.config.router_aux_loss_coef = router_aux_loss_coef

    def set_collect_stats(self, enabled: bool) -> None:
        for moe in self.moes():
            moe.collect_stats = enabled

    def set_moe_kernel(self, kernel: str) -> str:
        """Choose the expert compute path: "loop", "triton" (grouped GEMM, CUDA only) or "auto". Returns what is active."""
        if kernel not in {"loop", "triton", "auto"}:
            raise ValueError("kernel must be loop, triton or auto")
        from moe_kernels import kernels_available

        available = kernels_available(next(self.parameters()).device)
        if kernel == "triton" and not available:
            raise RuntimeError("the triton MoE kernels need a CUDA device and the triton package")
        active = "triton" if (kernel == "triton" or (kernel == "auto" and available)) else "loop"
        for moe in self.moes():
            moe.use_triton_kernels = active == "triton"
        return active

    def set_static_dispatch(self, enabled: bool) -> None:
        for moe in self.moes():
            moe.static_dispatch = enabled

    def routing_stats(self) -> list[dict[str, Tensor]]:
        return [moe.last_stats for moe in self.moes()]

    # ---- forward ----------------------------------------------------------
    def embed(self, input_ids: Tensor) -> Tensor:
        _, sequence = input_ids.shape
        if sequence > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        positions = torch.arange(sequence, device=input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]

    def run_blocks(self, hidden: Tensor, token_domains: Optional[Tensor] = None, num_domains: int = 0):
        """Run this model's transformer blocks; returns (hidden, balance, dropped, overflow, domain) lists."""
        sequence = hidden.size(1)
        mask = self.causal_mask[:sequence, :sequence]
        balance_losses, dropped_counts, overflow_counts, domain_losses = [], [], [], []
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                hidden, balance, dropped, overflow, domain = checkpoint(
                    block, hidden, mask, token_domains, num_domains, use_reentrant=False
                )
            else:
                hidden, balance, dropped, overflow, domain = block(hidden, mask, token_domains, num_domains)
            balance_losses.append(balance)
            dropped_counts.append(dropped)
            overflow_counts.append(overflow)
            domain_losses.append(domain)
        return hidden, balance_losses, dropped_counts, overflow_counts, domain_losses

    def head(self, hidden: Tensor) -> Tensor:
        return self.lm_head(self.final_norm(hidden))

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        domain_ids: Optional[Tensor] = None,
        num_domains: int = 0,
    ) -> dict[str, Tensor | list[Tensor]]:
        if self.pipeline is not None:
            raise RuntimeError("a pipeline-stage model holds only some layers; drive it with PipelineEngine")
        hidden = self.embed(input_ids)
        token_domains = None
        if domain_ids is not None and self.domain_specialization_coef != 0 and num_domains > 1:
            token_domains = domain_ids.repeat_interleave(input_ids.size(1))
        hidden, balance_losses, dropped_counts, overflow_counts, domain_losses = self.run_blocks(hidden, token_domains, num_domains)
        logits = self.head(hidden)
        result: dict[str, Tensor | list[Tensor]] = {
            "logits": logits,
            "router_aux_loss": torch.stack(balance_losses).mean(),
            "domain_loss": torch.stack(domain_losses).mean(),
            "expert_indices": [moe.last_top_indices for moe in self.moes()],
            "dropped_routes": torch.stack(dropped_counts).sum(),
            "overflow_routes": torch.stack(overflow_counts).sum(),
        }
        if labels is not None:
            result["loss"] = (
                F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
                + self.router_aux_loss_coef * result["router_aux_loss"]
                + self.domain_specialization_coef * result["domain_loss"]
            )
        return result
