"""Minimal configurable sparse-MoE causal language model for QuantaWeave."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


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


class SparseSwiGLU(nn.Module):
    def __init__(self, config: QuantaWeaveConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.ffn_size, bias=False)
        self.down = nn.Linear(config.ffn_size, config.hidden_size, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(hidden)) * self.up(hidden))


class TopKMoE(nn.Module):
    def __init__(self, config: QuantaWeaveConfig) -> None:
        super().__init__()
        if config.top_k < 1 or config.top_k > config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.capacity_factor = config.capacity_factor
        self.min_expert_capacity = config.min_expert_capacity
        self.drop_overflow_tokens = config.drop_overflow_tokens
        self.overflow_policy = config.overflow_policy
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SparseSwiGLU(config) for _ in range(config.num_experts)
        )

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, sequence, width = hidden.shape
        flat_hidden = hidden.reshape(-1, width)
        router_logits = self.router(flat_hidden)
        router_probs = router_logits.softmax(dim=-1)
        top_weights, top_indices = router_probs.topk(self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(flat_hidden)
        dropped_routes = torch.zeros((), device=hidden.device, dtype=torch.long)
        capacity = flat_hidden.size(0) * self.top_k
        if self.capacity_factor > 0:
            capacity = max(
                self.min_expert_capacity,
                int((flat_hidden.size(0) * self.top_k / self.num_experts) * self.capacity_factor + 0.999),
            )

        for route in range(self.top_k):
            route_indices = top_indices[:, route]
            route_weights = top_weights[:, route]
            for expert_index, expert in enumerate(self.experts):
                token_indices = (route_indices == expert_index).nonzero(as_tuple=False).flatten()
                if token_indices.numel() == 0:
                    continue
                if self.drop_overflow_tokens and self.overflow_policy == "drop" and token_indices.numel() > capacity:
                    dropped_routes += token_indices.numel() - capacity
                    token_indices = token_indices[:capacity]
                output[token_indices] += (
                    expert(flat_hidden[token_indices]) * route_weights[token_indices, None]
                )

        importance = router_probs.mean(dim=0)
        assignments = F.one_hot(top_indices, self.num_experts).float().mean(dim=(0, 1))
        balance_loss = self.num_experts * (importance * assignments).sum()
        return output.reshape(batch, sequence, width), balance_loss, top_indices, dropped_routes


class QuantaWeaveBlock(nn.Module):
    def __init__(self, config: QuantaWeaveConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.hidden_size)
        self.attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            batch_first=True,
        )
        self.moe_norm = nn.LayerNorm(config.hidden_size)
        self.moe = TopKMoE(config)

    def forward(self, hidden: Tensor, causal_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, attn_mask=causal_mask, need_weights=False
        )
        hidden = hidden + attended
        moe_output, balance_loss, expert_indices, dropped_routes = self.moe(self.moe_norm(hidden))
        return hidden + moe_output, balance_loss, expert_indices, dropped_routes


class QuantaWeaveMoEForCausalLM(nn.Module):
    def __init__(self, config: QuantaWeaveConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_sequence_length, config.hidden_size)
        self.blocks = nn.ModuleList(QuantaWeaveBlock(config) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_sequence_length, config.max_sequence_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(
        self, input_ids: Tensor, labels: Optional[Tensor] = None
    ) -> dict[str, Tensor | list[Tensor]]:
        _, sequence = input_ids.shape
        if sequence > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds max_sequence_length")
        positions = torch.arange(sequence, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        balance_losses = []
        expert_indices = []
        dropped_route_counts = []
        for block in self.blocks:
            hidden, balance_loss, indices, block_dropped_routes = block(
                hidden, self.causal_mask[:sequence, :sequence]
            )
            balance_losses.append(balance_loss)
            expert_indices.append(indices)
            dropped_route_counts.append(block_dropped_routes)
        logits = self.lm_head(self.final_norm(hidden))
        result: dict[str, Tensor | list[Tensor]] = {
            "logits": logits,
            "router_aux_loss": torch.stack(balance_losses).mean(),
            "expert_indices": expert_indices,
            "dropped_routes": torch.stack(dropped_route_counts).sum() if dropped_route_counts else torch.zeros((), device=input_ids.device, dtype=torch.long),
        }
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1)
            ) + self.config.router_aux_loss_coef * result["router_aux_loss"]
        return result
