"""LoRA adapters over frozen (optionally quantized) expert weights: QLoRA for the MoE.

The base model is quantized to int8/int4 (quantization.py), frozen, and each expert projection gets a trainable
low-rank update  y = base(x) + (alpha / rank) * B(A(x)).  Only the adapters (and optionally the routers and
norms) are trained, so gradients and optimizer state cover a small fraction of the parameters while the
quantized base stays compressed. Adapters save on their own; merge_lora_state() folds them back into a normal
full-precision state dict (dequantized base + low-rank delta).
"""

import math
from pathlib import Path
from typing import Optional

import torch

from checkpoint_io import load_checkpoint
import torch.nn.functional as F
from torch import Tensor, nn

from quantization import QuantizedLinear

PROJECTIONS = ("gate", "up", "down")


class LoRALinear(nn.Module):
    """Frozen base layer (nn.Linear or QuantizedLinear) plus a trainable low-rank update."""

    def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.scaling = rank, alpha / rank
        tensor = next(iter(base.parameters()), None)
        if tensor is None:
            tensor = next(iter(base.buffers()))
        device = tensor.device                                              # quantized bases hold buffers, not parameters
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, device=device))   # zero: the adapter starts as a no-op
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        return self.base(x) + update * self.scaling

    def base_weight(self) -> Tensor:
        return self.base.dequantize() if isinstance(self.base, QuantizedLinear) else self.base.weight.detach()

    def merged_weight(self) -> Tensor:
        return self.base_weight().float() + self.scaling * (self.lora_b.detach().float() @ self.lora_a.detach().float())


def add_lora(model, rank: int = 8, alpha: float = 16.0, include_lm_head: bool = False, train_router: bool = False,
             train_norms: bool = False) -> dict:
    """Freeze the model and wrap every expert projection (and optionally lm_head) with a LoRA adapter."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wrapped = 0
    for block in model.blocks:
        for expert in block.moe.experts:
            for name in PROJECTIONS:
                layer = getattr(expert, name)
                if isinstance(layer, LoRALinear):
                    raise ValueError("model already has LoRA adapters")
                setattr(expert, name, LoRALinear(layer, rank, alpha))
                wrapped += 1
    if include_lm_head:
        model.lm_head = LoRALinear(model.lm_head, rank, alpha)
        wrapped += 1
    extra = []
    for name, parameter in model.named_parameters():
        is_router = name.endswith("moe.router.weight")
        is_norm = "_norm." in name or name.startswith("final_norm.")
        if (train_router and is_router) or (train_norms and is_norm):
            parameter.requires_grad_(True)
            extra.append(name)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # quantized base weights are buffers, not parameters, but they are still model weights
    quantized = sum(m.in_features * m.out_features for m in model.modules() if isinstance(m, QuantizedLinear))
    total = sum(p.numel() for p in model.parameters()) + quantized
    return {"wrapped_layers": wrapped, "rank": rank, "alpha": alpha, "include_lm_head": include_lm_head,
            "extra_trainable": extra, "trainable_parameters": trainable, "total_parameters": total}


def adapter_state(model) -> dict[str, Tensor]:
    """Everything trainable: LoRA matrices plus any unfrozen router/norm tensors."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def save_adapter(model, path: Path, info: dict, base_checkpoint: Optional[str] = None) -> None:
    torch.save({"state": adapter_state(model), "info": info, "base_checkpoint": base_checkpoint}, path)


def load_adapter(model, path: Path) -> dict:
    payload = load_checkpoint(path, map_location="cpu")
    parameters = dict(model.named_parameters())
    missing = set(payload["state"]) - set(parameters)
    if missing:
        raise ValueError(f"adapter does not match this model; unknown tensors: {sorted(missing)[:3]}")
    with torch.no_grad():
        for name, tensor in payload["state"].items():
            parameters[name].copy_(tensor.to(parameters[name].device))
    return payload["info"]


def merge_lora_state(model) -> dict[str, Tensor]:
    """Full-precision state dict with the same keys as an ordinary QuantaWeave model."""
    merged: dict[str, Tensor] = {}
    for key, tensor in model.state_dict().items():
        belongs_to_adapter_layer = ".base." in key or key.endswith((".lora_a", ".lora_b"))
        if not belongs_to_adapter_layer:
            merged[key] = tensor.detach().clone()
    for name, layer in model.named_modules():
        if isinstance(layer, LoRALinear):
            merged[name + ".weight"] = layer.merged_weight().clone()
    return merged
