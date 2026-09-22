"""Weight-only int8 / int4 quantization for inference.

Expert feed-forward weights hold nearly all of an MoE's parameters, so they are what gets quantized. The
router, embeddings, norms and attention stay in floating point (the router in particular must keep
its precision: a flipped top-k choice changes which expert runs). Weights are dequantized on the fly, so
this saves memory and bandwidth, not FLOPs. It is an inference feature: training uses bf16/fp16 autocast.
"""

from pathlib import Path
from typing import Optional

import torch

from checkpoint_io import load_checkpoint
import torch.nn.functional as F
from torch import Tensor, nn

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


class QuantizedLinear(nn.Module):
    """Linear layer with symmetric per-group quantized weights.

    bits=8: int8, one scale per output row (or per ``group_size`` columns if given).
    bits=4: two 4-bit values packed per byte, one scale per ``group_size`` columns.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 8, group_size: Optional[int] = None) -> None:
        super().__init__()
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        group_size = group_size or (in_features if bits == 8 else 32)
        if bits == 4 and group_size % 2:
            raise ValueError("group_size must be even for 4-bit weights")
        self.in_features, self.out_features, self.bits, self.group_size = in_features, out_features, bits, group_size
        self.padded = -(-in_features // group_size) * group_size
        groups = self.padded // group_size
        stored = self.padded if bits == 8 else self.padded // 2
        self.register_buffer("qweight", torch.zeros(out_features, stored, dtype=torch.int8 if bits == 8 else torch.uint8))
        self.register_buffer("scales", torch.ones(out_features, groups))

    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 8, group_size: Optional[int] = None) -> "QuantizedLinear":
        if linear.bias is not None:
            raise ValueError("bias is not supported")
        layer = cls(linear.in_features, linear.out_features, bits, group_size).to(linear.weight.device)
        weight = linear.weight.detach().float()
        weight = F.pad(weight, (0, layer.padded - layer.in_features))
        grouped = weight.reshape(layer.out_features, -1, layer.group_size)
        limit = 127 if bits == 8 else 7
        scales = (grouped.abs().amax(dim=-1) / limit).clamp(min=1e-8)
        quantized = torch.round(grouped / scales[..., None]).clamp(-limit - (bits == 4), limit).reshape(layer.out_features, -1)
        if bits == 8:
            layer.qweight.copy_(quantized.to(torch.int8))
        else:
            unsigned = (quantized + 8).to(torch.uint8)
            layer.qweight.copy_(unsigned[:, 0::2] | (unsigned[:, 1::2] << 4))
        layer.scales.copy_(scales)
        return layer

    def dequantize(self) -> Tensor:
        if self.bits == 8:
            values = self.qweight.float()
        else:
            low = (self.qweight & 0xF).float() - 8
            high = (self.qweight >> 4).float() - 8
            values = torch.stack([low, high], dim=-1).reshape(self.out_features, -1)
        values = values.reshape(self.out_features, -1, self.group_size) * self.scales[..., None]
        return values.reshape(self.out_features, -1)[:, : self.in_features]

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.dequantize().to(x.dtype))

    def stored_bytes(self) -> int:
        return self.qweight.numel() * self.qweight.element_size() + self.scales.numel() * self.scales.element_size()


def quantize_model(model: QuantaWeaveMoEForCausalLM, bits: int = 8, group_size: Optional[int] = None, quantize_lm_head: bool = False) -> dict:
    """Replace expert weights (and optionally lm_head) in place. Returns size statistics."""
    if model.expert_parallel is not None:
        raise ValueError("consolidate an expert-parallel model before quantizing it")
    before = after = replaced = 0

    def swap(parent: nn.Module, name: str) -> None:
        nonlocal before, after, replaced
        linear = getattr(parent, name)
        if isinstance(linear, QuantizedLinear):
            raise ValueError("model is already quantized")
        quantized = QuantizedLinear.from_linear(linear, bits, group_size)
        before += linear.weight.numel() * linear.weight.element_size()
        after += quantized.stored_bytes()
        replaced += 1
        setattr(parent, name, quantized)

    for block in model.blocks:
        for expert in block.moe.experts:
            for name in ("gate", "up", "down"):
                swap(expert, name)
    if quantize_lm_head:
        swap(model, "lm_head")
    return {"bits": bits, "group_size": group_size, "replaced_layers": replaced, "bytes_before": before,
            "bytes_after": after, "compression": before / after if after else None}


def save_quantized(model: QuantaWeaveMoEForCausalLM, path: Path, stats: dict, quantize_lm_head: bool = False) -> None:
    torch.save({
        "config": dict(model.config.__dict__), "bits": stats["bits"], "group_size": stats["group_size"],
        "quantize_lm_head": quantize_lm_head, "state_dict": model.state_dict(),
    }, path)


def load_quantized(path: Path, device: torch.device = torch.device("cpu")) -> QuantaWeaveMoEForCausalLM:
    payload = load_checkpoint(path, map_location=device)
    model = QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(**payload["config"]))
    quantize_model(model, payload["bits"], payload["group_size"], payload["quantize_lm_head"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
