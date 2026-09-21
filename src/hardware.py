"""Hardware detection and safe-configuration helpers.

  python src/hardware.py                                # what this machine offers
  python src/hardware.py --probe --hidden-size 256 --layers 4 --total-experts 16 --sequence-length 128

Probing runs real forward/backward passes, so the batch size it returns has been proven to fit.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import torch

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


@dataclass
class DeviceInfo:
    backend: str  # cuda | rocm | xpu | cpu
    name: str
    count: int
    total_memory_bytes: int
    free_memory_bytes: int
    bf16: bool
    fp16: bool
    runtime: str
    compute_capability: Optional[str] = None

    @property
    def torch_device(self) -> torch.device:
        return torch.device("cuda" if self.backend == "rocm" else self.backend)


def _cpu_memory() -> tuple[int, int]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return memory.total, memory.available
    except ImportError:
        return 0, 0


def detect(requested: str = "auto") -> DeviceInfo:
    """Describe the device `requested` resolves to (auto: CUDA/ROCm, then XPU, then CPU)."""
    if requested == "rocm":
        requested = "cuda"
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        hip = getattr(torch.version, "hip", None)
        major, minor = torch.cuda.get_device_capability(0)
        return DeviceInfo(
            backend="rocm" if hip else "cuda",
            name=torch.cuda.get_device_name(0),
            count=torch.cuda.device_count(),
            total_memory_bytes=total,
            free_memory_bytes=free,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=True,
            runtime=f"hip {hip}" if hip else f"cuda {torch.version.cuda}",
            compute_capability=f"{major}.{minor}",
        )
    if requested in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        try:
            free, total = torch.xpu.mem_get_info()
        except Exception:
            free, total = props.total_memory, props.total_memory
        return DeviceInfo("xpu", props.name, torch.xpu.device_count(), total, free, True, True, f"xpu {torch.__version__}")
    if requested in {"cuda", "xpu"}:
        raise RuntimeError(f"{requested} was requested but is not available")
    total, free = _cpu_memory()
    return DeviceInfo("cpu", "cpu", 1, total, free, True, False, f"torch {torch.__version__}")


def choose_precision(info: DeviceInfo, preferred: str = "auto") -> str:
    """bf16 where supported, else fp16 on accelerators, fp32 on CPU. Explicit choices are validated."""
    if preferred == "auto":
        if info.backend == "cpu":
            return "fp32"
        return "bf16" if info.bf16 else "fp16"
    if preferred == "bf16" and not info.bf16:
        raise ValueError(f"{info.name} does not support bf16")
    if preferred == "fp16" and info.backend == "cpu":
        raise ValueError("fp16 autocast is not supported on CPU; use bf16 or fp32")
    if preferred not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be auto, bf16, fp16 or fp32")
    return preferred


def parameter_counts(config: QuantaWeaveConfig) -> dict[str, int]:
    """Exact counts from a meta-device model, so no memory is allocated."""
    with torch.device("meta"):
        model = QuantaWeaveMoEForCausalLM(config)
    total = sum(p.numel() for p in model.parameters())
    per_expert = sum(p.numel() for p in model.blocks[0].moe.experts[0].parameters())
    experts = config.layers * config.num_experts * per_expert
    return {
        "total": total,
        "experts": experts,
        "shared": total - experts,
        "active": total - config.layers * (config.num_experts - config.top_k) * per_expert,
        "per_expert": per_expert,
    }


def training_state_bytes(config: QuantaWeaveConfig) -> int:
    """fp32 weights + fp32 grads + AdamW's two fp32 moments."""
    return parameter_counts(config)["total"] * 16


def suggest_num_experts(
    info: DeviceInfo, hidden_size: int, ffn_size: int, layers: int, top_k: int,
    vocab_size: int, sequence_length: int, memory_fraction: float = 0.5, cap: int = 256,
) -> int:
    """Largest expert count (up to ``cap``) whose training state fits in ``memory_fraction`` of device memory."""
    config = QuantaWeaveConfig(
        vocab_size=vocab_size, hidden_size=hidden_size, layers=layers, ffn_size=ffn_size,
        num_experts=top_k, top_k=top_k, max_sequence_length=sequence_length + 1,
        attention_heads=next(h for h in (8, 4, 2, 1) if hidden_size % h == 0),
    )
    counts = parameter_counts(config)
    # parameters grow linearly with the expert count: each expert plus its router column, in every layer
    per_expert_total = layers * (counts["per_expert"] + hidden_size)
    fixed = counts["total"] - top_k * per_expert_total
    affordable = int((info.total_memory_bytes * memory_fraction / 16 - fixed) // per_expert_total)
    return max(top_k, min(cap, affordable))


def _is_oom(error: BaseException) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _peak_bytes(device: torch.device) -> Optional[int]:
    """Peak memory in use on the device since the last reset (process RSS on CPU when psutil is available)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return torch.xpu.max_memory_allocated()
    if device.type == "cpu":
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except ImportError:
            return None
    return None


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()


def _synchronize(device: torch.device) -> None:
    if device.type in {"cuda", "xpu"}:
        getattr(torch, device.type).synchronize()


def find_safe_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    sequence_length: int,
    autocast: Callable,
    start: int = 1,
    limit: int = 256,
    headroom: float = 0.85,
    optimizer_state_copies: int = 2,
    target: float = 0.95,
) -> dict:
    """Find a batch size that fits a real training step, and the one worth using.

    Doubles the batch size until a step runs out of memory or its peak passes ``headroom`` of device memory. Each
    candidate runs two full steps (forward, backward, gradient clipping) and times the second, so kernel compilation and
    allocator warm-up are excluded. The optimizer's moments are reserved up front (``optimizer_state_copies`` fp32 copies
    of every trainable parameter: 2 for AdamW), because they exist for the whole of real training and a probe that
    ignores them approves batches that then fail at the first optimizer step. The model's weights are not modified.

    Returns the largest safe size (``batch_size``), the smallest safe size reaching ``target`` of the best measured
    throughput (``recommended_batch_size``; ``target`` <= 0 means the largest safe size), and the measured curve.
    """
    vocab = model.config.vocab_size
    total_bytes = detect(device.type).total_memory_bytes or 0
    parameters = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.train()
    reserved = []                                   # stands in for the optimizer moments; freed in `finally`
    curve: list[dict] = []
    try:
        reserved = [torch.zeros_like(p, dtype=torch.float32) for p in parameters for _ in range(optimizer_state_copies)]
        size = start
        while size <= limit:
            try:
                _reset_peak(device)
                ids = torch.randint(0, vocab, (size, sequence_length + 1), device=device)
                elapsed = 0.0
                for iteration in range(2):
                    model.zero_grad(set_to_none=True)
                    _synchronize(device)
                    started = time.perf_counter()
                    with autocast():
                        loss = model(ids, labels=ids)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    _synchronize(device)
                    elapsed = time.perf_counter() - started
                peak = _peak_bytes(device)
                if peak is not None and total_bytes and peak > headroom * total_bytes:
                    break
                curve.append({"batch_size": size, "tokens_per_second": size * sequence_length / elapsed, "peak_memory_bytes": peak})
                size *= 2
            except Exception as error:
                if not _is_oom(error):
                    raise
                break
    finally:
        del reserved
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not curve:
        raise RuntimeError(f"even batch size {start} does not fit on {device}")
    best = max(point["tokens_per_second"] for point in curve)
    knee = curve[-1] if target <= 0 else next(p for p in curve if p["tokens_per_second"] >= target * best)
    return {
        "batch_size": curve[-1]["batch_size"], "recommended_batch_size": knee["batch_size"],
        "tokens_per_second": knee["tokens_per_second"], "peak_memory_bytes": knee["peak_memory_bytes"],
        "largest_peak_memory_bytes": curve[-1]["peak_memory_bytes"], "curve": curve, "headroom": headroom,
    }


def recommend(info: DeviceInfo, config: QuantaWeaveConfig, sequence_length: int, probe: bool = False, preferred_precision: str = "auto") -> dict:
    """Precision, memory estimate and (optionally measured) batch size for this device and model.

    With ``probe`` the model is really run, with and without activation checkpointing, and the setting that trains
    faster at its own best batch size is recommended (checkpointing only wins when it unlocks a much larger batch)."""
    precision = choose_precision(info, preferred_precision)
    counts = parameter_counts(config)
    state = counts["total"] * 16
    result = {
        "device": asdict(info),
        "precision": precision,
        "parameters": counts,
        "training_state_bytes": state,
        "fits_training_state": info.total_memory_bytes == 0 or state < 0.8 * info.total_memory_bytes,
        "max_sequence_length": config.max_sequence_length - 1,
        "sequence_length": sequence_length,
        "activation_checkpointing_recommended": bool(info.total_memory_bytes and state > 0.4 * info.total_memory_bytes),
    }
    if probe:
        from train_quantweave_moe import autocast_context

        device = info.torch_device
        model = QuantaWeaveMoEForCausalLM(config).to(device)
        autocast = lambda: autocast_context(device, precision)  # noqa: E731
        runs = {}
        for checkpointing in (False, True):
            model.activation_checkpointing = checkpointing
            try:
                runs[checkpointing] = find_safe_batch_size(model, device, sequence_length, autocast)
            except RuntimeError:
                continue
        if not runs:
            raise RuntimeError(f"even batch size 1 does not fit on {device}")
        best_without = runs.get(False, {}).get("tokens_per_second", 0.0)
        best_with = runs.get(True, {}).get("tokens_per_second", 0.0)
        use_checkpointing = best_with > 1.05 * best_without
        chosen = runs[use_checkpointing]
        result["probe"] = chosen
        result["probe_without_checkpointing"] = runs.get(False)
        result["probe_with_checkpointing"] = runs.get(True)
        result["activation_checkpointing_recommended"] = use_checkpointing
        result["batch_size"] = chosen["recommended_batch_size"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "rocm", "xpu", "cpu"))
    parser.add_argument("--probe", action="store_true", help="run real passes to find a safe batch size")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-size", type=int, default=128)
    parser.add_argument("--total-experts", type=int, default=16)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=7168)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--precision", default="auto", choices=("auto", "bf16", "fp16", "fp32"))
    args = parser.parse_args()
    info = detect(args.device)
    config = QuantaWeaveConfig(
        vocab_size=args.vocab_size, hidden_size=args.hidden_size, layers=args.layers, ffn_size=args.ffn_size,
        num_experts=args.total_experts, top_k=args.active_experts, max_sequence_length=args.sequence_length + 1,
    )
    report = recommend(info, config, args.sequence_length, probe=args.probe, preferred_precision=args.precision)
    report["suggested_num_experts"] = suggest_num_experts(
        info, args.hidden_size, args.ffn_size, args.layers, args.active_experts, args.vocab_size, args.sequence_length
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
