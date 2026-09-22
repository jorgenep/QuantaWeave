"""Fine-tune a quantized QuantaWeave checkpoint with LoRA adapters (QLoRA for the MoE).

  python src/finetune_quantweave_moe.py --checkpoint artifacts/outputs/quantweave-moe-out \\
      --data new_domain.jsonl --output artifacts/outputs/qlora --bits 4 --rank 8 --steps 500 \\
      --merge-output artifacts/outputs/qlora-merged

The experts are quantized to int4/int8 and frozen; only LoRA adapters (and optionally routers/norms) train.
--optimizer adamw8bit keeps optimizer moments in 8 bits via bitsandbytes (CUDA only). The output directory holds
adapter.pt plus config/tokenizer, so `load_finetuned` can rebuild the model from the untouched base checkpoint.
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import torch

from checkpoint_io import load_checkpoint

from data_pipeline import BatchStream, WindowDataset, build_corpus, load_tokenizer
from hardware import choose_precision, detect
from run_bundle import add_archive_arguments, create_run_bundle, reproduce_command, resolve_archive_dir
from lora import add_lora, load_adapter, merge_lora_state, save_adapter
from quantization import quantize_model
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import autocast_context, build_optimizer, select_device, write_checkpoint_metadata


def load_base(checkpoint: Path, device: torch.device) -> QuantaWeaveMoEForCausalLM:
    config = QuantaWeaveConfig(**json.loads((checkpoint / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(load_checkpoint(checkpoint / "model.pt", map_location="cpu")["model"])
    return model.to(device)


@torch.no_grad()
def evaluate(model, dataset: WindowDataset, device: torch.device, precision: str, batch_size: int = 8, batches: int = 8) -> float:
    """Mean loss over the first `batches` batches of windows (fixed, so before/after numbers compare)."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, min(len(dataset), batch_size * batches), batch_size):
        batch = dataset.batch(range(start, min(len(dataset), start + batch_size))).to(device)
        with autocast_context(device, precision):
            total += float(model(batch, labels=batch)["loss"]) * batch.size(0)
        count += batch.size(0)
    model.train()
    return total / max(1, count)


def load_finetuned(base_checkpoint: Path, adapter_dir: Path, device: Optional[torch.device] = None) -> QuantaWeaveMoEForCausalLM:
    """Rebuild a fine-tuned model: base checkpoint, quantized as trained, plus the saved adapter."""
    device = device or torch.device("cpu")
    info = json.loads((adapter_dir / "finetune_metadata.json").read_text())
    model = load_base(base_checkpoint, device)
    if info["bits"]:
        quantize_model(model, info["bits"], info.get("group_size"))
    add_lora(model, info["rank"], info["alpha"], info["include_lm_head"], info["train_router"], info["train_norms"])
    load_adapter(model, adapter_dir / "adapter.pt")
    return model.eval()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merge-output", type=Path, help="also write a full-precision checkpoint with the adapters merged in")
    parser.add_argument("--bits", type=int, choices=(0, 4, 8), default=4, help="quantize the base experts (0 = keep fp32)")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--include-lm-head", action="store_true")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--train-norms", action="store_true")
    parser.add_argument("--optimizer", choices=("adamw", "adamw8bit"), default="adamw")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--examples", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    add_archive_arguments(parser)
    return parser


def run_finetune(args) -> dict:
    started_at = time.time()
    device = select_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.checkpoint)
    model = load_base(args.checkpoint, device)
    if args.sequence_length + 1 > model.config.max_sequence_length:
        raise ValueError(f"--sequence-length {args.sequence_length} exceeds the checkpoint's context {model.config.max_sequence_length - 1}")
    dataset = WindowDataset(build_corpus([args.data], tokenizer, args.examples), args.sequence_length)
    if len(dataset) == 0:
        raise ValueError("the fine-tuning data holds fewer tokens than one window")
    precision = choose_precision(detect(device.type), args.precision)

    fp32_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    quant = quantize_model(model, args.bits, args.group_size) if args.bits else None
    info = add_lora(model, args.rank, args.alpha, args.include_lm_head, args.train_router, args.train_norms)
    base_loss = evaluate(model, dataset, device, precision)
    quantized_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + sum(b.numel() * b.element_size() for n, b in model.named_buffers() if "qweight" in n or "scales" in n)

    optimizer = build_optimizer(args.optimizer, [p for p in model.parameters() if p.requires_grad], args.lr)
    stream = BatchStream(dataset, args.batch_size, args.seed)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    model.train()
    losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch, _ = stream.batch(step)
        batch = batch.to(device)
        with autocast_context(device, precision):
            loss = model(batch, labels=batch)["loss"]
        optimizer.zero_grad(set_to_none=True)
        (scaler.scale(loss) if scaler else loss).backward()
        if scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        losses.append(float(loss))
        if step % 10 == 0 or step == args.steps:
            print(f"finetune step={step}/{args.steps} loss={losses[-1]:.4f}")
    final_loss = evaluate(model, dataset, device, precision)

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_checkpoint": str(args.checkpoint), "bits": args.bits, "group_size": args.group_size, "rank": args.rank,
        "alpha": args.alpha, "include_lm_head": args.include_lm_head, "train_router": args.train_router,
        "train_norms": args.train_norms, "optimizer": args.optimizer, "steps": args.steps,
        "trainable_parameters": info["trainable_parameters"], "total_parameters": info["total_parameters"],
        "trainable_fraction": info["trainable_parameters"] / info["total_parameters"],
        "fp32_parameter_bytes": fp32_bytes, "quantized_model_bytes": quantized_bytes,
        "quantization": quant, "loss_before": base_loss, "loss_after": final_loss, "seconds": time.perf_counter() - started,
        "peak_device_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
    }
    save_adapter(model, args.output / "adapter.pt", info, str(args.checkpoint))
    (args.output / "finetune_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    for name in ("config.json", "vocab.json", "tokenizer.json", "tokenizer_meta.json"):
        if (args.checkpoint / name).exists():
            shutil.copy(args.checkpoint / name, args.output / name)

    if args.merge_output is not None:
        merged = merge_lora_state(model)
        args.merge_output.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.cpu() for k, v in merged.items()}, "optimizer": None, "step": None, "rng_state": {}, "extra": {"merged_from": str(args.output)}},
                   args.merge_output / "model.pt")
        write_checkpoint_metadata(args.merge_output, QuantaWeaveConfig(**json.loads((args.checkpoint / "config.json").read_text())),
                                  tokenizer.vocab if tokenizer.kind == "char" else tokenizer, 0, None)
    if args.archive_dir is not None and not args.no_archive:
        finished = time.time()
        models = {"adapter": args.output, **({"merged": args.merge_output} if args.merge_output is not None else {})}
        bundle = create_run_bundle(
            args.archive_dir, kind="finetune", model_dirs=models, benchmark_model=args.merge_output, data_paths=[args.data], options=vars(args),
            summary={**metadata, "final_loss": metadata["loss_after"], "steps": args.steps}, started=started_at, finished=finished,
            reproduce=reproduce_command("finetune_quantweave_moe.py", build_parser(), args, skip=("archive_dir", "no_archive")),
            skip_rows=args.examples, benchmark_examples=args.archive_benchmark_examples, device=args.device, precision=args.precision,
            data_limit_mb=args.archive_data_limit_mb, data_card=args.data_card,
        )
        metadata["archive"] = str(bundle)
        print(f"archived run to {bundle}")
    return metadata


def main() -> None:
    args = build_parser().parse_args()
    resolve_archive_dir(args)
    metadata = run_finetune(args)
    print(json.dumps({k: metadata[k] for k in ("loss_before", "loss_after", "trainable_fraction", "fp32_parameter_bytes", "quantized_model_bytes")}, indent=2))


if __name__ == "__main__":
    main()
