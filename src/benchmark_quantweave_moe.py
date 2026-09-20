"""Benchmark a saved QuantaWeave sparse-MoE checkpoint."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import CharacterDataset, select_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an QuantaWeave MoE checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--data", type=Path, default=Path("data/smoke/tinystories.jsonl"))
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.checkpoint / "config.json"
    vocab_path = args.checkpoint / "vocab.json"
    model_path = args.checkpoint / "model.pt"
    for path in (config_path, vocab_path, model_path, args.data):
        if not path.exists():
            parser.error(f"missing required path: {path}")

    device = select_device(args.device)
    config = QuantaWeaveConfig(**json.loads(config_path.read_text()))
    vocab = json.loads(vocab_path.read_text())
    dataset = CharacterDataset(args.data, vocab, config.max_sequence_length - 1, args.examples)
    if not dataset:
        parser.error("benchmark dataset contains no usable sequences")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = QuantaWeaveMoEForCausalLM(config).to(device)
    if device.type in {"cuda", "xpu"}:
        model = model.bfloat16()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    shared_parameters = (
        config.vocab_size * config.hidden_size * 2
        + config.max_sequence_length * config.hidden_size
        + config.layers * (4 * config.hidden_size**2)
    )
    expert_parameters = config.layers * config.num_experts * 3 * config.hidden_size * config.ffn_size
    active_expert_parameters = config.layers * config.top_k * 3 * config.hidden_size * config.ffn_size
    active_parameters = shared_parameters + active_expert_parameters

    total_loss = 0.0
    total_tokens = 0
    total_router_aux = 0.0
    expert_counts = [0] * config.num_experts
    batches = 0
    routed_tokens = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            token_count = batch.size(0) * (batch.size(1) - 1)
            total_loss += float(outputs["loss"].item()) * token_count
            total_router_aux += float(outputs["router_aux_loss"].item())
            total_tokens += token_count
            batches += 1
            routed_tokens += batch.numel()
            for indices in outputs["expert_indices"]:
                counts = torch.bincount(indices.reshape(-1), minlength=config.num_experts).cpu().tolist()
                expert_counts = [left + right for left, right in zip(expert_counts, counts)]
    synchronize(device)
    elapsed = time.perf_counter() - started

    mean_loss = total_loss / total_tokens
    active_assignments = routed_tokens * config.layers * config.top_k
    utilization = [count / active_assignments for count in expert_counts]
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "examples": len(dataset),
        "tokens": total_tokens,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "router_aux_loss": total_router_aux / batches,
        "total_parameters": total_parameters,
        "shared_parameters_estimate": shared_parameters,
        "expert_parameters_estimate": expert_parameters,
        "active_parameters_per_token_estimate": active_parameters,
        "num_experts": config.num_experts,
        "active_experts_per_token": config.top_k,
        "inactive_experts_per_token": config.num_experts - config.top_k,
        "expert_utilization_min": min(utilization),
        "expert_utilization_max": max(utilization),
        "expert_utilization_mean": sum(utilization) / len(utilization),
        "training_step_in_checkpoint": checkpoint.get("step"),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()
