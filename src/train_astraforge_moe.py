"""Train the standalone AstraForge sparse MoE on JSONL text data."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from astraforge_moe_model import AstraForgeConfig, AstraForgeMoEForCausalLM


class CharacterDataset(Dataset):
    def __init__(self, path: Path, vocab: dict[str, int], sequence_length: int, limit: int | None) -> None:
        self.samples = []
        eos_id = vocab["<eos>"]
        for line_number, line in enumerate(path.open(encoding="utf-8")):
            if limit is not None and line_number >= limit:
                break
            row = json.loads(line)
            text = row.get("teacher_text") or row.get("completion") or row.get("text")
            if not text:
                text = row.get("prompt", "") + row.get("response", "")
            if not text:
                continue
            token_ids = [vocab.get(character, vocab["<unk>"]) for character in text]
            token_ids.append(eos_id)
            if len(token_ids) >= sequence_length + 1:
                self.samples.append(token_ids[: sequence_length + 1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(self.samples[index], dtype=torch.long)


def build_vocab(path: Path, vocab_size: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in path.open(encoding="utf-8"):
        for character in json.loads(line)["text"]:
            counts[character] = counts.get(character, 0) + 1
    reserved = {"<unk>": 0, "<eos>": 1}
    room = max(0, vocab_size - len(reserved))
    common_characters = sorted(counts, key=counts.get, reverse=True)[:room]
    return {**reserved, **{character: index + len(reserved) for index, character in enumerate(common_characters)}}


def save_checkpoint(
    path: Path,
    model: AstraForgeMoEForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    vocab: dict[str, int],
    device: torch.device,
) -> None:
    import shutil

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    temporary_path.mkdir(parents=True)
    rng_state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        rng_state["accelerator"] = torch.cuda.get_rng_state()
    elif device.type == "xpu":
        rng_state["accelerator"] = torch.xpu.get_rng_state()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "rng_state": rng_state},
        temporary_path / "model.pt",
    )
    (temporary_path / "config.json").write_text(json.dumps(model.config.__dict__, indent=2) + "\n")
    (temporary_path / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")
    if path.exists():
        shutil.rmtree(path)
    temporary_path.rename(path)


def load_checkpoint(path: Path, model: AstraForgeMoEForCausalLM, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    checkpoint = torch.load(path / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng_state = checkpoint.get("rng_state", {})
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"])
    if "accelerator" in rng_state and device.type == "cuda":
        torch.cuda.set_rng_state(rng_state["accelerator"])
    elif "accelerator" in rng_state and device.type == "xpu":
        torch.xpu.set_rng_state(rng_state["accelerator"])
    return int(checkpoint["step"])


def select_device(requested: str) -> torch.device:
    if requested == "rocm":
        requested = "cuda"
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/smoke/tinystories.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/outputs/astraforge-moe-out"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--examples", type=int, default=10000)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ffn-size", type=int, default=128)
    parser.add_argument("--total-experts", type=int, default=184)
    parser.add_argument("--active-experts", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=7168)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto"
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/astraforge-moe-checkpoint"))
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = select_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but Intel XPU support is not available")
    random.seed(0)
    torch.manual_seed(0)
    vocab = build_vocab(args.data, args.vocab_size)
    dataset = CharacterDataset(args.data, vocab, args.sequence_length, args.examples)
    if not dataset:
        raise ValueError("No examples are long enough for the selected sequence length")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    config = AstraForgeConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        ffn_size=args.ffn_size,
        num_experts=args.total_experts,
        top_k=args.active_experts,
        max_sequence_length=args.sequence_length + 1,
    )
    model = AstraForgeMoEForCausalLM(config).to(device)
    if device.type in {"cuda", "xpu"}:
        model = model.bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume and (args.checkpoint_dir / "model.pt").exists():
        start_step = load_checkpoint(args.checkpoint_dir, model, optimizer, device)
        print(f"resumed from {args.checkpoint_dir} at step {start_step}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} parameters total={parameters:,} experts={args.total_experts} active/token={args.active_experts}")

    iterator = iter(loader)
    model.train()
    if start_step >= args.steps:
        print(f"checkpoint already reached requested step {args.steps}")
        return
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"step={step}/{args.steps} loss={loss.item():.4f} router_aux={outputs['router_aux_loss'].item():.4f}")
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint_dir, model, optimizer, step, vocab, device)
            print(f"checkpoint saved at step {step} to {args.checkpoint_dir}")

    save_checkpoint(args.output, model, optimizer, args.steps, vocab, device)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
