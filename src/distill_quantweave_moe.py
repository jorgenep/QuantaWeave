"""Distill teacher-generated text into an QuantaWeave MoE student checkpoint.

This is sequence-level distillation: the teacher dataset supplies target text.
For true logit distillation, teacher and student tokenizers/vocabularies must
also be aligned and a teacher model must be available at training time.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import save_checkpoint, select_device


class TeacherTextDataset(Dataset):
    def __init__(self, path: Path, vocab: dict[str, int], sequence_length: int, limit: int | None) -> None:
        self.samples = []
        eos_id = vocab["<eos>"]
        for line_number, line in enumerate(path.open(encoding="utf-8")):
            if limit is not None and line_number >= limit:
                break
            row = json.loads(line)
            text = row.get("teacher_text") or row.get("completion") or row.get("text")
            if not text:
                prompt = row.get("prompt", "")
                completion = row.get("response", "")
                text = prompt + completion
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill teacher text into an QuantaWeave checkpoint")
    parser.add_argument("--student", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/outputs/quantweave-moe-distilled"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/quantweave-distill"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=100000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = select_device(args.device)
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but Intel XPU support is not available")
    config = QuantaWeaveConfig(**json.loads((args.student / "config.json").read_text()))
    vocab = json.loads((args.student / "vocab.json").read_text())
    dataset = TeacherTextDataset(args.teacher_data, vocab, args.sequence_length, args.examples)
    if not dataset:
        raise ValueError("Teacher dataset contains no usable text sequences")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = QuantaWeaveMoEForCausalLM(config).to(device)
    if device.type in {"cuda", "xpu"}:
        model = model.bfloat16()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume and (args.checkpoint_dir / "model.pt").exists():
        from train_quantweave_moe import load_checkpoint
        start_step = load_checkpoint(args.checkpoint_dir, model, optimizer, device)
        print(f"resumed distillation from step {start_step}")

    random.seed(0)
    torch.manual_seed(0)
    iterator = iter(loader)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        optimizer.zero_grad(set_to_none=True)
        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"distill_step={step}/{args.steps} loss={outputs['loss'].item():.4f} router_aux={outputs['router_aux_loss'].item():.4f}")
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint_dir, model, optimizer, step, vocab, device)
            print(f"distillation checkpoint saved at step {step}")

    save_checkpoint(args.output, model, optimizer, args.steps, vocab, device)
    print(f"distilled student saved to {args.output}")


if __name__ == "__main__":
    main()
