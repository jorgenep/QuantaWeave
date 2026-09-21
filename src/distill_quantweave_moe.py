"""Distill into a QuantaWeave MoE student checkpoint.

Two modes, chosen by whether --teacher-checkpoint is given:

* sequence-level (default): the student trains on teacher-generated text in --teacher-data.
* logit-level: a frozen QuantaWeave teacher checkpoint (a dense baseline or a larger MoE) scores the
  same batches and the student minimises  ce + alpha * T^2 * KL(teacher || student).  Teacher and
  student must share a tokenizer, because the KL is taken over aligned token distributions.
* cross-tokenizer (--hf-teacher): a Hugging Face causal LM with its own tokenizer scores the same *text*;
  the student is supervised where token boundaries of the two tokenizations coincide (see cross_tokenizer.py).

The student always starts from its trained weights (--student) with a fresh optimizer.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from cross_tokenizer import CrossTokenizerLoss, HFTeacher
from data_pipeline import BatchStream, WindowDataset, build_corpus, load_tokenizer
from run_bundle import add_archive_arguments, create_run_bundle, reproduce_command, resolve_archive_dir
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import (
    autocast_context,
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
    select_device,
    tokenizer_hash,
    tokenizer_payload,
)


def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float) -> Tensor:
    """Per-token KL(teacher || student) at ``temperature``, scaled by T^2 so its gradient magnitude
    stays comparable to cross-entropy. Positions line up with the LM loss (all but the last token)."""
    student = F.log_softmax(student_logits[:, :-1].float().reshape(-1, student_logits.size(-1)) / temperature, dim=-1)
    teacher = F.softmax(teacher_logits[:, :-1].float().reshape(-1, teacher_logits.size(-1)) / temperature, dim=-1)
    return F.kl_div(student, teacher, reduction="batchmean") * temperature**2


def load_model(checkpoint: Path, device: torch.device) -> tuple[QuantaWeaveMoEForCausalLM, object]:
    config = QuantaWeaveConfig(**json.loads((checkpoint / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config).to(device)
    state = torch.load(checkpoint / "model.pt", map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    return model, load_tokenizer(checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, help="enable logit-level distillation from this checkpoint")
    parser.add_argument("--hf-teacher", help="Hugging Face causal LM (directory or hub id) with its own tokenizer")
    parser.add_argument("--cross-loss", choices=("auto", "marginal", "uld"), default="auto",
                        help="loss for --hf-teacher: exact next-character marginal (char students) or sorted-distribution L1")
    parser.add_argument("--alpha", type=float, default=1.0, help="weight of the KL / cross-tokenizer term")
    parser.add_argument("--temperature", type=float, default=2.0, help="softmax temperature for the KL term")
    parser.add_argument("--output", type=Path, default=Path("artifacts/outputs/quantweave-moe-distilled"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/quantweave-distill"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--examples", type=int, default=100000)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_archive_arguments(parser)
    return parser


def run_distillation(args) -> dict:
    started = time.time()
    device = select_device(args.device)
    if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("XPU was requested but Intel XPU support is not available")
    for path in (args.student / "config.json", args.student / "model.pt"):
        if not path.exists():
            raise FileNotFoundError(f"missing student file: {path}")
    if args.hf_teacher and args.teacher_checkpoint:
        raise ValueError("choose one teacher: --teacher-checkpoint or --hf-teacher")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    student_config = QuantaWeaveConfig(**json.loads((args.student / "config.json").read_text()))
    tokenizer = load_tokenizer(args.student)
    if args.sequence_length + 1 > student_config.max_sequence_length:
        raise ValueError(
            f"--sequence-length {args.sequence_length} needs max_sequence_length "
            f">= {args.sequence_length + 1}, but the student has {student_config.max_sequence_length}"
        )
    dataset = WindowDataset(build_corpus([args.teacher_data], tokenizer, args.examples), args.sequence_length)
    if len(dataset) == 0:
        raise ValueError("Teacher dataset holds fewer tokens than one training window")
    stream = BatchStream(dataset, args.batch_size, args.seed)

    teacher: Optional[QuantaWeaveMoEForCausalLM] = None
    if args.teacher_checkpoint is not None:
        teacher, teacher_tokenizer = load_model(args.teacher_checkpoint, device)
        if tokenizer_hash(teacher_tokenizer) != tokenizer_hash(tokenizer) or teacher.config.vocab_size != student_config.vocab_size:
            raise ValueError(
                "logit distillation needs the teacher and student to share a tokenizer and vocabulary size "
                f"(teacher vocab {teacher.config.vocab_size}, student vocab {student_config.vocab_size})"
            )
        if args.sequence_length + 1 > teacher.config.max_sequence_length:
            raise ValueError("teacher max_sequence_length is shorter than the distillation sequence length")
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    cross_loss = None
    if args.hf_teacher:
        cross_loss = CrossTokenizerLoss(HFTeacher(args.hf_teacher, device), tokenizer, args.cross_loss)

    model = QuantaWeaveMoEForCausalLM(student_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume and checkpoint_exists(args.checkpoint_dir):
        start_step = load_checkpoint(args.checkpoint_dir, model, optimizer, device)
        print(f"resumed distillation from step {start_step}")
    else:
        # Fine-tune the student: start from its trained weights, with a fresh optimizer.
        state = torch.load(args.student / "model.pt", map_location=device, weights_only=False)["model"]
        model.load_state_dict(state)
        print(f"initialised from student weights at {args.student / 'model.pt'}")

    mode = "cross-tokenizer" if cross_loss is not None else "logit" if teacher is not None else "sequence"
    print(f"distillation mode: {mode}")
    model.train()
    last = {}
    for step in range(start_step + 1, args.steps + 1):
        batch, _ = stream.batch(step)
        batch = batch.to(device)
        with autocast_context(device, args.precision):
            outputs = model(batch, labels=batch)
            kl = None
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(batch)["logits"]
                kl = distillation_loss(outputs["logits"], teacher_logits, args.temperature)
            elif cross_loss is not None:
                kl, cross_stats = cross_loss(outputs["logits"], batch)
        loss = outputs["loss"] + (args.alpha * kl if kl is not None else 0.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {
            "loss": float(loss.item()), "ce_loss": float(outputs["loss"].item()),
            "kl": float(kl.item()) if kl is not None else None,
        }
        kl_text = f" kl={last['kl']:.4f}" if kl is not None else ""
        if cross_loss is not None:
            last["aligned_fraction"] = cross_stats["aligned_fraction"]
            kl_text += f" aligned={cross_stats['aligned_fraction']:.0%}"
        print(f"distill_step={step}/{args.steps} loss={last['loss']:.4f}{kl_text} router_aux={outputs['router_aux_loss'].item():.4f}")
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save_checkpoint(args.checkpoint_dir, model, optimizer, step, tokenizer_payload(tokenizer), device)
            print(f"distillation checkpoint saved at step {step}")

    save_checkpoint(args.output, model, optimizer, args.steps, tokenizer_payload(tokenizer), device)
    print(f"distilled student saved to {args.output}")
    result = {"mode": mode, "steps": args.steps, **last}
    if args.archive_dir is not None and not args.no_archive:
        finished = time.time()
        bundle = create_run_bundle(
            args.archive_dir, kind="distill", model_dirs={"model": args.output}, benchmark_model=args.output, data_paths=[args.teacher_data],
            options=vars(args), summary={**result, "final_loss": result.get("loss")}, started=started, finished=finished,
            reproduce=reproduce_command("distill_quantweave_moe.py", build_parser(), args, skip=("archive_dir", "no_archive")),
            skip_rows=args.examples, benchmark_examples=args.archive_benchmark_examples, device=args.device, precision=args.precision,
            data_limit_mb=args.archive_data_limit_mb,
        )
        result["archive"] = str(bundle)
        print(f"archived run to {bundle}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    resolve_archive_dir(args)
    run_distillation(args)


if __name__ == "__main__":
    main()
