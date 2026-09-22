"""Train the standalone QuantaWeave sparse MoE on JSONL text or a packed token corpus.

`run_training(args)` is the library entry point (used by the sweep and experiment manager);
`main()` parses the command line and calls it. Multi-process modes are launched with torchrun:
--expert-parallel [--tensor-parallel T] [--shard-optimizer] [--straggler-routing] [--straggler-capacity], or --pipeline-parallel.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch

from checkpoint_io import load_checkpoint as load_checkpoint_file

from data_pipeline import (
    BatchStream,
    BPETokenizer,
    CharacterDataset,  # noqa: F401  (re-exported: benchmark and tests import it from here)
    CharTokenizer,
    CurriculumConfig,
    DomainMixture,
    WindowDataset,
    build_corpus,
    difficulty_scores,
    extract_text,  # noqa: F401
    load_token_bin,
    load_tokenizer,
    parse_domain_weights,
    read_rows,
    token_classes,
    SentencePieceTokenizer,
)
from hardware import choose_precision, detect, find_safe_batch_size, parameter_counts
from moe_schedules import ScheduleConfig, TrainingController
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from routing_diagnostics import RoutingMonitor
from training_metrics import MetricsLogger

# settings that change routing behaviour on purpose during a run; they do not identify a run
MUTABLE_CONFIG_FIELDS = {
    "capacity_factor", "drop_overflow_tokens", "overflow_policy", "router_temperature", "router_aux_loss_coef",
}


def build_vocab(path: Path, vocab_size: int) -> dict[str, int]:
    return CharTokenizer.build((text for _, text in read_rows([path])), vocab_size).vocab


def save_checkpoint(
    path: Path,
    model: QuantaWeaveMoEForCausalLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    vocab,
    device: torch.device,
    extra: Optional[dict] = None,
) -> None:
    """Atomically write model, optimizer, RNG, tokenizer and (optionally) training-state metadata.

    ``vocab`` is a character vocabulary dict or a tokenizer object. ``extra`` (schedule state, data
    position, resume fingerprint, ...) is stored in model.pt and mirrored, when JSON-serialisable,
    in metadata.json for inspection.
    """
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
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "rng_state": rng_state,
            "extra": extra or {},
        },
        temporary_path / "model.pt",
    )
    write_checkpoint_metadata(temporary_path, model.config, vocab, step, extra)
    previous_path = path.with_name(f".{path.name}.previous")
    if previous_path.exists():
        shutil.rmtree(previous_path)
    if path.exists():
        path.rename(previous_path)
    try:
        temporary_path.rename(path)
    except Exception:
        if path.exists():
            shutil.rmtree(path)
        if previous_path.exists():
            previous_path.rename(path)
        raise
    if previous_path.exists():
        shutil.rmtree(previous_path)


def write_checkpoint_metadata(directory: Path, config: QuantaWeaveConfig, vocab, step: int, extra: Optional[dict]) -> None:
    (directory / "config.json").write_text(json.dumps(config.__dict__, indent=2) + "\n")
    if isinstance(vocab, dict):
        (directory / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n")
        (directory / "tokenizer_meta.json").write_text(json.dumps({"type": "char"}) + "\n")
    else:
        vocab.save(directory)
    if extra:
        metadata = {"step": step, "torch": torch.__version__, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def load_checkpoint(
    path: Path,
    model: QuantaWeaveMoEForCausalLM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    extra_out: Optional[dict] = None,
) -> int:
    """Restore weights, optimizer and RNG; returns the step. ``extra_out`` receives the saved extra state."""
    checkpoint_path = path / "model.pt"
    if not checkpoint_path.exists():
        fallback = path.with_name(f".{path.name}.previous") / "model.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(f"no valid checkpoint found under {path}")
    checkpoint = load_checkpoint_file(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint.get("rng_state", {}), device)
    if extra_out is not None:
        extra_out.update(checkpoint.get("extra", {}))
    return int(checkpoint["step"])


def restore_rng(rng_state: dict, device: torch.device) -> None:
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "torch" in rng_state:
        torch.set_rng_state(rng_state["torch"].cpu())
    if "accelerator" in rng_state and device.type == "cuda":
        torch.cuda.set_rng_state(rng_state["accelerator"].cpu())
    elif "accelerator" in rng_state and device.type == "xpu":
        torch.xpu.set_rng_state(rng_state["accelerator"].cpu())


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


def autocast_context(device: torch.device, precision: str = "auto"):
    """Mixed precision: weights and optimizer stay fp32, the forward pass runs in bf16/fp16.

    ``auto`` means bf16 on accelerators and fp32 on CPU.
    """
    if precision == "auto":
        precision = "bf16" if device.type in {"cuda", "xpu"} else "fp32"
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def evaluate_validation(model, val_stream: "BatchStream", device: torch.device, precision: str, start_step: int,
                        batches: int, num_domains: int) -> float:
    """Mean loss over ``batches`` batches of the validation split, deterministic given val_stream's seed.

    Uses a disjoint slice of steps (offset far past any plausible training step count) so it never draws the
    same windows as an earlier validation pass at a different training step, and never touches the training
    RNG stream. The model's mode is restored on the way out.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    try:
        for offset in range(batches):
            batch, domain_ids = val_stream.batch(start_step * 100_000 + offset)
            batch, domain_ids = batch.to(device), domain_ids.to(device)
            with autocast_context(device, precision):
                outputs = model(batch, labels=batch, domain_ids=domain_ids, num_domains=num_domains)
            total += float(outputs["loss"].item())
            count += 1
    finally:
        model.train(was_training)
    return total / max(1, count)


def build_optimizer(kind: str, parameters, lr: float):
    """A plain AdamW, (CUDA only, needs bitsandbytes) an 8-bit AdamW whose moments are stored in 8 bits, or an
    AdamW whose moments live in pinned CPU memory instead of device memory (see cpu_offload_optimizer.py)."""
    parameters = list(parameters)
    if kind == "adamw":
        return torch.optim.AdamW(parameters, lr=lr)
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as error:
            raise RuntimeError("--optimizer adamw8bit needs the bitsandbytes package") from error
        if not torch.cuda.is_available():
            raise RuntimeError("--optimizer adamw8bit needs a CUDA device")
        return bnb.optim.AdamW8bit(parameters, lr=lr)
    if kind == "adamw_cpu_offload":
        from cpu_offload_optimizer import CPUOffloadAdamW

        return CPUOffloadAdamW(parameters, lr=lr)
    raise ValueError("optimizer must be adamw, adamw8bit or adamw_cpu_offload")


def checkpoint_exists(path: Path) -> bool:
    """True if a completed checkpoint, or the previous one left by an interrupted save, exists."""
    previous_path = path.with_name(f".{path.name}.previous")
    return (path / "model.pt").exists() or (previous_path / "model.pt").exists()


def tokenizer_hash(tokenizer) -> str:
    if isinstance(tokenizer, CharTokenizer):
        text = json.dumps(tokenizer.vocab, sort_keys=True)
    elif tokenizer.kind == "sentencepiece":
        text = tokenizer.processor.serialized_model_proto().hex()
    else:
        text = tokenizer.tokenizer.to_str()
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def tokenizer_payload(tokenizer):
    """A char tokenizer is saved as a plain vocab dict (the long-standing on-disk format)."""
    return tokenizer.vocab if isinstance(tokenizer, CharTokenizer) else tokenizer


def resume_fields(args, config: QuantaWeaveConfig, dataset: WindowDataset, tokenizer, world_size: int) -> dict:
    """What must be unchanged for a resumed run to continue the same experiment."""
    fields = {key: value for key, value in config.__dict__.items() if key not in MUTABLE_CONFIG_FIELDS}
    fields.update(
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sequence_length=args.sequence_length,
        seed=args.seed,
        world_size=world_size,
        data_fingerprint=dataset.fingerprint(),
        tokenizer_hash=tokenizer_hash(tokenizer),
        curriculum=[args.curriculum, args.curriculum_steps, args.curriculum_start_fraction],
        domain_weights=[args.domain_weights, args.domain_weights_end, args.domain_weights_steps],
        parallel=[bool(args.expert_parallel), args.tensor_parallel, bool(args.pipeline_parallel), args.microbatches, bool(args.shard_optimizer), args.pipeline_schedule],
    )
    return fields


def config_hash(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


def diff_fields(saved: dict, current: dict) -> dict:
    return {
        key: (saved.get(key), current.get(key))
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--data", type=Path, nargs="+", default=[Path("data/smoke/tinystories.jsonl")])
    data.add_argument("--token-data", type=Path, help="packed corpus directory from prepare_tokens.py (replaces --data)")
    data.add_argument("--tokenizer", choices=("char", "bpe", "sentencepiece"), default="char")
    data.add_argument("--tokenizer-path", type=Path, help="BPE/SentencePiece tokenizer directory; trained from --data if missing")
    data.add_argument("--tokenizer-algorithm", choices=("unigram", "bpe"), default="unigram",
                      help="--tokenizer sentencepiece only: sentencepiece's own unigram or BPE algorithm")
    data.add_argument("--examples", type=int, default=10000, help="max JSONL rows read per file")
    data.add_argument("--redact-pii", action="store_true",
                      help="best-effort regex redaction of emails/phones/SSNs/credit cards/IPs before tokenization "
                           "(pii_redact.py) — not a substitute for reviewing your data, see SECURITY.md")
    data.add_argument("--sequence-length", type=int, default=128)
    data.add_argument("--vocab-size", type=int, default=7168, help="character vocabulary size, or BPE size when training one")
    data.add_argument("--curriculum", choices=("none", "rarity", "entropy", "uncommon"), default="none")
    data.add_argument("--curriculum-steps", type=int, default=0, help="steps to widen from the easiest windows to all")
    data.add_argument("--curriculum-start-fraction", type=float, default=0.25)
    data.add_argument("--domain-weights", help="sampling weights, e.g. code=0.7,stories=0.3")
    data.add_argument("--domain-weights-end", help="weights to move to linearly over --domain-weights-steps")
    data.add_argument("--domain-weights-steps", type=int, default=0)
    data.add_argument("--domain-specialization-coef", type=float, default=0.0,
                      help="> 0 penalises domains that share experts; < 0 rewards sharing")
    data.add_argument("--val-fraction", type=float, help="hold out this fraction of each domain's windows for validation (domain-stratified)")
    data.add_argument("--val-interval", type=int, default=200, help="--val-fraction only: steps between validation passes")
    data.add_argument("--val-batches", type=int, default=8, help="--val-fraction only: batches averaged per validation pass")
    data.add_argument("--val-seed", type=int, default=0, help="--val-fraction only: which windows are held out")

    model = parser.add_argument_group("model")
    model.add_argument("--auto-architecture", action="store_true",
                       help="pick --hidden-size/--layers/--ffn-size/--attention-heads/--total-experts/--active-experts/"
                            "--sequence-length for the training device instead of taking them explicitly (see hardware.py)")
    model.add_argument("--auto-architecture-quality", choices=("capacity", "balanced", "dense"), default="balanced",
                       help="--auto-architecture only: favour stored capacity, a balance, or a dense (single-expert) model")
    model.add_argument("--auto-architecture-memory-fraction", type=float, default=0.45,
                       help="--auto-architecture only: share of device memory to target")
    model.add_argument("--hidden-size", type=int, default=64)
    model.add_argument("--layers", type=int, default=2)
    model.add_argument("--ffn-size", type=int, default=128)
    model.add_argument("--attention-heads", type=int, default=4)
    model.add_argument("--total-experts", type=int, default=184)
    model.add_argument("--active-experts", type=int, default=1)
    model.add_argument("--capacity-factor", type=float, default=1.25)
    model.add_argument("--min-expert-capacity", type=int, default=4)
    model.add_argument("--drop-overflow-tokens", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--overflow-policy", choices=("drop", "residual"), default="drop")
    model.add_argument("--router-temperature", type=float, default=1.0)
    model.add_argument("--router-aux-coef", type=float, default=0.01)

    optim = parser.add_argument_group("optimisation and schedules")
    optim.add_argument("--steps", type=int, default=10)
    optim.add_argument("--schedule-steps", type=int, help="horizon of the LR/aux schedules (default --steps)")
    optim.add_argument("--batch-size", type=int, default=2)
    optim.add_argument("--gradient-accumulation-steps", type=int, default=1)
    optim.add_argument("--lr", type=float, default=3e-4)
    optim.add_argument("--warmup-steps", type=int, default=0)
    optim.add_argument("--lr-decay", choices=("constant", "cosine", "linear"), default="constant")
    optim.add_argument("--min-lr-ratio", type=float, default=0.1)
    optim.add_argument("--plateau-patience", type=int, default=0, help="controller intervals without progress before halving LR")
    optim.add_argument("--aux-coef-end", type=float)
    optim.add_argument("--aux-adapt", action="store_true", help="tighten/relax the balance loss weight from measured expert imbalance")
    optim.add_argument("--capacity-adapt", action="store_true", help="grow capacity when routes overflow, shrink when they do not")
    optim.add_argument("--capacity-min", type=float, default=1.0)
    optim.add_argument("--capacity-max", type=float, default=4.0)
    optim.add_argument("--drop-threshold", type=float, default=0.01)
    optim.add_argument("--capacity-release-step", type=int, default=0, help="stop enforcing capacity from this step (0 = never)")
    optim.add_argument("--temperature-start", type=float, help="initial router temperature, annealed to --router-temperature")
    optim.add_argument("--temperature-steps", type=int, default=0)
    optim.add_argument("--temperature-adapt", action="store_true", help="raise router temperature when experts go unused")
    optim.add_argument("--controller-interval", type=int, default=50)
    optim.add_argument("--optimizer", choices=("adamw", "adamw8bit", "adamw_cpu_offload"), default="adamw",
                       help="adamw8bit (needs bitsandbytes + CUDA) keeps AdamW's moments in 8 bits instead of fp32; "
                            "adamw_cpu_offload keeps them in pinned system RAM instead of device memory, at the cost "
                            "of a host<->device transfer every step; neither is combinable with --shard-optimizer")

    system = parser.add_argument_group("system")
    system.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    system.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    system.add_argument("--activation-checkpointing", action="store_true")
    system.add_argument("--auto-batch-size", action="store_true", help="probe the device (full training steps, optimizer state included) for a batch size that fits")
    system.add_argument("--auto-batch-target", type=float, default=0.95,
                        help="with --auto-batch-size, take the smallest batch reaching this fraction of the best throughput (0 = largest that fits)")
    system.add_argument("--expert-parallel", action="store_true", help="shard experts across ranks (launch with torchrun)")
    system.add_argument("--tensor-parallel", type=int, default=1, help="split attention and expert FFNs across T ranks (combines with expert parallelism)")
    system.add_argument("--pipeline-parallel", action="store_true", help="split layers across ranks in pipeline stages (launch with torchrun)")
    system.add_argument("--microbatches", type=int, default=1, help="micro-batches per step for --pipeline-parallel (must divide --batch-size)")
    system.add_argument("--pipeline-schedule", choices=("gpipe", "1f1b"), default="gpipe",
                        help="gpipe: all forwards then all backwards (O(microbatches) activation memory); "
                             "1f1b: interleave, same bubble but O(stages) activation memory")
    system.add_argument("--shard-optimizer", action="store_true", help="ZeRO-style: keep each replicated parameter's optimizer state on one rank")
    system.add_argument("--straggler-routing", action="store_true", help="bias routers away from slow or overloaded expert-parallel ranks")
    system.add_argument("--straggler-capacity", action="store_true", help="also grow/shrink each rank's MoE token capacity to match its measured speed")
    system.add_argument("--straggler-capacity-min", type=float, default=0.5, help="with --straggler-capacity, the smallest allowed capacity multiplier")
    system.add_argument("--straggler-capacity-max", type=float, default=2.0, help="with --straggler-capacity, the largest allowed capacity multiplier")
    system.add_argument("--straggler-strength", type=float, default=0.5)
    system.add_argument("--straggler-max-bias", type=float, default=2.0)
    system.add_argument("--straggler-interval", type=int, default=10)
    system.add_argument("--device-metrics", action="store_true", help="log per-rank expert time and rows (expert parallelism)")
    system.add_argument("--simulate-slow-rank", help="RANK:SECONDS_PER_ROW, charge that expert-parallel rank extra time per row (testing)")
    system.add_argument("--moe-kernel", choices=("loop", "triton", "auto"), default="loop",
                        help="expert compute: per-expert loop, Triton grouped GEMM, or auto (Triton on CUDA with bf16/fp16)")
    system.add_argument("--seed", type=int, default=0)

    io = parser.add_argument_group("outputs")
    io.add_argument("--output", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    io.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/quantweave-moe-checkpoint"))
    io.add_argument("--checkpoint-interval", type=int, default=1000)
    io.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    io.add_argument("--allow-config-change", action="store_true", help="resume even if the run fingerprint differs")
    io.add_argument("--metrics-file", type=Path, help="append JSONL metrics here")
    io.add_argument("--log-interval", type=int, default=1)
    io.add_argument("--diagnostics-dir", type=Path, help="write routing_log.jsonl and SVG heatmaps here")
    io.add_argument("--diagnostics-interval", type=int, default=0, help="steps between routing snapshots (0 = off unless a controller needs them)")
    io.add_argument("--archive-dir", type=Path, help="when the run finishes, save weights, data, benchmarks and more in ARCHIVE_DIR/<epoch seconds>/ "
                                                      "(the command line turns this on with artifacts/runs unless --no-archive)")
    io.add_argument("--no-archive", action="store_true", help="do not create a run archive")
    io.add_argument("--archive-data-limit-mb", type=float, default=200.0, help="copy the training data into the archive if it is at most this big (0 = never)")
    io.add_argument("--archive-benchmark-examples", type=int, default=500, help="rows per benchmark in the archive")
    io.add_argument("--data-card", type=Path, help="a JSON file describing this data's source/license/provenance; "
                                                    "embedded verbatim into the archive's data manifest.json (see SECURITY.md)")
    return parser


def default_args(**overrides) -> argparse.Namespace:
    """Parser defaults with overrides applied; the programmatic way to configure run_training."""
    args = build_parser().parse_args([])
    for name, value in overrides.items():
        if not hasattr(args, name):
            raise TypeError(f"unknown training option '{name}'")
        setattr(args, name, value)
    return args


def schedule_config_from(args) -> ScheduleConfig:
    return ScheduleConfig(
        lr=args.lr, total_steps=args.schedule_steps or args.steps, warmup_steps=args.warmup_steps,
        lr_decay=args.lr_decay, min_lr_ratio=args.min_lr_ratio, plateau_patience=args.plateau_patience,
        aux_coef=args.router_aux_coef, aux_coef_end=args.aux_coef_end, aux_adapt=args.aux_adapt,
        capacity_factor=args.capacity_factor, capacity_adapt=args.capacity_adapt, capacity_min=args.capacity_min,
        capacity_max=args.capacity_max, drop_threshold=args.drop_threshold,
        capacity_release_step=args.capacity_release_step, temperature_start=args.temperature_start,
        temperature_end=args.router_temperature, temperature_steps=args.temperature_steps,
        temperature_adapt=args.temperature_adapt, interval=args.controller_interval,
    )


def load_data(args, is_main: bool):
    """Return (tokenizer, corpus, redactor). ``redactor`` is None unless --redact-pii was passed."""
    redactor = None
    if args.redact_pii:
        from pii_redact import PIIRedactor

        redactor = PIIRedactor()
    if args.token_data:
        corpus, tokenizer = load_token_bin(args.token_data)
        return tokenizer, corpus, redactor
    if args.tokenizer in {"bpe", "sentencepiece"}:
        if args.tokenizer_path is None:
            raise ValueError(f"--tokenizer {args.tokenizer} needs --tokenizer-path")
        marker = "tokenizer.json" if args.tokenizer == "bpe" else "tokenizer.model"
        if (args.tokenizer_path / marker).exists():
            tokenizer = load_tokenizer(args.tokenizer_path)
        elif args.tokenizer == "bpe":
            tokenizer = BPETokenizer.train((text for _, text in read_rows(args.data, args.examples, redactor)), args.vocab_size)
            if is_main:
                tokenizer.save(args.tokenizer_path)
        else:
            tokenizer = SentencePieceTokenizer.train(
                (text for _, text in read_rows(args.data, args.examples, redactor)), args.vocab_size, args.tokenizer_algorithm
            )
            if is_main:
                tokenizer.save(args.tokenizer_path)
    else:
        tokenizer = CharTokenizer.build((text for _, text in read_rows(args.data, redactor=redactor)), args.vocab_size)
    return tokenizer, build_corpus(args.data, tokenizer, args.examples, redactor), redactor


def run_training(args) -> dict:
    """Train according to ``args`` (a namespace from build_parser()); returns a summary dict.

    Run archiving borrows a few option fields (metrics file, diagnostics directory) and creates temporary files for them;
    however the run ends, the caller's options are restored and those files are removed."""
    original_outputs = (args.metrics_file, args.diagnostics_dir, args.diagnostics_interval)
    temporary_paths: list[Path] = []
    try:
        return _run_training(args, original_outputs, temporary_paths)
    finally:
        args.metrics_file, args.diagnostics_dir, args.diagnostics_interval = original_outputs
        for path in temporary_paths:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def _run_training(args, original_outputs: tuple, temporary_paths: list) -> dict:
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be positive")
    ep = pp = None
    tensor_parallel = args.tensor_parallel
    if tensor_parallel < 1:
        raise ValueError("--tensor-parallel must be positive")
    use_expert_parallel = args.expert_parallel or tensor_parallel > 1
    if use_expert_parallel and args.pipeline_parallel:
        raise ValueError("--pipeline-parallel cannot be combined with --expert-parallel/--tensor-parallel")
    if (args.shard_optimizer or args.straggler_routing or args.straggler_capacity or args.device_metrics) and not use_expert_parallel:
        raise ValueError("--shard-optimizer, --straggler-routing, --straggler-capacity and --device-metrics need --expert-parallel (or --tensor-parallel)")
    if use_expert_parallel or args.pipeline_parallel:
        if args.auto_batch_size:
            raise ValueError("--auto-batch-size is not supported with multi-process parallelism")
        if args.aux_adapt or args.temperature_adapt:
            raise ValueError("--aux-adapt/--temperature-adapt need cross-rank routing statistics and are not supported with multi-process parallelism")
        if args.precision == "fp16":
            raise ValueError("fp16 loss scaling is not supported with multi-process parallelism; use bf16 or fp32")
    if use_expert_parallel:
        from expert_parallel import init_expert_parallel

        ep = init_expert_parallel(args.device, tensor_parallel)
        device, rank, world_size = ep.device, ep.rank, ep.world_size
        is_main = ep.global_rank == 0
    elif args.pipeline_parallel:
        from pipeline_parallel import init_pipeline_parallel

        if args.gradient_accumulation_steps != 1:
            raise ValueError("--pipeline-parallel micro-batches replace gradient accumulation; leave --gradient-accumulation-steps at 1")
        if args.domain_specialization_coef or args.diagnostics_interval or args.diagnostics_dir:
            raise ValueError("routing diagnostics and the domain loss need every layer on one rank and are not supported with --pipeline-parallel")
        if args.val_fraction is not None:
            raise ValueError("--val-fraction needs a full forward pass and is not supported with --pipeline-parallel")
        if args.batch_size % args.microbatches:
            raise ValueError(f"--microbatches {args.microbatches} must divide --batch-size {args.batch_size}")
        pp = init_pipeline_parallel(args.device, args.layers)
        device, rank, world_size = pp.device, 0, 1        # every stage draws the same batch
        is_main = pp.stage == 0
    else:
        device, rank, world_size = select_device(args.device), 0, 1
        is_main = True
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "xpu" and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
            raise RuntimeError("XPU was requested but Intel XPU support is not available")
    ctx = ep if ep is not None else pp
    say = print if is_main else (lambda *a, **k: None)

    started = time.time()
    archiving = args.archive_dir is not None and not args.no_archive and is_main
    if archiving:                                    # the archive needs metrics and diagnostics even if the caller did not ask for files
        if args.metrics_file is None:
            handle, name = tempfile.mkstemp(suffix=".jsonl")
            os.close(handle)
            args.metrics_file = Path(name)
            temporary_paths.append(args.metrics_file)
        if pp is None and args.diagnostics_dir is None:
            args.diagnostics_dir = Path(tempfile.mkdtemp(prefix="diagnostics-"))
            temporary_paths.append(args.diagnostics_dir)
            args.diagnostics_interval = args.diagnostics_interval or max(1, args.steps // 20)
    reproduce_args = argparse.Namespace(**vars(args))    # what the user asked for, before defaults are resolved below
    reproduce_args.metrics_file, reproduce_args.diagnostics_dir, reproduce_args.diagnostics_interval = original_outputs

    if args.auto_architecture:
        from hardware import auto_architecture as _auto_architecture

        shape = _auto_architecture(detect(device.type), args.vocab_size, args.auto_architecture_memory_fraction, args.auto_architecture_quality)
        args.hidden_size, args.layers, args.ffn_size = shape["hidden_size"], shape["layers"], shape["ffn_size"]
        args.attention_heads = shape["attention_heads"]
        args.total_experts, args.active_experts = shape["total_experts"], shape["active_experts"]
        args.sequence_length = shape["sequence_length"]
        say(f"auto architecture ({args.auto_architecture_quality}): hidden={shape['hidden_size']} layers={shape['layers']} "
            f"ffn={shape['ffn_size']} heads={shape['attention_heads']} experts={shape['total_experts']}/{shape['active_experts']} "
            f"sequence_length={shape['sequence_length']} ({shape['training_state_fraction_of_budget']:.0%} of the {args.auto_architecture_memory_fraction:.0%} memory target)")
        reproduce_args.auto_architecture = False
        for name in ("hidden_size", "layers", "ffn_size", "attention_heads", "total_experts", "active_experts", "sequence_length"):
            setattr(reproduce_args, name, getattr(args, name))

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, corpus, redactor = load_data(args, is_main)
    if redactor is not None and is_main:
        redacted = redactor.report()
        if any(redacted.values()):
            say(f"PII redaction: {redacted}")
    dataset = WindowDataset(corpus, args.sequence_length)
    if len(dataset) == 0:
        raise ValueError("The corpus holds fewer tokens than one training window")
    val_dataset = None
    if args.val_fraction is not None:
        dataset, val_dataset = dataset.split_train_val(args.val_fraction, args.val_seed)
        say(f"validation split: {len(val_dataset)} windows held out ({args.val_fraction:.1%} of each domain), {len(dataset)} left for training")
    # trained subword tokenizers (bpe/sentencepiece) have a fixed vocab; only the open-ended char tokenizer needs headroom
    vocab_size = tokenizer.vocab_size if tokenizer.kind in ("bpe", "sentencepiece") else max(args.vocab_size, tokenizer.vocab_size)
    config = QuantaWeaveConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        ffn_size=args.ffn_size,
        num_experts=args.total_experts,
        top_k=args.active_experts,
        attention_heads=args.attention_heads,
        max_sequence_length=args.sequence_length + 1,
        router_aux_loss_coef=args.router_aux_coef,
        capacity_factor=args.capacity_factor,
        min_expert_capacity=args.min_expert_capacity,
        drop_overflow_tokens=args.drop_overflow_tokens,
        overflow_policy=args.overflow_policy,
        router_temperature=args.router_temperature,
    )

    # every distinct shard (experts, tensor slices, pipeline stages) must initialise differently
    torch.manual_seed(args.seed + (ep.global_rank if ep is not None else pp.stage if pp is not None else 0))
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ep, pipeline=pp).to(device)
    if ep is not None:
        from expert_parallel import broadcast_shared_parameters

        broadcast_shared_parameters(model, ep)
    model.domain_specialization_coef = args.domain_specialization_coef
    model.activation_checkpointing = args.activation_checkpointing
    precision = choose_precision(detect(device.type), args.precision)
    scaler = torch.amp.GradScaler(device.type) if precision == "fp16" else None
    autocast = lambda: autocast_context(device, precision)  # noqa: E731
    if args.moe_kernel != "loop" and ep is None:
        kernel = args.moe_kernel
        if kernel == "auto" and precision == "fp32":
            kernel = "loop"                    # the grouped kernels only win with tensor-core dtypes
        say(f"moe kernel: {model.set_moe_kernel(kernel)}")
    elif args.moe_kernel == "triton" and ep is not None:
        raise ValueError("--moe-kernel triton applies to single-process and pipeline runs; expert parallelism uses its own dispatch")

    if args.auto_batch_size:
        found = find_safe_batch_size(model, device, args.sequence_length, autocast, target=args.auto_batch_target)
        args.batch_size = found["recommended_batch_size"]
        say(f"auto batch size: {args.batch_size} ({found['tokens_per_second']:.0f} tok/s in the probe; "
            f"largest that fits: {found['batch_size']}, peak {found['largest_peak_memory_bytes'] / 1e9:.2f} GB)")

    scores = None
    if args.curriculum != "none":
        if args.curriculum_steps <= 0:
            raise ValueError("--curriculum needs --curriculum-steps > 0")
        scores = difficulty_scores(dataset, args.curriculum, vocab_size)
    curriculum = CurriculumConfig(args.curriculum if scores is not None else "rarity", args.curriculum_start_fraction, args.curriculum_steps)
    mixture = None
    if args.domain_weights:
        mixture = DomainMixture(
            dataset.domains, parse_domain_weights(args.domain_weights),
            parse_domain_weights(args.domain_weights_end), args.domain_weights_steps,
        )
    stream = BatchStream(dataset, args.batch_size, args.seed, scores, curriculum, mixture, rank, world_size)
    domains_active = stream.num_domains > 1
    val_stream = None
    if val_dataset is not None:
        val_stream = BatchStream(val_dataset, args.batch_size, args.val_seed, rank=rank, world_size=world_size)

    if ep is not None and args.shard_optimizer:
        if args.optimizer != "adamw":
            raise ValueError(f"--optimizer {args.optimizer} is not combinable with --shard-optimizer")
        from sharded_optimizer import ShardedAdamW

        optimizer = ShardedAdamW(model, ep, lr=args.lr)
    else:
        optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    tracker = None
    if ep is not None and ep.world_size > 1 and (args.straggler_routing or args.straggler_capacity or args.device_metrics):
        from expert_parallel import DeviceLoadTracker

        tracker = DeviceLoadTracker(model, ep, strength=args.straggler_strength if args.straggler_routing else 0.0,
                                    max_bias=args.straggler_max_bias, interval=args.straggler_interval,
                                    adapt_capacity=args.straggler_capacity,
                                    capacity_scale_min=args.straggler_capacity_min, capacity_scale_max=args.straggler_capacity_max)
    if args.simulate_slow_rank:
        slow_rank, cost = args.simulate_slow_rank.split(":")
        if ep is not None and ep.rank == int(slow_rank):
            for moe in model.moes():
                moe.simulated_cost_per_row = float(cost)
    engine = None
    if pp is not None:
        from pipeline_parallel import PipelineEngine

        engine = PipelineEngine(model, pp, args.microbatches, autocast, schedule=args.pipeline_schedule)
    counts = parameter_counts(config)
    controller = TrainingController(schedule_config_from(args), base_temperature=args.router_temperature)
    adaptive = args.aux_adapt or args.temperature_adapt
    diagnostics_interval = args.diagnostics_interval or (args.controller_interval if adaptive else 0)
    monitor = None
    if diagnostics_interval and is_main:
        monitor = RoutingMonitor(
            config.layers, config.num_experts, config.top_k, dataset.domains,
            token_classes(tokenizer), args.diagnostics_dir,
        )
    logger = MetricsLogger(args.metrics_file if is_main else None, counts["total"], counts["active"])

    fields = resume_fields(args, config, dataset, tokenizer, world_size)
    if ctx is not None:
        from expert_parallel import sharded_checkpoint_exists

        resumable = sharded_checkpoint_exists(args.checkpoint_dir)
    else:
        resumable = checkpoint_exists(args.checkpoint_dir)
    start_step = 0
    if args.resume and resumable:
        extra: dict = {}
        if ctx is not None:
            from expert_parallel import load_sharded_checkpoint

            start_step = load_sharded_checkpoint(args.checkpoint_dir, model, optimizer, ctx, extra)
        else:
            start_step = load_checkpoint(args.checkpoint_dir, model, optimizer, device, extra)
        changed = diff_fields(extra.get("resume_fields", fields), fields)
        if changed and not args.allow_config_change:
            raise ValueError(
                f"checkpoint {args.checkpoint_dir} was made by a different run configuration: {changed}. "
                "Pass --allow-config-change to resume anyway, or use a fresh --checkpoint-dir."
            )
        if "schedule" in extra:
            controller.load_state_dict(extra["schedule"])
        if tracker is not None and extra.get("straggler"):
            tracker.load_state_dict(extra["straggler"])
        if scaler is not None and extra.get("scaler"):
            scaler.load_state_dict(extra["scaler"])
        say(f"resumed from {args.checkpoint_dir} at step {start_step}")
    say(
        f"device={device} precision={precision} parameters total={counts['total']:,} active/token={counts['active']:,} "
        f"experts={args.total_experts} top_k={args.active_experts}"
        + (f" expert_parallel={world_size} tensor_parallel={tensor_parallel}" if ep is not None else "")
        + (f" pipeline_stages={pp.num_stages} microbatches={args.microbatches}" if pp is not None else "")
        + (" sharded_optimizer" if ep is not None and args.shard_optimizer else "")
    )
    if start_step >= args.steps:
        say(f"checkpoint already reached requested step {args.steps}")
        return {"steps": start_step, "resumed_only": True}

    def checkpoint_extra(step: int) -> dict:
        return {
            "resume_fields": fields,
            "config_hash": config_hash(fields),
            "schedule": controller.state_dict(),
            "data": stream.state(step),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "straggler": tracker.state_dict() if tracker is not None else None,
            "routing_controls": {
                "capacity_factor": model.config.capacity_factor,
                "router_temperature": model.config.router_temperature,
                "router_aux_loss_coef": model.router_aux_loss_coef,
            },
        }

    def save(path: Path, step: int) -> None:
        if ctx is not None:
            from expert_parallel import save_sharded_checkpoint

            save_sharded_checkpoint(path, model, optimizer, step, tokenizer_payload(tokenizer), ctx, checkpoint_extra(step))
        else:
            save_checkpoint(path, model, optimizer, step, tokenizer_payload(tokenizer), device, checkpoint_extra(step))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    routes_per_step = args.batch_size * args.sequence_length * config.layers * config.top_k
    window_metrics = {"imbalance": None, "active_fraction": None}
    pending_tokens = 0
    last_log: dict = {}
    val_loss = best_val_loss = None
    for step in range(start_step + 1, args.steps + 1):
        applied = controller.apply(model, optimizer, step)
        batch, domain_ids = stream.batch(step)
        batch, domain_ids = batch.to(device), domain_ids.to(device)
        # Synchronized across ranks (depends only on diagnostics_interval/step, identical under torchrun), not on
        # `monitor is not None` (main rank only) — collect_stats must turn on everywhere so gather_expert_utilization,
        # a collective all_gather below, finds a populated last_stats["expert_load"] on every rank, not just main.
        collect = bool(diagnostics_interval) and step % diagnostics_interval == 0
        model.set_collect_stats(collect)
        grad_norm = None
        if engine is not None:
            result = engine.train_step(batch)
            loss_value, aux_value = result["loss"], result["router_aux_loss"]
            dropped_value, overflow_value, domain_value = result["dropped_routes"], result["overflow_routes"], 0.0
            from pipeline_parallel import clip_grad_norm_pipeline

            grad_norm = clip_grad_norm_pipeline(model, pp, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            overflow_fraction = overflow_value / max(1, routes_per_step)
        else:
            with autocast():
                outputs = model(batch, labels=batch, domain_ids=domain_ids, num_domains=stream.num_domains if domains_active else 0)
            loss = outputs["loss"]
            scaled = loss / args.gradient_accumulation_steps
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            if step % args.gradient_accumulation_steps == 0 or step == args.steps:
                if ep is not None:
                    from expert_parallel import clip_grad_norm_parallel, sync_gradients

                    sharded = bool(args.shard_optimizer)
                    sync_gradients(model, ep, optimizer if sharded else None)
                    grad_norm = clip_grad_norm_parallel(model, ep, 1.0, sharded)
                else:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_value = float(loss.item())
            aux_value = float(outputs["router_aux_loss"].item())
            dropped_value, overflow_value = int(outputs["dropped_routes"].item()), int(outputs["overflow_routes"].item())
            domain_value = float(outputs["domain_loss"].item())
            overflow_fraction = overflow_value / max(1, routes_per_step)
            if ep is not None:
                from expert_parallel import all_reduce_mean

                loss_value, overflow_fraction = all_reduce_mean([loss_value, overflow_fraction], ep)
        device_metrics = tracker.update(step) if tracker is not None else None
        controller.observe(loss_value, overflow_fraction)
        pending_tokens += args.batch_size * args.sequence_length * world_size

        # gather_expert_utilization is a collective (all_gather), so it must run on every rank in lockstep; `collect`
        # is now synchronized (see above), so gating on it here is safe.
        if tracker is not None and collect:
            per_device_utilization = tracker.gather_expert_utilization()
            if monitor is not None:
                monitor.record_device_utilization(per_device_utilization)

        if collect and monitor is not None:
            monitor.update(model, batch, domain_ids)
            window_metrics = {"imbalance": monitor.imbalance(), "active_fraction": monitor.active_fraction()}
            monitor.snapshot(step)
        if controller.should_update(step):
            changes = controller.update(step, **window_metrics)
            if changes:
                say(f"controller step={step}: {changes}")
            window_metrics = {"imbalance": None, "active_fraction": None}

        fresh_val_loss = None
        if val_stream is not None and (step % args.val_interval == 0 or step == args.steps):
            val_loss = fresh_val_loss = evaluate_validation(model, val_stream, device, precision, step, args.val_batches,
                                                             val_stream.num_domains if domains_active else 0)
            best_val_loss = val_loss if best_val_loss is None else min(best_val_loss, val_loss)
            say(f"step={step}/{args.steps} val_loss={val_loss:.4f}")

        if step % args.log_interval == 0 or step == args.steps:
            last_log = logger.log(
                step, pending_tokens, loss=loss_value, router_aux=aux_value,
                dropped_routes=dropped_value, overflow_routes=overflow_value,
                overflow_fraction=overflow_fraction, domain_loss=domain_value,
                grad_norm=float(grad_norm) if grad_norm is not None else None, **applied,
                **({"time_imbalance": device_metrics["time_imbalance"], "rows_imbalance": device_metrics["rows_imbalance"]} if device_metrics else {}),
                **({"val_loss": fresh_val_loss} if fresh_val_loss is not None else {}),
            )
            pending_tokens = 0
            say(
                f"step={step}/{args.steps} loss={loss_value:.4f} "
                f"router_aux={aux_value:.4f} "
                f"dropped_routes={dropped_value} "
                f"overflow_routes={overflow_value}"
            )
        if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
            save(args.checkpoint_dir, step)
            say(f"checkpoint saved at step {step} to {args.checkpoint_dir}")

    model.set_collect_stats(False)
    save(args.output if ctx is None else args.output / "shards", args.steps)
    if ctx is not None:
        from expert_parallel import consolidate_checkpoint, finish

        finish(ctx)
        if is_main:
            consolidate_checkpoint(args.output / "shards", args.output)
    say(f"saved checkpoint to {args.output}")
    if monitor is not None and args.diagnostics_dir is not None:
        say(f"routing diagnostics: {monitor.write_report()}")
    summary = logger.summary()
    summary.update(
        steps=args.steps, output=str(args.output), precision=precision, controller_history=controller.history,
        last_routing=monitor.snapshots[-1] if monitor is not None and monitor.snapshots else None,
        stream=stream.state(args.steps), last_log=last_log, val_loss=val_loss, best_val_loss=best_val_loss,
        val_windows=len(val_dataset) if val_dataset is not None else None,
    )
    if archiving:
        from run_bundle import create_run_bundle, reproduce_command

        if args.auto_batch_size:
            reproduce_args.auto_batch_size, reproduce_args.batch_size = False, args.batch_size    # pin what the probe chose
        finished = time.time()
        bundle = create_run_bundle(
            args.archive_dir, kind="train", model_dirs={"model": args.output}, benchmark_model=None if args.token_data else args.output,
            data_paths=[] if args.token_data else list(args.data), options=vars(args), summary=summary, started=started,
            reproduce=reproduce_command("train_quantweave_moe.py", build_parser(), reproduce_args, skip=("archive_dir", "no_archive")),
            reproduce_full=reproduce_command("train_quantweave_moe.py", build_parser(), reproduce_args, skip=("archive_dir", "no_archive"), explicit=True),
            finished=finished,
            metrics_file=args.metrics_file, diagnostics_dir=args.diagnostics_dir if monitor is not None else None,
            skip_rows=None if args.token_data else args.examples, benchmark_examples=args.archive_benchmark_examples,
            device=args.device if ctx is None else "auto", precision=args.precision, data_limit_mb=args.archive_data_limit_mb,
            data_card=args.data_card,
        )
        summary["archive"] = str(bundle)
        say(f"archived run to {bundle}")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    from run_bundle import resolve_archive_dir

    resolve_archive_dir(args)
    run_training(args)


if __name__ == "__main__":
    main()
