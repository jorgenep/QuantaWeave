"""Export a checkpoint as a self-contained inference bundle.

  python src/export_quantweave.py --checkpoint artifacts/outputs/quantweave-moe-out \\
      --output artifacts/exports/moe --quantize 8 --benchmark-data data/smoke/tinystories.jsonl

The bundle holds: a traced TorchScript graph with dynamic batch/sequence length (model.torchscript.pt) and its
text graph, an optional ONNX file (--onnx, needs the `onnx` package), an optional int8/int4 weight-only
copy, the tokenizer, config.json, and export_metadata.json (sizes, hashes, numerical parity, optional
benchmark numbers). Exported graphs evaluate every expert densely with router weights (no capacity limit,
no data-dependent control flow), so they are for inference correctness and portability, not sparse speed.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from data_pipeline import load_tokenizer
from quantization import load_quantized, quantize_model, save_quantized
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


class LogitsOnly(nn.Module):
    def __init__(self, model: QuantaWeaveMoEForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)["logits"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_model(checkpoint: Path) -> tuple[QuantaWeaveMoEForCausalLM, dict]:
    config = QuantaWeaveConfig(**json.loads((checkpoint / "config.json").read_text()))
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(state["model"])
    return model.eval(), state


def parity_shapes(config: QuantaWeaveConfig) -> list[tuple[int, int]]:
    longest = config.max_sequence_length
    return [(1, longest), (3, max(2, longest // 2)), (2, 2)]


def try_onnx(wrapper: LogitsOnly, example: torch.Tensor, path: Path) -> dict:
    try:
        import onnx  # noqa: F401
    except ImportError:
        return {"status": "skipped", "reason": "the `onnx` package is not installed"}
    try:
        torch.onnx.export(
            wrapper, (example,), str(path), input_names=["input_ids"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
            dynamo=False,
        )
    except Exception as error:  # exporter failures are informative, not fatal
        path.unlink(missing_ok=True)
        return {"status": "failed", "reason": f"{type(error).__name__}: {error}"[:300]}
    return {"status": "ok", "file": path.name}


def export_bundle(
    checkpoint: Path, output: Path, quantize_bits: Optional[int] = None, onnx: bool = False,
    benchmark_data: Optional[Path] = None, group_size: Optional[int] = None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    model, state = load_checkpoint_model(checkpoint)
    config = model.config
    tokenizer = load_tokenizer(checkpoint)
    model.set_static_dispatch(True)
    wrapper = LogitsOnly(model).eval()
    metadata: dict = {
        "format": 1, "source_checkpoint": str(checkpoint), "training_step": state.get("step"),
        "config": dict(config.__dict__), "tokenizer": tokenizer.kind,
        "parameters": {"total": sum(p.numel() for p in model.parameters())}, "files": {},
    }

    example = torch.randint(0, config.vocab_size, (1, min(8, config.max_sequence_length)))
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, example, check_trace=False)
    traced.save(str(output / "model.torchscript.pt"))
    (output / "model.graph.txt").write_text(str(traced.graph))

    # parity: traced vs eager on shapes the trace did not see, plus eager static vs the dynamic sparse path
    reloaded = torch.jit.load(str(output / "model.torchscript.pt"))
    worst = 0.0
    with torch.inference_mode():
        for batch, length in parity_shapes(config):
            ids = torch.randint(0, config.vocab_size, (batch, length))
            worst = max(worst, float((reloaded(ids) - wrapper(ids)).abs().max()))
        ids = torch.randint(0, config.vocab_size, (2, min(8, config.max_sequence_length)))
        static = wrapper(ids)
        model.set_static_dispatch(False)
        saved_capacity = (model.moes()[0].drop_overflow_tokens,)
        model.set_routing_controls(drop_overflow_tokens=False)
        dynamic = model(ids)["logits"]
        model.set_routing_controls(drop_overflow_tokens=saved_capacity[0])
        model.set_static_dispatch(True)
    metadata["parity"] = {
        "traced_vs_eager_max_abs_diff": worst,
        "static_vs_sparse_max_abs_diff": float((static - dynamic).abs().max()),
        "shapes_checked": [list(shape) for shape in parity_shapes(config)],
    }

    if onnx:
        metadata["onnx"] = try_onnx(wrapper, example, output / "model.onnx")

    if quantize_bits:
        quantized_model, _ = load_checkpoint_model(checkpoint)
        stats = quantize_model(quantized_model, quantize_bits, group_size)
        filename = f"model.int{quantize_bits}.pt"
        save_quantized(quantized_model, output / filename, stats)
        with torch.inference_mode():
            model.set_static_dispatch(False)
            model.set_routing_controls(drop_overflow_tokens=False)
            reference = model(ids)["logits"]
            approx = load_quantized(output / filename)(ids)["logits"]
            model.set_static_dispatch(True)
        cosine = torch.nn.functional.cosine_similarity(reference.flatten(), approx.flatten(), dim=0)
        metadata["quantized"] = {**stats, "file": filename, "logit_cosine_similarity": float(cosine)}

    for name in ("config.json", "tokenizer_meta.json", "vocab.json", "tokenizer.json"):
        if (checkpoint / name).exists():
            shutil.copy(checkpoint / name, output / name)

    if benchmark_data is not None:
        from benchmark_quantweave_moe import build_parser, run_benchmark

        report = run_benchmark(build_parser().parse_args([
            "--checkpoint", str(checkpoint), "--data", str(benchmark_data), "--device", "cpu", "--no-routing-stats",
        ]))
        metadata["benchmark"] = {key: report[key] for key in ("loss", "perplexity", "tokens_per_second", "tokens", "active_parameters_per_token_estimate")}

    for file in sorted(output.iterdir()):
        if file.is_file() and file.name != "export_metadata.json":
            metadata["files"][file.name] = {"bytes": file.stat().st_size, "sha256": sha256(file)}
    (output / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_exported(directory: Path):
    """Load a bundle for serving: returns (traced module, tokenizer, metadata). No model code needed."""
    metadata = json.loads((directory / "export_metadata.json").read_text())
    return torch.jit.load(str(directory / "model.torchscript.pt")).eval(), load_tokenizer(directory), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/outputs/quantweave-moe-out"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantize", type=int, choices=(4, 8), help="also write a weight-only int4/int8 copy")
    parser.add_argument("--group-size", type=int, help="quantization group size (default: per row for int8, 32 for int4)")
    parser.add_argument("--onnx", action="store_true", help="also try an ONNX export (needs the onnx package)")
    parser.add_argument("--benchmark-data", type=Path, help="record loss/perplexity/throughput of the source checkpoint")
    args = parser.parse_args()
    metadata = export_bundle(args.checkpoint, args.output, args.quantize, args.onnx, args.benchmark_data, args.group_size)
    print(json.dumps({k: metadata[k] for k in ("parity", "parameters") if k in metadata} | {"files": list(metadata["files"])}, indent=2))


if __name__ == "__main__":
    main()
