import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import benchmark_quantweave_moe as bench
import distill_quantweave_moe as distill
import export_quantweave as export
import generate_quantweave_moe as generate
import quantization as quant
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def write_corpus(path: Path, rows: int = 60) -> Path:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(rows)))
    return path


def train(tmp_path: Path, name: str, data: Path, **overrides) -> Path:
    values = dict(
        data=[data], steps=4, batch_size=2, sequence_length=16, examples=100, hidden_size=16, layers=2, ffn_size=32,
        total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0,
        output=tmp_path / name / "out", checkpoint_dir=tmp_path / name / "ckpt", log_interval=1,
    )
    values.update(overrides)
    trainer.run_training(trainer.default_args(**values))
    return values["output"]


@pytest.fixture(scope="module")
def student(tmp_path_factory):
    root = tmp_path_factory.mktemp("deploy")
    data = write_corpus(root / "d.jsonl")
    return root, data, train(root, "student", data, capacity_factor=0)


# ---- distillation -------------------------------------------------------------------------
def test_kl_is_zero_for_identical_logits_positive_otherwise_and_temperature_scaled():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    assert distill.distillation_loss(logits, logits, 2.0).item() == pytest.approx(0.0, abs=1e-6)
    other = torch.randn(2, 5, 11)
    kl = distill.distillation_loss(other, logits, 1.0)
    assert kl.item() > 0
    manual = torch.nn.functional.kl_div(
        torch.log_softmax(other[:, :-1].reshape(-1, 11), -1), torch.softmax(logits[:, :-1].reshape(-1, 11), -1), reduction="batchmean")
    assert kl.item() == pytest.approx(manual.item(), rel=1e-5)
    assert distill.distillation_loss(other, logits, 3.0).requires_grad is False


def distill_args(tmp_path, student_dir, data, **overrides):
    values = [
        "--student", str(student_dir), "--teacher-data", str(data), "--output", str(tmp_path / "distilled"),
        "--checkpoint-dir", str(tmp_path / "dckpt"), "--steps", "3", "--sequence-length", "16", "--batch-size", "2",
        "--device", "cpu", "--checkpoint-interval", "0",
    ]
    for key, value in overrides.items():
        values += [f"--{key.replace('_', '-')}", str(value)]
    return distill.build_parser().parse_args(values)


def test_logit_distillation_from_a_teacher_checkpoint_reduces_kl(tmp_path, student):
    root, data, student_dir = student
    # a teacher trained longer, sharing the student's data and therefore its tokenizer
    teacher_dir = train(root, "teacher", data, steps=30, hidden_size=16, capacity_factor=0, lr=3e-3)
    result = distill.run_distillation(distill_args(tmp_path, student_dir, data, teacher_checkpoint=teacher_dir, steps=25, lr=3e-3, alpha=1.0, temperature=2.0))
    assert result["mode"] == "logit" and result["kl"] is not None and result["kl"] >= 0

    # first-step KL vs last-step KL of the same recipe
    log = []
    original = distill.distillation_loss
    def spy(*a, **k):
        value = original(*a, **k)
        log.append(float(value.detach()))
        return value
    distill.distillation_loss = spy
    try:
        distill.run_distillation(distill_args(tmp_path, student_dir, data, teacher_checkpoint=teacher_dir, steps=25, lr=3e-3, output=tmp_path / "d2", checkpoint_dir=tmp_path / "dck2"))
    finally:
        distill.distillation_loss = original
    assert sum(log[-5:]) / 5 < sum(log[:5]) / 5


def test_logit_distillation_rejects_a_mismatched_tokenizer(tmp_path, student):
    root, data, student_dir = student
    other = root / "other.jsonl"
    other.write_text("".join(json.dumps({"text": f"ZYX {i} qwerty!! ~~ different alphabet 你好"}) + "\n" for i in range(60)))
    foreign_teacher = train(root, "foreign", other, capacity_factor=0)
    with pytest.raises(ValueError, match="share a tokenizer"):
        distill.run_distillation(distill_args(tmp_path, student_dir, data, teacher_checkpoint=foreign_teacher))


def test_sequence_distillation_still_works_and_validates_length(tmp_path, student):
    _, data, student_dir = student
    assert distill.run_distillation(distill_args(tmp_path, student_dir, data))["mode"] == "sequence"
    with pytest.raises(ValueError, match="max_sequence_length"):
        distill.run_distillation(distill_args(tmp_path, student_dir, data, sequence_length=64))


# ---- benchmark / generate -----------------------------------------------------------------
def test_benchmark_reports_routing_compute_memory_and_markdown(tmp_path, student):
    _, data, student_dir = student
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps({"step": i, "loss": 3.0 - 0.05 * i, "tokens_per_second": 100.0, "overflow_routes": 1}) + "\n" for i in range(1, 60)))
    args = bench.build_parser().parse_args(["--checkpoint", str(student_dir), "--data", str(data), "--examples", "30", "--device", "cpu",
                                            "--output", str(tmp_path / "r.json"), "--training-metrics", str(metrics)])
    report = bench.run_benchmark(args)
    routing = report["routing"]
    assert len(routing["per_layer"]) == 2
    layer = routing["per_layer"][0]
    assert sum(layer["utilization"]) == pytest.approx(1.0) and 0 <= layer["entropy_normalized"] <= 1.0001
    assert sum(layer["confidence_histogram"]) > 0 and sum(layer["utilization_histogram"]["counts"]) == 4
    assert report["expert_compute"]["moe_seconds_total"] > 0 and len(report["expert_compute"]["seconds_per_expert"]) == 4
    assert report["memory"]["parameter_bytes"] > 0 and report["memory"]["bytes_per_expert"] > 0
    assert report["seconds_per_batch"] > 0 and report["training_stability"]["logged_steps"] == 59
    markdown = bench.render_markdown(report)
    for heading in ("## Quality", "## Speed and memory", "## Routing", "### Per layer", "## Training stability"):
        assert heading in markdown

    bench.main.__globals__["sys"] = sys
    old = sys.argv
    sys.argv = ["bench", "--checkpoint", str(student_dir), "--data", str(data), "--examples", "10", "--device", "cpu", "--output", str(tmp_path / "out.json")]
    try:
        bench.main()
    finally:
        sys.argv = old
    assert (tmp_path / "out.json").exists() and (tmp_path / "out.md").read_text().startswith("# Benchmark")


def test_benchmark_can_skip_the_diagnostic_pass(tmp_path, student):
    _, data, student_dir = student
    args = bench.build_parser().parse_args(["--checkpoint", str(student_dir), "--data", str(data), "--examples", "10", "--device", "cpu", "--no-routing-stats"])
    report = bench.run_benchmark(args)
    assert "routing" not in report and report["loss"] > 0


def test_generate_samples_text_from_char_and_bpe_checkpoints(tmp_path, student):
    from data_pipeline import load_tokenizer
    _, data, student_dir = student
    config = QuantaWeaveConfig(**json.loads((student_dir / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(torch.load(student_dir / "model.pt", weights_only=False)["model"])
    model.eval()
    text = generate.sample_text(model, load_tokenizer(student_dir), "story", 12, 0.8, torch.device("cpu"))
    assert text.startswith("story") and len(text) > 5

    pytest.importorskip("tokenizers")
    bpe = train(tmp_path, "bpe", data, tokenizer="bpe", tokenizer_path=tmp_path / "bpe_tok", vocab_size=300, examples=60)
    bpe_model = QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(**json.loads((bpe / "config.json").read_text())))
    bpe_model.load_state_dict(torch.load(bpe / "model.pt", weights_only=False)["model"])
    out = generate.sample_text(bpe_model.eval(), load_tokenizer(bpe), "story", 8, 0.8, torch.device("cpu"))
    assert out.startswith("story")


# ---- quantization -------------------------------------------------------------------------
@pytest.mark.parametrize("bits,group,max_rel", [(8, None, 0.02), (4, 16, 0.2)])
def test_quantized_linear_approximates_the_original(bits, group, max_rel):
    torch.manual_seed(0)
    linear = torch.nn.Linear(40, 24, bias=False)        # 40 is not a multiple of the 16-wide groups: exercises padding
    q = quant.QuantizedLinear.from_linear(linear, bits, group)
    x = torch.randn(5, 40)
    error = (q(x) - linear(x)).norm() / linear(x).norm()
    assert error < max_rel
    assert q.dequantize().shape == linear.weight.shape
    assert q.stored_bytes() < linear.weight.numel() * 4 / (3.5 if bits == 8 else 4)   # int4 pays for fp32 group scales


def test_int4_packing_is_lossless_for_representable_weights():
    scale = 0.5
    linear = torch.nn.Linear(8, 1, bias=False)
    linear.weight.data = torch.tensor([[-7, -1, 0, 1, 7, 3, -4, 2]], dtype=torch.float32) * scale   # amax = 7 * scale
    q = quant.QuantizedLinear.from_linear(linear, 4, 8)
    assert q.scales.item() == pytest.approx(scale)
    assert torch.equal(q.dequantize(), linear.weight)
    with pytest.raises(ValueError):
        quant.QuantizedLinear(8, 1, bits=4, group_size=7)
    with pytest.raises(ValueError):
        quant.QuantizedLinear(8, 1, bits=3)


def test_quantize_model_shrinks_experts_keeps_router_and_roundtrips(tmp_path):
    torch.manual_seed(1)
    config = QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=2, ffn_size=32, num_experts=4, top_k=2, attention_heads=2,
                               max_sequence_length=16, capacity_factor=0)
    model = QuantaWeaveMoEForCausalLM(config).eval()
    ids = torch.randint(0, 32, (2, 8))
    reference = model(ids)["logits"]
    router_before = model.blocks[0].moe.router.weight.clone()
    stats = quant.quantize_model(model, 8)
    assert stats["replaced_layers"] == 2 * 4 * 3 and stats["compression"] > 3.0
    assert torch.equal(model.blocks[0].moe.router.weight, router_before)         # router untouched
    assert isinstance(model.blocks[0].moe.experts[0].gate, quant.QuantizedLinear)
    approx = model(ids)["logits"]
    assert torch.nn.functional.cosine_similarity(reference.flatten(), approx.flatten(), dim=0) > 0.99
    with pytest.raises(ValueError, match="already quantized"):
        quant.quantize_model(model, 8)

    quant.save_quantized(model, tmp_path / "m.pt", stats)
    loaded = quant.load_quantized(tmp_path / "m.pt")
    assert torch.equal(loaded(ids)["logits"], approx)


# ---- export -------------------------------------------------------------------------------
def test_export_bundle_traces_with_dynamic_shapes_and_records_parity(tmp_path, student):
    _, data, student_dir = student
    out = tmp_path / "bundle"
    metadata = export.export_bundle(student_dir, out, quantize_bits=8, onnx=True, benchmark_data=data)
    for name in ("model.torchscript.pt", "model.graph.txt", "model.int8.pt", "config.json", "vocab.json", "export_metadata.json"):
        assert (out / name).exists(), name
    assert metadata["parity"]["traced_vs_eager_max_abs_diff"] < 1e-4
    assert metadata["parity"]["static_vs_sparse_max_abs_diff"] < 1e-4        # no capacity: dense evaluation equals sparse dispatch
    assert metadata["quantized"]["logit_cosine_similarity"] > 0.99 and metadata["quantized"]["compression"] > 3
    assert metadata["benchmark"]["loss"] > 0
    assert metadata["onnx"]["status"] in {"skipped", "ok", "failed"}
    assert all(len(v["sha256"]) == 64 for v in metadata["files"].values())

    module, tokenizer, loaded_meta = export.load_exported(out)
    ids = torch.tensor([tokenizer.encode("story 1: the")])
    logits = module(ids)
    assert logits.shape == (1, ids.size(1), metadata["config"]["vocab_size"]) and loaded_meta["training_step"] == 4


def test_export_static_graph_matches_eager_model_without_capacity(tmp_path, student):
    _, _, student_dir = student
    out = tmp_path / "bundle"
    export.export_bundle(student_dir, out)
    module, _, _ = export.load_exported(out)
    model, _ = export.load_checkpoint_model(student_dir)
    ids = torch.randint(0, 40, (3, 11))
    with torch.inference_mode():
        assert torch.allclose(module(ids), model(ids)["logits"], atol=1e-4)
