"""Tests for the FUTURE_IDEAS.md items implemented after the original status table was written."""

import asyncio
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import hardware as hw
from quantweave_moe_model import QuantaWeaveConfig


def cpu_device(total_gb: float = 8.0) -> hw.DeviceInfo:
    return hw.DeviceInfo("cpu", "fake", 1, int(total_gb * 1e9), int(total_gb * 1e9), True, False, "test")


# ---- hardware.py: auto_architecture ---------------------------------------------------------------------
@pytest.mark.parametrize("quality", ["capacity", "balanced", "dense"])
def test_auto_architecture_hits_its_memory_budget_and_is_a_valid_config(quality):
    info = cpu_device(8.0)
    shape = hw.auto_architecture(info, vocab_size=7168, memory_fraction=0.45, quality=quality)
    # a valid, buildable config
    config = QuantaWeaveConfig(
        vocab_size=shape["vocab_size"], hidden_size=shape["hidden_size"], layers=shape["layers"],
        ffn_size=shape["ffn_size"], num_experts=shape["total_experts"], top_k=shape["active_experts"],
        attention_heads=shape["attention_heads"], max_sequence_length=shape["sequence_length"] + 1,
    )
    assert config.hidden_size % config.attention_heads == 0
    # it actually used a large share of the budget rather than leaving most of it idle
    assert 0.7 < shape["training_state_fraction_of_budget"] <= 1.05
    assert shape["training_state_bytes"] < 0.5 * info.total_memory_bytes  # respects memory_fraction, not the whole device


def test_auto_architecture_capacity_vs_dense_use_the_budget_very_differently():
    info = cpu_device(8.0)
    capacity = hw.auto_architecture(info, quality="capacity")
    dense = hw.auto_architecture(info, quality="dense")
    assert dense["total_experts"] == dense["active_experts"] == 1
    assert dense["active_parameters"] == dense["total_parameters"]
    assert capacity["active_experts"] == 1 and capacity["total_experts"] > 1
    # dense cannot lean on a large expert pool, so it must be much wider than the MoE shape at the same budget
    assert dense["hidden_size"] > capacity["hidden_size"]  # no expert pool to lean on, so width carries the budget instead


def test_auto_architecture_scales_with_more_memory():
    small = hw.auto_architecture(cpu_device(8.0), quality="balanced")
    large = hw.auto_architecture(cpu_device(40.0), quality="balanced")
    assert large["total_parameters"] > small["total_parameters"]
    assert large["hidden_size"] >= small["hidden_size"] and large["layers"] >= small["layers"]


def test_auto_architecture_rejects_bad_input():
    with pytest.raises(ValueError, match="quality"):
        hw.auto_architecture(cpu_device(8.0), quality="bogus")
    with pytest.raises(ValueError, match="memory"):
        hw.auto_architecture(hw.DeviceInfo("cpu", "fake", 1, 0, 0, True, False, "test"))


def test_auto_architecture_refuses_a_shape_that_does_not_actually_fit_the_device():
    # far below the anchor tables' tested range: even the smallest shape would not fit this device at all
    with pytest.raises(ValueError, match="no architecture fits"):
        hw.auto_architecture(cpu_device(1.0), memory_fraction=0.1, quality="dense")
    # merely over the requested fraction (not over the device) must not raise
    shape = hw.auto_architecture(cpu_device(8.0), memory_fraction=0.1, quality="dense")
    assert shape["training_state_bytes"] < cpu_device(8.0).total_memory_bytes


def test_auto_architecture_cli_flag(capsys):
    old_argv = sys.argv
    sys.argv = ["hardware", "--device", "cpu", "--auto-architecture", "--quality", "capacity"]
    try:
        hw.main()
    finally:
        sys.argv = old_argv
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["auto_architecture"]["quality"] == "capacity"
    assert report["parameters"]["total"] == report["auto_architecture"]["total_parameters"]


# ---- train_quantweave_moe.py: --auto-architecture --------------------------------------------------------
import json

import train_quantweave_moe as trainer


def write_corpus(path: Path, rows: int = 60) -> Path:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(rows)))
    return path


def test_trainer_auto_architecture_overrides_the_shape_and_pins_it_for_reproduce(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], steps=3, examples=60, device="cpu", checkpoint_interval=0, vocab_size=64,
        auto_architecture=True, auto_architecture_quality="dense", auto_architecture_memory_fraction=0.02,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
        archive_dir=tmp_path / "archive",
    )
    summary = trainer.run_training(args)
    assert summary["steps"] == 3
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["num_experts"] == 1 and config["top_k"] == 1          # dense quality really took effect
    assert args.hidden_size != 64 or args.layers != 2                   # the defaults were actually replaced
    reproduce = (Path(summary["archive"]) / "reproduce.sh").read_text()
    tokens = reproduce.replace("\\\n", " ").split()
    assert "--auto-architecture" not in tokens                          # reproduce.sh pins the resolved shape, not the flag
    assert f"--hidden-size {args.hidden_size}" in reproduce and f"--total-experts {args.total_experts}" in reproduce


# ---- data_pipeline.py: SentencePieceTokenizer -----------------------------------------------------------
import data_pipeline as dp


SP_CORPUS = [f"the quick brown fox {i} jumps over the lazy dog" for i in range(60)] + [f"def add{i}(a, b): return a + b" for i in range(60)]


@pytest.mark.parametrize("algorithm", ["unigram", "bpe"])
def test_sentencepiece_tokenizer_trains_saves_and_roundtrips(tmp_path, algorithm):
    pytest.importorskip("sentencepiece")
    tokenizer = dp.SentencePieceTokenizer.train(SP_CORPUS, 96, algorithm)
    assert tokenizer.kind == "sentencepiece" and tokenizer.vocab_size <= 96
    text = "the lazy fox: add(a, b)"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.eos_id is not None and tokenizer.unk_id is not None

    tokenizer.save(tmp_path)
    assert (tmp_path / "tokenizer.model").exists() and (tmp_path / "tokenizer_meta.json").exists()
    reloaded = dp.load_tokenizer(tmp_path)
    assert isinstance(reloaded, dp.SentencePieceTokenizer) and reloaded.encode(text) == ids
    assert dp.token_classes(reloaded)[0].numel() == reloaded.vocab_size


def test_sentencepiece_rejects_a_bad_algorithm():
    pytest.importorskip("sentencepiece")
    with pytest.raises(ValueError, match="algorithm"):
        dp.SentencePieceTokenizer.train(SP_CORPUS, 64, "bogus")


def test_sentencepiece_and_bpe_and_char_all_satisfy_the_same_tokenizer_interface():
    pytest.importorskip("sentencepiece")
    pytest.importorskip("tokenizers")
    char = dp.CharTokenizer.build(SP_CORPUS, 64)
    bpe = dp.BPETokenizer.train(SP_CORPUS, 264)
    sp = dp.SentencePieceTokenizer.train(SP_CORPUS, 96)
    for tok in (char, bpe, sp):
        assert isinstance(tok.vocab_size, int) and tok.vocab_size > 0
        assert isinstance(tok.eos_id, int) and isinstance(tok.unk_id, int)
        ids = tok.encode("the quick fox")
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        assert isinstance(tok.decode(ids), str)


# ---- train_quantweave_moe.py: --tokenizer sentencepiece --------------------------------------------------
def test_trainer_trains_end_to_end_with_a_sentencepiece_tokenizer(tmp_path):
    pytest.importorskip("sentencepiece")
    data = write_corpus(tmp_path / "d.jsonl", rows=80)
    args = trainer.default_args(
        data=[data], steps=6, batch_size=4, sequence_length=16, examples=80, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, device="cpu", checkpoint_interval=0, capacity_factor=0,
        tokenizer="sentencepiece", tokenizer_path=tmp_path / "tok", vocab_size=64,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    trainer.run_training(args)
    assert (tmp_path / "tok" / "tokenizer.model").exists()
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    tokenizer = dp.load_tokenizer(tmp_path / "out")
    assert isinstance(tokenizer, dp.SentencePieceTokenizer) and config["vocab_size"] == tokenizer.vocab_size

    # a second run reuses the already-trained tokenizer instead of retraining it
    tokenizer.processor.__class__  # sanity: real object
    args2 = trainer.default_args(**{**vars(args), "output": tmp_path / "out2", "checkpoint_dir": tmp_path / "ck2"})
    trainer.run_training(args2)
    assert dp.load_tokenizer(tmp_path / "out2").vocab_size == tokenizer.vocab_size


def test_sentencepiece_tokenizer_generates_through_the_chat_tool(tmp_path):
    pytest.importorskip("sentencepiece")
    import chat_quantweave_moe as chat

    data = write_corpus(tmp_path / "d.jsonl", rows=80)
    args = trainer.default_args(
        data=[data], steps=8, batch_size=4, sequence_length=16, examples=80, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, device="cpu", checkpoint_interval=0, capacity_factor=0, lr=3e-3,
        tokenizer="sentencepiece", tokenizer_path=tmp_path / "tok", vocab_size=64,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    trainer.run_training(args)
    model, tokenizer = chat.load_for_chat(tmp_path / "out", None, 0, torch.device("cpu"))
    assert tokenizer.kind == "sentencepiece"
    session = chat.ChatSession(model, tokenizer, torch.device("cpu"), "fp32", chat.Settings(temperature=0.0, tokens=15))
    reply = session.send("story")
    assert isinstance(reply.response, str) and reply.unknown_characters == []  # not applicable to a subword tokenizer


# ---- data_pipeline.py: domain-aware train/val split ------------------------------------------------------
def make_multi_domain_dataset(sequence_length=16):
    root_data = []
    for i in range(200):
        root_data.append({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again and again.", "domain": "stories"})
    for i in range(20):
        root_data.append({"text": f"def add{i}(a, b): return a + b", "domain": "code"})
    return root_data


def build_dataset(tmp_path: Path, rows: list[dict], sequence_length: int = 16) -> dp.WindowDataset:
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    tokenizer = dp.CharTokenizer.build([r["text"] for r in rows], 64)
    return dp.WindowDataset(dp.build_corpus([data], tokenizer), sequence_length)


def test_split_train_val_has_no_real_overlap_and_covers_every_window(tmp_path):
    dataset = build_dataset(tmp_path, make_multi_domain_dataset())
    train, val = dataset.split_train_val(val_fraction=0.1, seed=0)
    train_starts, val_starts = set(train.starts.tolist()), set(val.starts.tolist())
    assert not (train_starts & val_starts)                              # identity (offset), not content, must never overlap
    assert train_starts | val_starts == set(dataset.starts.tolist())    # every original window ends up in exactly one split
    assert len(train) + len(val) == len(dataset)


def test_split_train_val_is_stratified_by_domain():
    pass  # covered by the assertions below, kept for discoverability


def test_split_train_val_holds_out_every_domain_proportionally(tmp_path):
    dataset = build_dataset(tmp_path, make_multi_domain_dataset())
    train, val = dataset.split_train_val(val_fraction=0.1, seed=0)
    for name in dataset.domains:
        index = dataset.domains.index(name)
        total = int((dataset.window_domains == index).sum())
        val_count = int((val.window_domains == index).sum())
        assert val_count > 0, f"domain '{name}' got zero validation windows"                # even the small "code" domain
        assert 0.05 < val_count / total < 0.2, (name, val_count, total)                     # roughly the requested 10%


def test_split_train_val_is_reproducible_and_seed_sensitive(tmp_path):
    dataset = build_dataset(tmp_path, make_multi_domain_dataset())
    a, _ = dataset.split_train_val(val_fraction=0.1, seed=0)
    b, _ = dataset.split_train_val(val_fraction=0.1, seed=0)
    c, _ = dataset.split_train_val(val_fraction=0.1, seed=1)
    assert torch.equal(a.starts, b.starts)
    assert not torch.equal(a.starts, c.starts)


def test_split_train_val_views_are_usable_as_ordinary_datasets(tmp_path):
    dataset = build_dataset(tmp_path, make_multi_domain_dataset())
    train, val = dataset.split_train_val(val_fraction=0.1, seed=0)
    assert train.domains == dataset.domains == val.domains
    assert train.window.numel() if False else train.window == dataset.window
    assert torch.equal(train.batch(range(3)), torch.stack([train[i] for i in range(3)]))
    stream = dp.BatchStream(train, batch_size=4, seed=0)
    batch, domains = stream.batch(1)
    assert batch.shape == (4, train.window) and domains.shape == (4,)


def test_split_train_val_rejects_bad_fraction_and_too_small_a_domain(tmp_path):
    dataset = build_dataset(tmp_path, make_multi_domain_dataset())
    with pytest.raises(ValueError, match="val_fraction"):
        dataset.split_train_val(val_fraction=0.0)
    with pytest.raises(ValueError, match="val_fraction"):
        dataset.split_train_val(val_fraction=1.0)
    with pytest.raises(ValueError, match="too few"):
        dataset.split_train_val(val_fraction=0.99)                      # would need to hold out the whole domain


# ---- train_quantweave_moe.py: --val-fraction ----------------------------------------------------------------
def write_val_corpus(path: Path, rows: int = 300) -> Path:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(rows)))
    return path


def test_validation_loss_is_computed_only_on_its_own_interval_not_every_logged_step(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], steps=12, batch_size=4, sequence_length=16, examples=300, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, capacity_factor=0,
        val_fraction=0.1, val_interval=4, val_batches=2, log_interval=1, metrics_file=tmp_path / "m.jsonl",
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    summary = trainer.run_training(args)
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "val_loss" in r] == [4, 8, 12]     # exactly the val-interval steps, not stale carry-over
    assert summary["val_loss"] is not None and summary["best_val_loss"] is not None
    assert summary["best_val_loss"] <= summary["val_loss"]
    assert summary["val_windows"] > 0


def test_validation_windows_are_never_trained_on(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], steps=3, batch_size=4, sequence_length=16, examples=300, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, capacity_factor=0,
        val_fraction=0.2, val_interval=1, val_batches=2,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    # reconstruct the exact same split the trainer will use, independently, and confirm zero index overlap
    tokenizer = dp.CharTokenizer.build((t for _, t in dp.read_rows([data])), 64)
    dataset = dp.WindowDataset(dp.build_corpus([data], tokenizer), 16)
    train_view, val_view = dataset.split_train_val(0.2, seed=0)
    assert not (set(train_view.starts.tolist()) & set(val_view.starts.tolist()))
    trainer.run_training(args)  # just confirm it runs with this split size without error


def test_val_fraction_requires_a_val_interval_and_rejects_a_domain_too_small_to_split(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl", rows=5)   # too few windows to hold any out
    args = trainer.default_args(
        data=[data], steps=2, batch_size=1, sequence_length=16, examples=5, hidden_size=16, layers=1, ffn_size=24,
        total_experts=2, active_experts=1, vocab_size=64, device="cpu", checkpoint_interval=0,
        val_fraction=0.99, output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    with pytest.raises(ValueError, match="too few"):
        trainer.run_training(args)


def test_val_fraction_is_rejected_with_pipeline_parallelism(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], device="cpu", checkpoint_interval=0, val_fraction=0.1, pipeline_parallel=True,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    with pytest.raises(ValueError, match="pipeline-parallel"):
        trainer.run_training(args)


def test_no_val_fraction_means_no_validation_at_all(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], steps=3, batch_size=4, sequence_length=16, examples=300, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    summary = trainer.run_training(args)
    assert summary["val_loss"] is None and summary["best_val_loss"] is None and summary["val_windows"] is None


# ---- train_quantweave_moe.py: --optimizer adamw8bit -----------------------------------------------------
def test_build_optimizer_returns_the_right_class_and_validates_its_input():
    model = torch.nn.Linear(4, 4)
    assert isinstance(trainer.build_optimizer("adamw", model.parameters(), 1e-3), torch.optim.AdamW)
    with pytest.raises(ValueError, match="optimizer must be"):
        trainer.build_optimizer("sgd", model.parameters(), 1e-3)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA; monkeypatching is_available() "
                                                     "process-wide was found to corrupt bitsandbytes' CUDA state for later tests")
def test_build_optimizer_adamw8bit_needs_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        trainer.build_optimizer("adamw8bit", torch.nn.Linear(4, 4).parameters(), 1e-3)


def test_trainer_rejects_adamw8bit_combined_with_shard_optimizer(tmp_path):
    data = write_val_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], device="cpu", checkpoint_interval=0, optimizer="adamw8bit", expert_parallel=True, shard_optimizer=True,
        output=tmp_path / "out", checkpoint_dir=tmp_path / "ck",
    )
    with pytest.raises((ValueError, RuntimeError)):
        trainer.run_training(args)


def test_finetune_still_shares_the_same_build_optimizer(tmp_path):
    import finetune_quantweave_moe as ft

    assert ft.build_optimizer is trainer.build_optimizer          # no duplicate implementation


needs_cuda_bnb = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda_bnb
def test_trainer_trains_end_to_end_with_adamw8bit_and_uses_less_memory(tmp_path):
    pytest.importorskip("bitsandbytes")
    # bitsandbytes only quantizes tensors above a minimum size, so the model needs real width to show a difference
    data = write_val_corpus(tmp_path / "d.jsonl", rows=60)
    common = dict(data=[data], steps=4, batch_size=4, sequence_length=32, examples=60, hidden_size=256, layers=2,
                 ffn_size=512, total_experts=8, active_experts=2, vocab_size=7168, device="cuda", checkpoint_interval=0,
                 lr=3e-3, no_archive=True)
    torch.cuda.reset_peak_memory_stats()
    trainer.run_training(trainer.default_args(**common, optimizer="adamw", output=tmp_path / "o1", checkpoint_dir=tmp_path / "c1"))
    fp32_peak = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    summary = trainer.run_training(trainer.default_args(**common, optimizer="adamw8bit", output=tmp_path / "o2", checkpoint_dir=tmp_path / "c2"))
    bit8_peak = torch.cuda.max_memory_allocated()
    assert summary["steps"] == 4 and torch.isfinite(torch.tensor(summary["final_loss"]))
    assert bit8_peak < fp32_peak


# ---- export_quantweave.py: real (now installed) ONNX verification -----------------------------------------
import export_quantweave as exq


@pytest.fixture(scope="module")
def small_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx")
    data = write_val_corpus(root / "d.jsonl", rows=60)
    trainer.run_training(trainer.default_args(
        data=[data], steps=10, batch_size=4, sequence_length=16, examples=60, hidden_size=16, layers=2, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, lr=3e-3,
        capacity_factor=0, output=root / "model", checkpoint_dir=root / "ck", no_archive=True,
    ))
    return root / "model"


def test_onnx_export_is_skipped_cleanly_without_the_package(small_checkpoint, tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "onnx":
            raise ImportError("blocked for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    metadata = exq.export_bundle(small_checkpoint, tmp_path / "bundle", onnx=True)
    assert metadata["onnx"]["status"] == "skipped"


def test_onnx_export_is_genuinely_dynamic_across_batch_and_sequence_length(small_checkpoint, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    metadata = exq.export_bundle(small_checkpoint, tmp_path / "bundle", onnx=True)
    assert metadata["onnx"]["status"] == "ok"
    assert metadata["onnx"]["dynamic_shape_verified"] is True
    assert metadata["onnx"]["verified_max_abs_diff"] < 1e-4

    model, _ = exq.load_checkpoint_model(small_checkpoint)
    model.set_static_dispatch(True)
    session = ort.InferenceSession(str(tmp_path / "bundle" / "model.onnx"), providers=["CPUExecutionProvider"])
    for batch, seq in [(1, 8), (4, 12), (2, 1), (6, 15)]:              # includes batch=1, the historical failure mode
        ids = torch.randint(0, 64, (batch, seq))
        with torch.no_grad():
            eager = model(ids)["logits"]
        onnx_out = torch.from_numpy(session.run(None, {"input_ids": ids.numpy()})[0])
        assert torch.allclose(onnx_out, eager, atol=1e-3), (batch, seq)


def test_onnx_export_failure_is_reported_not_raised(small_checkpoint, tmp_path, monkeypatch):
    pytest.importorskip("onnx")

    def broken_export(*a, **k):
        raise RuntimeError("synthetic export failure")

    monkeypatch.setattr(torch.onnx, "export", broken_export)
    metadata = exq.export_bundle(small_checkpoint, tmp_path / "bundle", onnx=True)
    assert metadata["onnx"]["status"] == "failed" and "synthetic export failure" in metadata["onnx"]["reason"]
    assert not (tmp_path / "bundle" / "model.onnx").exists()


# ---- experiment_manager.py: multi-objective (Pareto) search -----------------------------------------------
import experiment_manager as em
import sweep as sw


def synthetic_summaries():
    return [
        {"run_id": "a", "status": "completed", "benchmark": {"loss": 1.0}, "train": {"tokens_per_second": 100}},
        {"run_id": "b", "status": "completed", "benchmark": {"loss": 0.8}, "train": {"tokens_per_second": 90}},
        {"run_id": "c", "status": "completed", "benchmark": {"loss": 1.2}, "train": {"tokens_per_second": 80}},
        {"run_id": "d", "status": "completed", "benchmark": {"loss": 0.9}, "train": {"tokens_per_second": 150}},
        {"run_id": "e", "status": "failed", "benchmark": {"loss": 0.1}, "train": {"tokens_per_second": 200}},
    ]


def test_parse_objectives_defaults_direction_and_validates():
    assert em.parse_objectives(["a:min", "b:max"]) == [("a", "min"), ("b", "max")]
    assert em.parse_objectives(["a", "b:max"]) == [("a", "min"), ("b", "max")]
    with pytest.raises(ValueError, match="at least two"):
        em.parse_objectives(["a"])
    with pytest.raises(ValueError, match="min.*max"):
        em.parse_objectives(["a:min", "b:sideways"])


def test_pareto_front_matches_hand_worked_example():
    objectives = [("benchmark.loss", "min"), ("train.tokens_per_second", "max")]
    front = {s["run_id"] for s in em.pareto_front(synthetic_summaries(), objectives)}
    assert front == {"b", "d"}                    # a and c are each dominated by d; failed run e is excluded


def test_dominates_requires_strict_improvement_on_at_least_one_objective():
    objectives = [("x", "min"), ("y", "max")]
    identical_a, identical_b = {"x": 1, "y": 1}, {"x": 1, "y": 1}
    assert not em.dominates(identical_a, identical_b, objectives)     # equal on everything: neither dominates
    assert em.dominates({"x": 1, "y": 2}, {"x": 1, "y": 1}, objectives)   # equal on x, strictly better on y
    assert not em.dominates({"x": 2, "y": 2}, {"x": 1, "y": 1}, objectives)  # worse on x cancels better on y


def test_pareto_front_excludes_summaries_missing_an_objective_value():
    partial = [{"run_id": "x", "status": "completed", "a": 1}]  # no "b" field at all
    assert em.pareto_front(partial, [("a", "min"), ("b", "max")]) == []


def test_compare_pareto_marks_the_front_and_lists_every_completed_run():
    table = em.compare_pareto(synthetic_summaries(), ["benchmark.loss:min", "train.tokens_per_second:max"])
    rows = table.splitlines()
    assert "pareto-optimal" in rows[0]
    by_run = {line.split("|")[1].strip(): line for line in rows[2:]}
    assert by_run["b"].rstrip().endswith("yes |") and by_run["d"].rstrip().endswith("yes |")
    assert not by_run["a"].rstrip().endswith("yes |") and "e" not in by_run       # failed run is excluded entirely


# ---- sweep.py: objectives config field --------------------------------------------------------------------
def sweep_corpus(tmp_path: Path) -> Path:
    return write_val_corpus(tmp_path / "d.jsonl", rows=60)


def test_grid_sweep_reports_a_pareto_front_when_objectives_are_configured(tmp_path):
    data = sweep_corpus(tmp_path)
    config = {
        "name": "mo", "strategy": "grid",
        "base": {"data": [str(data)], "steps": 3, "batch_size": 4, "sequence_length": 16, "examples": 60,
                 "hidden_size": 16, "layers": 1, "vocab_size": 64, "device": "cpu", "checkpoint_interval": 0},
        "benchmark": {"examples": 15, "no_routing_stats": True}, "metric": "benchmark.loss",
        "objectives": ["benchmark.loss:min", "train.active_parameter_ratio:max"],
        "space": {"ffn_size": [16, 24], "total_experts": [2, 4]},
    }
    outcome = sw.run_sweep(config, tmp_path / "runs")
    assert "pareto_front" in outcome and len(outcome["pareto_front"]) >= 1
    for entry in outcome["pareto_front"]:
        assert set(entry["values"]) == {"benchmark.loss", "train.active_parameter_ratio"}
    markdown = (Path(outcome["sweep_dir"]) / "sweep_results.md").read_text()
    assert "## Pareto front" in markdown and "pareto-optimal" in markdown


def test_sweep_without_objectives_has_no_pareto_front_key(tmp_path):
    data = sweep_corpus(tmp_path)
    config = {
        "name": "single", "strategy": "grid",
        "base": {"data": [str(data)], "steps": 2, "batch_size": 4, "sequence_length": 16, "examples": 60,
                 "hidden_size": 16, "layers": 1, "vocab_size": 64, "device": "cpu", "checkpoint_interval": 0},
        "benchmark": {"examples": 10, "no_routing_stats": True}, "metric": "benchmark.loss",
        "space": {"total_experts": [2, 4]},
    }
    outcome = sw.run_sweep(config, tmp_path / "runs")
    assert "pareto_front" not in outcome
    assert "Pareto" not in (Path(outcome["sweep_dir"]) / "sweep_results.md").read_text()


def test_compare_cli_supports_objectives_alongside_metric(tmp_path, monkeypatch):
    data = sweep_corpus(tmp_path)
    base = dict(data=[str(data)], steps=3, batch_size=4, sequence_length=16, examples=60, hidden_size=16, layers=1,
                ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0)
    a = em.run_experiment({"name": "a", "train": base}, tmp_path / "runs", run_id="a")
    b = em.run_experiment({"name": "b", "train": {**base, "total_experts": 2}}, tmp_path / "runs", run_id="b")
    monkeypatch.setattr(sys, "argv", ["em", "compare", str(Path(a["run_dir"])), str(Path(b["run_dir"])),
                                      "--objectives", "benchmark.loss:min", "train.active_parameter_ratio:max"])
    em.main()


# ---- sweep.py: bohb (multi-fidelity Bayesian) search -------------------------------------------------------
def bohb_config(tmp_path: Path, **overrides) -> dict:
    data = sweep_corpus(tmp_path)
    base = {"data": [str(data)], "steps": 2, "batch_size": 4, "sequence_length": 16, "examples": 60, "hidden_size": 16,
            "layers": 1, "ffn_size": 24, "vocab_size": 64, "device": "cpu", "checkpoint_interval": 0}
    config = {"name": "bohb", "strategy": "bohb", "seed": 0, "n_trials": 6, "bayes": {"n_init": 3},
              "halving": {"eta": 2, "rungs": 2}, "base": base, "benchmark": {"examples": 10, "no_routing_stats": True},
              "metric": "benchmark.loss",
              # a continuous dimension keeps the space large enough for 6 distinct trials even with few discrete choices
              "space": {"total_experts": [2, 4], "active_experts": [1, 2], "lr": {"loguniform": [1e-4, 1e-2]}}}
    config.update(overrides)
    return config


def test_bohb_runs_the_base_rung_with_bayesian_suggestions_then_promotes_survivors(tmp_path):
    outcome = sw.run_sweep(bohb_config(tmp_path), tmp_path / "runs")
    assert [r["rung"] for r in outcome["rungs"]] == [0, 1]
    assert outcome["rungs"][0]["trials"] == 6                                  # the full base-rung budget
    assert outcome["rungs"][1]["trials"] == 3                                  # top half promoted (eta=2)
    assert [r["steps"] for r in outcome["rungs"]] == [2, 4]                    # rung 1 runs at eta x the base steps
    assert len(outcome["trajectory"]) == 6 and all(v is not None for v in outcome["trajectory"])
    assert outcome["best"]["run_id"].startswith("trial-r1")                    # the reported best came from the promoted rung


def test_bohb_never_proposes_an_invalid_combination(tmp_path):
    # active_experts=8 is only valid when total_experts=8 too; a continuous lr keeps the space large enough for 6 distinct trials
    config = bohb_config(tmp_path, space={"total_experts": [2, 4, 8], "active_experts": [1, 2, 8], "lr": {"loguniform": [1e-4, 1e-2]}})
    outcome = sw.run_sweep(config, tmp_path / "runs")
    # if an invalid trial had been submitted, run_experiment would have failed it and "completed" would drop below "trials"
    assert all(r["completed"] == r["trials"] for r in outcome["rungs"])


def test_bohb_dry_run_previews_without_training(tmp_path):
    plan = sw.run_sweep(bohb_config(tmp_path), tmp_path / "runs", dry_run=True)
    assert plan["dry_run"] and len(plan["first_trials"]) == 3
    assert not (tmp_path / "runs").exists()


def test_bohb_supports_objectives_like_the_other_strategies(tmp_path):
    outcome = sw.run_sweep(bohb_config(tmp_path, objectives=["benchmark.loss:min", "train.active_parameter_ratio:max"]), tmp_path / "runs")
    assert "pareto_front" in outcome and len(outcome["pareto_front"]) >= 1
    assert "## Pareto front" in (Path(outcome["sweep_dir"]) / "sweep_results.md").read_text()


def test_bohb_typo_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="strategy"):
        sw.run_sweep({**bohb_config(tmp_path), "strategy": "bohb-typo"}, tmp_path / "runs")


# ---- serve_quantweave.py: HTTP serving ----------------------------------------------------------------------
fastapi_testclient = pytest.importorskip("fastapi.testclient")
import serve_quantweave as sq  # noqa: E402


@pytest.fixture(scope="module")
def serve_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("serve")
    data = write_val_corpus(root / "d.jsonl", rows=80)
    trainer.run_training(trainer.default_args(
        data=[data], steps=12, batch_size=4, sequence_length=16, examples=80, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=96, device="cpu", checkpoint_interval=0, lr=3e-3,
        capacity_factor=0, output=root / "model", checkpoint_dir=root / "ck", no_archive=True,
    ))
    return root / "model"


@pytest.fixture()
def client(serve_checkpoint):
    server = sq.Server(serve_checkpoint, None, torch.device("cpu"), "fp32", use_cache=True)
    return fastapi_testclient.TestClient(sq.build_app(server)), server


def test_health_reports_model_info(client):
    test_client, server = client
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["tokenizer"] == "char" and body["requests_served"] == 0
    assert body["checkpoint"] == str(server.checkpoint) and "decoding" in body


def test_generate_endpoint_is_deterministic_at_temperature_zero(client):
    test_client, _ = client
    payload = {"prompt": "story 3:", "tokens": 12, "temperature": 0}
    a = test_client.post("/generate", json=payload).json()
    b = test_client.post("/generate", json=payload).json()
    assert a["text"] == b["text"] and a["tokens"] == 12 and a["tokens_per_second"] > 0
    assert a["stop_reason"] in {"length", "eos", "stop"}


def test_generate_matches_the_chat_tools_own_session_for_the_same_settings(client, serve_checkpoint):
    import chat_quantweave_moe as chat

    test_client, _ = client
    response = test_client.post("/generate", json={"prompt": "story 5:", "tokens": 10, "temperature": 0}).json()
    model, tokenizer = chat.load_for_chat(serve_checkpoint, None, 0, torch.device("cpu"))
    session = chat.ChatSession(model, tokenizer, torch.device("cpu"), "fp32", chat.Settings(temperature=0, tokens=10))
    assert session.send("story 5:").response == response["text"]


def test_generate_validates_its_input(client):
    test_client, _ = client
    assert test_client.post("/generate", json={"prompt": "x", "tokens": 0}).status_code == 422       # tokens must be >= 1
    assert test_client.post("/generate", json={"prompt": "x", "temperature": -1}).status_code == 422
    assert test_client.post("/generate", json={"prompt": "x", "top_p": 0}).status_code == 422
    assert test_client.post("/generate", json={}).status_code == 422                                  # prompt is required


def test_chat_endpoint_requires_a_trailing_user_turn_and_accepts_history(client):
    test_client, _ = client
    assert test_client.post("/chat", json={"messages": []}).status_code == 422
    assert test_client.post("/chat", json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 422
    response = test_client.post("/chat", json={"messages": [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "second"},
    ], "tokens": 8, "temperature": 0})
    assert response.status_code == 200 and isinstance(response.json()["text"], str)


def test_stream_endpoint_yields_pieces_that_concatenate_to_the_final_text(client):
    test_client, _ = client
    with test_client.stream("POST", "/generate/stream", json={"prompt": "story 1:", "tokens": 16, "temperature": 0}) as response:
        lines = [line for line in response.iter_lines() if line]
    assert response.status_code == 200

    # every "data:" line up to (not including) "event: done" is a streamed text piece; the one after it is the summary
    done_index = lines.index("event: done")
    pieces = [json.loads(line[len("data: "):])["text"] for line in lines[:done_index]]
    final = json.loads(lines[done_index + 1][len("data: "):])
    assert "".join(pieces) == final["text"] and final["tokens"] == 16


def test_concurrent_requests_are_serialized_and_never_corrupt_each_other(serve_checkpoint):
    async def run():
        server = sq.Server(serve_checkpoint, None, torch.device("cpu"), "fp32", use_cache=True)
        request = sq.GenerateRequest(prompt="story 2:", tokens=20, temperature=0)
        results = await asyncio.gather(*[server.generate(request) for _ in range(6)])
        return results, server.requests_served

    results, served = asyncio.run(run())
    assert len({r["text"] for r in results}) == 1                    # identical deterministic requests, identical output
    assert served == 6


def test_build_app_exposes_the_server_on_app_state(client):
    test_client, server = client
    assert test_client.app.state.server is server


# ---- expert_parallel.py: reshard_checkpoint ----------------------------------------------------------------
import expert_parallel as epar


@pytest.fixture(scope="module")
def reshardable_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("reshard")
    data = write_val_corpus(root / "d.jsonl", rows=60)
    trainer.run_training(trainer.default_args(
        data=[data], steps=10, batch_size=4, sequence_length=16, examples=60, hidden_size=32, layers=2, ffn_size=48,
        total_experts=8, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, lr=3e-3,
        output=root / "model", checkpoint_dir=root / "ck", no_archive=True,
    ))
    return root / "model", data


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_reshard_then_consolidate_is_the_identity(reshardable_checkpoint, tmp_path, ep_size):
    model_dir, _ = reshardable_checkpoint
    original = torch.load(model_dir / "model.pt", weights_only=False)["model"]
    meta = epar.reshard_checkpoint(model_dir, tmp_path / f"shards-{ep_size}", ep_size=ep_size)
    assert len(meta["shards"]) == ep_size and meta["per_rank_experts"] == 8 // ep_size
    epar.consolidate_checkpoint(tmp_path / f"shards-{ep_size}", tmp_path / f"back-{ep_size}")
    reconsolidated = torch.load(tmp_path / f"back-{ep_size}" / "model.pt", weights_only=False)["model"]
    assert reconsolidated.keys() == original.keys()
    for key in original:
        assert torch.equal(reconsolidated[key], original[key]), key   # byte-exact, not just close


def test_reshard_with_tensor_parallel_too_is_also_the_identity(reshardable_checkpoint, tmp_path):
    model_dir, _ = reshardable_checkpoint
    original = torch.load(model_dir / "model.pt", weights_only=False)["model"]
    meta = epar.reshard_checkpoint(model_dir, tmp_path / "shards", ep_size=2, tp_size=2)
    assert meta["world_size"] == 4 and len(meta["shards"]) == 4
    epar.consolidate_checkpoint(tmp_path / "shards", tmp_path / "back")
    reconsolidated = torch.load(tmp_path / "back" / "model.pt", weights_only=False)["model"]
    for key in original:
        assert torch.equal(reconsolidated[key], original[key]), key


def test_reshard_rejects_shapes_that_do_not_divide_evenly(reshardable_checkpoint, tmp_path):
    model_dir, _ = reshardable_checkpoint
    with pytest.raises(ValueError, match="num_experts"):
        epar.reshard_checkpoint(model_dir, tmp_path / "bad1", ep_size=3)          # 8 experts, not divisible by 3
    with pytest.raises(ValueError, match="attention_heads"):
        epar.reshard_checkpoint(model_dir, tmp_path / "bad2", ep_size=2, tp_size=3)  # 4 heads, not divisible by 3


def test_reshard_copies_the_tokenizer_and_config_and_marks_weights_only(reshardable_checkpoint, tmp_path):
    model_dir, _ = reshardable_checkpoint
    epar.reshard_checkpoint(model_dir, tmp_path / "shards", ep_size=2)
    assert (tmp_path / "shards" / "config.json").exists() and (tmp_path / "shards" / "vocab.json").exists()
    shard = torch.load(epar.shard_path(tmp_path / "shards", 0), weights_only=False)
    assert shard["optimizer"] is None
    assert shard["layout"] == {"kind": "ep", "ep_rank": 0, "ep_size": 2, "tp_rank": 0, "tp_size": 1}


def test_load_sharded_checkpoint_tolerates_a_missing_optimizer_state(reshardable_checkpoint, tmp_path):
    """A regression test for a real bug: load_sharded_checkpoint used to crash on reshard_checkpoint's
    intentionally-None optimizer state instead of starting a fresh optimizer."""
    model_dir, _ = reshardable_checkpoint
    epar.reshard_checkpoint(model_dir, tmp_path / "shards", ep_size=2)
    ctx = epar.ExpertParallelContext(0, 1, torch.device("cpu"), None, False, 0, 1, None, global_rank=0, global_size=1)
    from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM

    # ep_size=2 was resharded, but load a *single*-rank view of one shard directly to keep this test non-distributed.
    # The router stays fully replicated (it scores every expert globally); only expert FFN weights are split, and
    # the model constructor works that out itself from ctx.world_size, so the config keeps the full expert count.
    config = QuantaWeaveConfig(**json.loads((model_dir / "config.json").read_text()))
    ctx_for_shard0 = epar.ExpertParallelContext(0, 2, torch.device("cpu"), None, False, 0, 1, None)
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx_for_shard0)
    optimizer = torch.optim.AdamW(model.parameters())
    step = epar.load_sharded_checkpoint(tmp_path / "shards", model, optimizer, ctx_for_shard0)
    assert step == 10
    assert optimizer.state == {}                                       # fresh optimizer, no crash


# ---- reshard_checkpoint: real multi-process load, matching the single-device original ----------------------
import socket

import torch.multiprocessing as mp


def _reshard_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _reshard_load_worker(rank, world, port, checkpoint_dir, original_state_path):
    import os

    from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = epar.init_expert_parallel("cpu")
    config = QuantaWeaveConfig(**json.loads((checkpoint_dir / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
    optimizer = torch.optim.AdamW(model.parameters())
    step = epar.load_sharded_checkpoint(checkpoint_dir, model, optimizer, ctx)  # this line used to crash (see the regression test above)

    torch.manual_seed(123)
    ids = torch.randint(0, config.vocab_size, (2, 10))
    model.eval()
    with torch.no_grad():
        out = model(ids)["logits"]

    reference = QuantaWeaveMoEForCausalLM(config)
    reference.load_state_dict(torch.load(original_state_path, weights_only=False))
    reference.eval()
    with torch.no_grad():
        expected = reference(ids)["logits"]
    assert step == 10 and float((out - expected).abs().max()) < 1e-4
    epar.finish(ctx)


def test_resharded_checkpoint_loads_correctly_across_real_distributed_ranks(reshardable_checkpoint, tmp_path):
    model_dir, _ = reshardable_checkpoint
    epar.reshard_checkpoint(model_dir, tmp_path / "shards", ep_size=2)
    state_path = tmp_path / "original_state.pt"
    torch.save(torch.load(model_dir / "model.pt", weights_only=False)["model"], state_path)
    mp.spawn(_reshard_load_worker, args=(2, _reshard_free_port(), tmp_path / "shards", state_path), nprocs=2, join=True)


# ---- DeviceLoadTracker: capacity adapts to device speed, and per-device routing gathering --------------------
def _speed_config() -> "QuantaWeaveConfig":
    return QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=1, ffn_size=24, num_experts=8, top_k=2,
                             attention_heads=4, max_sequence_length=12, capacity_factor=1.0,
                             min_expert_capacity=1, router_aux_loss_coef=0.01)


def _capacity_worker(rank, world, port):
    import os

    import torch.distributed as dist

    from quantweave_moe_model import QuantaWeaveMoEForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = epar.init_expert_parallel("cpu")
    torch.manual_seed(0)
    config = _speed_config()
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
    moe = model.moes()[0]
    moe.simulated_cost_per_row = 4e-3 if rank == 1 else 1e-3  # rank 1 is 4x slower per row

    tracker = epar.DeviceLoadTracker(model, ctx, strength=0.5, interval=1, adapt_capacity=True,
                                     capacity_scale_min=0.5, capacity_scale_max=2.0)
    torch.manual_seed(100 + rank)
    ids = torch.randint(0, config.vocab_size, (2, 10))

    metrics = None
    for step in range(1, 21):
        model.set_collect_stats(True)
        out = model(ids, labels=ids)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
        metrics = tracker.update(step)

    assert metrics is not None and "capacity_scale" in metrics
    if rank == 1:
        assert moe.capacity_scale < 1.0, moe.capacity_scale     # the slower rank sheds capacity
    else:
        assert moe.capacity_scale > 1.0, moe.capacity_scale     # the faster rank is granted more

    per_device = tracker.gather_expert_utilization()
    assert per_device.shape == (world, config.layers, config.num_experts)
    assert float(per_device.sum()) > 0                            # real routed tokens were counted, not zeros
    gathered = [torch.zeros_like(per_device) for _ in range(world)]
    dist.all_gather(gathered, per_device)
    assert all(torch.equal(g, gathered[0]) for g in gathered), "every rank must gather the same tensor"

    if rank == 0:
        from routing_diagnostics import RoutingMonitor

        monitor = RoutingMonitor(config.layers, config.num_experts, config.top_k, ["d"], None, None)
        monitor.record_device_utilization(per_device)
        monitor.record_device_utilization(per_device)
        assert monitor.device_load is not None
        assert torch.allclose(monitor.device_load, per_device.double() * 2)
    epar.finish(ctx)


def test_straggler_capacity_shrinks_the_slow_ranks_capacity_and_grows_the_fast_ranks():
    # also covers gather_expert_utilization's cross-rank agreement and RoutingMonitor.record_device_utilization,
    # asserted inside the worker itself (a second mp.spawn just to re-check the same thing would only add cost).
    mp.spawn(_capacity_worker, args=(2, _reshard_free_port()), nprocs=2, join=True)


def _no_adapt_capacity_worker(rank, world, port):
    import os

    from quantweave_moe_model import QuantaWeaveMoEForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = epar.init_expert_parallel("cpu")
    torch.manual_seed(0)
    config = _speed_config()
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
    moe = model.moes()[0]
    moe.simulated_cost_per_row = 4e-3 if rank == 1 else 1e-3
    tracker = epar.DeviceLoadTracker(model, ctx, strength=0.0, interval=1, adapt_capacity=False)
    torch.manual_seed(100 + rank)
    ids = torch.randint(0, config.vocab_size, (2, 10))
    metrics = None
    for step in range(1, 6):
        model.set_collect_stats(False)
        out = model(ids, labels=ids)
        out["loss"].backward()
        model.zero_grad(set_to_none=True)
        metrics = tracker.update(step)
    assert metrics is not None and "capacity_scale" not in metrics
    assert moe.capacity_scale == 1.0
    epar.finish(ctx)


def test_device_load_tracker_without_adapt_capacity_leaves_capacity_scale_at_one():
    mp.spawn(_no_adapt_capacity_worker, args=(2, _reshard_free_port()), nprocs=2, join=True)


# ---- train_quantweave_moe.py CLI/training-loop wiring for --straggler-capacity and per-device diagnostics ----
def _straggler_capacity_trainer_worker(rank, world, port, tmp):
    import os

    import train_quantweave_moe as trainer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    tmp = Path(tmp)
    args = trainer.default_args(
        data=[tmp / "d.jsonl"], steps=8, batch_size=2, sequence_length=16, examples=60, hidden_size=16, layers=2,
        ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0,
        output=tmp / "out", checkpoint_dir=tmp / "ckpt", metrics_file=tmp / "metrics.jsonl",
        expert_parallel=True, straggler_capacity=True, straggler_capacity_min=0.5, straggler_capacity_max=2.0,
        diagnostics_interval=2, diagnostics_dir=tmp / "diag",
    )
    trainer.run_training(args)


def test_trainer_wires_straggler_capacity_and_writes_per_device_diagnostics(tmp_path):
    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60))
    )
    mp.spawn(_straggler_capacity_trainer_worker, args=(2, _reshard_free_port(), tmp_path), nprocs=2, join=True)
    diag_files = sorted(p.name for p in (tmp_path / "diag").iterdir())
    assert any(name.startswith("device_utilization_layer") for name in diag_files), diag_files
    assert "routed_tokens_by_device.svg" in diag_files, diag_files
    # every device_utilization SVG is non-trivial content, not an empty placeholder
    for name in diag_files:
        if name.startswith("device_utilization_layer"):
            assert (tmp_path / "diag" / name).stat().st_size > 200


def test_straggler_capacity_needs_expert_parallel():
    import train_quantweave_moe as trainer

    with pytest.raises(ValueError, match="straggler-capacity"):
        trainer.run_training(trainer.default_args(
            data=[Path("unused.jsonl")], straggler_capacity=True, expert_parallel=False, tensor_parallel=1,
        ))


def test_bar_chart_svg_renders_a_bar_per_value():
    from routing_diagnostics import bar_chart_svg

    svg = bar_chart_svg({"device 0": 10.0, "device 1": 40.0}, "totals")
    assert svg.startswith("<svg") and svg.count("<rect") == 2
    assert "device 0" in svg and "device 1" in svg


# ---- cpu_offload_optimizer.py: AdamW with moments in pinned CPU memory instead of device memory ----------------
def test_cpu_offload_adamw_matches_torch_adamw_step_for_step():
    from cpu_offload_optimizer import CPUOffloadAdamW

    torch.manual_seed(0)
    ref_params = [torch.randn(8, 4, requires_grad=True), torch.randn(16, requires_grad=True)]
    off_params = [p.detach().clone().requires_grad_() for p in ref_params]
    ref_opt = torch.optim.AdamW(ref_params, lr=1e-2, weight_decay=0.01)
    off_opt = CPUOffloadAdamW(off_params, lr=1e-2, weight_decay=0.01)

    torch.manual_seed(1)
    for _ in range(10):
        step_grads = [torch.randn_like(p) for p in ref_params]
        for p, g in zip(ref_params, step_grads):
            p.grad = g.clone()
        for p, g in zip(off_params, step_grads):
            p.grad = g.clone()
        ref_opt.step()
        off_opt.step()
        ref_opt.zero_grad()
        off_opt.zero_grad()

    for r, o in zip(ref_params, off_params):
        assert torch.allclose(r, o, atol=1e-6), (r - o).abs().max()


def test_cpu_offload_adamw_state_lives_off_the_autograd_graph_and_round_trips():
    from cpu_offload_optimizer import CPUOffloadAdamW

    torch.manual_seed(0)
    params = [torch.randn(6, requires_grad=True)]
    opt = CPUOffloadAdamW(params, lr=1e-2)
    params[0].grad = torch.randn(6)
    opt.step()
    state = next(iter(opt.state.values()))
    assert state["exp_avg"].device.type == "cpu" and state["exp_avg_sq"].device.type == "cpu"
    assert opt.state_numel() == 12  # exp_avg + exp_avg_sq, 6 elements each

    restored_params = [p.detach().clone().requires_grad_() for p in params]
    restored = CPUOffloadAdamW(restored_params, lr=1e-2)
    restored.load_state_dict(opt.state_dict())
    restored_state = next(iter(restored.state.values()))
    assert torch.equal(restored_state["exp_avg"], state["exp_avg"])
    assert torch.equal(restored_state["exp_avg_sq"], state["exp_avg_sq"])
    assert restored_state["step"] == state["step"] == 1


def test_build_optimizer_accepts_adamw_cpu_offload():
    from cpu_offload_optimizer import CPUOffloadAdamW
    from train_quantweave_moe import build_optimizer

    params = [torch.nn.Parameter(torch.randn(4))]
    optimizer = build_optimizer("adamw_cpu_offload", params, lr=1e-3)
    assert isinstance(optimizer, CPUOffloadAdamW)


def test_trainer_trains_end_to_end_with_adamw_cpu_offload_and_resumes(tmp_path):
    import train_quantweave_moe as trainer

    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60))
    )
    shared = dict(
        data=[tmp_path / "d.jsonl"], batch_size=2, sequence_length=16, examples=60, hidden_size=16, layers=2,
        ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=4,
        checkpoint_dir=tmp_path / "ckpt", optimizer="adamw_cpu_offload",
    )
    first = trainer.run_training(trainer.default_args(**shared, steps=8, output=tmp_path / "out1", metrics_file=tmp_path / "m1.jsonl"))
    assert first["final_loss"] > 0
    second = trainer.run_training(trainer.default_args(**shared, steps=12, output=tmp_path / "out2", metrics_file=tmp_path / "m2.jsonl"))
    assert second["steps"] == 12  # resumed from the step-8 checkpoint and trained 4 more steps, not restarted


def _shard_optimizer_rejects_cpu_offload_worker(rank, world, port, data_path):
    import os

    import train_quantweave_moe as trainer

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    try:
        with pytest.raises(ValueError, match="adamw_cpu_offload"):
            trainer.run_training(trainer.default_args(
                data=[Path(data_path)], expert_parallel=True, shard_optimizer=True, optimizer="adamw_cpu_offload",
                device="cpu", hidden_size=16, layers=2, ffn_size=24, total_experts=4, active_experts=2,
                vocab_size=64, batch_size=2, sequence_length=16, examples=60, steps=1,
            ))
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def test_shard_optimizer_rejects_adamw_cpu_offload(tmp_path):
    # reached through the same code path as real expert-parallel training, so it needs a real (if tiny) process
    # group and real data rather than plain args, matching this file's other expert-parallel tests.
    data_path = tmp_path / "d.jsonl"
    data_path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60)))
    mp.spawn(_shard_optimizer_rejects_cpu_offload_worker, args=(2, _reshard_free_port(), str(data_path)), nprocs=2, join=True)


# ---- pipeline_parallel.py: 1F1B schedule -------------------------------------------------------------------
def _pipeline_config(layers, coef=0.0):
    return QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=layers, ffn_size=24, num_experts=4, top_k=2,
                             attention_heads=4, max_sequence_length=12, capacity_factor=0.0, min_expert_capacity=1,
                             router_aux_loss_coef=coef)


def _pipeline_stage_state(full_state, ctx, model):
    out = {}
    for name in model.state_dict():
        match = epar.BLOCK_KEY.match(name)
        full_name = f"blocks.{ctx.layer_start + int(match.group(1))}." + name[match.end():] if match else name
        out[name] = full_state[full_name]
    return out


def _1f1b_worker(rank, world, port, layers, microbatches, schedule):
    import os

    import pipeline_parallel as pl
    from quantweave_moe_model import QuantaWeaveMoEForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = pl.init_pipeline_parallel("cpu", layers)
    config = _pipeline_config(layers)
    torch.manual_seed(0)
    reference = QuantaWeaveMoEForCausalLM(config)
    model = QuantaWeaveMoEForCausalLM(config, pipeline=ctx)
    model.load_state_dict(_pipeline_stage_state(reference.state_dict(), ctx, model))

    torch.manual_seed(9)
    batch = torch.randint(0, 32, (microbatches * 2, 10))
    result = pl.PipelineEngine(model, ctx, microbatches, schedule=schedule).train_step(batch)

    expected = reference(batch, labels=batch)
    assert abs(result["loss"] - expected["loss"].item()) < 1e-4, (schedule, result["loss"], expected["loss"].item())
    expected["loss"].backward()
    reference_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        match = epar.BLOCK_KEY.match(name)
        full = f"blocks.{ctx.layer_start + int(match.group(1))}." + name[match.end():] if match else name
        assert param.grad is not None, name
        assert torch.allclose(param.grad, reference_params[full].grad, atol=1e-4, rtol=1e-3), f"gradient of {full}"
    epar.finish(ctx)


@pytest.mark.parametrize("world,layers,microbatches", [
    (2, 4, 1),    # microbatches == 1: 1F1B degenerates to the same thing as GPipe
    (2, 4, 4),    # this exact shape deadlocked before the isend fix (regression coverage)
    (3, 6, 3),    # microbatches == stages
    (3, 6, 1),
    (4, 8, 2),    # microbatches < stages - 1: the warmup count's min(M, ...) clamp
])
def test_1f1b_matches_single_device_reference_gradients(world, layers, microbatches):
    mp.spawn(_1f1b_worker, args=(world, _reshard_free_port(), layers, microbatches, "1f1b"), nprocs=world, join=True)


def _1f1b_repeat_worker(rank, world, port, layers, microbatches):
    """Regression test for the isend deadlock: several 1F1B steps back to back on the same engine must not hang."""
    import os

    import pipeline_parallel as pl
    from quantweave_moe_model import QuantaWeaveMoEForCausalLM

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = pl.init_pipeline_parallel("cpu", layers)
    config = _pipeline_config(layers)
    torch.manual_seed(0)
    model = QuantaWeaveMoEForCausalLM(config, pipeline=ctx)
    engine = pl.PipelineEngine(model, ctx, microbatches, schedule="1f1b")
    for step in range(3):
        torch.manual_seed(step)
        batch = torch.randint(0, 32, (microbatches * 2, 10))
        for p in model.parameters():
            p.grad = None
        result = engine.train_step(batch)
        assert result["loss"] > 0
    epar.finish(ctx)


def test_1f1b_does_not_deadlock_across_repeated_steps():
    mp.spawn(_1f1b_repeat_worker, args=(3, _reshard_free_port(), 6, 3), nprocs=3, join=True)


def test_pipeline_engine_rejects_an_unknown_schedule():
    from pipeline_parallel import PipelineContext, PipelineEngine
    from quantweave_moe_model import QuantaWeaveMoEForCausalLM

    ctx = PipelineContext(0, 1, torch.device("cpu"), 0, 4)
    model = QuantaWeaveMoEForCausalLM(_pipeline_config(4))
    model.pipeline = ctx
    with pytest.raises(ValueError, match="schedule"):
        PipelineEngine(model, ctx, 1, schedule="bogus")


def test_trainer_accepts_pipeline_schedule_flag():
    import train_quantweave_moe as trainer

    args = trainer.default_args(data=[Path("unused.jsonl")], pipeline_schedule="1f1b")
    assert args.pipeline_schedule == "1f1b"
    with pytest.raises(SystemExit):
        trainer.build_parser().parse_args(["--data", "x.jsonl", "--pipeline-schedule", "bogus"])
