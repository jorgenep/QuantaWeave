import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import hardware
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def write_corpus(path: Path, rows: int = 60, domains: tuple[str, ...] = ()) -> Path:
    lines = []
    for i in range(rows):
        row = {"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again and again."}
        if domains:
            row["domain"] = domains[i % len(domains)]
            if row["domain"] == "code":
                row["text"] = f"def f{i}(x): return x * {i} + 1  # compute {i}; end of function {i}."
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    return path


def train(tmp_path: Path, name: str, data: Path, **overrides):
    values = dict(
        data=[data], steps=6, batch_size=2, sequence_length=16, examples=100, hidden_size=16, layers=2,
        ffn_size=32, total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0,
        output=tmp_path / name / "out", checkpoint_dir=tmp_path / name / "ckpt", log_interval=1,
    )
    values.update(overrides)
    return trainer.run_training(trainer.default_args(**values))


def weights(directory: Path) -> dict:
    return torch.load(directory / "model.pt", weights_only=False)["model"]


def test_resume_reproduces_an_uninterrupted_run_exactly(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    common = dict(lr_decay="cosine", warmup_steps=2, schedule_steps=6, checkpoint_interval=3, curriculum="entropy",
                  curriculum_steps=6, plateau_patience=1, controller_interval=2)
    train(tmp_path, "straight", data, steps=6, **common)

    # first leg stops at step 3 (its checkpoint at step 3 is the resume point), second leg finishes
    train(tmp_path, "split", data, steps=3, **common)
    resumed = train(tmp_path, "split", data, steps=6, output=tmp_path / "split" / "out2", **common)
    assert resumed["steps"] == 6

    straight, split = weights(tmp_path / "straight" / "out"), weights(tmp_path / "split" / "out2")
    for name, tensor in straight.items():
        assert torch.equal(tensor, split[name]), name


def test_resume_refuses_a_different_run_configuration(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    train(tmp_path, "run", data, steps=2, checkpoint_interval=2)
    with pytest.raises(ValueError, match="different run configuration"):
        train(tmp_path, "run", data, steps=4, batch_size=4, output=tmp_path / "run" / "out2")
    # explicit opt-in continues
    result = train(tmp_path, "run", data, steps=4, batch_size=4, output=tmp_path / "run" / "out3", allow_config_change=True)
    assert result["steps"] == 4


def test_checkpoint_carries_rich_metadata(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    train(tmp_path, "run", data, steps=4, checkpoint_interval=2, capacity_adapt=True, controller_interval=2)
    meta = json.loads((tmp_path / "run" / "ckpt" / "metadata.json").read_text())
    assert meta["step"] == 4 and len(meta["config_hash"]) == 16
    assert meta["data"]["samples_consumed"] == 4 * 2 and meta["data"]["seed"] == 0
    assert meta["schedule"]["capacity_factor"] > 0 and "routing_controls" in meta
    state = torch.load(tmp_path / "run" / "ckpt" / "model.pt", weights_only=False)
    assert state["extra"]["schedule"]["config"]["capacity_adapt"] is True


def test_metrics_and_diagnostics_are_written(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    summary = train(tmp_path, "run", data, steps=8, metrics_file=tmp_path / "m.jsonl",
                    diagnostics_dir=tmp_path / "diag", diagnostics_interval=2)
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines] == list(range(1, 9))
    assert {"loss", "lr", "capacity_factor", "router_temperature", "tokens_per_second", "overflow_fraction"} <= set(lines[0])
    assert summary["active_parameter_ratio"] < 1 and summary["steps_logged"] == 8
    assert (tmp_path / "diag" / "routing_log.jsonl").exists() and (tmp_path / "diag" / "index.html").exists()
    assert len((tmp_path / "diag" / "routing_log.jsonl").read_text().splitlines()) == 4


def test_capacity_controller_grows_capacity_in_the_saved_model(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    train(tmp_path, "run", data, steps=20, capacity_factor=0.3, min_expert_capacity=1, capacity_adapt=True,
          capacity_min=0.3, capacity_max=3.0, controller_interval=2, drop_threshold=0.0)
    config = json.loads((tmp_path / "run" / "out" / "config.json").read_text())
    assert config["capacity_factor"] > 0.3          # the controller saw overflow and raised it


def test_capacity_release_switches_enforcement_off(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    train(tmp_path, "run", data, steps=6, capacity_factor=0.3, min_expert_capacity=1, capacity_release_step=4,
          controller_interval=2, metrics_file=tmp_path / "m.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert lines[0]["overflow_routes"] > 0 and lines[-1]["overflow_routes"] == 0
    assert json.loads((tmp_path / "run" / "out" / "config.json").read_text())["drop_overflow_tokens"] is False


def test_aux_and_temperature_adaptation_run_and_stay_finite(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    summary = train(tmp_path, "run", data, steps=12, aux_adapt=True, temperature_adapt=True, controller_interval=3,
                    temperature_start=2.0, temperature_steps=6, router_temperature=1.0, metrics_file=tmp_path / "m.jsonl")
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert lines[0]["router_temperature"] == pytest.approx(2.0 - 1 / 6)      # annealing 2.0 -> 1.0 over 6 steps
    assert lines[-1]["router_temperature"] < lines[0]["router_temperature"]
    assert all(torch.isfinite(torch.tensor(line["loss"])) for line in lines)
    assert summary["last_routing"] is not None


def test_multi_domain_training_tracks_domains_and_specialization_loss(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl", rows=80, domains=("stories", "code"))
    summary = train(tmp_path, "run", data, steps=6, domain_weights="stories=0.5,code=0.5",
                    domain_specialization_coef=1.0, diagnostics_dir=tmp_path / "diag", diagnostics_interval=2,
                    metrics_file=tmp_path / "m.jsonl", batch_size=4)
    assert set(summary["stream"]["domain_weights"]) == {"stories", "code"}
    lines = [json.loads(line) for line in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert any(line["domain_loss"] > 0 for line in lines)
    assert (tmp_path / "diag" / "domain_usage_layer0.svg").exists()

    shifting = train(tmp_path, "shift", data, steps=4, domain_weights="stories=1,code=0", domain_weights_end="stories=0,code=1",
                     domain_weights_steps=4)
    assert shifting["stream"]["domain_weights"]["code"] == 1.0


def test_bpe_tokenizer_and_packed_token_corpus_paths(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("numpy")
    data = write_corpus(tmp_path / "d.jsonl")
    bpe_dir = tmp_path / "bpe"
    train(tmp_path, "bpe", data, steps=3, tokenizer="bpe", tokenizer_path=bpe_dir, vocab_size=300, examples=60)
    assert (bpe_dir / "tokenizer.json").exists()
    config = json.loads((tmp_path / "bpe" / "out" / "config.json").read_text())
    assert 258 < config["vocab_size"] <= 300         # merges were learned on top of the byte alphabet
    assert (tmp_path / "bpe" / "out" / "tokenizer.json").exists()

    import data_pipeline as dp
    with pytest.raises(ValueError, match="at least 258"):
        dp.BPETokenizer.train(["abc"], 200)
    packed = tmp_path / "packed"
    dp.write_token_bin([data], dp.load_tokenizer(bpe_dir), packed)
    result = train(tmp_path, "packed", data, steps=3, token_data=packed)
    assert result["steps"] == 3
    reloaded = dp.load_tokenizer(tmp_path / "packed" / "out")
    assert isinstance(reloaded, dp.BPETokenizer)


def test_activation_checkpointing_does_not_change_training(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    train(tmp_path, "plain", data, steps=4)
    train(tmp_path, "ckpt", data, steps=4, activation_checkpointing=True)
    a, b = weights(tmp_path / "plain" / "out"), weights(tmp_path / "ckpt" / "out")
    for name, tensor in a.items():
        assert torch.allclose(tensor, b[name], atol=1e-6), name


def test_bf16_autocast_training_on_cpu_stays_finite(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    summary = train(tmp_path, "bf16", data, steps=4, precision="bf16")
    assert summary["precision"] == "bf16" and summary["final_loss"] == summary["final_loss"]
    with pytest.raises(ValueError, match="fp16"):
        train(tmp_path, "fp16", data, steps=2, precision="fp16")


def test_curriculum_requires_steps_and_unknown_options_are_rejected(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    with pytest.raises(ValueError, match="curriculum-steps"):
        train(tmp_path, "run", data, curriculum="rarity")
    with pytest.raises(TypeError, match="unknown training option"):
        trainer.default_args(not_an_option=1)


# ---- hardware ---------------------------------------------------------------
def test_detect_and_precision_choice():
    info = hardware.detect("cpu")
    assert info.backend == "cpu" and info.count == 1 and info.total_memory_bytes > 0
    assert hardware.choose_precision(info) == "fp32" and hardware.choose_precision(info, "bf16") == "bf16"
    with pytest.raises(ValueError):
        hardware.choose_precision(info, "fp16")
    gpu = hardware.DeviceInfo("cuda", "old gpu", 1, 8 << 30, 8 << 30, bf16=False, fp16=True, runtime="cuda 12")
    assert hardware.choose_precision(gpu) == "fp16"
    with pytest.raises(ValueError, match="bf16"):
        hardware.choose_precision(gpu, "bf16")
    modern = hardware.DeviceInfo("cuda", "new gpu", 1, 8 << 30, 8 << 30, bf16=True, fp16=True, runtime="cuda 12")
    assert hardware.choose_precision(modern) == "bf16"
    with pytest.raises(RuntimeError):
        hardware.detect("xpu") if not (hasattr(torch, "xpu") and torch.xpu.is_available()) else (_ for _ in ()).throw(RuntimeError())


def test_parameter_counts_match_a_real_model():
    config = QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=2, ffn_size=24, num_experts=5, top_k=2,
                               attention_heads=2, max_sequence_length=9)
    counts = hardware.parameter_counts(config)
    model = QuantaWeaveMoEForCausalLM(config)
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["experts"] == 2 * 5 * 3 * 16 * 24 and counts["active"] == counts["total"] - 2 * 3 * counts["per_expert"]


def test_safe_batch_size_probe_and_recommendation_on_cpu():
    config = QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=1, ffn_size=16, num_experts=4, top_k=1,
                               attention_heads=2, max_sequence_length=9)
    model = QuantaWeaveMoEForCausalLM(config)
    found = hardware.find_safe_batch_size(model, torch.device("cpu"), 8, lambda: trainer.autocast_context(torch.device("cpu")), limit=8)
    assert found["batch_size"] == 8 and found["tokens_per_second"] > 0
    report = hardware.recommend(hardware.detect("cpu"), config, 8, probe=True)
    assert report["precision"] == "fp32" and report["batch_size"] >= 1 and report["fits_training_state"]

    class Exploding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config

        def forward(self, ids, labels=None):
            raise torch.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="does not fit"):
        hardware.find_safe_batch_size(Exploding(), torch.device("cpu"), 8, lambda: trainer.autocast_context(torch.device("cpu")))


def test_suggested_expert_count_grows_with_memory_and_respects_cap():
    small = hardware.DeviceInfo("cuda", "s", 1, 200 << 20, 200 << 20, True, True, "cuda")
    large = hardware.DeviceInfo("cuda", "l", 1, 2000 << 20, 2000 << 20, True, True, "cuda")
    few = hardware.suggest_num_experts(small, 64, 128, 4, 2, 4096, 128)
    many = hardware.suggest_num_experts(large, 64, 128, 4, 2, 4096, 128)
    assert 2 <= few < many <= 256


def test_auto_batch_size_flag_picks_a_probed_size(tmp_path):
    data = write_corpus(tmp_path / "d.jsonl")
    args = trainer.default_args(
        data=[data], steps=2, sequence_length=8, examples=60, hidden_size=16, layers=1, ffn_size=16, total_experts=2,
        active_experts=1, vocab_size=64, device="cpu", checkpoint_interval=0, auto_batch_size=True,
        output=tmp_path / "o", checkpoint_dir=tmp_path / "c",
    )
    trainer.run_training(args)
    assert args.batch_size >= 2
