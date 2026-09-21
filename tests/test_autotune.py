import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import hardware as hw
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def tiny_model():
    return QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=1, ffn_size=16, num_experts=4, top_k=1,
                                                      attention_heads=2, max_sequence_length=9, capacity_factor=0))


CPU = torch.device("cpu")
plain = lambda: trainer.autocast_context(CPU)  # noqa: E731


def fake_device(monkeypatch, total_gb: float, peak_gb, tps=None):
    """Pretend the device has ``total_gb`` memory; a step of batch b peaks at peak_gb(b) and runs at tps(b) tokens/s."""
    monkeypatch.setattr(hw, "detect", lambda name="auto": hw.DeviceInfo("cpu", "fake", 1, int(total_gb * 1e9), int(total_gb * 1e9), True, False, "test"))
    state = {"batch": 0}
    real_randint = torch.randint

    def spy_randint(*args, **kwargs):
        if len(args) >= 3 and isinstance(args[2], tuple):
            state["batch"] = args[2][0]
        return real_randint(*args, **kwargs)

    monkeypatch.setattr(hw.torch, "randint", spy_randint)
    monkeypatch.setattr(hw, "_peak_bytes", lambda device: int(peak_gb(state["batch"]) * 1e9))
    if tps is not None:
        clock = {"t": 0.0, "calls": 0}
        def perf_counter():
            # the probe reads the clock twice per step (start and stop); fake the elapsed time of the current batch
            clock["calls"] += 1
            if clock["calls"] % 2 == 0:
                clock["t"] += state["batch"] * 8 / tps(state["batch"])
            return clock["t"]
        monkeypatch.setattr(hw.time, "perf_counter", perf_counter)


def test_probe_stops_at_the_headroom_limit_and_reports_the_largest_safe_size(monkeypatch):
    fake_device(monkeypatch, total_gb=10, peak_gb=lambda b: 0.2 * b)               # 85% of 10 GB = 8.5 GB -> batch 32 (6.4) ok, 64 (12.8) not
    result = hw.find_safe_batch_size(tiny_model(), CPU, 8, plain, limit=256, target=0)
    assert result["batch_size"] == 32 and result["recommended_batch_size"] == 32
    assert [p["batch_size"] for p in result["curve"]] == [1, 2, 4, 8, 16, 32]


def test_probe_recommends_the_knee_of_the_throughput_curve_not_the_largest_batch(monkeypatch):
    # throughput saturates at 16 and then falls: the largest safe batch (64) is slower than the knee
    speed = {1: 1e3, 2: 2e3, 4: 4e3, 8: 8e3, 16: 16e3, 32: 16.2e3, 64: 15e3}
    fake_device(monkeypatch, total_gb=100, peak_gb=lambda b: 0.1 * b, tps=lambda b: speed.get(b, 1e3))
    result = hw.find_safe_batch_size(tiny_model(), CPU, 8, plain, limit=64, target=0.95)
    assert result["batch_size"] == 64                                              # everything up to 64 fits
    assert result["recommended_batch_size"] == 16                                  # smallest batch within 95% of the best speed
    assert result["tokens_per_second"] == pytest.approx(16e3)
    assert hw.find_safe_batch_size(tiny_model(), CPU, 8, plain, limit=64, target=0)["recommended_batch_size"] == 64


def test_probe_reserves_optimizer_state_for_every_trainable_parameter_and_leaves_weights_alone():
    model = tiny_model()
    trainable = [p for p in model.parameters() if p.requires_grad]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    was_training = model.training
    allocated = []
    original = torch.zeros_like

    def counting_zeros_like(tensor, *args, **kwargs):
        allocated.append((tuple(tensor.shape), kwargs.get("dtype")))
        return original(tensor, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hw.torch, "zeros_like", counting_zeros_like)
        hw.find_safe_batch_size(model, CPU, 8, plain, limit=2)
    from collections import Counter
    copies = Counter(shape for shape, dtype in allocated if dtype == torch.float32)
    needed = Counter(tuple(p.shape) for p in trainable)
    for shape, count in needed.items():
        assert copies[shape] >= 2 * count, f"no room reserved for the optimizer moments of {shape}"      # two AdamW moments per parameter
    for key, value in model.state_dict().items():
        assert torch.equal(before[key], value), f"{key} was modified by the probe"
    assert all(p.grad is None for p in model.parameters()) and model.training is was_training


def test_probe_treats_out_of_memory_as_the_end_of_the_search_and_other_errors_as_errors(monkeypatch):
    fake_device(monkeypatch, total_gb=1000, peak_gb=lambda b: 0.0)
    model = tiny_model()
    original_forward = model.forward

    def forward(ids, labels=None):
        if ids.size(0) >= 8:
            raise torch.OutOfMemoryError("CUDA out of memory")
        return original_forward(ids, labels=labels)

    model.forward = forward
    assert hw.find_safe_batch_size(model, CPU, 8, plain, limit=64)["batch_size"] == 4

    def broken(ids, labels=None):
        raise RuntimeError("shape mismatch")

    model.forward = broken
    with pytest.raises(RuntimeError, match="shape mismatch"):
        hw.find_safe_batch_size(model, CPU, 8, plain)


def test_recommendation_uses_activation_checkpointing_only_when_it_is_measurably_faster(monkeypatch):
    config = tiny_model().config
    info = hw.DeviceInfo("cpu", "fake", 1, 8 << 30, 8 << 30, True, False, "test")

    def run(without_tps, with_tps):
        def fake(model, device, sequence_length, autocast, **kwargs):
            tps = with_tps if model.activation_checkpointing else without_tps
            batch = 64 if model.activation_checkpointing else 8
            return {"batch_size": batch, "recommended_batch_size": batch, "tokens_per_second": tps, "curve": []}
        monkeypatch.setattr(hw, "find_safe_batch_size", fake)
        return hw.recommend(info, config, 8, probe=True)

    slower = run(100.0, 90.0)
    assert slower["activation_checkpointing_recommended"] is False and slower["batch_size"] == 8
    barely = run(100.0, 103.0)
    assert barely["activation_checkpointing_recommended"] is False              # within 5%: not worth the recompute
    faster = run(100.0, 140.0)
    assert faster["activation_checkpointing_recommended"] is True and faster["batch_size"] == 64


def test_recommendation_survives_a_setting_that_does_not_fit(monkeypatch):
    info = hw.DeviceInfo("cpu", "fake", 1, 8 << 30, 8 << 30, True, False, "test")

    def fake(model, device, sequence_length, autocast, **kwargs):
        if not model.activation_checkpointing:
            raise RuntimeError("even batch size 1 does not fit")
        return {"batch_size": 2, "recommended_batch_size": 2, "tokens_per_second": 5.0, "curve": []}

    monkeypatch.setattr(hw, "find_safe_batch_size", fake)
    result = hw.recommend(info, tiny_model().config, 8, probe=True)
    assert result["activation_checkpointing_recommended"] is True and result["batch_size"] == 2
    monkeypatch.setattr(hw, "find_safe_batch_size", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no")))
    with pytest.raises(RuntimeError, match="does not fit"):
        hw.recommend(info, tiny_model().config, 8, probe=True)


def test_trainer_applies_the_recommended_batch_size_on_cpu(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("".join('{"text": "story %d: the quick brown fox jumps over the lazy dog, again."}\n' % i for i in range(40)))
    args = trainer.default_args(data=[data], steps=2, sequence_length=8, examples=40, hidden_size=16, layers=1, ffn_size=16, total_experts=2,
                                active_experts=1, vocab_size=64, device="cpu", checkpoint_interval=0, auto_batch_size=True, auto_batch_target=0,
                                output=tmp_path / "o", checkpoint_dir=tmp_path / "c")
    trainer.run_training(args)
    assert args.batch_size >= 8                                                      # target 0 = the largest that fits (CPU: up to the limit)


# ---- the regression: the batch sizes the probe approves must survive real training on the actual device ----------
needs_gpu = pytest.mark.skipif(not torch.cuda.is_available() or (torch.cuda.is_available() and torch.cuda.mem_get_info()[0] < 6e9),
                               reason="needs a CUDA GPU with 6 GB free")


@needs_gpu
@pytest.mark.parametrize("target", [0.95, 0.0])
def test_autotuned_batch_sizes_train_on_the_gpu_without_running_out_of_memory(tmp_path, target):
    """The probe once ignored optimizer state and approved a batch that ran out of memory at the first optimizer step.
    The model is sized so the largest batch that fits is close to the device limit."""
    total = torch.cuda.mem_get_info()[1]
    experts = 32 if total < 12e9 else 64
    data = tmp_path / "d.jsonl"
    data.write_text("".join('{"text": "story %d: the quick brown fox jumps over the lazy dog, again and again and again and again."}\n' % i for i in range(3000)))
    args = trainer.default_args(data=[data], steps=4, sequence_length=256, examples=3000, hidden_size=256, layers=6, ffn_size=512, total_experts=experts,
                                active_experts=2, vocab_size=7168, device="cuda", checkpoint_interval=0, capacity_factor=0, auto_batch_size=True,
                                auto_batch_target=target, output=tmp_path / "o", checkpoint_dir=tmp_path / "c")
    summary = trainer.run_training(args)                                             # raises torch.OutOfMemoryError if the choice was unsafe
    assert summary["steps"] == 4 and torch.isfinite(torch.tensor(summary["final_loss"]))
    assert args.batch_size >= 1
    assert torch.cuda.max_memory_allocated() < 0.98 * total
