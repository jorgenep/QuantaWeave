import json
import sys
import xml.dom.minidom
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import moe_schedules as ms
import routing_diagnostics as rd
import training_metrics as tm
from data_pipeline import CharTokenizer, token_classes
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def controller(**overrides) -> ms.TrainingController:
    return ms.TrainingController(ms.ScheduleConfig(**overrides))


def test_lr_warmup_then_cosine_decay_to_floor():
    c = controller(lr=1.0, warmup_steps=10, total_steps=110, lr_decay="cosine", min_lr_ratio=0.1)
    assert c.lr_at(0) == pytest.approx(0.1) and c.lr_at(9) == pytest.approx(1.0)
    assert c.lr_at(10) == pytest.approx(1.0)
    assert c.lr_at(60) == pytest.approx(0.55)              # halfway through the cosine
    assert c.lr_at(110) == pytest.approx(0.1) and c.lr_at(999) == pytest.approx(0.1)
    linear = controller(lr=1.0, total_steps=100, lr_decay="linear", min_lr_ratio=0.0)
    assert linear.lr_at(50) == pytest.approx(0.5)
    assert controller(lr=2.0).lr_at(12345) == 2.0


def test_plateau_decays_lr_and_respects_floor():
    c = controller(lr=1.0, plateau_patience=2, plateau_factor=0.5, min_lr_scale=0.3, interval=1)
    for step in range(1, 12):
        c.observe(loss=1.0, dropped_fraction=0.0)   # never improves
        c.update(step)
    assert c.lr_scale == pytest.approx(0.3)
    improving = controller(plateau_patience=2, interval=1)
    for step, loss in enumerate([3, 2, 1, 0.5, 0.2], start=1):
        improving.observe(loss, 0.0)
        improving.update(step)
    assert improving.lr_scale == 1.0


def test_capacity_grows_under_drops_and_relaxes_without():
    c = controller(capacity_factor=1.0, capacity_adapt=True, capacity_max=1.3, capacity_min=1.0, drop_threshold=0.01)
    for step in range(1, 6):
        c.observe(1.0, 0.2)
        c.update(step)
    assert c.capacity_factor == pytest.approx(1.3)          # capped
    for step in range(6, 60):
        c.observe(1.0, 0.0)
        c.update(step)
    assert c.capacity_factor == pytest.approx(1.0)          # relaxed back to the floor


def test_capacity_release_disables_enforcement_from_step():
    c = controller(capacity_release_step=100)
    c.update(50)
    assert not c.released
    changes = c.update(100)
    assert c.released and changes["capacity_released"] == 1.0


def test_aux_weight_tracks_imbalance_and_anneals():
    c = controller(aux_coef=0.01, aux_adapt=True, balance_low=0.2, balance_high=1.0, aux_factor_max=1.2)
    for step in range(1, 10):
        c.update(step, imbalance=2.0)
    assert c.aux_factor == pytest.approx(1.2)
    for step in range(10, 100):
        c.update(step, imbalance=0.0)
    assert c.aux_factor == pytest.approx(0.25)
    c.update(100, imbalance=0.5)                            # inside the band: unchanged
    assert c.aux_factor == pytest.approx(0.25)
    annealed = controller(aux_coef=0.1, aux_coef_end=0.0, total_steps=100)
    assert annealed.aux_coef_at(50) == pytest.approx(0.05) and annealed.aux_coef_at(100) == 0.0


def test_temperature_schedule_and_adaptive_boost():
    c = controller(temperature_start=3.0, temperature_end=1.0, temperature_steps=100, temperature_adapt=True)
    assert c.temperature_at(0) == 3.0 and c.temperature_at(50) == pytest.approx(2.0) and c.temperature_at(200) == 1.0
    c.update(1, active_fraction=0.1)                        # experts dying: explore more
    assert c.temperature_boost > 1.0 and c.temperature_at(200) > 1.0
    for step in range(2, 200):
        c.update(step, active_fraction=1.0)
    assert c.temperature_boost == 1.0


def test_controller_state_roundtrip_and_apply_to_model():
    c = controller(lr=0.5, capacity_factor=1.5, capacity_adapt=True, capacity_max=3.0, interval=1,
                   capacity_release_step=7, aux_coef=0.02)
    c.observe(1.0, 0.5)
    c.update(1)
    model = QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(vocab_size=16, hidden_size=8, layers=1, ffn_size=8, num_experts=4, top_k=1, attention_heads=2, max_sequence_length=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    values = c.apply(model, optimizer, step=3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5) == values["lr"]
    assert model.moes()[0].capacity_factor == pytest.approx(c.capacity_factor)
    assert c.capacity_factor > 1.5
    assert model.router_aux_loss_coef == pytest.approx(0.02)

    restored = controller(lr=0.5, capacity_factor=1.5, capacity_adapt=True, capacity_max=3.0)
    restored.load_state_dict(json.loads(json.dumps(c.state_dict())))     # must survive JSON/torch.save
    assert restored.capacity_factor == c.capacity_factor and restored.history == c.history
    with pytest.raises(ValueError):
        ms.ScheduleConfig(lr_decay="bogus")
    with pytest.raises(ValueError):
        ms.ScheduleConfig(capacity_min=5, capacity_max=1)


def test_plateau_and_stability_detection():
    falling = [5.0 - 0.1 * i for i in range(40)]
    flat = [1.0] * 40
    assert not tm.detect_plateau(falling) and tm.detect_plateau(flat) and not tm.detect_plateau(flat[:10])
    steps = list(range(1, 101))
    losses = [3.0 * 0.9 ** i for i in range(40)] + [0.05] * 60
    stable = tm.first_stable_step(steps, losses)
    assert stable is not None and 20 <= stable <= 60
    assert tm.first_stable_step(steps[:30], [3.0 - 0.1 * i for i in range(30)]) is None


def test_metrics_logger_writes_jsonl_and_summarises(tmp_path):
    logger = tm.MetricsLogger(tmp_path / "logs" / "metrics.jsonl", total_parameters=100, active_parameters=25)
    for step in range(1, 5):
        logger.log(step, tokens=128, loss=2.0 / step, lr=0.1)
    lines = [json.loads(line) for line in (tmp_path / "logs" / "metrics.jsonl").read_text().splitlines()]
    assert [line["step"] for line in lines] == [1, 2, 3, 4] and lines[0]["tokens_per_second"] > 0
    summary = logger.summary()
    assert summary["active_parameter_ratio"] == 0.25 and summary["best_loss"] == 0.5


def monitored_model(capacity_factor=0.5):
    torch.manual_seed(0)
    config = QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=2, ffn_size=16, num_experts=4, top_k=2,
                               attention_heads=2, max_sequence_length=16, capacity_factor=capacity_factor, min_expert_capacity=1)
    return QuantaWeaveMoEForCausalLM(config), config


def test_monitor_accumulates_stats_specialization_and_domains(tmp_path):
    tokenizer = CharTokenizer.build(["abc 123 xyz"], 32)
    model, config = monitored_model()
    monitor = rd.RoutingMonitor(2, 4, 2, domains=["x", "y"], token_classes=token_classes(tokenizer), out_dir=tmp_path)
    ids = torch.randint(0, tokenizer.vocab_size, (4, 8))
    domains = torch.tensor([0, 0, 1, 1])
    model.set_collect_stats(True)
    for _ in range(3):
        model(ids)
        monitor.update(model, ids, domains)

    tokens = 4 * 8
    assert monitor.window_assigned.sum(dim=1).tolist() == [3 * tokens * 2] * 2
    assert monitor.class_usage.sum().item() == 2 * 3 * tokens * 2       # every route counted per class
    assert monitor.domain_usage[0].sum(dim=1).tolist() == [3 * 16 * 2, 3 * 16 * 2]
    assert monitor.imbalance() >= 0 and 0 <= monitor.active_fraction() <= 1

    record = monitor.snapshot(step=10)
    assert record["step"] == 10 and len(record["layers"]) == 2
    layer = record["layers"][0]
    assert sum(layer["utilization"]) == pytest.approx(1.0) and 0 <= layer["drop_rate"] < 1
    assert layer["entropy"] <= layer["entropy_max"] + 1e-6
    assert (tmp_path / "routing_log.jsonl").exists()
    assert monitor.window_updates == 0                                    # window reset after snapshot
    if layer["drop_rate"] > 0:
        assert record["drops_by_token_class"]


def test_monitor_requires_collected_stats():
    model, _ = monitored_model()
    monitor = rd.RoutingMonitor(2, 4, 2)
    ids = torch.randint(0, 32, (2, 8))
    model(ids)
    with pytest.raises(RuntimeError, match="set_collect_stats"):
        monitor.update(model, ids)


def test_report_writes_valid_svg_and_index(tmp_path):
    tokenizer = CharTokenizer.build(["abc 123 xyz"], 32)
    model, _ = monitored_model()
    monitor = rd.RoutingMonitor(2, 4, 2, domains=["x", "y"], token_classes=token_classes(tokenizer), out_dir=tmp_path)
    ids = torch.randint(0, tokenizer.vocab_size, (4, 8))
    model.set_collect_stats(True)
    for step in (10, 20):
        model(ids)
        monitor.update(model, ids, torch.tensor([0, 0, 1, 1]))
        monitor.snapshot(step)
    index = monitor.write_report()
    names = {p.name for p in tmp_path.iterdir()}
    for expected in ("expert_usage_by_layer.svg", "specialization_layer0.svg", "domain_usage_layer1.svg",
                     "drop_rate_over_time.svg", "router_entropy_over_time.svg", "utilization_layer0_over_time.svg", "index.html"):
        assert expected in names, expected
    for svg in tmp_path.glob("*.svg"):
        xml.dom.minidom.parseString(svg.read_text())                      # well-formed XML
    assert "expert_usage_by_layer.svg" in index.read_text()
    with pytest.raises(ValueError):
        rd.RoutingMonitor(1, 2, 1).write_report()
