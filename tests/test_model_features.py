import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM, TopKMoE


def small_config(**overrides) -> QuantaWeaveConfig:
    values = dict(
        vocab_size=32, hidden_size=16, layers=2, ffn_size=32, num_experts=6, top_k=2,
        attention_heads=2, max_sequence_length=16, capacity_factor=0, min_expert_capacity=1,
    )
    values.update(overrides)
    return QuantaWeaveConfig(**values)


def reference_moe(moe: TopKMoE, hidden: torch.Tensor, capacity: int) -> torch.Tensor:
    """Straightforward per-expert loop the vectorised dispatch must agree with."""
    flat = hidden.reshape(-1, hidden.size(-1))
    probs = (moe.router(flat) / moe.router_temperature).softmax(-1)
    weights, indices = probs.topk(moe.top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    output = torch.zeros_like(flat)
    for e, expert in enumerate(moe.experts):
        token, rank = (indices == e).nonzero(as_tuple=True)
        if token.numel() == 0:
            continue
        w = weights[token, rank]
        keep = w.argsort(descending=True)[:capacity]
        output[token[keep]] += expert(flat[token[keep]]) * w[keep, None]
    return output.reshape(hidden.shape)


@pytest.mark.parametrize("capacity_factor", [0, 0.6])
def test_sorted_dispatch_matches_reference_loop(capacity_factor):
    torch.manual_seed(0)
    config = small_config(capacity_factor=capacity_factor, min_expert_capacity=2)
    moe = TopKMoE(config)
    hidden = torch.randn(3, 8, 16)
    output, *_ = moe(hidden)
    expected = reference_moe(moe, hidden, moe.capacity(24))
    assert torch.allclose(output, expected, atol=1e-6)


def test_static_dispatch_matches_dynamic_when_capacity_is_off():
    torch.manual_seed(1)
    moe = TopKMoE(small_config())
    hidden = torch.randn(2, 8, 16)
    dynamic, *_ = moe(hidden)
    moe.static_dispatch = True
    static, *_ = moe(hidden)
    assert torch.allclose(dynamic, static, atol=1e-6)


def test_router_temperature_flattens_routing_distribution():
    torch.manual_seed(2)
    moe = TopKMoE(small_config())
    moe.collect_stats = True
    hidden = torch.randn(2, 8, 16) * 3
    moe(hidden)
    sharp = moe.last_stats["entropy_mean"].item()
    moe.router_temperature = 4.0
    moe(hidden)
    assert moe.last_stats["entropy_mean"].item() > sharp


def test_set_routing_controls_updates_modules_and_saved_config():
    model = QuantaWeaveMoEForCausalLM(small_config())
    model.set_routing_controls(
        capacity_factor=2.0, overflow_policy="residual", router_temperature=1.5, router_aux_loss_coef=0.5,
        drop_overflow_tokens=False,
    )
    for moe in model.moes():
        assert (moe.capacity_factor, moe.overflow_policy, moe.router_temperature) == (2.0, "residual", 1.5)
        assert moe.drop_overflow_tokens is False
    assert model.config.capacity_factor == 2.0 and model.config.router_temperature == 1.5
    assert model.router_aux_loss_coef == 0.5 and model.config.router_aux_loss_coef == 0.5
    with pytest.raises(ValueError):
        model.set_routing_controls(overflow_policy="bogus")
    with pytest.raises(ValueError):
        model.set_routing_controls(router_temperature=0)


def test_stats_are_collected_only_on_request_and_describe_the_batch():
    config = small_config(capacity_factor=0.5, min_expert_capacity=1)
    model = QuantaWeaveMoEForCausalLM(config)
    ids = torch.randint(0, 32, (2, 8))
    model(ids)
    assert all(not stats for stats in model.routing_stats())

    model.set_collect_stats(True)
    out = model(ids)
    for stats in model.routing_stats():
        assert stats["expert_assigned"].sum().item() == 2 * 8 * config.top_k
        assert stats["expert_load"].sum().item() + stats["dropped_tokens"].numel() == 2 * 8 * config.top_k
        assert stats["confidence_hist"].sum().item() == 16
        assert 0 <= stats["entropy_mean"].item() <= torch.log(torch.tensor(6.0)).item() + 1e-5
    assert out["overflow_routes"].item() == sum(s["dropped_tokens"].numel() for s in model.routing_stats())


def test_activation_checkpointing_gives_identical_gradients():
    torch.manual_seed(3)
    model = QuantaWeaveMoEForCausalLM(small_config())
    ids = torch.randint(0, 32, (2, 8))
    model(ids, labels=ids)["loss"].backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model.activation_checkpointing = True
    model.train()
    model(ids, labels=ids)["loss"].backward()
    for name, grad in reference.items():
        assert torch.allclose(grad, dict(model.named_parameters())[name].grad, atol=1e-6), name


def test_domain_specialization_loss_measures_routing_overlap():
    torch.manual_seed(4)
    moe = TopKMoE(small_config())
    probs = torch.tensor([[1.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0]])
    same = moe._domain_loss(probs, torch.tensor([0, 0, 1, 1]), 2)      # domains use different experts
    overlap = moe._domain_loss(probs, torch.tensor([0, 1, 0, 1]), 2)   # both domains use both experts
    assert same.item() == pytest.approx(0.0) and overlap.item() == pytest.approx(1.0)
    single = moe._domain_loss(probs, torch.tensor([0, 0, 0, 0]), 2)
    assert single.item() == 0.0


def test_domain_loss_flows_into_training_loss_and_gradients():
    model = QuantaWeaveMoEForCausalLM(small_config(capacity_factor=0))
    ids = torch.randint(0, 32, (4, 8))
    domains = torch.tensor([0, 0, 1, 1])
    base = model(ids, labels=ids, domain_ids=domains, num_domains=2)["loss"].item()
    model.domain_specialization_coef = 5.0
    out = model(ids, labels=ids, domain_ids=domains, num_domains=2)
    assert out["domain_loss"].item() > 0
    assert out["loss"].item() == pytest.approx(base + 5.0 * out["domain_loss"].item(), rel=1e-4)
    out["loss"].backward()
    assert model.blocks[0].moe.router.weight.grad.abs().sum() > 0


def test_expert_profiling_accumulates_time_per_expert():
    moe = TopKMoE(small_config())
    moe.profile_experts = True
    moe(torch.randn(2, 8, 16))
    assert sum(moe.expert_seconds) > 0


def test_router_runs_in_fp32_under_autocast():
    moe = TopKMoE(small_config())
    hidden = torch.randn(1, 4, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, balance, *_ = moe(hidden)
    assert balance.dtype == torch.float32 and out.isfinite().all()
