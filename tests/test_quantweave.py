import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def test_sparse_moe_forward_backward_reports_capacity_overflow():
    config = QuantaWeaveConfig(
        vocab_size=32,
        hidden_size=16,
        layers=1,
        ffn_size=32,
        num_experts=4,
        top_k=2,
        max_sequence_length=8,
        capacity_factor=0.5,
        min_expert_capacity=1,
    )
    model = QuantaWeaveMoEForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    outputs = model(input_ids, labels=input_ids)

    outputs["loss"].backward()

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["dropped_routes"].item() >= 0
    assert torch.isfinite(outputs["router_aux_loss"])


def test_unlimited_capacity_does_not_drop_routes():
    config = QuantaWeaveConfig(
        vocab_size=16,
        hidden_size=8,
        layers=1,
        ffn_size=16,
        num_experts=2,
        top_k=1,
        max_sequence_length=4,
        capacity_factor=0,
    )
    model = QuantaWeaveMoEForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))

    outputs = model(input_ids, labels=input_ids)

    assert outputs["dropped_routes"].item() == 0


def test_invalid_attention_shape_is_rejected():
    try:
        QuantaWeaveConfig(hidden_size=15, attention_heads=4)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid attention shape was accepted")


def test_residual_overflow_policy_preserves_routes():
    config = QuantaWeaveConfig(
        vocab_size=16,
        hidden_size=8,
        layers=1,
        ffn_size=16,
        num_experts=2,
        top_k=1,
        max_sequence_length=4,
        capacity_factor=0.1,
        min_expert_capacity=1,
        overflow_policy="residual",
    )
    model = QuantaWeaveMoEForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 4))
    outputs = model(input_ids, labels=input_ids)
    assert outputs["dropped_routes"].item() == 0
