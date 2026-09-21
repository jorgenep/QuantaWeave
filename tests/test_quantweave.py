import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import benchmark_quantweave_moe
import distill_quantweave_moe
import model_scaling
import train_quantweave_moe
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM, TopKMoE


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
    # capacity is still enforced: overflow is skipped, just not reported as dropped
    assert outputs["overflow_routes"].item() > 0


class Recorder(nn.Module):
    """Stands in for an expert: passes tokens through and remembers which it saw."""

    def __init__(self) -> None:
        super().__init__()
        self.seen = torch.empty(0)

    def forward(self, hidden):
        self.seen = hidden[:, 0].clone()
        return hidden


def make_two_expert_moe(**overrides) -> tuple[TopKMoE, list[Recorder]]:
    config = QuantaWeaveConfig(
        vocab_size=8, hidden_size=2, layers=1, ffn_size=2, num_experts=2, top_k=2,
        attention_heads=1, max_sequence_length=4, capacity_factor=0.5, min_expert_capacity=1,
        **overrides,
    )
    moe = TopKMoE(config)
    recorders = [Recorder(), Recorder()]
    moe.experts = nn.ModuleList(recorders)
    # expert 0 prefers tokens with a large first feature, expert 1 the opposite
    moe.router.weight.data = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    return moe, recorders


def test_capacity_keeps_highest_weight_routes_not_first_tokens():
    moe, (expert_0, expert_1) = make_two_expert_moe()
    features = torch.tensor([0.5, 3.0, -2.0, 1.5])
    hidden = torch.stack([features, torch.zeros(4)], dim=-1)[None]

    _, _, dropped, overflow, _ = moe(hidden)

    # top_k == num_experts, so each expert is routed all 4 tokens but holds only 2
    assert overflow.item() == 4 and dropped.item() == 4
    assert sorted(expert_0.seen.tolist()) == [1.5, 3.0]
    assert sorted(expert_1.seen.tolist()) == [-2.0, 0.5]


def test_residual_policy_enforces_capacity_without_reporting_drops():
    moe, (expert_0, expert_1) = make_two_expert_moe(overflow_policy="residual")
    hidden = torch.stack([torch.tensor([0.5, 3.0, -2.0, 1.5]), torch.zeros(4)], dim=-1)[None]

    _, _, dropped, overflow, _ = moe(hidden)

    assert dropped.item() == 0 and overflow.item() == 4
    assert expert_0.seen.numel() == 2 and expert_1.seen.numel() == 2


def test_drop_overflow_tokens_false_disables_capacity():
    moe, (expert_0, expert_1) = make_two_expert_moe(drop_overflow_tokens=False)
    hidden = torch.stack([torch.tensor([0.5, 3.0, -2.0, 1.5]), torch.zeros(4)], dim=-1)[None]

    _, _, dropped, overflow, _ = moe(hidden)

    assert dropped.item() == 0 and overflow.item() == 0
    assert expert_0.seen.numel() == 4 and expert_1.seen.numel() == 4


def test_skipped_route_contributes_exactly_zero():
    config = QuantaWeaveConfig(
        vocab_size=8, hidden_size=2, layers=1, ffn_size=4, num_experts=2, top_k=1,
        attention_heads=1, max_sequence_length=4, capacity_factor=0.5, min_expert_capacity=1,
    )
    moe = TopKMoE(config)
    # every token routes to expert 0, which has room for exactly one of the 4 tokens
    moe.router.weight.data = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    hidden = torch.tensor([[[5.0, 1.0], [4.0, 1.0], [3.0, 1.0], [2.0, 1.0]]])

    output, _, _, overflow, _ = moe(hidden)

    # skipped tokens get no MoE output, so the block's residual add leaves them unchanged
    assert overflow.item() == 3
    assert (output[0].abs().sum(-1) > 0).sum().item() == 1


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_character_dataset_packs_documents_instead_of_truncating(tmp_path):
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [{"text": "aaaa"}, {"text": "bb"}, {"text": "ab"}])
    vocab = {"<unk>": 0, "<eos>": 1, "a": 2, "b": 3}

    dataset = train_quantweave_moe.CharacterDataset(data, vocab, sequence_length=3, limit=None)

    # stream: aaaa <eos> bb <eos> ab <eos> = 11 tokens -> two windows of 4, tail dropped
    assert len(dataset) == 2
    assert dataset[0].tolist() == [2, 2, 2, 2]
    assert dataset[1].tolist() == [1, 3, 3, 1]
    with pytest.raises(IndexError):
        dataset[2]


def test_character_dataset_too_short_is_empty(tmp_path):
    data = tmp_path / "data.jsonl"
    write_jsonl(data, [{"text": "ab"}])
    dataset = train_quantweave_moe.CharacterDataset(
        data, {"<unk>": 0, "<eos>": 1, "a": 2, "b": 3}, sequence_length=8, limit=None
    )
    assert len(dataset) == 0 and not dataset


def tiny_model() -> QuantaWeaveMoEForCausalLM:
    return QuantaWeaveMoEForCausalLM(
        QuantaWeaveConfig(
            vocab_size=16, hidden_size=8, layers=1, ffn_size=16, num_experts=2, top_k=1,
            max_sequence_length=4,
        )
    )


def test_checkpoint_roundtrip_and_interrupted_save_fallback(tmp_path):
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint_dir = tmp_path / "ckpt"
    train_quantweave_moe.save_checkpoint(checkpoint_dir, model, optimizer, 7, {"<unk>": 0}, torch.device("cpu"))
    assert train_quantweave_moe.checkpoint_exists(checkpoint_dir)

    # a save interrupted between "rename current to .previous" and "rename tmp to current"
    checkpoint_dir.rename(tmp_path / ".ckpt.previous")
    assert train_quantweave_moe.checkpoint_exists(checkpoint_dir)

    restored = tiny_model()
    step = train_quantweave_moe.load_checkpoint(
        checkpoint_dir, restored, torch.optim.AdamW(restored.parameters()), torch.device("cpu")
    )
    assert step == 7
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])

    assert not train_quantweave_moe.checkpoint_exists(tmp_path / "missing")


@pytest.fixture(scope="module")
def trained_student(tmp_path_factory):
    root = tmp_path_factory.mktemp("student")
    data = root / "data.jsonl"
    write_jsonl(data, [{"text": f"story {i}: the quick brown fox jumps over the lazy dog."} for i in range(40)])
    output = root / "out"
    argv = [
        "train", "--data", str(data), "--output", str(output), "--steps", "3", "--batch-size", "2",
        "--sequence-length", "8", "--examples", "40", "--hidden-size", "16", "--layers", "1",
        "--ffn-size", "32", "--total-experts", "4", "--active-experts", "2", "--vocab-size", "64",
        "--device", "cpu", "--checkpoint-dir", str(root / "ckpt"), "--checkpoint-interval", "0",
        "--no-resume",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        train_quantweave_moe.main()
    finally:
        sys.argv = old_argv
    return root, data, output


def run_main(module, argv: list[str]) -> None:
    old_argv = sys.argv
    sys.argv = argv
    try:
        module.main()
    finally:
        sys.argv = old_argv


def test_distillation_starts_from_student_weights(trained_student, tmp_path):
    _, data, student = trained_student
    output = tmp_path / "distilled"
    # lr=0: one optimizer step must leave the student's weights untouched
    run_main(distill_quantweave_moe, [
        "distill", "--student", str(student), "--teacher-data", str(data), "--output", str(output),
        "--checkpoint-dir", str(tmp_path / "ckpt"), "--steps", "1", "--sequence-length", "8",
        "--batch-size", "2", "--lr", "0", "--device", "cpu", "--checkpoint-interval", "0",
    ])

    student_state = torch.load(student / "model.pt", weights_only=False)["model"]
    distilled_state = torch.load(output / "model.pt", weights_only=False)["model"]
    assert student_state.keys() == distilled_state.keys()
    for name, tensor in student_state.items():
        assert torch.equal(tensor, distilled_state[name]), name


def test_benchmark_reports_exact_parameter_counts(trained_student, tmp_path):
    _, data, student = trained_student
    report_path = tmp_path / "report.json"
    run_main(benchmark_quantweave_moe, [
        "benchmark", "--checkpoint", str(student), "--data", str(data), "--examples", "20",
        "--device", "cpu", "--output", str(report_path),
    ])

    report = json.loads(report_path.read_text())
    config = QuantaWeaveConfig(**json.loads((student / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    per_expert = sum(parameter.numel() for parameter in model.blocks[0].moe.experts[0].parameters())
    inactive = config.layers * (config.num_experts - config.top_k) * per_expert

    assert report["total_parameters"] == total
    assert report["active_parameters_per_token_estimate"] == total - inactive
    assert "overflow_routes" in report and report["training_step_in_checkpoint"] == 3


def test_preset_l_is_the_30b_a1b_target():
    spec = model_scaling.make_spec("moe", "l")
    assert 30e9 < spec.total_params < 31e9
    assert 0.95e9 < spec.active_params < 1.05e9


def test_pack_dataset_emits_only_whole_chunks_with_labels():
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")
    from datasets import Dataset

    import pack_dataset

    length = pack_dataset.SEQ_LENGTH
    ids = Dataset.from_dict({"input_ids": [[1] * (length + 5), [2] * (length - 5), [3] * 7]})
    packed = ids.map(pack_dataset.group_texts, batched=True, batch_size=3, remove_columns=["input_ids"])

    # 2*length + 7 tokens -> two whole chunks, 7-token tail dropped
    assert len(packed) == 2
    assert set(packed.column_names) == {"input_ids", "attention_mask", "labels"}
    assert all(len(row["input_ids"]) == length for row in packed)
    assert packed[0]["labels"] == packed[0]["input_ids"]
    assert set(packed[0]["attention_mask"]) == {1}

    short = Dataset.from_dict({"input_ids": [[1] * 10]})
    assert len(short.map(pack_dataset.group_texts, batched=True, remove_columns=["input_ids"])) == 0
