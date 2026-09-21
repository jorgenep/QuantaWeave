import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import cross_tokenizer as ct
import data_pipeline as dp
import distill_quantweave_moe as distill
import train_quantweave_moe as trainer

TEMPLATE = "story {i}: the quick brown fox jumps over the lazy dog, again and again."


def corpus(path: Path, rows: int = 60) -> Path:
    path.write_text("".join(json.dumps({"text": TEMPLATE.format(i=i % 10)}) + "\n" for i in range(rows)))
    return path


# ---- pure alignment and losses (no transformers needed) ---------------------------------------------
def test_alignment_pairs_positions_that_end_at_the_same_character():
    # text "ab cd": student chars end at 1,2,3,4,5; teacher tokens "ab"(2) " cd"(5)
    pairs = ct.align_positions([1, 1, 1, 1, 1], [2, 5])
    assert pairs == [(1, 0)]                          # only the boundary after "ab"; offset 5 is the end of text (no successor)
    assert ct.align_positions([1, 1, 1, 1, 1], [1, 2, 3, 4, 5]) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert ct.align_positions([3, 2], [5]) == []      # the only boundary is the end of the text
    assert ct.align_positions([2, 0, 2, 1], [2, 4, 5]) == [(0, 0), (2, 1)]      # zero-length pieces never align
    assert ct.align_positions([1, 1, 1], [1, 1, 3]) == [(0, 1)]                 # duplicate end offset: the later teacher token wins


def test_uld_is_permutation_invariant_zero_for_matching_shapes_and_differentiable():
    teacher = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]])
    permuted_student = torch.tensor([[0.1, 0.7, 0.2], [0.0, 0.5, 0.5]])
    assert ct.uld_loss(permuted_student, teacher).item() == pytest.approx(0.0, abs=1e-7)
    flat = torch.full((2, 3), 1 / 3)
    assert ct.uld_loss(flat, teacher).item() > 0.2
    wider = torch.tensor([[0.7, 0.2, 0.1, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0]])       # different vocabulary sizes
    assert ct.uld_loss(wider, teacher).item() == pytest.approx(0.0, abs=1e-7)
    logits = torch.randn(2, 5, requires_grad=True)
    ct.uld_loss(logits.softmax(-1), teacher).backward()
    assert logits.grad.abs().sum() > 0


class StubTeacher:
    """A teacher with a fixed 4-token vocabulary over the text 'ab ab', tokenised as 'ab',' ab'."""
    vocab_size = 4

    def token_strings(self):
        return ["ab", " ab", "b", " "]

    def next_token_probs(self, text):
        assert text == "ab ab"
        probs = torch.tensor([[0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
        return probs, [2, 5]


def test_marginal_loss_uses_the_teachers_exact_next_character_distribution():
    tokenizer = dp.CharTokenizer.build(["ab "], 16)
    teacher = StubTeacher()
    matrix = ct.first_char_matrix(teacher.token_strings(), tokenizer, torch.device("cpu"))
    v = tokenizer.vocab
    # after "ab": next char is 'a' w.p. 0.1 (token "ab"), ' ' w.p. 0.6+0.1 (" ab", " "), 'b' w.p. 0.2
    target = teacher.next_token_probs("ab ab")[0][0] @ matrix
    assert target[v["a"]].item() == pytest.approx(0.1) and target[v[" "]].item() == pytest.approx(0.7) and target[v["b"]].item() == pytest.approx(0.2)

    ids = torch.tensor([[v["a"], v["b"], v[" "], v["a"], v["b"], tokenizer.eos_id]])
    logits = torch.randn(1, 6, tokenizer.vocab_size, requires_grad=True)
    loss_fn = ct.CrossTokenizerLoss(teacher, tokenizer, "marginal")
    loss, stats = loss_fn(logits, ids)
    assert stats["aligned_positions"] == 1 and stats["considered_positions"] == 4 and stats["mode"] == "marginal"
    expected = torch.nn.functional.kl_div(torch.log_softmax(logits[0, 1], -1), target, reduction="sum")   # student position 1 = after "ab"
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    loss.backward()
    assert logits.grad[0, 1].abs().sum() > 0 and logits.grad[0, 0].abs().sum() == 0      # only aligned positions get gradient


def test_cross_loss_configuration_errors_and_no_signal_case():
    char = dp.CharTokenizer.build(["ab "], 16)
    with pytest.raises(ValueError, match="auto, marginal or uld"):
        ct.CrossTokenizerLoss(StubTeacher(), char, "bogus")
    assert ct.CrossTokenizerLoss(StubTeacher(), char, "auto").mode == "marginal"
    tokens_only = ids = torch.tensor([[char.vocab["a"], char.eos_id, char.vocab["b"]]])          # segments too short to align
    loss, stats = ct.CrossTokenizerLoss(StubTeacher(), char)(torch.randn(1, 3, char.vocab_size, requires_grad=True), tokens_only)
    assert loss.item() == 0.0 and stats["aligned_positions"] == 0


# ---- with a real Hugging Face model and tokenizer ---------------------------------------------------
@pytest.fixture(scope="module")
def hf_teacher_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    root = tmp_path_factory.mktemp("hf")
    data = corpus(root / "d.jsonl", 80)
    texts = [TEMPLATE.format(i=i % 10) for i in range(80)]
    bpe = dp.BPETokenizer.train(texts, 300)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=bpe.tokenizer, unk_token="<unk>", eos_token="<eos>")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=bpe.vocab_size, n_embd=48, n_layer=2, n_head=2, n_positions=128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = torch.tensor([bpe.encode(t) + [bpe.eos_id] for t in texts[:10]])
    for _ in range(120):                                             # enough to make the teacher's next-token distribution informative
        loss = model(input_ids=batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    directory = root / "teacher"
    directory.mkdir()
    # Written by hand rather than model.save_pretrained: that path imports deepspeed via accelerate, which some
    # environments cannot import (no CUDA_HOME). Loading with from_pretrained is unaffected.
    from safetensors.torch import save_file
    model.config.save_pretrained(directory)
    save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}, str(directory / "model.safetensors"))
    fast.save_pretrained(directory)
    return root, data, directory, float(loss)


def test_hf_teacher_exposes_probabilities_offsets_and_token_strings(hf_teacher_dir):
    _, _, directory, teacher_loss = hf_teacher_dir
    assert teacher_loss < 1.5                                        # it learned the template
    teacher = ct.HFTeacher(str(directory), torch.device("cpu"))
    text = "story 3: the quick brown fox"
    probs, ends = teacher.next_token_probs(text)
    assert probs.shape == (len(ends), teacher.vocab_size) and ends[-1] == len(text) and ends == sorted(ends)
    assert torch.allclose(probs.sum(-1), torch.ones(len(ends)), atol=1e-4)
    strings = teacher.token_strings()
    assert len(strings) == teacher.vocab_size and "".join(strings[i] for i in teacher.tokenizer(text, add_special_tokens=False)["input_ids"]) == text
    with pytest.raises(ValueError):
        teacher.tokenizer.__class__.is_fast = False                  # a slow tokenizer cannot provide offsets
        try:
            ct.HFTeacher(str(directory), torch.device("cpu"))
        finally:
            teacher.tokenizer.__class__.is_fast = True


@pytest.fixture(scope="module")
def char_student(hf_teacher_dir, tmp_path_factory):
    root, data, _, _ = hf_teacher_dir
    output = root / "student"
    trainer.run_training(trainer.default_args(data=[data], steps=20, batch_size=4, sequence_length=48, examples=80, hidden_size=32, layers=2,
                                              ffn_size=64, total_experts=4, active_experts=2, vocab_size=96, device="cpu", checkpoint_interval=0,
                                              lr=3e-3, capacity_factor=0, output=output, checkpoint_dir=root / "sck"))
    return output


def distill_args(tmp_path, student, data, teacher_dir, **overrides):
    values = ["--student", str(student), "--teacher-data", str(data), "--hf-teacher", str(teacher_dir), "--output", str(tmp_path / "out"),
              "--checkpoint-dir", str(tmp_path / "ck"), "--steps", "40", "--sequence-length", "48", "--batch-size", "4", "--device", "cpu",
              "--checkpoint-interval", "0", "--lr", "3e-3"]
    for key, value in overrides.items():
        values += [f"--{key.replace('_', '-')}", str(value)]
    return distill.build_parser().parse_args(values)


def collect_losses(monkeypatch):
    recorded = []
    original = ct.CrossTokenizerLoss.__call__

    def spy(self, *args, **kwargs):
        loss, stats = original(self, *args, **kwargs)
        recorded.append((float(loss.detach()), stats["aligned_fraction"]))
        return loss, stats

    monkeypatch.setattr(ct.CrossTokenizerLoss, "__call__", spy)
    return recorded


def test_char_student_learns_the_teachers_next_character_distribution(hf_teacher_dir, char_student, tmp_path, monkeypatch):
    _, data, directory, _ = hf_teacher_dir
    recorded = collect_losses(monkeypatch)
    result = distill.run_distillation(distill_args(tmp_path, char_student, data, directory))
    assert result["mode"] == "cross-tokenizer" and 0 < result["aligned_fraction"] < 1
    losses = [loss for loss, _ in recorded]
    assert sum(losses[-8:]) / 8 < 0.75 * sum(losses[:8]) / 8         # the student moved toward the teacher's next-character distribution


def test_uld_loss_runs_with_a_char_student_and_a_bpe_student(hf_teacher_dir, char_student, tmp_path, monkeypatch):
    root, data, directory, _ = hf_teacher_dir
    recorded = collect_losses(monkeypatch)
    distill.run_distillation(distill_args(tmp_path, char_student, data, directory, cross_loss="uld", steps=25))
    assert all(0 <= loss <= 2.0001 for loss, _ in recorded) and sum(l for l, _ in recorded[-5:]) < sum(l for l, _ in recorded[:5])

    bpe_dir = root / "bpe_student"
    trainer.run_training(trainer.default_args(data=[data], steps=6, batch_size=4, sequence_length=32, examples=80, hidden_size=32, layers=2,
                                              ffn_size=64, total_experts=4, active_experts=2, device="cpu", checkpoint_interval=0, capacity_factor=0,
                                              tokenizer="bpe", tokenizer_path=root / "bpe_tok", vocab_size=320, output=bpe_dir, checkpoint_dir=root / "bck"))
    recorded.clear()
    args = distill_args(tmp_path / "b", bpe_dir, data, directory, steps=4, sequence_length=32)
    args.output, args.checkpoint_dir = tmp_path / "bo", tmp_path / "bc"
    distill.run_distillation(args)
    assert recorded and all(loss == loss for loss, _ in recorded)
    with pytest.raises(ValueError, match="only for character-level"):
        ct.CrossTokenizerLoss(StubTeacher(), dp.load_tokenizer(bpe_dir), "marginal")


def test_conflicting_or_broken_teacher_options_are_rejected(hf_teacher_dir, char_student, tmp_path):
    _, data, directory, _ = hf_teacher_dir
    args = distill_args(tmp_path, char_student, data, directory)
    args.teacher_checkpoint = char_student
    with pytest.raises(ValueError, match="choose one teacher"):
        distill.run_distillation(args)
