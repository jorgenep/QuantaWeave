import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import data_pipeline as dp


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def char_tokenizer(text: str = "abcdefgh") -> dp.CharTokenizer:
    return dp.CharTokenizer.build([text], 64)


def test_char_tokenizer_roundtrip_and_legacy_vocab_loading(tmp_path):
    tokenizer = char_tokenizer("hello world")
    ids = tokenizer.encode("hello ?")
    assert tokenizer.decode(ids[:5]) == "hello" and ids[-1] == tokenizer.unk_id
    tokenizer.save(tmp_path / "new")
    assert dp.load_tokenizer(tmp_path / "new").vocab == tokenizer.vocab

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "vocab.json").write_text(json.dumps(tokenizer.vocab))  # no tokenizer_meta.json
    assert isinstance(dp.load_tokenizer(legacy), dp.CharTokenizer)


def test_bpe_tokenizer_trains_saves_and_roundtrips(tmp_path):
    pytest.importorskip("tokenizers")
    corpus = ["the quick brown fox jumps over the lazy dog"] * 50 + ["def add(a, b): return a + b"] * 50
    tokenizer = dp.BPETokenizer.train(corpus, 300)
    text = "the lazy fox: add(a, b)"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text and len(ids) < len(text)
    assert tokenizer.eos_id == 1 and tokenizer.unk_id == 0
    tokenizer.save(tmp_path)
    reloaded = dp.load_tokenizer(tmp_path)
    assert isinstance(reloaded, dp.BPETokenizer) and reloaded.encode(text) == ids


def test_domain_windows_are_pure_and_partial_tails_dropped(tmp_path):
    data = write_rows(tmp_path / "d.jsonl", [
        {"text": "aaaa", "domain": "x"}, {"text": "bbbbbbbbb", "domain": "y"}, {"text": "aa", "domain": "x"},
    ])
    tokenizer = char_tokenizer("ab")
    dataset = dp.WindowDataset(dp.build_corpus([data], tokenizer), sequence_length=3)  # window = 4
    assert dataset.domains == ["x", "y"]
    a, b, eos = tokenizer.vocab["a"], tokenizer.vocab["b"], tokenizer.eos_id
    # x stream: aaaa <eos> aa <eos> = 8 tokens -> 2 windows; y stream: b*9 <eos> = 10 -> 2 windows
    assert len(dataset) == 4 and dataset.window_domains.tolist() == [0, 0, 1, 1]
    assert dataset[0].tolist() == [a, a, a, a] and dataset[1].tolist() == [eos, a, a, eos]
    assert dataset[2].tolist() == [b] * 4


def test_multiple_files_use_file_stem_as_domain(tmp_path):
    one = write_rows(tmp_path / "stories.jsonl", [{"text": "aaaaaaaa"}])
    two = write_rows(tmp_path / "code.jsonl", [{"text": "bbbbbbbb"}])
    corpus = dp.build_corpus([one, two], char_tokenizer("ab"))
    assert corpus.domains == ["stories", "code"]


def test_memmap_corpus_matches_in_memory_corpus(tmp_path):
    pytest.importorskip("numpy")
    data = write_rows(tmp_path / "d.jsonl", [
        {"text": "abcabcabc", "domain": "x"}, {"text": "hgfhgfhgf", "domain": "y"}, {"text": "abcabc", "domain": "x"},
    ])
    tokenizer = char_tokenizer("abcfgh")
    meta = dp.write_token_bin([data], tokenizer, tmp_path / "bin")
    assert meta["dtype"] == "uint16" and meta["total_tokens"] == (9 + 1 + 6 + 1) + (9 + 1)
    memory = dp.WindowDataset(dp.build_corpus([data], tokenizer), 4)
    mapped_corpus, mapped_tokenizer = dp.load_token_bin(tmp_path / "bin")
    mapped = dp.WindowDataset(mapped_corpus, 4)
    assert mapped_tokenizer.vocab == tokenizer.vocab
    assert len(mapped) == len(memory) and mapped.domains == memory.domains
    for i in range(len(memory)):
        assert torch.equal(mapped[i], memory[i])
    assert mapped.fingerprint() == memory.fingerprint()


def test_difficulty_scores_rank_repetitive_windows_easier():
    tokenizer = char_tokenizer("abcdefgh")
    # window 1: one repeated common letter. window 2: all different letters.
    text = "aaaaaaaa" + "abcdefgh"
    corpus = dp.TokenCorpus(torch.tensor(tokenizer.encode(text), dtype=torch.int32), [dp.Segment("default", 0, len(text))])
    dataset = dp.WindowDataset(corpus, sequence_length=7)
    assert len(dataset) == 2
    for method in ("rarity", "entropy", "uncommon"):
        scores = dp.difficulty_scores(dataset, method, tokenizer.vocab_size)
        assert scores[0] <= scores[1], method
    with pytest.raises(ValueError):
        dp.difficulty_scores(dataset, "nope", tokenizer.vocab_size)


def curriculum_stream(tmp_path, curriculum, batch_size=8):
    tokenizer = char_tokenizer("abcdefgh")
    # 20 windows of increasing character diversity
    text = "".join((("abcdefgh"[: 1 + i % 8]) * 8)[:8] for i in range(20))
    corpus = dp.TokenCorpus(torch.tensor(tokenizer.encode(text), dtype=torch.int32), [dp.Segment("default", 0, len(text))])
    dataset = dp.WindowDataset(corpus, 7)
    scores = dp.difficulty_scores(dataset, "entropy", tokenizer.vocab_size)
    return dp.BatchStream(dataset, batch_size, seed=3, scores=scores, curriculum=curriculum), scores, dataset


def test_curriculum_starts_easy_and_widens(tmp_path):
    curriculum = dp.CurriculumConfig(method="entropy", start_fraction=0.25, steps=10_000)
    stream, scores, dataset = curriculum_stream(tmp_path, curriculum)
    order = torch.argsort(scores, stable=True)
    easiest = set(order[: 5].tolist())          # 25% of 20 windows
    seen_early = set()
    for step in range(20):
        tokens, _ = stream.batch(step)
        for row in tokens:
            seen_early |= {i for i in range(len(dataset)) if torch.equal(dataset[i], row)}
    assert seen_early <= easiest | {i for i in range(len(dataset)) if scores[i] <= scores[order[4]]}
    seen_late = set()
    for step in range(10_000, 10_040):
        tokens, _ = stream.batch(step)
        for row in tokens:
            seen_late |= {i for i in range(len(dataset)) if torch.equal(dataset[i], row)}
    assert len(seen_late) > len(seen_early)

    assert curriculum.fraction(0) == 0.25 and curriculum.fraction(5_000) == pytest.approx(0.625)
    assert curriculum.fraction(50_000) == 1.0 and curriculum.stage(0) == 1 and curriculum.stage(50_000) == 4
    assert not dp.CurriculumConfig().enabled and dp.CurriculumConfig().fraction(7) == 1.0
    with pytest.raises(ValueError):
        dp.BatchStream(dataset, 2, curriculum=curriculum)  # no scores


def test_batch_stream_is_deterministic_and_rank_slices_partition_the_global_batch(tmp_path):
    stream, _, dataset = curriculum_stream(tmp_path, dp.CurriculumConfig())
    a, b = stream.batch(5), stream.batch(5)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(stream.batch(5)[0], stream.batch(6)[0])

    ranks = [dp.BatchStream(dataset, 4, seed=3, rank=r, world_size=2) for r in range(2)]
    whole = dp.BatchStream(dataset, 8, seed=3)
    assert torch.equal(torch.cat([r.batch(9)[0] for r in ranks]), whole.batch(9)[0])


def test_domain_mixture_weights_and_sampling(tmp_path):
    data = write_rows(tmp_path / "d.jsonl", [{"text": "a" * 64, "domain": "x"}, {"text": "b" * 64, "domain": "y"}])
    dataset = dp.WindowDataset(dp.build_corpus([data], char_tokenizer("ab")), 7)
    only_y = dp.DomainMixture(dataset.domains, {"x": 0, "y": 1})
    stream = dp.BatchStream(dataset, 16, mixture=only_y)
    assert set(stream.batch(0)[1].tolist()) == {1}

    shifting = dp.DomainMixture(dataset.domains, {"x": 1, "y": 0}, {"x": 0, "y": 1}, steps=10)
    assert shifting.weights(0).tolist() == [1.0, 0.0]
    assert shifting.weights(5).tolist() == pytest.approx([0.5, 0.5])
    assert shifting.weights(99).tolist() == [0.0, 1.0]

    with pytest.raises(ValueError, match="unknown"):
        dp.DomainMixture(dataset.domains, {"x": 1, "z": 1})
    with pytest.raises(ValueError, match="cover every"):
        dp.DomainMixture(dataset.domains, {"x": 1})
    assert dp.parse_domain_weights("x=0.7, y=0.3") == {"x": 0.7, "y": 0.3}
    assert dp.parse_domain_weights(None) is None
    with pytest.raises(ValueError):
        dp.parse_domain_weights("x")


def test_stream_state_reports_position_and_stage(tmp_path):
    stream, _, dataset = curriculum_stream(tmp_path, dp.CurriculumConfig(start_fraction=0.5, steps=10))
    state = stream.state(5)
    assert state["samples_consumed"] == 5 * 8 and state["fingerprint"] == dataset.fingerprint()
    assert state["curriculum_stage"] == 3 and state["domain_weights"] == {"default": 1.0}


def test_token_classes_label_character_kinds():
    tokenizer = dp.CharTokenizer.build(["ab 1.Z"], 32)
    classes, names = dp.token_classes(tokenizer)
    label = lambda ch: names[classes[tokenizer.vocab[ch]]]
    assert (label("a"), label("1"), label(" "), label("."), label("Z")) == ("letter", "digit", "space", "punct", "letter")
    assert names[classes[tokenizer.eos_id]] == "special"


def test_empty_stream_is_rejected(tmp_path):
    data = write_rows(tmp_path / "d.jsonl", [{"text": "ab"}])
    dataset = dp.WindowDataset(dp.build_corpus([data], char_tokenizer("ab")), 32)
    assert len(dataset) == 0
    with pytest.raises(ValueError):
        dp.BatchStream(dataset, 2)
