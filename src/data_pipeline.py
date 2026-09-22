"""Tokenizers, domain-tagged token corpora, windowed datasets, curriculum and domain sampling.

Everything here is a pure function of its inputs: ``BatchStream.batch(step)`` depends only on
(seed, step, world layout), so a resumed run reproduces the batches it would have seen.
"""

import hashlib
import json
import math
import shutil
import unicodedata
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

SPECIAL_TOKENS = ("<unk>", "<eos>")


# --------------------------------------------------------------------------- text rows
def extract_text(row: dict[str, str]) -> str:
    text = row.get("teacher_text") or row.get("completion") or row.get("text")
    if text:
        return text
    return row.get("prompt", "") + row.get("response", "")


def read_rows(paths: Sequence[Path], limit: Optional[int] = None) -> Iterator[tuple[str, str]]:
    """Yield (domain, text). ``limit`` caps the JSONL rows read per file.

    A row's domain is its "domain" field; without one it is "default" for a single input
    file and the file stem when several files are given.
    """
    for path in paths:
        fallback = "default" if len(paths) == 1 else Path(path).stem
        with Path(path).open(encoding="utf-8") as lines:
            for line_number, line in enumerate(lines):
                if limit is not None and line_number >= limit:
                    break
                row = json.loads(line)
                text = extract_text(row)
                if text:
                    yield str(row.get("domain") or fallback), text


# --------------------------------------------------------------------------- tokenizers
class CharTokenizer:
    kind = "char"

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.unk_id = vocab["<unk>"]
        self.eos_id = vocab["<eos>"]
        self.inverse = {index: token for token, index in vocab.items()}

    @property
    def vocab_size(self) -> int:
        return max(self.vocab.values()) + 1

    @classmethod
    def build(cls, texts: Iterable[str], vocab_size: int) -> "CharTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for character in text:
                counts[character] = counts.get(character, 0) + 1
        reserved = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        common = sorted(counts, key=counts.get, reverse=True)[: max(0, vocab_size - len(reserved))]
        return cls({**reserved, **{character: index + len(reserved) for index, character in enumerate(common)}})

    def encode(self, text: str) -> list[int]:
        unknown = self.unk_id
        return [self.vocab.get(character, unknown) for character in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.inverse.get(int(i), "?") for i in ids)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2) + "\n")
        (directory / "tokenizer_meta.json").write_text(json.dumps({"type": "char"}) + "\n")


class BPETokenizer:
    """Byte-level BPE trained from scratch with the ``tokenizers`` library."""

    kind = "bpe"

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.token_to_id("<unk>")
        self.eos_id = tokenizer.token_to_id("<eos>")

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        minimum = 256 + len(SPECIAL_TOKENS)  # every byte plus the special tokens
        if vocab_size < minimum:
            raise ValueError(f"byte-level BPE needs a vocab size of at least {minimum}, got {vocab_size}")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=False)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_meta.json").write_text(json.dumps({"type": "bpe"}) + "\n")


class SentencePieceTokenizer:
    """Unigram or BPE tokenizer trained from scratch with Google's ``sentencepiece`` library.

    Unlike the byte-level ``BPETokenizer`` above, sentencepiece's default ``unigram`` algorithm is a probabilistic
    subword model (Kudo, 2018) rather than greedy merges, and works directly on raw text with its own internal
    normalization instead of a byte-level pre-tokenizer. Pass ``algorithm="bpe"`` for a sentencepiece-flavoured BPE
    tokenizer instead, if you want merges but with sentencepiece's normalization/whitespace handling.
    """

    kind = "sentencepiece"

    def __init__(self, processor) -> None:
        self.processor = processor
        self.unk_id = processor.unk_id()
        self.eos_id = processor.eos_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int, algorithm: str = "unigram") -> "SentencePieceTokenizer":
        import io

        import sentencepiece as spm

        if algorithm not in {"unigram", "bpe"}:
            raise ValueError("algorithm must be 'unigram' or 'bpe'")
        buffer = io.BytesIO()
        spm.SentencePieceTrainer.Train(
            sentence_iterator=iter(texts), model_writer=buffer, vocab_size=vocab_size, model_type=algorithm,
            unk_id=0, bos_id=-1, eos_id=1, pad_id=-1, unk_piece="<unk>", eos_piece="<eos>",
        )
        return cls(spm.SentencePieceProcessor(model_proto=buffer.getvalue()))

    def encode(self, text: str) -> list[int]:
        return self.processor.encode(text)

    def decode(self, ids: Iterable[int]) -> str:
        return self.processor.decode([int(i) for i in ids])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer.model").write_bytes(self.processor.serialized_model_proto())
        (directory / "tokenizer_meta.json").write_text(json.dumps({"type": "sentencepiece"}) + "\n")


def load_tokenizer(directory: Path):
    """Load the tokenizer saved beside a checkpoint. A bare vocab.json means a character tokenizer."""
    directory = Path(directory)
    meta_path = directory / "tokenizer_meta.json"
    kind = json.loads(meta_path.read_text())["type"] if meta_path.exists() else "char"
    if kind == "bpe":
        from tokenizers import Tokenizer

        return BPETokenizer(Tokenizer.from_file(str(directory / "tokenizer.json")))
    if kind == "sentencepiece":
        import sentencepiece as spm

        return SentencePieceTokenizer(spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model")))
    return CharTokenizer(json.loads((directory / "vocab.json").read_text()))


def token_classes(tokenizer) -> tuple[Tensor, list[str]]:
    """Coarse class per token id (letter/digit/space/punct/special/other) for specialization maps."""
    names = ["special", "letter", "digit", "space", "punct", "other"]
    classes = torch.full((tokenizer.vocab_size,), names.index("other"), dtype=torch.long)
    for token_id in range(tokenizer.vocab_size):
        if isinstance(tokenizer, CharTokenizer):
            text = tokenizer.inverse.get(token_id, "")
        elif isinstance(tokenizer, SentencePieceTokenizer):
            text = tokenizer.processor.id_to_piece(token_id).replace("▁", " ")
        else:
            text = (tokenizer.tokenizer.id_to_token(token_id) or "").replace("Ġ", " ").replace("Ċ", "\n")
        if text in SPECIAL_TOKENS:
            label = "special"
        elif not text.strip():
            label = "space" if text else "other"
        elif text.strip()[0].isalpha():
            label = "letter"
        elif text.strip()[0].isdigit():
            label = "digit"
        elif unicodedata.category(text.strip()[0]).startswith("P") or text.strip()[0] in "<>=+-*/%&|^~$#@`":
            label = "punct"
        else:
            label = "other"
        classes[token_id] = names.index(label)
    return classes, names


# --------------------------------------------------------------------------- corpus
@dataclass
class Segment:
    domain: str
    start: int
    length: int


class TokenCorpus:
    """A flat token stream split into per-domain segments; backed by memory or a memmap."""

    def __init__(self, data, segments: list[Segment]) -> None:
        self._data = data
        self.segments = segments
        self.domains: list[str] = []
        for segment in segments:
            if segment.domain not in self.domains:
                self.domains.append(segment.domain)

    def read(self, start: int, stop: int) -> Tensor:
        if isinstance(self._data, Tensor):
            return self._data[start:stop].long()
        import numpy as np

        return torch.from_numpy(np.array(self._data[start:stop], dtype=np.int64))

    def __len__(self) -> int:
        return sum(segment.length for segment in self.segments)


def build_corpus(paths: Sequence[Path], tokenizer, limit: Optional[int] = None) -> TokenCorpus:
    """Tokenize JSONL rows into per-domain streams, each document followed by <eos>."""
    streams: dict[str, array] = {}
    for domain, text in read_rows(paths, limit):
        stream = streams.setdefault(domain, array("i"))
        stream.extend(tokenizer.encode(text))
        stream.append(tokenizer.eos_id)
    joined = array("i")
    segments = []
    for domain, stream in streams.items():
        segments.append(Segment(domain, len(joined), len(stream)))
        joined.extend(stream)
    data = torch.frombuffer(joined, dtype=torch.int32) if len(joined) else torch.empty(0, dtype=torch.int32)
    corpus = TokenCorpus(data, segments)
    corpus._keepalive = joined  # torch.frombuffer does not own the memory
    return corpus


def write_token_bin(
    paths: Sequence[Path], tokenizer, out_dir: Path, limit: Optional[int] = None
) -> dict:
    """Tokenize JSONL into a memory-mappable ``tokens.bin`` plus ``meta.json`` and the tokenizer.

    Rows are appended to one temporary file per domain, so memory use does not grow with the corpus.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if tokenizer.vocab_size <= 2**16 else np.uint32
    parts: dict[str, Path] = {}
    handles = {}
    lengths: dict[str, int] = {}
    try:
        for domain, text in read_rows(paths, limit):
            if domain not in handles:
                parts[domain] = out_dir / f".tokens.{len(parts)}.part"
                handles[domain] = parts[domain].open("wb")
                lengths[domain] = 0
            ids = np.asarray(tokenizer.encode(text) + [tokenizer.eos_id], dtype=dtype)
            ids.tofile(handles[domain])
            lengths[domain] += ids.size
    finally:
        for handle in handles.values():
            handle.close()
    segments = []
    with (out_dir / "tokens.bin").open("wb") as target:
        for domain, part in parts.items():
            segments.append({"domain": domain, "start": target.tell() // np.dtype(dtype).itemsize, "length": lengths[domain]})
            with part.open("rb") as source:
                shutil.copyfileobj(source, target)
            part.unlink()
    tokenizer.save(out_dir / "tokenizer")
    meta = {
        "format": 1,
        "dtype": np.dtype(dtype).name,
        "tokenizer": tokenizer.kind,
        "vocab_size": tokenizer.vocab_size,
        "eos_id": tokenizer.eos_id,
        "segments": segments,
        "total_tokens": sum(lengths.values()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_token_bin(directory: Path) -> tuple[TokenCorpus, object]:
    import numpy as np

    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    if meta["total_tokens"] == 0:
        return TokenCorpus(torch.empty(0, dtype=torch.int32), []), load_tokenizer(directory / "tokenizer")
    data = np.memmap(directory / "tokens.bin", dtype=np.dtype(meta["dtype"]), mode="r")
    segments = [Segment(item["domain"], item["start"], item["length"]) for item in meta["segments"]]
    return TokenCorpus(data, segments), load_tokenizer(directory / "tokenizer")


# --------------------------------------------------------------------------- windows
class WindowDataset(Dataset):
    """Non-overlapping windows of ``sequence_length + 1`` tokens (the extra one is the shifted label).

    Windows never cross a domain boundary; each domain's trailing partial window is dropped.
    """

    def __init__(self, corpus: TokenCorpus, sequence_length: int) -> None:
        self.corpus = corpus
        self.window = sequence_length + 1
        self.domains = corpus.domains
        starts, window_domains = [], []
        for segment in corpus.segments:
            count = segment.length // self.window
            starts.append(segment.start + torch.arange(count, dtype=torch.long) * self.window)
            window_domains.append(torch.full((count,), self.domains.index(segment.domain), dtype=torch.long))
        self.starts = torch.cat(starts) if starts else torch.empty(0, dtype=torch.long)
        self.window_domains = torch.cat(window_domains) if window_domains else torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def __getitem__(self, index: int) -> Tensor:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = int(self.starts[index])
        return self.corpus.read(start, start + self.window)

    def batch(self, indices: Sequence[int]) -> Tensor:
        return torch.stack([self[int(i)] for i in indices])

    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{len(self)}|{self.window}|{self.domains}".encode())
        return digest.hexdigest()[:16]

    @classmethod
    def _view(cls, corpus: TokenCorpus, window: int, domains: list[str], starts: Tensor, window_domains: Tensor) -> "WindowDataset":
        """A restricted view over a subset of windows, sharing the corpus rather than rescanning it."""
        view = cls.__new__(cls)
        view.corpus, view.window, view.domains = corpus, window, domains
        view.starts, view.window_domains = starts, window_domains
        return view

    def split_train_val(self, val_fraction: float = 0.02, seed: int = 0, min_val_windows_per_domain: int = 1) -> tuple["WindowDataset", "WindowDataset"]:
        """Split into (train, val), holding out ``val_fraction`` of each domain's windows.

        Domain-aware: the split is stratified per domain rather than taken globally, so a small domain isn't
        left with zero validation windows (or entirely absent from training) purely by chance. Deterministic
        for a given ``seed``, and the two views never overlap. Raises if any domain has too few windows to hold
        any out at all (``val_fraction`` too small, or the domain itself too small).
        """
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        generator = torch.Generator().manual_seed(seed)
        train_parts, val_parts = [], []
        for domain_index in range(len(self.domains)):
            positions = (self.window_domains == domain_index).nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            order = positions[torch.randperm(positions.numel(), generator=generator)]
            held_out = max(min_val_windows_per_domain, round(positions.numel() * val_fraction))
            if held_out >= positions.numel():
                raise ValueError(
                    f"domain '{self.domains[domain_index]}' has only {positions.numel()} window(s), too few to "
                    f"both train on and hold out {held_out} for validation; lower val_fraction or add data"
                )
            val_parts.append(order[:held_out])
            train_parts.append(order[held_out:])
        train_indices = torch.cat(train_parts).sort().values
        val_indices = torch.cat(val_parts).sort().values
        train = self._view(self.corpus, self.window, self.domains, self.starts[train_indices], self.window_domains[train_indices])
        val = self._view(self.corpus, self.window, self.domains, self.starts[val_indices], self.window_domains[val_indices])
        return train, val


class CharacterDataset(WindowDataset):
    """Character-level windows over JSONL text (kept for the trainer, benchmark and distillation)."""

    def __init__(self, path: Path, vocab: dict[str, int], sequence_length: int, limit: Optional[int]) -> None:
        super().__init__(build_corpus([Path(path)], CharTokenizer(vocab), limit), sequence_length)


# --------------------------------------------------------------------------- curriculum
def difficulty_scores(dataset: WindowDataset, method: str, vocab_size: int, chunk: int = 4096) -> Tensor:
    """Per-window difficulty; higher is harder.

    rarity   mean negative log unigram frequency of the window's tokens
    entropy  Shannon entropy (nats) of the window's token histogram
    uncommon share of tokens whose corpus frequency is in the rarest 10%
    """
    if method not in {"rarity", "entropy", "uncommon"}:
        raise ValueError("curriculum method must be rarity, entropy or uncommon")
    windows = len(dataset)
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    for offset in range(0, windows, chunk):
        batch = dataset.batch(range(offset, min(windows, offset + chunk)))
        counts += torch.bincount(batch.reshape(-1), minlength=vocab_size).double()
    frequency = counts / counts.sum().clamp(min=1)
    neg_log = -(frequency.clamp_min(1e-12)).log()
    seen = counts[counts > 0]
    rare_cutoff = torch.quantile(seen, 0.1) if seen.numel() else torch.tensor(0.0, dtype=torch.float64)
    scores = torch.empty(windows, dtype=torch.float64)
    for offset in range(0, windows, chunk):
        batch = dataset.batch(range(offset, min(windows, offset + chunk)))
        if method == "rarity":
            scores[offset : offset + len(batch)] = neg_log[batch].mean(dim=1)
        elif method == "uncommon":
            scores[offset : offset + len(batch)] = (counts[batch] <= rare_cutoff).double().mean(dim=1)
        else:
            for row, tokens in enumerate(batch):
                histogram = torch.bincount(tokens).double()
                probabilities = histogram[histogram > 0] / tokens.numel()
                scores[offset + row] = -(probabilities * probabilities.log()).sum()
    return scores.float()


@dataclass
class CurriculumConfig:
    """Linear pacing: start with the easiest ``start_fraction`` of windows and grow to all of them."""

    method: str = "rarity"
    start_fraction: float = 0.25
    steps: int = 0  # 0 disables the curriculum

    @property
    def enabled(self) -> bool:
        return self.steps > 0

    def fraction(self, step: int) -> float:
        if not self.enabled:
            return 1.0
        return min(1.0, self.start_fraction + (1.0 - self.start_fraction) * step / self.steps)

    def stage(self, step: int, tiers: int = 4) -> int:
        """Difficulty tier reached at ``step`` (1..tiers)."""
        return max(1, math.ceil(self.fraction(step) * tiers - 1e-9))


# --------------------------------------------------------------------------- domain mixtures
def parse_domain_weights(spec: Optional[str]) -> Optional[dict[str, float]]:
    if not spec:
        return None
    weights = {}
    for item in spec.split(","):
        name, _, value = item.partition("=")
        if not name.strip() or not value:
            raise ValueError(f"bad domain weight '{item}', expected name=weight")
        weights[name.strip()] = float(value)
        if weights[name.strip()] < 0:
            raise ValueError("domain weights must be non-negative")
    return weights


class DomainMixture:
    """Domain sampling weights, optionally moving linearly from ``start`` to ``end`` over ``steps``."""

    def __init__(self, domains: Sequence[str], start: dict[str, float], end: Optional[dict[str, float]] = None, steps: int = 0):
        for weights in (start, end or {}):
            unknown = set(weights) - set(domains)
            if unknown:
                raise ValueError(f"unknown domains {sorted(unknown)}; dataset has {list(domains)}")
            missing = set(domains) - set(weights)
            if weights and missing:
                raise ValueError(f"domain weights must cover every domain; missing {sorted(missing)}")
        self.domains = list(domains)
        self.start = torch.tensor([start[d] for d in domains], dtype=torch.double)
        self.end = torch.tensor([end[d] for d in domains], dtype=torch.double) if end else None
        self.steps = steps
        if self.start.sum() <= 0 or (self.end is not None and self.end.sum() <= 0):
            raise ValueError("domain weights must not all be zero")

    def weights(self, step: int) -> Tensor:
        if self.end is None or self.steps <= 0:
            return self.start / self.start.sum()
        mix = min(1.0, step / self.steps)
        blended = (1 - mix) * self.start / self.start.sum() + mix * self.end / self.end.sum()
        return blended / blended.sum()


# --------------------------------------------------------------------------- batches
class BatchStream:
    """Deterministic batch source: ``batch(step)`` is a pure function of (seed, step, rank layout).

    Each step draws a domain per sample (mixture weights, or the corpus proportions), then a window
    from that domain's pool. With a curriculum the pool is the easiest fraction of the domain's
    windows for that step. Ranks take disjoint slices of one global draw.
    """

    def __init__(
        self,
        dataset: WindowDataset,
        batch_size: int,
        seed: int = 0,
        scores: Optional[Tensor] = None,
        curriculum: Optional[CurriculumConfig] = None,
        mixture: Optional[DomainMixture] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset holds fewer tokens than one training window")
        if curriculum is not None and curriculum.enabled and scores is None:
            raise ValueError("a curriculum needs difficulty scores")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.curriculum = curriculum or CurriculumConfig()
        self.mixture = mixture
        self.rank, self.world_size = rank, world_size
        self.pools: list[Tensor] = []
        for domain_index in range(len(dataset.domains)):
            indices = (dataset.window_domains == domain_index).nonzero(as_tuple=True)[0]
            if scores is not None and indices.numel():
                indices = indices[torch.argsort(scores[indices], stable=True)]
            self.pools.append(indices)
        self.natural = torch.tensor([pool.numel() for pool in self.pools], dtype=torch.double)

    @property
    def num_domains(self) -> int:
        return len(self.dataset.domains)

    def domain_weights(self, step: int) -> Tensor:
        weights = self.mixture.weights(step) if self.mixture is not None else self.natural
        weights = weights.clone()
        weights[self.natural == 0] = 0  # a domain too short for one window cannot be sampled
        if weights.sum() <= 0:
            raise ValueError("no sampleable domain has non-zero weight")
        return weights / weights.sum()

    def batch(self, step: int) -> tuple[Tensor, Tensor]:
        total = self.batch_size * self.world_size
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + step)
        domains = torch.multinomial(self.domain_weights(step), total, replacement=True, generator=generator)
        uniform = torch.rand(total, generator=generator, dtype=torch.double)
        fraction = self.curriculum.fraction(step)
        chosen = []
        for domain, u in zip(domains.tolist(), uniform.tolist()):
            pool = self.pools[domain]
            reachable = max(1, math.ceil(fraction * pool.numel()))
            chosen.append(int(pool[min(int(u * reachable), reachable - 1)]))
        mine = slice(self.rank * self.batch_size, (self.rank + 1) * self.batch_size)
        return self.dataset.batch(chosen[mine]), domains[mine]

    def state(self, step: int) -> dict:
        """Everything needed to verify and describe the stream position at ``step``."""
        return {
            "seed": self.seed,
            "step": step,
            "samples_consumed": step * self.batch_size * self.world_size,
            "fingerprint": self.dataset.fingerprint(),
            "curriculum_fraction": self.curriculum.fraction(step),
            "curriculum_stage": self.curriculum.stage(step) if self.curriculum.enabled else None,
            "domain_weights": {d: float(w) for d, w in zip(self.dataset.domains, self.domain_weights(step))},
        }
