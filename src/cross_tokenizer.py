"""Logit distillation from a teacher with a different tokenizer (e.g. a Hugging Face causal LM).

Teacher and student read the same *text*. Wherever a student token boundary coincides with a teacher token
boundary (a character offset where both tokenizations agree), the teacher's next-token distribution and the
student's describe the same event: "what comes next after this text". Those positions are the supervision;
the others are skipped. Two losses:

marginal  (char-level students, exact): the teacher's next-token distribution is collapsed onto the first
          character of each candidate token, giving the teacher's true next-character distribution, and the
          student minimises KL(teacher_next_char || student). No approximation beyond the alignment.
uld       (any student): universal-logit-distillation style. Both distributions are sorted in descending order
          and compared with L1, so no vocabulary mapping is needed; this matches the *shape* of the teacher's
          distribution (confidence, tail mass), not which token gets which probability.

Documents are split at <eos> and each segment is scored independently. For BPE students each token must decode
to valid text on its own (true for ASCII data); otherwise a clear error is raised. Only the fast-tokenizer
character offsets of the teacher are used, so slow tokenizers are rejected.
"""

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

REPLACEMENT = "�"


class HFTeacher:
    """A Hugging Face causal LM (local directory or hub id) scored on raw text."""

    def __init__(self, path: str, device: torch.device, precision_dtype: Optional[torch.dtype] = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("--hf-teacher needs the `transformers` package") from error
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("the teacher needs a fast tokenizer: character offsets are used to align it with the student")
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=precision_dtype).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.vocab_size = self.model.get_output_embeddings().weight.size(0)
        limit = getattr(self.model.config, "max_position_embeddings", None) or getattr(self.model.config, "n_positions", None)
        self.max_tokens = int(limit) if limit else 2048
        self._strings: Optional[list[str]] = None

    def token_strings(self) -> list[str]:
        if self._strings is None:
            self._strings = [self.tokenizer.decode([i]) for i in range(self.vocab_size)]
        return self._strings

    @torch.no_grad()
    def next_token_probs(self, text: str) -> tuple[Tensor, list[int]]:
        """(probs [T, V], end offset of each of the T tokens): probs[k] predicts the token after token k."""
        encoding = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, return_tensors="pt")
        ids = encoding["input_ids"][:, : self.max_tokens].to(self.device)
        ends = [int(end) for _, end in encoding["offset_mapping"][0][: self.max_tokens].tolist()]
        logits = self.model(input_ids=ids).logits[0].float()
        return logits.softmax(dim=-1), ends


def student_pieces(tokenizer) -> list[str]:
    """The text each student token id stands for."""
    if tokenizer.kind == "char":
        pieces = [tokenizer.inverse.get(i, REPLACEMENT) for i in range(tokenizer.vocab_size)]
        pieces[tokenizer.unk_id] = REPLACEMENT
        pieces[tokenizer.eos_id] = ""
        return pieces
    pieces = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    pieces[tokenizer.eos_id] = ""
    return pieces


def align_positions(piece_lengths: Sequence[int], teacher_ends: Sequence[int]) -> list[tuple[int, int]]:
    """Pairs (student position j, teacher position k) predicting the same next event.

    Student position j is the state after student token j (it predicts token j+1). It aligns with teacher
    position k when teacher token k ends at exactly the character offset where student token j ends. The
    last student token has no successor, so it is excluded.
    """
    by_end: dict[int, int] = {}
    for index, end in enumerate(teacher_ends):
        by_end[end] = index                      # equal ends: the later token has consumed more, so it wins
    pairs, offset = [], 0
    for j, length in enumerate(piece_lengths[:-1]):
        offset += length
        if length and offset in by_end:
            pairs.append((j, by_end[offset]))
    return pairs


def first_char_matrix(teacher_strings: Sequence[str], student_tokenizer, device: torch.device, width: Optional[int] = None) -> Tensor:
    """M [teacher vocab, width]: M[t, c] = 1 when teacher token t begins with student character c.

    ``width`` is the student's output dimension (the model's vocab size, which may exceed the tokenizer's)."""
    matrix = torch.zeros(len(teacher_strings), width or student_tokenizer.vocab_size)
    for index, text in enumerate(teacher_strings):
        target = student_tokenizer.vocab.get(text[0], student_tokenizer.unk_id) if text else student_tokenizer.unk_id
        matrix[index, target] = 1.0
    return matrix.to(device)


def uld_loss(student_probs: Tensor, teacher_probs: Tensor) -> Tensor:
    """Mean over positions of the L1 distance between the descending-sorted distributions (zero-padded)."""
    width = max(student_probs.size(-1), teacher_probs.size(-1))
    student_sorted = F.pad(student_probs.sort(dim=-1, descending=True).values, (0, width - student_probs.size(-1)))
    teacher_sorted = F.pad(teacher_probs.sort(dim=-1, descending=True).values, (0, width - teacher_probs.size(-1)))
    return (student_sorted - teacher_sorted).abs().sum(dim=-1).mean()


def marginal_kl(student_logits: Tensor, teacher_probs: Tensor, matrix: Tensor) -> Tensor:
    """KL(teacher's next-character distribution || student) at each aligned position, averaged."""
    target = teacher_probs @ matrix
    log_student = F.log_softmax(student_logits.float(), dim=-1)
    return F.kl_div(log_student, target, reduction="batchmean")


class CrossTokenizerLoss:
    def __init__(self, teacher, student_tokenizer, mode: str = "auto") -> None:
        if mode == "auto":
            mode = "marginal" if student_tokenizer.kind == "char" else "uld"
        if mode not in {"marginal", "uld"}:
            raise ValueError("cross-tokenizer loss must be auto, marginal or uld")
        if mode == "marginal" and student_tokenizer.kind != "char":
            raise ValueError("the marginal loss is exact only for character-level students; use uld for BPE students")
        self.teacher, self.tokenizer, self.mode = teacher, student_tokenizer, mode
        self.pieces = student_pieces(student_tokenizer)
        self.matrix: Optional[Tensor] = None

    def _segments(self, ids: Sequence[int]):
        """Split at <eos>: yield (start position in the window, token ids) for each non-trivial segment."""
        start, current = 0, []
        for position, token in enumerate(list(ids) + [self.tokenizer.eos_id]):
            if token == self.tokenizer.eos_id:
                if len(current) > 1:
                    yield start, current
                start, current = position + 1, []
            else:
                current.append(token)

    def __call__(self, student_logits: Tensor, batch: Tensor) -> tuple[Tensor, dict]:
        """student_logits [B, L+1, V_student]; batch [B, L+1] token ids. Returns (loss, stats)."""
        rows, targets, considered = [], [], 0
        for b in range(batch.size(0)):
            for start, ids in self._segments(batch[b].tolist()):
                text = "".join(self.pieces[i] for i in ids)
                if self.tokenizer.kind != "char" and self.tokenizer.decode(ids) != text:
                    raise ValueError("student tokens do not decode to text piece by piece (byte-level splits); "
                                     "cross-tokenizer distillation needs tokens that are valid text on their own")
                probs, ends = self.teacher.next_token_probs(text)
                pairs = align_positions([len(self.pieces[i]) for i in ids], ends)
                considered += len(ids) - 1
                for j, k in pairs:
                    rows.append(student_logits[b, start + j])
                    targets.append(probs[k])
        stats = {"aligned_positions": len(rows), "considered_positions": considered,
                 "aligned_fraction": len(rows) / considered if considered else 0.0, "mode": self.mode}
        if not rows:
            return student_logits.sum() * 0.0, stats
        logits, teacher_probs = torch.stack(rows), torch.stack(targets).to(student_logits.device)
        if self.mode == "marginal":
            if self.matrix is None:
                self.matrix = first_char_matrix(self.teacher.token_strings(), self.tokenizer, student_logits.device, student_logits.size(-1))
            return marginal_kl(logits, teacher_probs, self.matrix), stats
        return uld_loss(F.softmax(logits.float(), dim=-1), teacher_probs), stats
