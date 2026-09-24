"""Build a right-sized code-pretraining corpus for the standalone QuantaWeave trainer.

Streams a language-weighted slice of bigcode/starcoderdata (code) plus a small slice of
HuggingFaceFW/fineweb-edu (natural-language grounding: docs, explanations, Q&A) and writes
plain {"text": ...} JSONL rows the standalone data pipeline (data_pipeline.py's extract_text)
already reads directly -- no Axolotl, no external tokenizer required.

Token counts for code are estimated at ~4 characters/token (a standard rough BPE approximation);
the exact count is fixed later when prepare_tokens.py packs the corpus with a real trained
tokenizer. FineWeb-Edu rows carry an exact `token_count` field, so that side of the budget is exact.
"""

import argparse
import json
import time
from pathlib import Path

from datasets import load_dataset

CHARS_PER_TOKEN_ESTIMATE = 4.0

CODE_LANGUAGE_WEIGHTS = {
    "python": 0.40,
    "javascript": 0.15,
    "java": 0.15,
    "cpp": 0.10,
    "go": 0.10,
    "rust": 0.05,
    "shell": 0.05,
}

MIN_CHARS = 50
MAX_CHARS = 100_000  # drop pathological single-file blobs (minified bundles, generated code, data dumps)


def stream_code(language: str, target_tokens: int, output_path: Path) -> int:
    dataset = load_dataset("bigcode/starcoderdata", data_dir=language, split="train", streaming=True)
    written_tokens = 0
    written_rows = 0
    with output_path.open("a", encoding="utf-8") as out:
        for row in dataset:
            if written_tokens >= target_tokens:
                break
            text = row.get("content", "")
            n = len(text)
            if n < MIN_CHARS or n > MAX_CHARS:
                continue
            out.write(json.dumps({"text": text, "domain": f"code_{language}"}, ensure_ascii=False) + "\n")
            written_tokens += int(n / CHARS_PER_TOKEN_ESTIMATE)
            written_rows += 1
    return written_tokens, written_rows


def stream_fineweb(target_tokens: int, output_path: Path) -> int:
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True)
    written_tokens = 0
    written_rows = 0
    with output_path.open("a", encoding="utf-8") as out:
        for row in dataset:
            if written_tokens >= target_tokens:
                break
            text = row.get("text", "")
            if len(text) < MIN_CHARS:
                continue
            out.write(json.dumps({"text": text, "domain": "docs_nl"}, ensure_ascii=False) + "\n")
            written_tokens += int(row.get("token_count") or len(text) / CHARS_PER_TOKEN_ESTIMATE)
            written_rows += 1
    return written_tokens, written_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-tokens", type=int, default=2_500_000_000)
    parser.add_argument("--code-fraction", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=Path("data/raw/quantweave-code-corpus.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("")  # truncate/start fresh

    code_budget = int(args.total_tokens * args.code_fraction)
    nl_budget = args.total_tokens - code_budget

    print(f"target: {args.total_tokens:,} tokens ({code_budget:,} code / {nl_budget:,} natural language)")

    grand_total_tokens = 0
    grand_total_rows = 0
    t0 = time.time()
    for language, weight in CODE_LANGUAGE_WEIGHTS.items():
        lang_target = int(code_budget * weight)
        tokens, rows = stream_code(language, lang_target, args.output)
        grand_total_tokens += tokens
        grand_total_rows += rows
        elapsed = time.time() - t0
        print(f"[{elapsed:6.0f}s] {language:12} target={lang_target:>12,} got={tokens:>12,} tok  "
              f"({rows:>8,} files)  running_total={grand_total_tokens:,}")

    tokens, rows = stream_fineweb(nl_budget, args.output)
    grand_total_tokens += tokens
    grand_total_rows += rows
    elapsed = time.time() - t0
    print(f"[{elapsed:6.0f}s] {'fineweb-edu':12} target={nl_budget:>12,} got={tokens:>12,} tok  "
          f"({rows:>8,} docs)  running_total={grand_total_tokens:,}")

    print(f"\nwrote {grand_total_rows:,} rows, ~{grand_total_tokens:,} tokens (char-based estimate for code) "
          f"to {args.output} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
