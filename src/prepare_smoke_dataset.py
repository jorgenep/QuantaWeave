"""Download a bounded TinyStories sample for local smoke tests."""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=Path("data/smoke/tinystories.jsonl"))
    args = parser.parse_args()

    if args.examples < 1:
        parser.error("--examples must be greater than zero")

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        written = 0
        for row in dataset:
            if written >= args.examples:
                break
            text = row.get("text", "").strip()
            if text:
                output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} examples to {args.output}")


if __name__ == "__main__":
    main()
