"""Train a tokenizer and/or pack JSONL text into a memory-mapped token corpus.

  python src/prepare_tokens.py bpe  --data a.jsonl --vocab-size 4096 --output artifacts/tokenizers/bpe4k
  python src/prepare_tokens.py sentencepiece --data a.jsonl --vocab-size 4096 --output artifacts/tokenizers/sp4k
  python src/prepare_tokens.py pack --data a.jsonl b.jsonl --tokenizer artifacts/tokenizers/bpe4k \\
        --output data/tokens/mixed
  python src/prepare_tokens.py pack --data a.jsonl --tokenizer char --vocab-size 7168 --output data/tokens/chars

Rows may carry a "domain" field; otherwise the domain is "default" (one file) or the file stem.
"""

import argparse
import json
from pathlib import Path

from data_pipeline import CharTokenizer, BPETokenizer, SentencePieceTokenizer, load_tokenizer, read_rows, write_token_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    bpe = commands.add_parser("bpe", help="train a byte-level BPE tokenizer")
    bpe.add_argument("--data", type=Path, nargs="+", required=True)
    bpe.add_argument("--vocab-size", type=int, default=4096)
    bpe.add_argument("--examples", type=int, help="max JSONL rows read per file")
    bpe.add_argument("--output", type=Path, required=True)

    sp = commands.add_parser("sentencepiece", help="train a sentencepiece (unigram or BPE) tokenizer")
    sp.add_argument("--data", type=Path, nargs="+", required=True)
    sp.add_argument("--vocab-size", type=int, default=4096)
    sp.add_argument("--algorithm", choices=("unigram", "bpe"), default="unigram")
    sp.add_argument("--examples", type=int, help="max JSONL rows read per file")
    sp.add_argument("--output", type=Path, required=True)

    pack = commands.add_parser("pack", help="tokenize JSONL into tokens.bin + meta.json")
    pack.add_argument("--data", type=Path, nargs="+", required=True)
    pack.add_argument("--tokenizer", default="char", help="'char' or a directory saved by the bpe command")
    pack.add_argument("--vocab-size", type=int, default=7168, help="character vocabulary size (char only)")
    pack.add_argument("--examples", type=int, help="max JSONL rows read per file")
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    texts = lambda: (text for _, text in read_rows(args.data, args.examples))
    if args.command == "bpe":
        tokenizer = BPETokenizer.train(texts(), args.vocab_size)
        tokenizer.save(args.output)
        print(f"trained BPE tokenizer with {tokenizer.vocab_size} tokens -> {args.output}")
        return
    if args.command == "sentencepiece":
        tokenizer = SentencePieceTokenizer.train(texts(), args.vocab_size, args.algorithm)
        tokenizer.save(args.output)
        print(f"trained sentencepiece ({args.algorithm}) tokenizer with {tokenizer.vocab_size} tokens -> {args.output}")
        return

    tokenizer = CharTokenizer.build(texts(), args.vocab_size) if args.tokenizer == "char" else load_tokenizer(Path(args.tokenizer))
    meta = write_token_bin(args.data, tokenizer, args.output, args.examples)
    print(json.dumps({key: meta[key] for key in ("tokenizer", "vocab_size", "dtype", "total_tokens")}))
    for segment in meta["segments"]:
        print(f"  domain {segment['domain']}: {segment['length']:,} tokens")


if __name__ == "__main__":
    main()
