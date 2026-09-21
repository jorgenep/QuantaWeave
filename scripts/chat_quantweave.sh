#!/usr/bin/env bash
# Send messages to a model: an interactive chat by default, or pass options through.
#   ./scripts/chat_quantweave.sh                                  # newest archived run, interactive
#   ./scripts/chat_quantweave.sh -m "Once upon a time" --temperature 0
#   MOE_CHECKPOINT=artifacts/outputs/quantweave-moe-out ./scripts/chat_quantweave.sh --messages-file smoke.jsonl
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [ -n "${MOE_CHECKPOINT:-}" ]; then
  MODEL_ARGS=(--checkpoint "$MOE_CHECKPOINT")
else
  MODEL_ARGS=(--latest)
fi

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/chat_quantweave_moe.py "${MODEL_ARGS[@]}" --device "${MOE_DEVICE:-auto}" "$@"
