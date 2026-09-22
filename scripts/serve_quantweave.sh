#!/usr/bin/env bash
# Serve a checkpoint over HTTP (FastAPI + uvicorn). One model, one process; requests are serialized.
#   ./scripts/serve_quantweave.sh                                  # newest archived run
#   MOE_CHECKPOINT=artifacts/outputs/quantweave-moe-out MOE_PORT=8080 ./scripts/serve_quantweave.sh
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

if [ -n "${MOE_CHECKPOINT:-}" ]; then
  MODEL_ARGS=(--checkpoint "$MOE_CHECKPOINT")
else
  MODEL_ARGS=(--latest)
fi

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/serve_quantweave.py "${MODEL_ARGS[@]}" \
  --device "${MOE_DEVICE:-auto}" --host "${MOE_HOST:-127.0.0.1}" --port "${MOE_PORT:-8000}" "$@"
