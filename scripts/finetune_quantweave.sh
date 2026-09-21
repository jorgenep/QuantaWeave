#!/usr/bin/env bash
# QLoRA-style fine-tuning: quantize the experts, freeze them, train low-rank adapters.
#   FT_DATA=new_domain.jsonl FT_BITS=4 ./scripts/finetune_quantweave.sh
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/finetune_quantweave_moe.py \
  --checkpoint "${FT_CHECKPOINT:-artifacts/outputs/quantweave-moe-out}" \
  --data "${FT_DATA:?Set FT_DATA to a JSONL file}" \
  --output "${FT_OUTPUT:-artifacts/outputs/quantweave-moe-qlora}" \
  --bits "${FT_BITS:-4}" \
  --rank "${FT_RANK:-8}" \
  --steps "${FT_STEPS:-200}" \
  --optimizer "${FT_OPTIMIZER:-adamw}" \
  --device "${MOE_DEVICE:-auto}" \
  ${FT_MERGE_OUTPUT:+--merge-output "$FT_MERGE_OUTPUT"} \
  ${FT_EXTRA_ARGS:-}
