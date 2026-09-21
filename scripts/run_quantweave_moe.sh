#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/train_quantweave_moe.py \
  --data data/smoke/tinystories.jsonl \
  --output "${MOE_OUTPUT:-artifacts/outputs/quantweave-moe-out}" \
  --steps "${MOE_STEPS:-10}" \
  --examples "${MOE_EXAMPLES:-10000}" \
  --sequence-length "${MOE_SEQUENCE_LENGTH:-128}" \
  --batch-size "${MOE_BATCH_SIZE:-2}" \
  --total-experts "${MOE_TOTAL_EXPERTS:-184}" \
  --active-experts "${MOE_ACTIVE_EXPERTS:-1}" \
  --gradient-accumulation-steps "${MOE_GRADIENT_ACCUMULATION_STEPS:-1}" \
  --capacity-factor "${MOE_CAPACITY_FACTOR:-1.25}" \
  --min-expert-capacity "${MOE_MIN_EXPERT_CAPACITY:-4}" \
  --drop-overflow-tokens \
  --overflow-policy "${MOE_OVERFLOW_POLICY:-drop}" \
  --device "${MOE_DEVICE:-auto}" \
  --checkpoint-dir "${MOE_CHECKPOINT_DIR:-artifacts/checkpoints/quantweave-moe-checkpoint}" \
  --checkpoint-interval "${MOE_CHECKPOINT_INTERVAL:-1000}" \
  --resume \
  ${MOE_EXTRA_ARGS:-}
