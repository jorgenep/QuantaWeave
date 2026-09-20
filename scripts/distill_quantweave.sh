#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/distill_quantweave_moe.py \
  --student "${DISTILL_STUDENT:-artifacts/outputs/quantweave-moe-out}" \
  --teacher-data "${DISTILL_TEACHER_DATA:?Set DISTILL_TEACHER_DATA to a teacher-generated JSONL file}" \
  --output "${DISTILL_OUTPUT:-artifacts/outputs/quantweave-moe-distilled}" \
  --checkpoint-dir "${DISTILL_CHECKPOINT_DIR:-artifacts/checkpoints/quantweave-distill}" \
  --steps "${DISTILL_STEPS:-10000}" \
  --examples "${DISTILL_EXAMPLES:-100000}" \
  --sequence-length "${DISTILL_SEQUENCE_LENGTH:-128}" \
  --batch-size "${DISTILL_BATCH_SIZE:-2}" \
  --checkpoint-interval "${DISTILL_CHECKPOINT_INTERVAL:-1000}" \
  --lr "${DISTILL_LR:-0.0001}" \
  --device "${MOE_DEVICE:-auto}" \
  --resume
