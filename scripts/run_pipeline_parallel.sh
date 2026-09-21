#!/usr/bin/env bash
# Pipeline-parallel training: layers split across ranks, micro-batches flow through them (GPipe).
#   MOE_GPUS=2 MOE_LAYERS=8 MOE_MICROBATCHES=4 MOE_BATCH_SIZE=8 ./scripts/run_pipeline_parallel.sh
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_TORCHRUN:-.axolotl-venv/bin/torchrun}" --nproc_per_node "${MOE_GPUS:-2}" src/train_quantweave_moe.py \
  --pipeline-parallel \
  --microbatches "${MOE_MICROBATCHES:-2}" \
  --data "${MOE_DATA:-data/smoke/tinystories.jsonl}" \
  --output "${MOE_OUTPUT:-artifacts/outputs/quantweave-moe-pp-out}" \
  --steps "${MOE_STEPS:-10}" \
  --examples "${MOE_EXAMPLES:-10000}" \
  --sequence-length "${MOE_SEQUENCE_LENGTH:-128}" \
  --batch-size "${MOE_BATCH_SIZE:-4}" \
  --layers "${MOE_LAYERS:-4}" \
  --total-experts "${MOE_TOTAL_EXPERTS:-16}" \
  --active-experts "${MOE_ACTIVE_EXPERTS:-2}" \
  --device "${MOE_DEVICE:-auto}" \
  --checkpoint-dir "${MOE_CHECKPOINT_DIR:-artifacts/checkpoints/quantweave-moe-pp-checkpoint}" \
  --checkpoint-interval "${MOE_CHECKPOINT_INTERVAL:-1000}" \
  ${MOE_EXTRA_ARGS:-}
