#!/usr/bin/env bash
# Multi-process training: experts sharded across ranks, optionally with tensor parallelism and a sharded optimizer.
#   MOE_GPUS=4 MOE_TENSOR_PARALLEL=2 MOE_TOTAL_EXPERTS=16 MOE_EXTRA_ARGS="--shard-optimizer --straggler-routing" ./scripts/run_expert_parallel.sh
# MOE_GPUS is the total number of ranks (= expert-parallel size x MOE_TENSOR_PARALLEL).
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_TORCHRUN:-.axolotl-venv/bin/torchrun}" --nproc_per_node "${MOE_GPUS:-2}" src/train_quantweave_moe.py \
  --expert-parallel \
  --tensor-parallel "${MOE_TENSOR_PARALLEL:-1}" \
  --data "${MOE_DATA:-data/smoke/tinystories.jsonl}" \
  --output "${MOE_OUTPUT:-artifacts/outputs/quantweave-moe-ep-out}" \
  --steps "${MOE_STEPS:-10}" \
  --examples "${MOE_EXAMPLES:-10000}" \
  --sequence-length "${MOE_SEQUENCE_LENGTH:-128}" \
  --batch-size "${MOE_BATCH_SIZE:-2}" \
  --total-experts "${MOE_TOTAL_EXPERTS:-16}" \
  --active-experts "${MOE_ACTIVE_EXPERTS:-2}" \
  --device "${MOE_DEVICE:-auto}" \
  --checkpoint-dir "${MOE_CHECKPOINT_DIR:-artifacts/checkpoints/quantweave-moe-ep-checkpoint}" \
  --checkpoint-interval "${MOE_CHECKPOINT_INTERVAL:-1000}" \
  ${MOE_EXTRA_ARGS:-}
