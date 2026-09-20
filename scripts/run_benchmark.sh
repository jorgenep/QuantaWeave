#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/benchmark_quantweave_moe.py \
  --checkpoint "${MOE_CHECKPOINT:-artifacts/outputs/quantweave-moe-out}" \
  --data "${MOE_BENCHMARK_DATA:-data/smoke/tinystories.jsonl}" \
  --examples "${MOE_BENCHMARK_EXAMPLES:-1000}" \
  --batch-size "${MOE_BENCHMARK_BATCH_SIZE:-2}" \
  --device "${MOE_DEVICE:-auto}" \
  --output "${MOE_BENCHMARK_REPORT:-benchmark-report.json}"
