#!/usr/bin/env bash
# Export a checkpoint as an inference bundle (TorchScript + optional int8/int4 + tokenizer + metadata).
#   MOE_QUANTIZE=8 ./scripts/export_quantweave.sh
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

"${MOE_PYTHON:-.axolotl-venv/bin/python}" src/export_quantweave.py \
  --checkpoint "${MOE_CHECKPOINT:-artifacts/outputs/quantweave-moe-out}" \
  --output "${MOE_EXPORT_DIR:-artifacts/exports/quantweave-moe}" \
  ${MOE_QUANTIZE:+--quantize "$MOE_QUANTIZE"} \
  ${MOE_EXPORT_ONNX:+--onnx} \
  ${MOE_BENCHMARK_DATA:+--benchmark-data "$MOE_BENCHMARK_DATA"}
