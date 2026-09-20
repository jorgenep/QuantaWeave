#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(/usr/bin/dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

.venv/bin/python src/model_scaling.py \
  --mode moe \
  --size xs \
  --hidden-size 64 \
  --layers 2 \
  --vocab-size 7168 \
  --ffn-multiplier 2 \
  --total-experts 184 \
  --active-experts 1 \
  --no-tied-embeddings \
  --output data/smoke/quantweave-10m-a1m.json

.venv/bin/python src/prepare_smoke_dataset.py \
  --examples "${SMOKE_EXAMPLES:-100000}" \
  --output data/smoke/tinystories.jsonl

python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data/smoke/quantweave-10m-a1m.json").read_text())
assert manifest["mode"] == "moe"
assert manifest["num_experts"] == 184
assert manifest["top_k"] == 1
assert manifest["inactive_experts_per_token"] == 183
assert 9_900_000 <= manifest["estimated_total_params"] <= 10_100_000
assert 900_000 <= manifest["estimated_active_params"] <= 1_100_000

with Path("data/smoke/tinystories.jsonl").open() as dataset:
    first = json.loads(next(dataset))
assert first["text"]
print("QuantaWeave 10M/A1M smoke test passed")
PY
