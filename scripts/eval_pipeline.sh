#!/usr/bin/env bash
# Usage: ./eval_pipeline.sh ./quantweave-out/checkpoint-2000

set -euo pipefail

CHECKPOINT_DIR=${1:-}

if [ -z "$CHECKPOINT_DIR" ]; then
  echo "Error: Please provide the path to the checkpoint folder."
  exit 1
fi

echo "======================================"
echo " Phase 1: Quantitative Benchmarking   "
echo "======================================"
# Requires: pip install lm-eval
# Runs HumanEval to get the Python pass@1 score
lm_eval --model hf \
    --model_args "pretrained=$CHECKPOINT_DIR" \
    --tasks humaneval \
    --batch_size 8 \
    --output_path "./quantweave_eval_results_${CHECKPOINT_DIR##*-}"

echo "======================================"
echo " Phase 2: Local Vibe Check Conversion "
echo "======================================"
# Requires llama.cpp cloned in the same directory
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements.txt
fi

# Convert the raw Hugging Face tensors to a Q8 GGUF for local inference
GGUF_OUT="${CHECKPOINT_DIR}/quantweave-30b-eval.gguf"
python3 llama.cpp/convert_hf_to_gguf.py "$CHECKPOINT_DIR" \
  --outfile "$GGUF_OUT" \
    --outtype q8_0

echo "======================================"
echo " Phase 3: Engine Registration         "
echo "======================================"
# Creates a Modelfile and registers it so it immediately appears in your IDE extensions
MODEL_NAME="quantweave-30b-test-${CHECKPOINT_DIR##*-}"

echo "FROM $GGUF_OUT" > Modelfile.eval
echo "TEMPLATE \"\"\"{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n\"\"\"" >> Modelfile.eval
echo "PARAMETER stop \"<|im_end|>\"" >> Modelfile.eval

ollama create $MODEL_NAME -f Modelfile.eval
rm Modelfile.eval

echo "Done! QuantaWeave-30B is scored and ready for local testing."
echo "You can now select '$MODEL_NAME' in Continue/Aider to test its code generation capabilities."