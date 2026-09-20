# QuantaWeave

QuantaWeave is a configurable language-model research framework for experimenting with dense and sparse Mixture-of-Experts (MoE) causal language models. It is designed to run small local experiments and scale toward distributed training on NVIDIA CUDA, AMD ROCm, or Intel XPU hardware.

The repository contains two training paths:

- **Standalone QuantaWeave MoE:** the portable implementation in `src/` with explicit top-k routing, expert balancing, resumable checkpoints, and benchmarking.
- **Axolotl integration:** a separate dense-model baseline for validating a larger production training stack. It does not create the custom QuantaWeave MoE architecture.

## How The Framework Works

QuantaWeave models contain shared transformer parameters plus a pool of expert feed-forward networks.

For a dense model, every token uses the same feed-forward network:

```text
parameters stored = parameters active for each token
```

For a sparse MoE model, a router scores every expert and selects `top_k` experts for each token:

```text
parameters stored = shared parameters + all expert parameters
parameters active/token = shared parameters + selected expert parameters
```

For example, with `64` total experts and `2` active experts, every token uses `2/64` of the expert pool while all 64 experts remain stored in the checkpoint. The router auxiliary loss encourages tokens to be distributed across experts instead of collapsing onto a few of them.

The current standalone model includes:

- Decoder-only causal language modeling
- Multi-head causal self-attention
- SwiGLU-style expert feed-forward networks
- Configurable total experts and active experts
- Top-k sparse routing
- Configurable expert capacity and overflow handling
- Router load-balancing loss
- CUDA, ROCm, XPU, and CPU device selection
- Periodic atomic checkpoints with optimizer and RNG state
- Evaluation loss, perplexity, throughput, and expert-utilization benchmarks

## Repository Layout

```text
src/
  quantweave_moe_model.py       Model and sparse router
  train_quantweave_moe.py       Standalone trainer and checkpoint recovery
  benchmark_quantweave_moe.py   Checkpoint benchmark tool
  model_scaling.py              Architecture calculator and manifest builder
  prepare_smoke_dataset.py       Bounded TinyStories downloader
  data.py                        Large production dataset compiler
  pack_dataset.py                Production dataset packer

scripts/
  run_quantweave_moe.sh       Standalone MoE training launcher
  benchmark_quantweave.sh               Benchmark launcher
  smoke_quantweave.sh                  Bounded data and manifest smoke test
  eval_pipeline.sh              Axolotl checkpoint evaluation pipeline

configs/
  architecture_presets.json     xs through xxl architecture presets
  quantweave_moe.yaml           Axolotl dense baseline configuration
  quantweave_smoke.yaml         Small Axolotl LoRA integration test
  deepspeed_zero3.json          DeepSpeed ZeRO-3 settings

data/
  raw/                          Large JSONL source data
  smoke/                        Small local test data and manifests
  packed/                       Production packed training data

artifacts/
  checkpoints/                  Resumable intermediate checkpoints
  outputs/                      Final models, adapters, and reports
  prepared/                     Axolotl prepared datasets

axolotl/                         Existing Axolotl checkout
.venv/                           Lightweight data/smoke environment
.axolotl-venv/                   Python 3.12 Axolotl environment
```

## Environments

Python 3.12 is recommended for the Axolotl environment. The pinned native dependencies can fail to build on Python 3.14.

Create the lightweight environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python datasets transformers tqdm huggingface_hub
```

Create the Axolotl environment from the existing checkout:

```bash
uv venv --python 3.12 .axolotl-venv
uv pip install --python .axolotl-venv/bin/python -e './axolotl[deepspeed]'
```

The standalone trainer does not require Axolotl. Use `.axolotl-venv` because it already contains the compatible PyTorch stack used by the project.

## Device Backends

The same standalone code supports multiple PyTorch device backends:

```text
cuda  NVIDIA CUDA
rocm  AMD ROCm, exposed through PyTorch's CUDA API
xpu   Intel GPU through Intel's XPU-enabled PyTorch build
cpu   CPU fallback and development testing
```

Check the backend on each machine:

```bash
# NVIDIA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# AMD ROCm
python -c "import torch; print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Intel
python -c "import torch; print(hasattr(torch, 'xpu')); print(torch.xpu.is_available())"
```

Use the matching vendor-specific PyTorch build and driver/runtime. Do not reuse a CUDA-only virtual environment on an Intel machine. `MOE_DEVICE=auto` selects CUDA/ROCm first, then XPU, then CPU.

## Configure An Architecture

The architecture builder supports dense and MoE modes with `xs`, `s`, `m`, `l`, `xl`, and `xxl` presets. Every value can be overridden for a custom experiment.

```bash
# Dense preset
python3 src/model_scaling.py \
  --mode dense \
  --size xs \
  --output artifacts/outputs/dense-xs.json

# MoE preset
python3 src/model_scaling.py \
  --mode moe \
  --size l \
  --output artifacts/outputs/moe-l.json

# Fully custom MoE dimensions
python3 src/model_scaling.py \
  --mode moe \
  --size m \
  --hidden-size 1536 \
  --layers 24 \
  --total-experts 64 \
  --active-experts 2 \
  --ffn-multiplier 2.75 \
  --output artifacts/outputs/custom-moe.json
```

Important controls:

- `--hidden-size`: transformer representation width
- `--layers`: number of transformer blocks
- `--ffn-multiplier`: expert feed-forward width relative to hidden size
- `--total-experts`: number of expert FFNs stored per MoE block
- `--active-experts`: number of experts selected per token
- `--vocab-size`: vocabulary capacity
- `--tied-embeddings`: share input and output embeddings
- `--capacity-factor`: target routed capacity per expert; `0` disables the limit
- `--min-expert-capacity`: minimum token slots per expert
- `--overflow-policy`: `drop` to enforce capacity or `residual` to keep overflow on the block residual path

The calculator estimates parameter counts. The final model implementation is the authority for exact counts because attention projections, biases, embeddings, and auxiliary layers affect totals.

## Run The Standalone MoE

The standard local experiment uses a compact character vocabulary and TinyStories data. Run the default 10M-total/1M-active profile:

```bash
MOE_DEVICE=auto ./scripts/run_quantweave_moe.sh
```

A longer experiment with 64 total experts and 2 active experts per token:

```bash
MOE_DEVICE=auto \
MOE_STEPS=10000 \
MOE_EXAMPLES=100000 \
MOE_SEQUENCE_LENGTH=128 \
MOE_BATCH_SIZE=2 \
MOE_TOTAL_EXPERTS=64 \
MOE_ACTIVE_EXPERTS=2 \
./scripts/run_quantweave_moe.sh
```

The launcher accepts these environment variables:

```text
MOE_DEVICE              auto, cuda, rocm, xpu, or cpu
MOE_STEPS               target training step
MOE_EXAMPLES            maximum JSONL examples to load
MOE_SEQUENCE_LENGTH     sequence length excluding the shifted label token
MOE_BATCH_SIZE          micro-batch size
MOE_TOTAL_EXPERTS       stored expert count
MOE_ACTIVE_EXPERTS      routed experts per token
MOE_CAPACITY_FACTOR     expert capacity multiplier; 0 disables overflow limits
MOE_MIN_EXPERT_CAPACITY minimum routed slots per expert
MOE_CHECKPOINT_DIR      recovery checkpoint directory
MOE_CHECKPOINT_INTERVAL save frequency in steps
MOE_OUTPUT              final output directory; use this for parallel experiments
```

## Checkpoint Recovery

The trainer saves an atomic recovery checkpoint containing model weights, optimizer state, the current step, and random-number-generator state. Default locations are:

```text
artifacts/checkpoints/quantweave-moe-checkpoint/
artifacts/outputs/quantweave-moe-out/
```

Run a long job with periodic recovery:

```bash
MOE_STEPS=1000000 \
MOE_EXAMPLES=100000 \
MOE_CHECKPOINT_INTERVAL=1000 \
./scripts/run_quantweave_moe.sh
```

If the process stops, rerun the same command. It resumes from the newest completed checkpoint. To start a separate run:

```bash
MOE_CHECKPOINT_DIR=artifacts/checkpoints/experiment-02 \
MOE_STEPS=10000 \
./scripts/run_quantweave_moe.sh
```

## Benchmark A Checkpoint

Benchmark a completed model with the default evaluation slice:

```bash
./scripts/benchmark_quantweave.sh
```

Use more examples or a specific backend/checkpoint:

```bash
MOE_CHECKPOINT=artifacts/outputs/quantweave-moe-out \
MOE_BENCHMARK_EXAMPLES=10000 \
MOE_DEVICE=cuda \
./scripts/benchmark_quantweave.sh
```

The report is saved as `benchmark-report.json` by default and includes:

- Evaluation loss and perplexity
- Evaluation throughput in tokens per second
- Total parameter count
- Estimated active parameters per token
- Total, active, and inactive expert counts
- Router auxiliary loss
- Dropped-route count and fraction from expert capacity limits
- Minimum, maximum, and mean expert utilization
- Training step recorded in the checkpoint

For comparisons, keep the data slice, batch size, sequence length, and device consistent.

## Teacher-Data Distillation

Distillation lets a stronger teacher improve an existing QuantaWeave student by training the student on teacher-produced answers. The current implementation is sequence-level distillation: it learns from teacher text in a JSONL file while preserving the student's tokenizer and MoE architecture.

Teacher data can use any of these records:

```json
{"teacher_text": "A carefully written teacher answer."}
{"prompt": "Explain routing.", "completion": "A router selects..."}
{"prompt": "Write a story.", "response": "Once upon a time..."}
{"text": "Already-generated teacher text."}
```

Run distillation on NVIDIA, AMD, or Intel:

```bash
DISTILL_TEACHER_DATA=data/teacher/teacher_answers.jsonl \
MOE_PYTHON=.intel-venv/bin/python \
MOE_DEVICE=xpu \
DISTILL_STEPS=10000 \
DISTILL_EXAMPLES=100000 \
./scripts/distill_quantweave.sh
```

The distilled model is written to `artifacts/outputs/quantweave-moe-distilled/`. Distillation also saves recovery checkpoints under `artifacts/checkpoints/quantweave-distill/` and resumes automatically when rerun.

Benchmark the distilled student:

```bash
MOE_CHECKPOINT=artifacts/outputs/quantweave-moe-distilled \
MOE_BENCHMARK_DATA=data/teacher/teacher_answers.jsonl \
MOE_DEVICE=xpu \
./scripts/benchmark_quantweave.sh
```

Compare the original and distilled students using the same benchmark command and evaluation slice. Lower loss/perplexity is better; router auxiliary loss should remain finite and expert utilization should not collapse.

This path uses teacher-generated text rather than teacher logits. True logit-level distillation would require aligned teacher/student tokenizers and a teacher forward pass during training.

## Generate A Showcase Sample

After benchmarking, generate text from the saved checkpoint:

```bash
MOE_DEVICE=xpu \
.intel-venv/bin/python src/generate_quantweave_moe.py \
  --checkpoint artifacts/outputs/quantweave-moe-out \
  --prompt "Once upon a time" \
  --tokens 300 \
  --temperature 0.8
```

The generator uses the checkpoint vocabulary and model configuration, so it is suitable for a reproducible demo after each training run.

## Small Smoke Test

The bounded smoke test downloads at most 100,000 TinyStories examples and writes only under `data/smoke/`:

```bash
./scripts/smoke_quantweave.sh
```

Use fewer examples for a fast check:

```bash
SMOKE_EXAMPLES=1000 ./scripts/smoke_quantweave.sh
```

This verifies the architecture manifest and dataset preparation. It does not train the standalone MoE; use `scripts/run_quantweave_moe.sh` for that.

## Large Dataset Pipeline

The production data tools are intentionally separate from the local MoE smoke path. They are designed for a large code/reasoning mixture and require substantial storage, memory, network bandwidth, and compute.

Compile raw data:

```bash
python3 src/data.py
```

Pack the compiled data:

```bash
python3 src/pack_dataset.py
```

The resulting data is written to `data/packed/packed_100B_dataset/`. Do not run these commands for a local smoke test.

## Axolotl Baseline

The Axolotl configuration is a dense baseline and integration path. It is not the standalone sparse QuantaWeave MoE.

```bash
source .axolotl-venv/bin/activate
accelerate launch -m axolotl.cli.train configs/quantweave_moe.yaml
```

DeepSpeed ZeRO-3 partitions memory but does not create expert routing. FlashAttention, bitsandbytes, Triton, and some DeepSpeed operators may be vendor-specific. Use the standalone trainer when cross-vendor CUDA/ROCm/XPU portability is the priority.

## Current Scope

The standalone implementation is a research and scaling framework, not a production trillion-parameter training system. A real large-scale deployment would additionally require distributed expert parallelism, tensor or pipeline parallelism, sharded checkpoints, efficient fused routing kernels, capacity management, stronger tokenization, validation splits, and production-grade monitoring.
