# QuantaWeave

QuantaWeave is a configurable language-model research framework for experimenting with dense and sparse Mixture-of-Experts (MoE) causal language models. It is designed to run small local experiments and scale toward distributed training on NVIDIA CUDA, AMD ROCm, or Intel XPU hardware.

The repository contains two training paths:

- **Standalone QuantaWeave MoE:** the portable implementation in `src/` with explicit top-k routing, expert balancing, resumable checkpoints, and benchmarking.
- **Axolotl integration:** a separate dense-model baseline for validating a larger production training stack. It does not create the custom QuantaWeave MoE architecture.

**Picking training flags and sizing a run to your GPU?** See **[PARAMETERS.md](PARAMETERS.md)** — every training flag explained, plus how to estimate VRAM usage before you run so you don't hit `CUDA out of memory`.

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
- Router load-balancing loss, with schedules and feedback controllers for LR, capacity, aux weight and router temperature
- Routing diagnostics, curriculum and multi-domain sampling, BPE tokenizers and packed corpora
- Expert parallelism, mixed precision, activation checkpointing, int8/int4 inference, TorchScript export
- CUDA, ROCm, XPU, and CPU device selection
- Periodic atomic checkpoints with optimizer, RNG, schedule and data-position state, and exact resume
- Evaluation loss, perplexity, throughput, and expert-utilization benchmarks

## Repository Layout

```text
src/
  quantweave_moe_model.py       Model, sparse router, dispatch, runtime routing controls
  train_quantweave_moe.py       Trainer (run_training), checkpoints, resume validation
  moe_schedules.py              LR / aux-weight / capacity / router-temperature schedules and controllers
  routing_diagnostics.py        Router monitor, JSONL log, SVG heatmaps
  training_metrics.py           Metrics log, plateau and stability detection
  data_pipeline.py              Tokenizers, token corpora, memmap, curriculum, domain sampling, BatchStream
  prepare_tokens.py             Train a BPE tokenizer / pack a memory-mapped token corpus
  hardware.py                   Device detection, precision choice, safe batch-size probe
  expert_parallel.py            Expert + tensor parallelism, straggler-aware routing, sharded checkpoints
  tensor_parallel.py            Megatron-style attention/FFN slicing and collectives
  pipeline_parallel.py          Pipeline stages, GPipe micro-batch schedule
  sharded_optimizer.py          ZeRO-style optimizer-state sharding
  moe_kernels.py                Triton grouped-GEMM expert kernels
  lora.py, finetune_quantweave_moe.py   QLoRA-style fine-tuning of quantized experts
  cross_tokenizer.py            Distillation from teachers with a different tokenizer
  bayes_opt.py                  Gaussian-process Bayesian optimisation
  distill_quantweave_moe.py     Sequence-level and logit-level distillation
  benchmark_quantweave_moe.py   Checkpoint benchmark (JSON + Markdown)
  chat_quantweave_moe.py        Send messages: interactive chat, one-off, or batch test messages
  fast_decode.py                KV-cache / CUDA-graph decoder behind the chat tool
  generate_quantweave_moe.py    Text sampling (simple, one prompt)
  quantization.py               Weight-only int8 / int4 for inference
  run_bundle.py                 Per-run archive folders named by epoch time
  export_quantweave.py          Inference bundle: TorchScript, int8/int4, tokenizer, metadata
  experiment_manager.py         Tracked runs, comparison, best-run pointer
  sweep.py                      Grid / random / successive-halving search
  model_scaling.py              Architecture calculator and manifest builder
  prepare_smoke_dataset.py      Bounded TinyStories downloader
  data.py, pack_dataset.py      Large production dataset compiler and packer (Axolotl path)

scripts/
  run_quantweave_moe.sh         Standalone MoE training launcher (MOE_EXTRA_ARGS passes any trainer option)
  run_expert_parallel.sh        torchrun launcher: expert (+ tensor) parallelism
  run_pipeline_parallel.sh      torchrun launcher: pipeline parallelism
  finetune_quantweave.sh        QLoRA fine-tuning launcher
  benchmark_quantweave.sh       Benchmark launcher
  chat_quantweave.sh            Chat / send messages launcher
  distill_quantweave.sh         Distillation launcher
  export_quantweave.sh          Export launcher
  smoke_quantweave.sh           Bounded data and manifest smoke test
  eval_pipeline.sh              Axolotl checkpoint evaluation pipeline

configs/
  architecture_presets.json     xs through xxl architecture presets
  exp_moe_balanced.yaml         Example experiment-manager config
  sweep_example.yaml            Example successive-halving sweep
  sweep_bayes.yaml              Example Bayesian sweep
  quantweave_moe.yaml           Axolotl dense baseline configuration
  quantweave_smoke.yaml         Small Axolotl LoRA integration test
  deepspeed_zero3.json          DeepSpeed ZeRO-3 settings

data/
  raw/                          Large JSONL source data
  smoke/                        Small local test data and manifests
  tokens/                       Packed memory-mapped token corpora
  packed/                       Production packed training data

artifacts/
  checkpoints/                  Resumable intermediate checkpoints
  outputs/                      Final models, adapters, and reports
  runs/                         One folder per finished run, named by epoch time (see Run Archives)
  experiments/                  Tracked experiment runs
  exports/                      Inference bundles
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
- `--overflow-policy`: `drop` or `residual`. Both enforce capacity the same way: each expert keeps its highest-weight routes, and a route that does not fit is skipped, so the token keeps its residual stream unchanged for that route. They differ only in reporting: `drop` counts skipped routes in `dropped_routes`, `residual` does not (both report them in `overflow_routes`)
- `--no-drop-overflow-tokens`: disable capacity enforcement entirely

The calculator estimates parameter counts. The final model implementation is the authority for exact counts because attention projections, biases, embeddings, and auxiliary layers affect totals.

## Run The Standalone MoE

The standard local experiment uses a compact character vocabulary and TinyStories data. Documents are joined with `<eos>` and cut into non-overlapping windows of `sequence_length + 1` characters, so every character is trained on (`MOE_EXAMPLES` caps the number of JSONL rows read, not the number of windows). On CUDA/XPU the weights and optimizer stay in fp32 and the forward pass runs under bf16 autocast. Run the default 10M-total/1M-active profile:

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
MOE_EXTRA_ARGS          any other trainer option, e.g. "--curriculum rarity --curriculum-steps 500 --capacity-adapt"
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

If the process stops, rerun the same command. It resumes from the newest completed checkpoint. Resuming is exact: the run continues the trajectory an uninterrupted run would have taken, because the data stream is a pure function of `(seed, step)` and every controller's state is restored. Each checkpoint also stores `metadata.json` (step, config hash, data position, curriculum stage, schedule/controller state, current routing controls). A checkpoint made under a different run fingerprint (model shape, batch size, sequence length, seed, data, tokenizer, curriculum, domain weights, world size) is refused unless you pass `--allow-config-change`. To start a separate run:

```bash
MOE_CHECKPOINT_DIR=artifacts/checkpoints/experiment-02 \
MOE_STEPS=10000 \
./scripts/run_quantweave_moe.sh
```

## Training Schedules And Adaptive Control

`--lr-decay {constant,cosine,linear}` with `--warmup-steps`, `--min-lr-ratio`, and `--schedule-steps` (the horizon, default `--steps`). Every `--controller-interval` steps the controllers look at what training measured:

| Flag | What it does |
|---|---|
| `--plateau-patience N` | Halve the LR after N controller intervals without loss improvement |
| `--aux-coef-end X` | Anneal the balance-loss weight from `--router-aux-coef` to X |
| `--aux-adapt` | Raise the balance-loss weight when measured expert-load imbalance is high, relax it when low |
| `--capacity-adapt` (`--capacity-min/-max`, `--drop-threshold`) | Grow the capacity factor 10% when overflow exceeds the threshold, shrink it 1% otherwise |
| `--capacity-release-step N` | Stop enforcing capacity from step N, so no route is skipped late in training |
| `--temperature-start T --temperature-steps N` | Start with a soft router at temperature T and anneal to `--router-temperature` |
| `--temperature-adapt` | Raise the router temperature when too few experts receive traffic |

`--overflow-policy drop|residual` only changes how skipped routes are reported (see above), so it is not scheduled; `--capacity-release-step` is the switch that changes what the model computes. The chosen values are applied every step and logged (`lr`, `capacity_factor`, `router_temperature`, `router_aux_loss_coef`), and the saved `config.json` always records the current routing settings.

## Data Pipeline

**Tokenizers.** `--tokenizer char` (default) or `--tokenizer bpe --tokenizer-path DIR` for a byte-level BPE tokenizer trained from scratch on your data (needs the `tokenizers` package; `--vocab-size` must be at least 258). Checkpoints carry their tokenizer, so benchmark, generate, distill and export work for either kind.

**Packed corpora.** For anything larger than RAM-friendly JSONL, pack once and train from a memory-mapped file:

```bash
python src/prepare_tokens.py bpe  --data a.jsonl b.jsonl --vocab-size 4096 --output artifacts/tokenizers/bpe4k
python src/prepare_tokens.py pack --data a.jsonl b.jsonl --tokenizer artifacts/tokenizers/bpe4k --output data/tokens/mixed
python src/train_quantweave_moe.py --token-data data/tokens/mixed --steps 5000
```

**Domains.** A JSONL row may carry a `"domain"` field (otherwise the domain is `default`, or the file name when several `--data` files are given). Windows never cross a domain boundary. `--domain-weights code=0.7,stories=0.3` sets the sampling mix, `--domain-weights-end` with `--domain-weights-steps` moves it linearly during training, and `--domain-specialization-coef X` adds `X ×` the mean pairwise cosine similarity between the domains' routing profiles to the loss (positive X pushes domains toward different experts; negative rewards sharing).

**Curriculum.** `--curriculum {rarity,entropy,uncommon} --curriculum-steps N` scores every window and starts sampling from the easiest 25% (`--curriculum-start-fraction`), widening linearly to all windows by step N. `rarity` is the mean negative log unigram frequency of the window's tokens, `entropy` the window's token entropy, `uncommon` the share of tokens from the rarest 10% of the vocabulary.

## Routing Diagnostics

`--diagnostics-dir DIR --diagnostics-interval N` records router statistics every N steps (the adaptive controllers turn this on by themselves). Output:

- `routing_log.jsonl`: per layer utilization, requested share, router entropy (and its maximum), top-1 confidence and its histogram, drop rate, drops per expert, dead experts, plus load imbalance and active-expert fraction and which token classes were dropped
- `index.html` with SVG heatmaps: expert usage by layer, per-layer expert utilization over time, expert share per token class (letter/digit/space/punct), expert share per domain, and drop-rate and entropy curves

`--metrics-file metrics.jsonl` writes one JSON object per logged step (loss, LR, routing controls, overflow fraction, gradient norm, tokens/s). The end-of-run summary includes plateau detection, the step training first became stable, and the active/total parameter ratio.

## Precision, Memory And Hardware

For a full walkthrough of every training flag, what it costs in VRAM, and how to size a model to your GPU before running it, see **[PARAMETERS.md](PARAMETERS.md)**.

`--precision {auto,bf16,fp16,fp32}` runs the forward pass in mixed precision over fp32 weights (`auto`: bf16 on accelerators, fp32 on CPU; fp16 uses a gradient scaler). The router always runs in fp32, because a flipped top-k choice changes which expert runs. `--activation-checkpointing` recomputes each block's activations in backward.

```bash
python src/hardware.py                       # device, memory, bf16/fp16 support, runtime
python src/hardware.py --probe --hidden-size 256 --layers 4 --total-experts 16 --sequence-length 128
python src/train_quantweave_moe.py --auto-batch-size ...   # measure, then train with the batch size it recommends
```

**How the probe works.** It runs real training steps (forward, backward, gradient clipping), not an estimate. Each candidate batch size, doubling from 1, runs twice and the second run is timed, so kernel compilation and allocator warm-up are excluded. The optimizer's moments (two fp32 copies of every trainable parameter, for AdamW) are reserved up front, because they occupy memory for the whole of training. A candidate is rejected if it runs out of memory or its peak passes 85% of device memory. The model's weights are never modified. The result has two batch sizes: `batch_size`, the largest that fits, and `recommended_batch_size`, the smallest one reaching 95% of the best measured throughput. Throughput usually flattens or even drops before memory runs out, so the recommended size is the one used; `--auto-batch-target 0` selects the largest that fits instead. `--probe` also tries activation checkpointing and recommends it only when it trains at least 5% faster at its own best batch size. This was validated on a 4060: an earlier version that ignored optimizer state approved batch 128 for an 81M-parameter model, which ran out of memory at the first optimizer step (batch 64 trains, 128 fails, and the probe now reports 64 as the largest that fits and 32 as the recommendation).

`hardware.py` also suggests the largest expert count whose fp32 training state (weights, grads, Adam moments) fits half of device memory. That is deliberately conservative: on the 4060 the 107 experts it suggested for a 6-layer, 256-wide model trained at a 5.2 GB peak (62% of the device), and 150 experts still fit, so the remaining half is left for activations and larger batches.

## Run Archives

Every command-line run of the trainer, `distill_quantweave_moe.py` and `finetune_quantweave_moe.py` ends by saving everything about it in its own folder, `artifacts/runs/<epoch seconds>/`, for example `artifacts/runs/1789960141/` (the second the archive was written; a run finishing in the same second gets a `-1` suffix and never overwrites another):

```text
README.md          headline results, what is in the folder, commands to use the model, and to reproduce it
summary.json       results, timings, warnings (machine-readable)
config.json        every option the run used
reproduce.sh       the command that recreates the run, with every option written out
environment.json   python/torch versions, git revision, device, platform
model/             weights, config, tokenizer, checkpoint metadata (fine-tuning: adapter/ and merged/)
data/              the training data (copied when small) and manifest.json: path, size, SHA-256, rows, rows used
benchmarks/       train-slice.{json,md}, held-out.{json,md}, and the held-out rows used
training/          metrics.jsonl and routing diagnostics (diagnostics/index.html)
samples.txt        a few generations from the finished model
```

The **held-out benchmark uses rows the model never saw**: training reads the first `--examples` rows of each file, so the archive benchmarks the rows after them and reports the generalisation gap (held-out minus training-slice loss). If the file has no rows past `--examples`, only the training-slice benchmark runs and the README says so. Data larger than `--archive-data-limit-mb` (default 200) is not copied; the manifest still records the source path, size, row count and a SHA-256 (over the first and last 16 MB for files above 512 MB, marked `partial`) so the run can be matched to its data. Archiving never fails a run: a benchmark or copy that goes wrong is recorded under `warnings` in `summary.json` and the README. Multi-process runs archive once, from rank 0, using the consolidated model. Runs from a packed `--token-data` corpus are archived without a benchmark (the benchmark reads JSONL); adapter-only fine-tunes likewise unless `--merge-output` is given.

`--no-archive` turns it off, `--archive-dir DIR` puts the folders elsewhere, and `--archive-benchmark-examples` sets the rows per benchmark. Calling `run_training()` from Python does not archive unless you pass `archive_dir`, so sweeps and tests do not fill the disk. The archived `model/` includes the optimizer state (the same checkpoint the run saved), so it can be resumed or fine-tuned as is.

## Multi-Process Parallelism

All of these are launched with `torchrun` and were verified against the single-device model on CPU (Gloo) processes: forward output, loss, gradients and global gradient norm match. NCCL/XCCL multi-GPU runs use the same collectives but have not been exercised on real multi-GPU hardware.

**Expert parallelism** (`--expert-parallel`). Each layer's experts are sharded across ranks; tokens reach the rank that owns their expert through a differentiable all-to-all and come back the same way. The expert count must divide evenly by the expert-parallel size. Replicated tensors (attention, embeddings, routers, norms) have their gradients averaged across ranks; each rank saves its shard and rank 0 consolidates them into an ordinary `model.pt` at the end of the run. Capacity is applied per source rank to its local batch. Gradient clipping uses the true global norm, so ranks clip identically and replicated weights never drift.

**Tensor parallelism** (`--tensor-parallel T`, combinable with expert parallelism). Attention heads and every expert's FFN width are split across T ranks, with one all-reduce per attention block and one per MoE layer. Ranks form an expert-parallel by tensor-parallel grid (world size = both). Heads and FFN width must be divisible by T. Embeddings, norms, routers and lm_head stay replicated (no vocabulary-parallel embedding), and the checkpoint is consolidated back to the ordinary layout.

**Optimizer-state sharding** (`--shard-optimizer`, ZeRO-2 style). Each replicated parameter is owned by one rank: gradients are reduced onto the owner, only the owner holds the AdamW moments and updates the parameter, then owners broadcast the result. With N expert-parallel ranks the moments for replicated parameters take about 1/N of the memory, and training is numerically identical to the unsharded optimizer. Use bf16 (fp16 loss scaling is not supported with multi-process modes).

**Straggler-aware routing** (`--straggler-routing`, plus `--device-metrics` for just the numbers). Every `--straggler-interval` steps each rank's expert time and rows are gathered. An integral controller lowers the routing bias of ranks that are slower or more loaded than the mean and raises it for fast ones; the bias steers which experts are *chosen*, not how much they are trusted. `time_imbalance` and `rows_imbalance` (max over mean across ranks) are logged. `--simulate-slow-rank RANK:SECONDS_PER_ROW` makes one rank artificially slow, which is how it is tested.

**Pipeline parallelism** (`--pipeline-parallel --microbatches M`). Layers are split into contiguous stages (embeddings on the first stage, final norm and lm_head on the last); micro-batches flow through with point-to-point sends on a GPipe schedule (all forwards, then all backwards), with a global gradient norm across stages. It has the usual pipeline bubble and holds M micro-batches of activations per stage (add `--activation-checkpointing` to trade compute for memory). The router balance loss is computed per micro-batch, so with M > 1 it differs slightly from the whole-batch value. Not combinable with expert/tensor parallelism; diagnostics, the domain loss and gradient accumulation are unavailable in this mode.

```bash
MOE_GPUS=4 MOE_TENSOR_PARALLEL=2 MOE_TOTAL_EXPERTS=16 MOE_EXTRA_ARGS="--shard-optimizer --straggler-routing" ./scripts/run_expert_parallel.sh
MOE_GPUS=2 MOE_LAYERS=8 MOE_MICROBATCHES=4 MOE_BATCH_SIZE=8 ./scripts/run_pipeline_parallel.sh
```

Adaptive controllers that need cross-rank routing statistics (`--aux-adapt`, `--temperature-adapt`) are not available with multi-process modes. Not implemented: vocabulary-parallel embeddings, combining pipeline with expert/tensor parallelism, and asynchronous expert scheduling.

## Triton Expert Kernels

`--moe-kernel triton` (or `auto`, which picks Triton only for bf16/fp16 on CUDA) replaces the per-expert loop with grouped GEMM: because routes are already sorted by expert, one kernel launch per projection computes every expert's rows, with matching grouped kernels for the input and weight gradients. It applies to single-process and pipeline runs and needs plain `nn.Linear` experts (quantized and LoRA experts use the loop). On an RTX 4060 the layer-level speedup under bf16 was 1.3x to 5.6x across the four shapes I tried (16 to 128 experts; more experts and smaller experts gain most), and 64-expert bf16 training ran 1.65x faster (73k vs 44k tokens/s) at the same loss. In fp32 it can lose to cuBLAS (0.7x with few experts), which is why `auto` skips it there. `python src/moe_kernels.py` benchmarks both paths on your GPU. Experts with no routed tokens receive an exactly-zero gradient rather than `None`, so AdamW still applies weight decay to them.

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

The report is saved as `benchmark-report.json` (with a Markdown twin, `benchmark-report.md`) and includes:

- Evaluation loss, perplexity, tokens/s, seconds per batch, and the precision and device used
- Exact parameter counts, active parameters per token, and the active/total ratio
- Memory: parameter bytes, bytes per expert, peak device memory, process RSS
- Dropped-route and overflow-route counts and fractions from expert capacity limits
- Per layer: expert utilization and its histogram, router entropy (raw and normalised), top-1 confidence distribution, load coefficient of variation, dead and underused experts, drop rate
- Per-expert compute time and the share of a forward pass spent in experts (from a separate profiled pass; `--no-routing-stats` skips it)
- Training stability (final/best loss, plateau, first stable step) when you pass `--training-metrics metrics.jsonl`
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

Distillation fine-tunes the student: it starts from the weights in `--student` (default `artifacts/outputs/quantweave-moe-out/`) with a fresh optimizer, and `--sequence-length` must fit the student's `max_sequence_length`. The distilled model is written to `artifacts/outputs/quantweave-moe-distilled/`. Distillation also saves recovery checkpoints under `artifacts/checkpoints/quantweave-distill/` and resumes automatically when rerun.

Benchmark the distilled student:

```bash
MOE_CHECKPOINT=artifacts/outputs/quantweave-moe-distilled \
MOE_BENCHMARK_DATA=data/teacher/teacher_answers.jsonl \
MOE_DEVICE=xpu \
./scripts/benchmark_quantweave.sh
```

Compare the original and distilled students using the same benchmark command and evaluation slice. Lower loss/perplexity is better; router auxiliary loss should remain finite and expert utilization should not collapse.

**Logit-level distillation.** Give the student a QuantaWeave teacher checkpoint (a dense baseline or a larger MoE) and it minimises `ce + alpha * T^2 * KL(teacher || student)` on the same batches:

```bash
DISTILL_TEACHER_DATA=data/smoke/tinystories.jsonl \
DISTILL_TEACHER_CHECKPOINT=artifacts/outputs/big-teacher \
DISTILL_ALPHA=1.0 DISTILL_TEMPERATURE=2.0 ./scripts/distill_quantweave.sh
```

Teacher and student must share a tokenizer and vocabulary size, because the KL is taken over aligned token distributions; otherwise distillation stops with an explanation.

**Cross-tokenizer distillation** (`--hf-teacher`). A Hugging Face causal LM (local directory or hub id) with its *own* tokenizer can teach a student that has a different one. Both models read the same text, and the student is supervised at the character positions where a student token boundary coincides with a teacher token boundary (found from the teacher's fast-tokenizer offsets); the other positions are skipped. `--cross-loss marginal` (default for character-level students, exact) collapses the teacher's next-token distribution onto the first character of each candidate token and minimises the KL to the student's next-character distribution. `--cross-loss uld` (any student) compares the descending-sorted distributions with L1, matching the shape of the teacher's distribution without a vocabulary mapping. Documents are split at `<eos>`; BPE students need tokens that decode to valid text one at a time (true for ASCII data). Needs the `transformers` package and a fast tokenizer.

```bash
python src/distill_quantweave_moe.py --student artifacts/outputs/quantweave-moe-out \
  --teacher-data data/smoke/tinystories.jsonl --hf-teacher /path/to/hf-model --alpha 1.0 --steps 500
```

## Export And Quantization

```bash
MOE_QUANTIZE=8 MOE_BENCHMARK_DATA=data/smoke/tinystories.jsonl ./scripts/export_quantweave.sh
```

writes a bundle under `artifacts/exports/quantweave-moe/`: a traced TorchScript model with dynamic batch and sequence length (`model.torchscript.pt`) and its graph text, an optional int8 or int4 weight-only copy, the tokenizer and config, and `export_metadata.json` (file sizes and SHA-256 hashes, traced-vs-eager numerical parity on several shapes, quantization compression and logit cosine similarity, optional benchmark numbers). Exported graphs evaluate every expert densely with its router weight (no capacity limit and no data-dependent control flow), so they are for portability and deployment-style evaluation, not for sparse speed. `--onnx` additionally tries an ONNX export and records the outcome; it needs the `onnx` package, which is not installed by default, so that path is untested here. Load a bundle without any model code via `export_quantweave.load_exported(dir)`.

Quantization (`src/quantization.py`) covers the expert weights, which hold nearly all parameters: int8 with one scale per row, or int4 packed two per byte with per-group scales. The router stays fp32. It saves memory and bandwidth (weights are dequantized on the fly) rather than FLOPs, and it is inference-only; training uses bf16/fp16 autocast.

## Quantized Fine-Tuning (QLoRA)

```bash
FT_DATA=new_domain.jsonl FT_BITS=4 FT_RANK=8 FT_MERGE_OUTPUT=artifacts/outputs/qlora-merged ./scripts/finetune_quantweave.sh
```

The experts are quantized to int4 or int8 and frozen, and each expert projection gets a trainable low-rank adapter `y = base(x) + (alpha/rank) * B(A(x))`; optionally `--include-lm-head`, `--train-router` and `--train-norms` too. Only the adapters train, so gradients and optimizer state cover a small fraction of the weights while the base stays compressed. `--optimizer adamw8bit` additionally keeps AdamW moments in 8 bits through `bitsandbytes` (CUDA only; tensors under 4096 elements stay 32-bit). The output holds `adapter.pt` and metadata (before/after loss, trainable fraction, memory); `load_finetuned(base_checkpoint, adapter_dir)` rebuilds the model from the untouched base, and `--merge-output` writes an ordinary full-precision checkpoint with the adapters folded in (dequantized base plus the low-rank delta). Merging reproduces the adapted model up to the quantization error already present in it.

## Experiments And Sweeps

```bash
python src/experiment_manager.py run     --config configs/exp_moe_balanced.yaml
python src/experiment_manager.py list
python src/experiment_manager.py compare artifacts/experiments/* --charts-dir artifacts/compare
python src/sweep.py --config configs/sweep_example.yaml --dry-run
python src/sweep.py --config configs/sweep_example.yaml
```

Each run gets a unique id and a directory with `config.json`, `metrics.jsonl`, `train.log`, routing diagnostics, checkpoints, the final model, a benchmark, `summary.json` and `report.md`; `best.json` in the runs directory points at the best run on the config's objective (default: benchmark loss). Failed runs are recorded with their traceback rather than aborting a sweep. `compare` ranks runs, shows only the options that differ between them, and draws loss curves and a metric bar chart as SVG. The benchmark evaluates on the training corpus unless `benchmark.data` names a held-out file.

Sweeps support `grid`, `random` (lists, `loguniform`, `uniform`, `int`), `halving` (successive halving: the top 1/eta of each rung continue with eta times the steps) and `bayes`. Bayesian search (`src/bayes_opt.py`, torch only) fits a Gaussian process (Matern-5/2 kernel, hyperparameters chosen by marginal likelihood) to the trials so far and runs the point with the highest expected improvement next, after `bayes.n_init` random trials; it never proposes invalid combinations or repeats a point, and a failed trial is remembered without teaching the model a fake value. Combinations that cannot work, such as more active than total experts, are skipped and listed.

## Send Messages To A Model

```bash
./scripts/chat_quantweave.sh                         # interactive chat with the newest archived run
python src/chat_quantweave_moe.py --checkpoint artifacts/outputs/quantweave-moe-out -m "Once upon a time" --temperature 0
python src/chat_quantweave_moe.py --run 1789960141 --messages-file smoke.jsonl --json
printf 'Once upon a time\nThe quick brown fox\n' | python src/chat_quantweave_moe.py --latest
```

**Which model:** `--checkpoint DIR`, `--run EPOCH` (an archived run by its epoch id or folder), `--latest` (the newest archived run with a model; fine-tune runs resolve to their merged model or base plus adapter), `--adapter DIR` (a QLoRA adapter over `--checkpoint`), and `--quantize 4|8` to try the compressed model. With none of them it uses `artifacts/outputs/quantweave-moe-out`.

**Three ways to send:** run with no messages in a terminal for an interactive session; pass `-m` (repeatable), `--messages-file`, or pipe lines on stdin for one-off and batch messages; `--interactive` forces the session even from a pipe. Inside the session `/help` lists the commands: `/set temperature|top_k|top_p|tokens|seed|system VALUE`, `/mode complete|chat`, `/reset`, `/show`, `/history`, `/save FILE`, `/quit`. Invalid settings are rejected and change nothing.

**Test messages.** A batch file is `.txt` (one message per line, `#` comments) or `.jsonl` with `{"message": ..., "expect": "text" or ["text", ...], "tokens": ..., "temperature": ..., "mode": ...}` per line. Every `expect` string must appear in the reply; the run prints PASS/FAIL per message and a summary, and **exits 1 if any check failed**, so it can gate a script or CI job. `--json` prints one JSON object per reply (message, prompt, response, tokens, tokens/s, stop reason, unknown characters, pass/fail); `--transcript FILE` appends every exchange to a JSONL log.

**Two modes.** `complete` (default) sends the message as the start of a text and shows the continuation, which is what a base model does. `chat` wraps the conversation as `User: ... / Assistant:` turns with an optional `--system` line, keeps the history in the session, and stops when the model starts a new `User:` line. Only a model trained on dialogue answers sensibly there; a base model plays along with the format at best. The trainer has no chat-format (role-tagged, assistant-only-loss) training yet.

**Sampling and reproducibility.** `--temperature 0` is greedy and fully deterministic; `--seed N` makes sampling reproducible; `--top-k`, `--top-p` and `--stop TEXT` (repeatable) are supported. Only tokens the tokenizer really has are ever sampled and `<unk>` is never produced: a character model's output range is far larger than its ~40 real characters, and the untrained rest used to appear as `?` in samples (the older `generate_quantweave_moe.py` had this too and now applies the same restriction). Characters in your message that the model never saw are replaced by `<unk>`, and the tool tells you which.

**Speed and the context window.** Replies are decoded incrementally (`src/fast_decode.py`): a KV cache so each token costs one token of compute, sampling inside the step so the GPU feeds itself and the host synchronises once per 8 tokens, and on CUDA the whole step recorded as a CUDA graph after `torch.compile` fuses its small kernels (a few seconds at start-up; it falls back to the plain graph if compilation fails). On an RTX 4060 with the 6-layer, 16-expert, 256-wide test model that is about **1,900 tokens/s, against about 75-110 for re-reading the whole context for every token (`--no-cache`)**, roughly 20-25x, and the very first reply is as fast as later ones because start-up warms the GPU. The step is launch-bound (about 0.37 ms per token, dominated by the number of tiny kernels rather than arithmetic or memory), so bigger models slow down less than proportionally; CPU decoding uses the cache without a graph. Sampling inside the graph draws Gumbel noise from the seeded generator, so `--seed` reproduces a run but not the same tokens as `--no-cache`.

Two deliberate differences from a training-style forward pass: **no expert capacity** at inference (a token is never dropped from its experts), and a context of **`max_sequence_length - 1` tokens**. The second one matters for quality: training computes its loss at positions 0 to L-1 of an L+1 token window, so the model's output at the very last position was never trained. On a trained model the loss at that position was 5.98 against 0.46-0.6 at every other, and feeding a full window and reading that slot made text fall apart after about 113 generated tokens. When the context fills, the oldest half is dropped and the rest re-read in one batched pass, which costs a few milliseconds every 64 tokens; long generations therefore stay coherent (a 1,200-token generation ran across many slides).

`generate_quantweave_moe.py` remains the simple one-prompt sampler (plain forward pass, no cache; it now also stays inside the trained context and samples only real tokens).

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

Compile raw data. This streams all languages of `bigcode/starcoderdata` (75%) and FineWeb-Edu (25%) into `data/raw/`:

```bash
python3 src/data.py
```

Pack the compiled data. Both raw files must exist. Each document gets an explicit EOS token, and the output uses Axolotl's pre-tokenized schema (`input_ids`, `attention_mask`, `labels`), which is why `configs/quantweave_moe.yaml` sets an empty dataset `type:`:

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

The standalone implementation is a research and scaling framework, not a production trillion-parameter training system. `FUTURE_IDEAS.md` tracks which of the roadmap ideas are implemented and which are not. Still open: multi-GPU validation of the parallel modes (they are verified on CPU processes only), vocabulary-parallel embeddings, combining pipeline with expert/tensor parallelism, asynchronous expert scheduling, and fused kernels beyond the grouped-GEMM expert path (for example a fused router or attention kernel).
