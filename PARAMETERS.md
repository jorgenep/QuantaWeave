# Parameter and VRAM Guide

What every training flag does, what it costs in GPU memory, and how to size a run for your card without hitting
`CUDA out of memory`. This covers `src/train_quantweave_moe.py`. Run `python src/train_quantweave_moe.py --help`
for the exact, current list; this file explains *why* each one exists and what moving it does.

Formulas here were checked against real measurements on an RTX 4060 Laptop (8 GB); see "Measured examples" below.

## The fast way to avoid OOM

You don't have to do the arithmetic by hand. Two commands do it for you:

```bash
# 1. What does this device offer, and roughly how many experts fit?
python src/hardware.py --device cuda

# 2. For a specific architecture, run real training steps and find a batch size that actually fits.
python src/hardware.py --probe --hidden-size 256 --layers 6 --ffn-size 512 --total-experts 16 --sequence-length 128
```

`--probe` doubles the batch size, running a real forward+backward+optimizer step each time (with the optimizer's
memory reserved up front), until a step runs out of memory or its peak passes 85% of device memory. It reports the
**largest** batch that fits and a **recommended** batch (the smallest one still within 95% of the best measured
throughput — usually smaller and faster than the largest, because throughput flattens before memory runs out).

Then train with `--auto-batch-size`, which runs the same probe before your real run starts and uses its
recommendation:

```bash
python src/train_quantweave_moe.py --auto-batch-size --hidden-size 256 --layers 6 --ffn-size 512 \
  --total-experts 16 --active-experts 2 --sequence-length 128 ...
```

This is the right tool 95% of the time. The rest of this document is for when you want to reason about *why* a
setting is expensive, pick a batch size by hand, or plan an architecture before you have data to train on.

## Suggested presets by VRAM

Starting points for common card sizes, each sized so the **training state uses about 45% of the tier's VRAM**,
leaving roughly half for activations, batch size, and sequence length. Every row was computed with
`src/hardware.py`'s exact parameter counter (`total_parameters × 16` for the training state); the **8 GB row was
also run for real** on an RTX 4060 Laptop with `--probe`, which found batch size 64 fits at a 4.73 GB peak (about
57% of the card, with another 1.2 GB already used by an unrelated process) — the other rows are the same
computation for larger cards I don't have, so **run `--probe` (or train with `--auto-batch-size`) before trusting
them on a long run**, especially at the sequence length and batch size you actually intend to use.

| VRAM tier | `--hidden-size` | `--layers` | `--ffn-size` | `--attention-heads` | `--total-experts` | `--active-experts` | `--vocab-size` | `--sequence-length` | total params | active params/token | training state |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 GB | 192 | 4 | 384 | 4 | 123 | 2 | 7,168 | 128 | 112M | 5.2M | 1.80 GB (45%) |
| 6 GB | 256 | 4 | 512 | 4 | 104 | 2 | 7,168 | 128 | 168M | 8.0M | 2.70 GB (45%) |
| 8 GB | 256 | 6 | 512 | 4 | 93 | 2 | 7,168 | 128 | 225M | 10.2M | 3.60 GB (45%) — **measured: batch 64 fits, 4.73 GB peak** |
| 12 GB | 384 | 8 | 768 | 8 | 43 | 2 | 32,000 | 256 | 334M | 43.7M | 5.34 GB (45%) |
| 16 GB | 384 | 10 | 768 | 8 | 47 | 2 | 32,000 | 256 | 447M | 48.5M | 7.15 GB (45%) |
| 24 GB | 512 | 12 | 1,024 | 8 | 33 | 2 | 32,000 | 512 | 669M | 83.6M | 10.70 GB (45%) |
| 32 GB | 640 | 14 | 1,280 | 8 | 24 | 2 | 32,000 | 512 | 890M | 133M | 14.24 GB (45%) |
| 40 GB | 768 | 16 | 1,536 | 8 | 18 | 2 | 32,000 | 512 | 1.11B | 201M | 17.71 GB (44%) |
| 80 GB | 1,024 | 20 | 2,048 | 8 | 16 | 2 | 32,000 | 1,024 | 2.16B | 403M | 34.63 GB (43%) |

Some cards these tiers correspond to (going by their advertised VRAM — check yours with `python src/hardware.py`
rather than assuming from the model name, since the same name sometimes ships in more than one memory size): **8
GB** — RTX 4060/4060 Ti 8 GB, RTX 3070; **12 GB** — RTX 3060 12 GB, RTX 4070; **16 GB** — RTX 4060 Ti 16 GB, RTX
4080 mobile; **24 GB** — RTX 3090, RTX 4090; **32 GB** — RTX 5090, Intel Arc Pro B70, AMD Radeon RX 9800 series; **40
GB** — A100 40 GB; **80 GB** — A100/H100 80 GB.

Ready-to-run command for any row (fill in the tier's numbers and let the probe pick the batch size):

```bash
python src/train_quantweave_moe.py --auto-batch-size --device cuda \
  --hidden-size 256 --layers 6 --ffn-size 512 --attention-heads 4 \
  --total-experts 93 --active-experts 2 --vocab-size 7168 --sequence-length 128 \
  --data data/smoke/tinystories.jsonl --steps 10000 ...
```

Notes on these presets:

- **They favor a high `--total-experts`** to make use of the available VRAM, since stored-but-inactive experts are
  the cheapest way to grow a model without raising per-token compute (see the `--total-experts` vs
  `--active-experts` distinction below). If you'd rather have a smaller, denser model with more headroom for a huge
  batch or long sequences, lower `--total-experts` — everything else about the row still fits.
- **The 4–8 GB rows use the repo's default character tokenizer** (`--vocab-size 7168`, the default; no
  `--tokenizer` flag needed). The 12 GB+ rows assume `--tokenizer bpe --tokenizer-path <dir> --vocab-size 32000` (a
  trained BPE tokenizer) — swap in your own vocabulary size and re-check the training state if you pick a different
  one, since embeddings scale with `2 × vocab_size × hidden_size`.
- **Raising `--sequence-length`** past a row's suggested value costs more than the same fractional change to
  `--batch-size` (attention scales with its square) — re-probe if you do.
- If a probe on your actual card recommends a much smaller batch than expected, or fails outright, that's real
  signal to shrink the row's `--total-experts` or `--hidden-size`, not just fight the batch size — see
  [Model-shape parameters](#model-shape-parameters-fix-the-training-state-cost) below.

## Training a dense (non-MoE) model

`train_quantweave_moe.py` doesn't have a separate "dense mode" flag, and it doesn't need one: **a dense model is
just an MoE model with one expert that always gets picked.** Set:

```bash
python src/train_quantweave_moe.py --total-experts 1 --active-experts 1 --router-aux-coef 0 ...
```

That's the whole recipe. `--total-experts 1 --active-experts 1` is what makes it dense; `--router-aux-coef 0` is a
cleanup, not a requirement (see below). Everything else — `--hidden-size`, `--layers`, `--ffn-size`,
`--sequence-length`, `--batch-size`, the schedules, curriculum, domain sampling, distillation, quantization, LoRA
fine-tuning, export, and the chat tool — works exactly as documented elsewhere in this file and the README, because
none of that code path changes; it's the same model with `num_experts` pinned to 1. This was verified end to end
(trained, benchmarked, and generated from) rather than inferred from the code:

```
$ python src/train_quantweave_moe.py --total-experts 1 --active-experts 1 ...
step=20/20 loss=1.7269 router_aux=1.0000 dropped_routes=0 overflow_routes=0
$ python src/benchmark_quantweave_moe.py --checkpoint ... --no-routing-stats
{"total_parameters": 116032, "active_parameters_per_token_estimate": 116032, "active_parameter_ratio": 1.0, ...}
```

`active_parameter_ratio: 1.0` is the whole point — every parameter runs on every token, which is the definition of
"dense."

**Why `--router-aux-coef 0` is worth adding, and why `--capacity-factor` doesn't matter here:** with one expert,
the router's softmax is over a single logit, which is mathematically always `1.0` regardless of what that logit is
— so the load-balancing loss is a constant with zero gradient (confirmed above: `router_aux=1.0000` never moves),
and capacity can never bind (every token's one "route" always fits within any capacity ≥ its own count). Neither
setting is wrong at its default; `--router-aux-coef 0` just skips computing a term that can't affect training.

**What does *not* carry over:** `--expert-parallel` has nothing to shard (it requires `--total-experts` to divide
evenly across ranks, and `1 % world_size == 0` only for `world_size = 1`), so it's not useful for a dense model —
use `--tensor-parallel` and/or `--pipeline-parallel` instead to split a large dense model across GPUs, exactly as
you would for the MoE model (see [Multi-process parallelism](#multi-process-parallelism-fitting-a-model-too-big-for-one-gpu)).

**The one thing genuinely different from the MoE table: dense couples storage and compute.** In the MoE presets
above, `--total-experts` grows the *stored* model for free — VRAM goes up, but a token still only touches
`--active-experts` of it, so compute per token stays small. A dense model has no such lever: every stored
parameter runs on every token, so `active params/token` always equals `total params`. That means, at the *same*
training-state budget, a dense preset has far fewer total parameters than its MoE counterpart (roughly the MoE
row's *active* count, not its *total* count) — and it will be noticeably slower per training step, because it does
proportionally more compute for the memory it uses. This is the actual trade-off MoE exists to make, not just a
naming difference.

### Dense model presets by VRAM

Same method as the MoE table: sized so the training state uses about 42–45% of the tier. `--ffn-size` follows the
repo's usual `2.75 × hidden_size` ratio (rounded); `--attention-heads` and `--vocab-size` match the MoE table's
rows for the same tier. None of these were run on real hardware (only the dense *mechanism* above was, on a tiny
model) — probe before a long run, same as always.

| VRAM tier | `--hidden-size` | `--layers` | `--ffn-size` | `--attention-heads` | `--vocab-size` | `--sequence-length` | total = active params | training state |
|---|---|---|---|---|---|---|---|---|
| 4 GB | 1,344 | 4 | 3,584 | 4 | 7,168 | 128 | 106M | 1.70 GB (42%) |
| 6 GB | 1,664 | 4 | 4,480 | 4 | 7,168 | 128 | 158M | 2.53 GB (42%) |
| 8 GB | 1,664 | 6 | 4,480 | 4 | 7,168 | 128 | 225M | 3.60 GB (45%) |
| 12 GB | 1,536 | 8 | 4,224 | 8 | 32,000 | 256 | 330M | 5.28 GB (44%) |
| 16 GB | 1,664 | 10 | 4,480 | 8 | 32,000 | 256 | 441M | 7.06 GB (44%) |
| 24 GB | 1,920 | 12 | 5,248 | 8 | 32,000 | 512 | 664M | 10.62 GB (44%) |
| 32 GB | 2,048 | 14 | 5,632 | 8 | 32,000 | 512 | 852M | 13.63 GB (43%) |
| 40 GB | 2,176 | 16 | 5,888 | 8 | 32,000 | 512 | 1,059M | 16.94 GB (42%) |
| 80 GB | 2,816 | 20 | 7,680 | 8 | 32,000 | 1,024 | 2,116M | 33.85 GB (42%) |

Ready-to-run command for any row:

```bash
python src/train_quantweave_moe.py --auto-batch-size --device cuda \
  --hidden-size 1664 --layers 6 --ffn-size 4480 --attention-heads 4 \
  --total-experts 1 --active-experts 1 --router-aux-coef 0 --vocab-size 7168 --sequence-length 128 \
  --data data/smoke/tinystories.jsonl --steps 10000 ...
```

Compare a row here with the *same VRAM tier's* row in the MoE table above — at 8 GB, they land on almost the same
total parameter count by coincidence (225M each), which makes the difference easy to see: the dense row's 225M
parameters are **all** active on every token; the MoE row's 225M parameters are spread across 93 experts, of which
only 2 run per token — **10.2M active**. Same VRAM, same total parameter count, a 22x difference in compute per
token. Neither is "better" in the abstract: dense gives every parameter a gradient on every step (simpler
optimization, no routing/capacity tuning, and it's how most non-MoE LLMs are trained and evaluated); MoE trades
that simplicity for far more raw stored capacity at the same memory cost, and lets that capacity specialize, at the
cost of the routing/capacity tuning covered elsewhere in this file.

**Comparing to the Axolotl baseline path:** this section trains a QuantaWeave dense model from scratch, with the
standalone trainer and everything above still applying. `configs/quantweave_moe.yaml` / `configs/quantweave_smoke.yaml`
(see the README's "Axolotl Baseline" section) are a different thing — they fine-tune an existing pretrained dense
checkpoint (Qwen) through Axolotl, for validating that larger production stack. Use this section's approach for a
from-scratch dense QuantaWeave model; use Axolotl for fine-tuning an existing model.

## How VRAM is spent

Training memory has two parts:

```
peak VRAM  =  training state (fixed, exact)  +  activations (grows with batch size × sequence length)
```

### 1. Training state — exact, and independent of batch size

Whatever you train with (bf16 autocast, fp16, or plain fp32), the trainer keeps an **fp32 master copy** of every
weight, because that's what `autocast_context` and AdamW require. For each trainable parameter that's:

| Copy | Bytes / parameter |
|---|---|
| fp32 weight | 4 |
| fp32 gradient | 4 |
| AdamW momentum (`m`) | 4 |
| AdamW variance (`v`) | 4 |
| **Total** | **16** |

```
training_state_bytes ≈ total_parameters × 16
```

Get `total_parameters` exactly (no memory allocated — it builds the model on a "meta" device) with:

```bash
python src/model_scaling.py --mode moe --size m          # a preset
python src/model_scaling.py --mode moe --size m --hidden-size 512 --total-experts 24   # or custom dimensions
```

or from `src/hardware.py`'s `parameter_counts()` / `training_state_bytes()`, which is what `--probe` and
`--auto-batch-size` use internally.

**This is usually 60–90% of your peak memory**, and it is the part you can compute exactly before running anything.
If `training_state_bytes` alone is close to your device's total memory, no batch size will save you — you must
shrink the model (fewer/smaller experts, smaller hidden size) or shard it (see [Multi-process parallelism](#multi-process-parallelism-fitting-a-model-too-big-for-one-gpu)).

### 2. Activations — grows with batch size and sequence length, and is *not* a clean formula

Activations (attention scores, MoE dispatch buffers, intermediate tensors kept for backward) scale with
`batch_size × sequence_length`, and the attention score matrix specifically scales with `sequence_length²` per
layer — **doubling sequence length costs more than doubling batch size**. They also depend on `hidden_size`,
`ffn_size`, `layers`, and how many experts a batch actually routes to.

This part does **not** reduce to one clean per-token constant — the numbers below (measured, not estimated) show
why:

| hidden | layers | ffn | experts (total/active) | seq | batch | training state | peak VRAM | activations |
|---|---|---|---|---|---|---|---|---|
| 256 | 4 | 512 | 16 / 2 | 128 | 8 | 479 MB | 616 MB | 137 MB |
| 256 | 4 | 512 | 16 / 2 | 128 | 32 | 479 MB | 1,059 MB | 580 MB |
| 256 | 6 | 512 | 32 / 2 | 128 | 8 | 1,293 MB | 1,635 MB | 342 MB |
| 256 | 6 | 512 | 32 / 2 | 128 | 32 | 1,293 MB | 1,931 MB | 638 MB |
| 512 | 6 | 1024 | 32 / 2 | 256 | 8 | 5,054 MB | 6,356 MB | 1,302 MB |
| 512 | 6 | 1024 | 32 / 2 | 256 | 32 | 5,054 MB | 7,017 MB | 1,962 MB |

Activation memory per token *shrinks* as batch size grows (there's a roughly fixed allocator/kernel-launch overhead
that amortizes), and it grows faster than linearly with sequence length. This is exactly why `--probe` /
`--auto-batch-size` measure it instead of estimating it: get `training_state_bytes` exactly by formula, then let the
probe measure the rest for your actual shapes.

### Rule of thumb, if you need one before running anything

```
budget = 0.8 × your GPU's total VRAM              # leave headroom for the CUDA context, fragmentation, other processes
if training_state_bytes > budget:                  # shrink the model; no batch size fixes this
    reduce hidden_size / layers / ffn_size / total_experts
else:
    remaining = budget - training_state_bytes
    start batch_size at 1-2, double it, watch nvidia-smi (or use --auto-batch-size to do this properly)
```

## Model-shape parameters (fix the training-state cost)

These set `total_parameters`, so they set the fixed, exact part of VRAM. Changing any of them changes the model
architecture — a checkpoint trained with one shape cannot resume with a different one (the trainer checks this and
refuses unless you pass `--allow-config-change`).

| Flag | Default | What increasing it does | VRAM impact |
|---|---|---|---|
| `--hidden-size` | 64 | Wider token representation. Must be divisible by `--attention-heads`. Raises model capacity roughly proportional to its square (touches attention, embeddings, every expert). | High — quadratic-ish in parameter count |
| `--layers` | 2 | More transformer blocks stacked. Roughly linear increase in parameters and in activation memory (and depth of the residual stream). | High — linear |
| `--ffn-size` | 128 | Width of each expert's SwiGLU. Each expert holds `3 × hidden_size × ffn_size` parameters (gate, up, down). | High — linear per expert, and there are `total_experts × layers` of them |
| `--attention-heads` | 4 | More, narrower attention heads at the same `hidden_size`. Must evenly divide `hidden_size`. | Negligible on its own |
| `--total-experts` | 184 | **Stored** experts per layer. This is the single biggest VRAM lever in an MoE model — every expert's weights, gradients, and Adam state exist whether or not a token ever routes to it. | Very high — linear, and usually dominates |
| `--active-experts` | 1 | Experts **routed per token** (top-k). Does **not** change stored memory (all experts are always allocated); it changes compute per token and how much of the model is "active." Must be ≤ `--total-experts`. | None on stored memory; raises activation memory and compute a bit (more experts run per token) |
| `--vocab-size` | 7168 | Character-tokenizer vocabulary size (ignored for BPE, which uses the trained tokenizer's own size). Adds `2 × vocab_size × hidden_size` parameters (input + output embedding, since they aren't tied). | Matters mainly at small `hidden_size` with a large vocab (e.g. a 151,936-token BPE vocab at `hidden_size=384` is ~117M embedding parameters alone) |

**The `--total-experts` vs `--active-experts` distinction is the one to internalize**: `total-experts` is what you
pay in memory; `active-experts` is what you pay in compute per token. A model with 184 total experts and 1 active
(the smoke-test default) stores 184 experts' worth of weights but only runs 1 per token — VRAM reflects the 184,
throughput reflects the 1. Setting `--total-experts 1 --active-experts 1` is a dense (non-MoE) model — see
[Training a dense (non-MoE) model](#training-a-dense-non-moe-model) below.

Use the architecture presets as reference points (`python src/model_scaling.py --mode moe --size <xs|s|m|l|xl|xxl>`,
or override any field):

| preset | hidden | layers | experts (total/active) | vocab | total params | active params/token | fp32 training state |
|---|---|---|---|---|---|---|---|
| xs | 192 | 4 | 4 / 1 | 32,000 | 0.01B | 0.01B | ~160 MB |
| s | 384 | 8 | 8 / 1 | 151,936 | 0.20B | 0.13B | ~3.2 GB |
| m | 768 | 16 | 16 / 1 | 151,936 | 1.52B | 0.35B | ~24.3 GB |
| l | 1408 | 24 | 76 / 1 | 151,936 | 30.45B | 1.01B | ~487 GB |
| xl | 4096 | 40 | 64 / 2 | 151,936 | 358.26B | 15.00B | ~5.7 TB |
| xxl | 8192 | 80 | 22 / 2 | 151,936 | 998.38B | 112.55B | ~16.0 TB |

An 8 GB card fits `xs` comfortably and struggles with `s` at anything but a tiny batch size — check
`training_state_bytes` against your card before picking a preset.

## Data and context parameters (affect activation memory and what the model learns from)

| Flag | Default | What increasing it does | VRAM impact |
|---|---|---|---|
| `--sequence-length` | 128 | Tokens of context per training window. Attention memory grows roughly with the **square** of this, so raising it is the most expensive way to use more VRAM. Also: the model's usable context at inference is `sequence-length - 1` (the last position of a training window is never trained; see the chat tool's docs). | High, and non-linear |
| `--batch-size` | 2 | Sequences trained per step. Activation memory scales roughly linearly with this (see the measured table above). | Moderate, roughly linear |
| `--gradient-accumulation-steps` | 1 | Accumulates gradients over this many micro-batches before stepping the optimizer, so `effective_batch = batch_size × gradient_accumulation_steps` without the memory cost of a bigger `--batch-size`. **Use this instead of raising `--batch-size` when you're VRAM-limited but want a larger effective batch.** | None extra — this is the trick for training a large effective batch in small memory |
| `--examples` | 10,000 | Max JSONL rows read per data file (not tokens; controls how much of the corpus is used before wrapping). | None |
| `--vocab-size` | 7168 | See model-shape table above (char tokenizer only). | See above |
| `--tokenizer` / `--tokenizer-path` | char | `bpe` trains/loads a byte-level BPE tokenizer instead of characters — usually a smaller, better vocabulary for the same text, meaning more information per token. | Indirect, via `--vocab-size` |
| `--curriculum*` | none | Orders training windows from easiest to hardest (see README's "Data Pipeline" section). Affects what the model sees, not memory. | None |
| `--domain-*` | — | Multi-domain sampling weights and a domain-specialization loss term. Affects data mix, not memory. | None |

## Routing and capacity parameters (affect stability and compute, not stored memory)

| Flag | Default | What increasing it does | Notes |
|---|---|---|---|
| `--capacity-factor` | 1.25 | How many routed tokens each expert may process per batch, relative to an even split (`0` disables the limit entirely — every routed token is processed). Raising it reduces dropped/overflowed routes at the cost of a bit more compute and activation memory (more tokens actually go through experts up to the cap). | Small VRAM effect; mainly a training-stability knob |
| `--min-expert-capacity` | 4 | Floor on the above, so tiny batches don't starve experts entirely. | Negligible |
| `--drop-overflow-tokens` | on | Whether overflow is enforced at all; `--no-drop-overflow-tokens` processes every routed token regardless of capacity. | Raises memory/compute somewhat if capacity would otherwise have limited things |
| `--overflow-policy` | drop | `drop` vs `residual` — both skip the same overflowing routes; they only differ in whether that's counted as `dropped_routes` (drop) or just `overflow_routes` (residual). Doesn't change what's computed. | None |
| `--router-temperature` | 1.0 | Softens (`>1`) or sharpens (`<1`) routing decisions. A training/exploration knob. | None |
| `--router-aux-coef` | 0.01 | Weight of the load-balancing auxiliary loss. Higher pushes routing toward evenness harder. | None |
| `--capacity-adapt`, `--aux-adapt`, `--temperature-adapt`, `--plateau-patience`, `--capacity-release-step`, `--temperature-start/-steps`, `--controller-interval` | off | Adaptive schedules/controllers for the above (see README's "Training Schedules And Adaptive Control"). Training-dynamics knobs, not memory. | None |

## System / device parameters (the ones that actually change what fits)

| Flag | Default | What it does | VRAM impact |
|---|---|---|---|
| `--precision` | auto | `auto` picks bf16 on CUDA/XPU (fp32 on CPU). `bf16`/`fp16` run the *forward pass* in half precision under autocast — **the fp32 master weights, gradients, and Adam state are unaffected**, so this saves activation memory and bandwidth, not the fixed training-state cost. `fp16` also adds a `GradScaler`. `fp32` disables autocast (slower, and no memory saved over bf16's activations). | Moderate — activations only |
| `--activation-checkpointing` | off | Recomputes each block's activations during backward instead of storing them, trading compute for memory. Worth it once activations are a large share of peak memory (typically bigger models / longer sequences); `--auto-batch-size` and `python src/hardware.py --probe` measure both ways and pick whichever trains faster at its own best batch size. | Reduces activation memory, at some speed cost |
| `--auto-batch-size` / `--auto-batch-target` | off / 0.95 | See "The fast way to avoid OOM" above. `--auto-batch-target 0` picks the *largest* batch that fits instead of the fastest-per-VRAM one. | This is the tool for the whole "activations" half of the budget |
| `--moe-kernel` | loop | `triton` replaces the per-expert Python loop with a fused grouped-GEMM kernel (CUDA + the `triton` package; faster, especially with many experts, under bf16/fp16). `auto` picks it automatically on CUDA with bf16/fp16. | Roughly neutral to slightly lower (fewer intermediate tensors) |
| `--device` | auto | `cuda`, `rocm` (via CUDA API), `xpu`, `cpu`, or `auto` (picks the best available). | Determines which memory pool everything above is measured against |
| `--seed` | 0 | Reproducibility only. | None |

## Multi-process parallelism (fitting a model too big for one GPU)

If `training_state_bytes` alone exceeds your GPU, no single-GPU batch size will save you — the model itself has to
be sharded across devices. Launched with `torchrun`; see the README's "Multi-Process Parallelism" section for full
detail.

| Flag | What it does | Effect on per-GPU VRAM |
|---|---|---|
| `--expert-parallel` | Shards experts across ranks: each rank stores only `total_experts / world_size` of them. `--total-experts` must divide evenly by the number of ranks. | Divides the biggest cost (expert storage) by the rank count |
| `--tensor-parallel T` | Additionally splits attention heads and each expert's FFN width across `T` ranks (combinable with `--expert-parallel`; `attention_heads` and `ffn_size` must be divisible by `T`). | Further divides stored weights and activations per rank |
| `--shard-optimizer` | ZeRO-2-style: each rank's *replicated* parameters (attention, embeddings, routers, norms) have their Adam state owned by one rank instead of duplicated on every rank. | Cuts replicated-parameter optimizer memory roughly by the expert-parallel rank count |
| `--pipeline-parallel --microbatches M` | Splits **layers** across ranks (each rank holds a contiguous slice of the model) instead of splitting experts. Not combinable with expert/tensor parallelism. | Divides both stored weights and activations by the number of pipeline stages, at the cost of pipeline bubble idle time |

For a model whose `training_state_bytes` would need, say, 4 GPUs' worth of memory: `--expert-parallel` on 4 ranks
(with `--total-experts` divisible by 4) is the first thing to reach for, since expert storage is usually what
dominates.

## System RAM and CPU offloading: can I train bigger than my VRAM by using RAM too?

**Update: the standalone trainer now has a CPU-offload path.** `--optimizer adamw_cpu_offload`
(`src/cpu_offload_optimizer.py`) keeps AdamW's two moments (`exp_avg`, `exp_avg_sq` — 8 of the 16
bytes/parameter in [training state](#1-training-state--exact-and-independent-of-batch-size)) in pinned system RAM
instead of device memory, freeing that much VRAM for a bigger model or batch. Weights and gradients stay on the
device as usual — this offloads optimizer *state* only, the same scope as DeepSpeed's `offload_optimizer` with
`offload_param: none` (see below). It is verified step-for-step against `torch.optim.AdamW` on identical gradients
(max diff ~1e-7, fp32 rounding only) and end-to-end through the real trainer, including checkpoint save/resume.

It is not free: every optimizer step now copies each parameter's gradient to a pinned CPU staging buffer, runs the
AdamW update there, and copies the result back — a real host↔device transfer per step, not just a memory trick.
Expect a real wall-clock cost, worse on a slower PCIe link or with a model large enough that the transfer dominates
the step. Not combinable with `--shard-optimizer` (which offloads differently, across ranks rather than across
device/host) or `--optimizer adamw8bit` (bitsandbytes' 8-bit moments already live in VRAM; the two approaches are
alternatives, not additive).

**Before this**, the situation was: no CPU-offload code path (checked directly: nothing in `src/*.py` moved
weights, gradients, or optimizer state to CPU during training), so system RAM was used for the data loader and
general process overhead only, not for `training_state_bytes`. The "32 GB" row in the
[presets](#suggested-presets-by-vram) and [dense presets](#dense-model-presets-by-vram) tables still assumes that
(the whole VRAM budget, no RAM offload) since it's the safe default; `--optimizer adamw_cpu_offload` is how you go
beyond it deliberately, at the transfer-time cost above.

**The Axolotl integration path is different, and this repo already has it half set up.**
`configs/deepspeed_zero3.json` enables DeepSpeed ZeRO-3 with **optimizer-state offload to CPU**:

```json
"offload_optimizer": { "device": "cpu", "pin_memory": true },
"offload_param": { "device": "none" }
```

Recall from [training state](#1-training-state--exact-and-independent-of-batch-size) that the 16 bytes/parameter
splits into 4 (weight) + 4 (grad) + 4 + 4 (AdamW's two moments). `offload_optimizer: cpu` moves those last 8
bytes/parameter — half the total — into pinned system RAM, freeing that much GPU memory for a bigger model or
batch. Setting `offload_param.device` to `"cpu"` as well (it's currently `"none"`, so weights stay on GPU) would
offload the remaining 4 bytes/parameter of weights too, for up to 12 of the 16 bytes/parameter living in RAM instead
of VRAM.

**What that would mean for your 32 GB + 32 GB machine, honestly:** I don't have that hardware to test this on, so
take this as how ZeRO-3 offload generally behaves, not a number specific to this repo. Don't expect a clean 32 + 32
= 64 GB budget:

- **System RAM isn't 100% available.** The OS, the training process itself, and ZeRO-3's pinned-memory buffers
  (needed for fast host↔device transfer, and pinned memory itself isn't swappable) all eat into the 32 GB before any
  of it is free for offloaded state. Plan for meaningfully less than the full 32 GB being usable.
- **It costs real speed, not just capacity.** Every optimizer step now moves the offloaded state over PCIe.
  Depending on how much you offload and your PCIe generation/lane count, this ranges from a moderate slowdown to
  the dominant cost of each step — this is standard, well-documented DeepSpeed ZeRO-Offload behavior, and worth
  benchmarking on your own hardware before committing a long run to it.
- **This path fine-tunes an existing pretrained dense checkpoint (Qwen)** via Axolotl (see the README's "Axolotl
  Baseline" section), not the from-scratch QuantaWeave MoE architecture the tables above are sized for. There's no
  preset table for it here, because sizing it is governed by DeepSpeed's own tuning (`auto` fields in the JSON, plus
  `stage3_max_live_parameters` etc.), not this repo's `hardware.py`.

For the **standalone QuantaWeave MoE trainer**, `--optimizer adamw_cpu_offload` (above) is that feature: AdamW's
moments live in pinned host memory, with each step's gradient and update crossing PCIe to get there and back. It
offloads less than the Axolotl/DeepSpeed ZeRO-3 path can (only the two AdamW moments, not weights or gradients too,
and there's no equivalent of `offload_param: cpu`), so the same "don't expect a clean sum of the two budgets"
caution above still applies — plan for real speed cost and less than the nominal extra headroom.

## Output / archiving parameters

`--output`, `--checkpoint-dir`, `--checkpoint-interval`, `--resume`, `--allow-config-change`, `--metrics-file`,
`--log-interval`, `--diagnostics-dir`, `--diagnostics-interval` control where things are saved and how often, and
don't touch GPU memory. See the README's "Checkpoint Recovery" and "Routing Diagnostics" sections.

`--archive-dir` (on by default from the command line; `--no-archive` turns it off) is the one exception:
**archiving a finished run allocates a second full model on the GPU, in the same process, while the training model
is still alive** — for its benchmark and sample-generation passes (see the README's "Run Archives"). Measured on a
225M-parameter model: GPU memory allocated right after training was 1,078 MB; it peaked at **2,551 MB during
archiving** — more than double. If a run's training loop fit with little headroom, archiving right after it can
still OOM even though training itself never came close. See
[OOM troubleshooting](#it-ran-fine-for-a-while-then-oomd-partway-through) below.

## Worked example

Say you have an 8 GB card and want the biggest model that trains comfortably.

```bash
# 1. See what the device offers and get a rough expert-count suggestion
python src/hardware.py --device cuda
#   -> total_memory_bytes ~8.3e9; suggested_num_experts uses half the device for fp32 training state

# 2. Compute the exact parameter count for a candidate shape
python src/model_scaling.py --mode moe --size m --hidden-size 256 --layers 6 --total-experts 32 --top-k 2
#   -> read estimated_total_params; total_params * 16 is your fixed cost

# 3. If that fits comfortably under ~0.8 x 8 GB, let the probe find the batch size
python src/hardware.py --probe --hidden-size 256 --layers 6 --ffn-size 704 --total-experts 32 \
  --active-experts 2 --sequence-length 128 --device cuda

# 4. Train with what it recommends
python src/train_quantweave_moe.py --auto-batch-size --hidden-size 256 --layers 6 --ffn-size 704 \
  --total-experts 32 --active-experts 2 --sequence-length 128 --device cuda ...
```

If step 2's fixed cost alone is already close to 6.6 GB (0.8 × 8 GB), stop and shrink `--total-experts` or
`--hidden-size` before going further — no batch size will fix a model that doesn't fit by itself.

## OOM troubleshooting

Sizing a run in advance (everything above) prevents most `CUDA out of memory` errors, but not all of them — some
only show up once training is already running. This section is for those: what each one looks like, why it
happens, and the fix. The first three were reproduced and measured for this guide, not guessed.

### Read the error first

PyTorch's OOM message tells you most of what you need:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 896.00 MiB.
GPU 0 has a total capacity of 7.70 GiB of which 348.12 MiB is free. Process 3085 has 98.00 MiB memory in use.
```

- **"Tried to allocate" is small (tens to hundreds of MB) but almost nothing is free:** you're right at the edge —
  drop `--batch-size` by half, or use `--auto-batch-size` which finds this edge safely instead of finding it by
  crashing.
- **Another process already holds real memory** (the `Process ... has ... memory in use` line, or check
  `nvidia-smi` yourself): that's not your training run's fault. Close it, or subtract its usage from what you plan
  for (`hardware.py`'s `free_memory_bytes` already accounts for this if it's running when you probe).
- **The allocation itself is huge** (multiple GB in one "Tried to allocate"): that's usually `--sequence-length` (a
  giant attention matrix) or a batch size far beyond what was probed — check whether something changed those since
  you last sized the run.

### It ran fine for a while, then OOM'd partway through

This is the confusing case, because the run already proved it fits. Two real causes, both specific to this
project:

**`--capacity-adapt` raised the effective batch of expert work over time.** The capacity controller grows
`capacity_factor` toward `--capacity-max` (default 4.0) whenever routes are overflowing, and a higher capacity
means *more tokens actually pass through experts* per step, not just a higher limit. Measured on a 6-layer,
32-expert model: peak memory at `capacity_factor=1.25` was 2,061 MB; at `capacity_factor=2.0` (which
`--capacity-adapt` can reach within its first several controller intervals) it was 2,728 MB — **33% more**, for the
same batch size, just from the controller doing its job. If you use `--capacity-adapt` (or
`--capacity-release-step`, which disables the cap entirely from some step on), **probe at the *worst-case* capacity
you'll allow, not the starting one**:

```bash
# size for what --capacity-adapt can grow into, not --capacity-factor's starting value
python src/hardware.py --probe --capacity-factor 4.0 ...   # match --capacity-max (or 0, if you use --capacity-release-step)
```

**Archiving after the run finishes built a second model on the GPU.** Covered above under
[Output / archiving parameters](#output--archiving-parameters): a run whose training loop just barely fit can OOM
during its own archiving step, right after printing `saved checkpoint to ...`, because the trained model is still
resident when the archive benchmarks a fresh copy of it. Fixes, in order of preference:
- Leave more headroom when sizing the run (target 70–75% of VRAM for training state + activations instead of the
  85% the probe allows, if you know you'll archive).
- `--archive-benchmark-examples` smaller (fewer rows loaded per benchmark batch).
- `--no-archive`, then benchmark separately afterward with `python src/benchmark_quantweave_moe.py --checkpoint ...`
  once the training process has exited and released its memory.

**Not a real risk, in case you're wondering:** resuming from a checkpoint (`--resume`, the default) does **not**
add extra memory beyond a normal step. It was measured directly: loading a checkpoint peaked at 1,312 MB, which is
just `training_state_bytes` reached once (1,290 MB expected), *before* any batch's activations are added —
comfortably under the 2,365 MB steady-state peak of ordinary training on the same model. If a run trained, its
resumes will too, unless you also raise `--batch-size` on resume (which needs `--allow-config-change`, since the
trainer otherwise refuses to resume with different settings — at that point size the new batch size like any other
change).

### It fails immediately, on step 1

The model plus its first batch doesn't fit at all. `--auto-batch-size` will find this (and tell you plainly if even
batch size 1 doesn't fit) rather than you discovering it from a crash. If batch size 1 still doesn't fit:
`training_state_bytes` alone is too big for the card — shrink `--total-experts` or `--hidden-size` (see
[Model-shape parameters](#model-shape-parameters-fix-the-training-state-cost)), or shard the model across multiple
GPUs (see [Multi-process parallelism](#multi-process-parallelism-fitting-a-model-too-big-for-one-gpu)).

### General fixes, roughly in order of how much they help

1. **`--sequence-length` down**, if you can afford it — attention memory scales with its square, so this is the
   single biggest lever after the model shape itself.
2. **`--batch-size` down**, and use **`--gradient-accumulation-steps`** to recover the effective batch size you
   lost — this costs no extra memory (see the data-parameters table above).
3. **`--activation-checkpointing`** — trades compute for memory; `--probe` / `--auto-batch-size` already try this
   automatically and tell you whether it was worth it for your shape.
4. **`--total-experts` down** — remember this is stored memory regardless of `--active-experts`, so it's often the
   biggest single number to reduce if the model itself (not just activations) is too large.
5. **Fragmentation, on a long-running process**: if `nvidia-smi` shows less free memory than "allocated" would
   suggest, set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in your environment before launching — this is
   PyTorch's own suggestion in the OOM error message, not specific to this project.
6. **Check for other processes** holding GPU memory (`nvidia-smi`) before assuming your configuration is at fault —
   a leftover training run, a chat session (`chat_quantweave_moe.py`), or another job on a shared machine all count
   against your budget.
