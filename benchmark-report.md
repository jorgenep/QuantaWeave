# Benchmark: artifacts/outputs/quantweave-moe-out

- device: NVIDIA GeForce RTX 4060 Laptop GPU (cuda), precision bf16, tokenizer char
- checkpoint step: 2

## Quality

| metric | value |
|---|---|
| loss | 9.0804 |
| perplexity | 8781.11 |
| tokens evaluated | 935,296 |

## Speed and memory

| metric | value |
|---|---|
| tokens / second | 108,484 |
| seconds / batch | 0.0024 |
| parameter memory | 4.6 MB |
| memory per expert (all layers) | 0.2 MB |
| peak device memory | 49.3 MB |
| process RSS | 1,636.0 MB |

## Parameters

| metric | value |
|---|---|
| total | 1,156,800 |
| active per token | 1,009,344 |
| active / total | 87.3% |
| experts (total / active) | 4 / 1 |

## Routing

| metric | value |
|---|---|
| aux (balance) loss | 1.0220 |
| overflow routes | 47,901 (2.54%) |
| dropped routes | 0 (0.00%) |
| capacity factor | 1.25 |
| expert utilization min / mean / max | 0.2221 / 0.2500 / 0.2777 |

### Per layer

| layer | entropy (norm.) | confidence | load CV | dead | underused | drop rate |
|---|---|---|---|---|---|---|
| 0 | 1.287 (93%) | 0.397 | 0.143 | 0 | 0 | 0.10% |
| 1 | 1.287 (93%) | 0.400 | 0.327 | 0 | 0 | 4.99% |

MoE layers spent 1.904s across experts (20% of the profiled pass).
