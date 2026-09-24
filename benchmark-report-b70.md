# Benchmark: artifacts/outputs/b70-test

- device: xpu (xpu), precision bf16, tokenizer char
- checkpoint step: 50

## Quality

| metric | value |
|---|---|
| loss | 7.3370 |
| perplexity | 1536.10 |
| tokens evaluated | 1,696,256 |

## Speed and memory

| metric | value |
|---|---|
| tokens / second | 23,597 |
| seconds / batch | 0.0867 |
| parameter memory | 40.1 MB |
| memory per expert (all layers) | 0.2 MB |
| peak device memory | n/a |
| process RSS | 2,969.8 MB |

## Parameters

| metric | value |
|---|---|
| total | 10,035,392 |
| active per token | 1,040,576 |
| active / total | 10.4% |
| experts (total / active) | 184 / 1 |

## Routing

| metric | value |
|---|---|
| aux (balance) loss | 1.0789 |
| overflow routes | 497,033 (14.59%) |
| dropped routes | 497,033 (14.59%) |
| capacity factor | 1.25 |
| expert utilization min / mean / max | 0.0012 / 0.0054 / 0.0228 |

### Per layer

| layer | entropy (norm.) | confidence | load CV | dead | underused | drop rate |
|---|---|---|---|---|---|---|
| 0 | 5.060 (97%) | 0.020 | 0.402 | 0 | 1 | 9.90% |
| 1 | 5.105 (98%) | 0.016 | 0.823 | 0 | 16 | 19.29% |

MoE layers spent 69.087s across experts (88% of the profiled pass).
