# Security

This is a research/experimentation framework, not a hardened multi-tenant service. Read this before deploying
anything built here where the inputs (checkpoints, training data, HTTP requests) might come from someone else.

## Checkpoint loading is equivalent to unpickling an untrusted file

Every checkpoint this project reads or writes (`model.pt`, `adapter.pt`, sharded weights, quantized weights) is a
plain `torch.save`'d Python dict that mixes tensors with ordinary objects (RNG state, config snapshots, optimizer
state, run metadata). `torch.load`'s default `weights_only=True` restricted unpickler rejects that, so every
loader in `src/` goes through one place, `checkpoint_io.load_checkpoint`, which uses `weights_only=False`.

**`weights_only=False` means a malicious checkpoint file can execute arbitrary code the moment it is loaded**,
before any of this project's own validation runs — the same risk as `pickle.load` on a file you don't control.

**Rule:** never load a checkpoint (`--checkpoint`, `--adapter`, `--run`, `--latest`, a fine-tuned adapter
directory, an exported bundle) that you did not produce yourself or do not otherwise trust, exactly as you would
never unpickle a file from an untrusted source. This applies to every script that takes `--checkpoint`/`--adapter`
(`chat_quantweave_moe.py`, `serve_quantweave.py`, `benchmark_quantweave_moe.py`, `distill_quantweave_moe.py`,
`export_quantweave.py`, `finetune_quantweave_moe.py`, and the trainer's own `--resume`), and to any future workflow
that downloads or receives a checkpoint from somewhere else. If that need arises, harden
`checkpoint_io.load_checkpoint` first (a restricted unpickler allow-listing only the classes this project actually
saves, or moving to `safetensors` for the tensor payload and a separate signed/validated file for the rest) rather
than working around it per call site.

## `serve_quantweave.py` is a reference server, not a hardened edge service

- **No authentication or rate limiting beyond what you put in front of it.** Run it behind a reverse proxy that
  terminates TLS and enforces auth/rate limits, on a network the caller doesn't reach directly. It has no notion
  of separate users or API keys of its own.
- **No content moderation or safety filtering.** The model is a from-scratch base/pretrain pipeline with no
  RLHF/instruction-alignment step and no output filtering. Nothing stops it from generating whatever its training
  data and sampling settings produce. Do not point it at real users without adding a moderation layer in front of
  it (input and output), appropriate to what you're deploying it for.
- **No prompt-injection defenses.** `/chat` and `/generate` pass the caller's text straight into the model's
  context with no isolation between "system" and "user" content beyond the chat template itself.
- **Single-process, single-model, no request isolation.** One slow or stuck generation can starve every other
  caller (there is no per-request timeout in the base server; see the request-timeout/backpressure work tracked
  separately). Do not run it multi-tenant (different trust levels sharing one instance) without adding those
  controls.

## Training data

`data_pipeline.py` has no PII-scrubbing or license-provenance tracking built in by default (a basic, explicitly
best-effort regex-based redaction pass now exists — see README's "Data governance" section — but it is not a
substitute for reviewing your data). Treat any corpus you point the trainer at as becoming, in a diffuse and
unrecoverable way, part of the model's weights. Don't train on data you don't have the rights to train on, or that
contains information you would not want a downstream user of the model to be able to extract.

## Dependencies

No automated dependency vulnerability scanning is configured. `pyproject.toml`'s optional dependency groups pull
in `torch`, `transformers`, `fastapi`, `uvicorn`, `onnxruntime`, and others; keep them updated and run your own
scanner (`pip-audit`, GitHub's Dependabot, etc.) before treating a deployment as production-hardened.

## Reporting

This is a research project without a formal disclosure program. If you find an issue, open one where the rest of
the project's issues live, or contact whoever operates your deployment of it directly — don't file it somewhere
public if it's a real exploit against a live deployment you don't control.
