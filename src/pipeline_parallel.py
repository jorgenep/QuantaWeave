"""Pipeline parallelism: consecutive layers live on consecutive ranks and micro-batches flow through them (GPipe).

Launch:  torchrun --nproc_per_node=S src/train_quantweave_moe.py --pipeline-parallel --microbatches M ...
World size = number of stages S. Stage 0 holds the embeddings, the last stage the final norm and lm_head, and layers are
split evenly (earlier stages take the remainder). Every rank draws the same batch; only the first stage reads the
tokens and the last stage the labels.

One step, for M micro-batches of the batch:
  forward   micro-batch i enters stage 0, activations move stage to stage with point-to-point sends;
  backward  micro-batches drain in reverse: the last stage starts backprop from the loss, each stage receives the
            gradient of its output, backpropagates, and sends the gradient of its input upstream.
The router balance loss of every layer joins the loss on the stage that owns it, weighted so the total equals the
single-device loss (mean over all layers). Gradients accumulate across micro-batches, scaled by 1/M.

This is the GPipe schedule by default: all forwards, then all backwards. It has the usual pipeline bubble and holds
M micro-batches of activations per stage (use --activation-checkpointing to trade compute for that memory). Expert
capacity is applied per micro-batch. Not combinable with expert/tensor parallelism. Tested with CPU (Gloo)
processes.

``schedule="1f1b"`` (``--pipeline-schedule 1f1b``) switches to the standard non-interleaved 1F1B ("PipeDream-flush")
schedule instead: the pipeline bubble ratio is the same as GPipe's, but peak activation memory drops from O(M) to
roughly O(S) (the number of stages), since a micro-batch's backward runs as soon as its gradient is available
rather than only after every micro-batch has finished its forward. Each stage independently computes a warm-up
count ``min(M, S - stage - 1)`` forwards, then alternates one forward with one backward (of the oldest still-
pending micro-batch) until every forward has been issued, then drains the remaining backwards — the textbook
schedule (used by Megatron-LM and PipeDream), proven not to deadlock as long as every stage follows it. It computes
exactly the same gradients as GPipe (same sends/receives, just reordered in time), up to floating-point summation
order, which is why it is verified against GPipe's own output rather than an independent reference.
"""

import contextlib
import os
from collections import deque
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PipelineContext:
    stage: int
    num_stages: int
    device: torch.device
    layer_start: int
    layer_end: int
    owns_process_group: bool = False

    @property
    def global_rank(self) -> int:
        return self.stage

    @property
    def global_size(self) -> int:
        return self.num_stages

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.num_stages - 1

    def layout(self) -> dict:
        return {"kind": "pp", "stage": self.stage, "stages": self.num_stages, "layer_start": self.layer_start, "layer_end": self.layer_end}


def partition_layers(total_layers: int, stages: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) layer ranges per stage; the first (total % stages) stages get one extra layer."""
    if stages < 1 or total_layers < stages:
        raise ValueError(f"cannot split {total_layers} layers over {stages} pipeline stages")
    base, extra = divmod(total_layers, stages)
    ranges, start = [], 0
    for stage in range(stages):
        size = base + (1 if stage < extra else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def init_pipeline_parallel(device_arg: str, total_layers: int) -> PipelineContext:
    owns = False
    if not dist.is_initialized():
        missing = [name for name in ("RANK", "WORLD_SIZE") if name not in os.environ]
        if missing:
            raise RuntimeError("--pipeline-parallel needs a process group; launch with torchrun (missing " + ", ".join(missing) + ")")
        local = int(os.environ.get("LOCAL_RANK", 0))
        if device_arg in {"auto", "cuda", "rocm"} and torch.cuda.is_available():
            device, backend = torch.device("cuda", local), "nccl"
            torch.cuda.set_device(device)
        elif device_arg in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
            device, backend = torch.device("xpu", local), "ccl"
        else:
            device, backend = torch.device("cpu"), "gloo"
        dist.init_process_group(backend, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        owns = True
    else:
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else torch.device("cpu")
    stages, stage = dist.get_world_size(), dist.get_rank()
    start, end = partition_layers(total_layers, stages)[stage]
    return PipelineContext(stage, stages, device, start, end, owns)


class PipelineEngine:
    def __init__(
        self, model, ctx: PipelineContext, microbatches: int, autocast: Callable = contextlib.nullcontext,
        schedule: str = "gpipe",
    ) -> None:
        if microbatches < 1:
            raise ValueError("microbatches must be positive")
        if model.pipeline is not ctx and model.pipeline != ctx:
            raise ValueError("the model was not built for this pipeline stage")
        if schedule not in ("gpipe", "1f1b"):
            raise ValueError("schedule must be gpipe or 1f1b")
        self.model, self.ctx, self.microbatches, self.autocast, self.schedule = model, ctx, microbatches, autocast, schedule
        self._totals = {"ce": 0.0, "aux": 0.0, "dropped": 0, "overflow": 0}
        self._pending_sends: list = []   # [(request, tensor)] kept alive until _drain_sends(); see _isend's docstring

    def _isend(self, tensor: Tensor, dst: int) -> None:
        """Non-blocking send, waited on at the end of the step (_drain_sends), not here.

        1F1B interleaves forward (downstream) and backward (upstream) traffic, so two adjacent stages can each be
        mid-send to the other at the same time with neither having posted its matching recv yet. A blocking
        ``dist.send`` deadlocks in exactly that situation (verified: reproduced live with a 2-stage, 4-microbatch
        1F1B run before this fix). ``isend`` returns immediately regardless of the peer's state, so this rank can
        keep making progress (in particular, reach the ``recv`` that unblocks the peer) instead of blocking on a
        send the peer isn't ready for yet; GPipe's strictly one-direction-per-phase traffic was never at risk of
        this, but uses the same helper for consistency and because there is no downside to it here.
        """
        request = dist.isend(tensor, dst=dst)
        self._pending_sends.append((request, tensor))   # tensor kept alive so isend's buffer stays valid until wait()

    def _drain_sends(self) -> None:
        for request, _ in self._pending_sends:
            request.wait()
        self._pending_sends = []

    def _forward_step(self, batch: Tensor, index: int, size: int) -> tuple:
        """Run one micro-batch's forward pass; returns the (hidden_in, output, aux_term) tuple its backward needs."""
        model, ctx, M = self.model, self.ctx, self.microbatches
        width = model.config.hidden_size
        coef = model.router_aux_loss_coef
        total_layers = model.config.layers
        tokens = batch[index * size : (index + 1) * size].to(ctx.device)
        hidden_in = None
        if ctx.is_first:
            with self.autocast():
                hidden = model.embed(tokens)
        else:
            hidden_in = torch.empty(size, tokens.size(1), width, dtype=torch.float32, device=ctx.device)
            dist.recv(hidden_in, src=ctx.stage - 1)
            hidden_in.requires_grad_(True)
            hidden = hidden_in
        with self.autocast():
            hidden, balance, dropped, overflow, _ = model.run_blocks(hidden)
        aux = torch.stack(balance).sum()
        aux_term = coef * aux / total_layers / M
        self._totals["aux"] += float(aux.detach()) / M
        self._totals["dropped"] += int(torch.stack(dropped).sum())
        self._totals["overflow"] += int(torch.stack(overflow).sum())
        if ctx.is_last:
            with self.autocast():
                logits = model.head(hidden)
            cross_entropy = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), tokens[:, 1:].reshape(-1))
            loss = cross_entropy / M + aux_term
            self._totals["ce"] += float(cross_entropy.detach()) / M
            return (hidden_in, loss, None)
        self._isend(hidden.detach().float().contiguous(), dst=ctx.stage + 1)
        return (hidden_in, hidden, aux_term)

    def _backward_step(self, saved: tuple) -> None:
        """Run one micro-batch's backward pass from its saved forward tuple."""
        ctx = self.ctx
        hidden_in, output, aux_term = saved
        if ctx.is_last:
            output.backward()
        else:
            grad = torch.empty(output.shape, dtype=torch.float32, device=ctx.device)
            dist.recv(grad, src=ctx.stage + 1)
            torch.autograd.backward([output, aux_term], [grad.to(output.dtype), None])
        if not ctx.is_first:
            self._isend(hidden_in.grad.contiguous(), dst=ctx.stage - 1)

    def train_step(self, batch: Tensor) -> dict:
        """Forward and backward over one batch [B, L+1]; gradients accumulate in the parameters."""
        ctx, M = self.ctx, self.microbatches
        if batch.size(0) % M:
            raise ValueError(f"batch size {batch.size(0)} must be divisible by --microbatches {M}")
        size = batch.size(0) // M
        self._totals = {"ce": 0.0, "aux": 0.0, "dropped": 0, "overflow": 0}

        if self.schedule == "1f1b":
            self._run_1f1b(batch, size)
        else:
            self._run_gpipe(batch, size)
        self._drain_sends()

        totals = torch.tensor(
            [self._totals["ce"], self._totals["aux"], float(self._totals["dropped"]), float(self._totals["overflow"])],
            dtype=torch.float64, device=ctx.device,
        )
        dist.all_reduce(totals)
        ce, aux_sum, dropped, overflow = totals.tolist()
        router_aux = aux_sum / self.model.config.layers
        coef = self.model.router_aux_loss_coef
        return {"loss": ce + coef * router_aux, "ce_loss": ce, "router_aux_loss": router_aux,
                "dropped_routes": int(dropped), "overflow_routes": int(overflow)}

    def _run_gpipe(self, batch: Tensor, size: int) -> None:
        saved = [self._forward_step(batch, index, size) for index in range(self.microbatches)]
        for index in reversed(range(self.microbatches)):
            self._backward_step(saved[index])

    def _run_1f1b(self, batch: Tensor, size: int) -> None:
        """Non-interleaved 1F1B: warm up min(M, S - stage - 1) forwards, alternate forward/backward, then drain."""
        stage, stages, M = self.ctx.stage, self.ctx.num_stages, self.microbatches
        warmup = min(M, stages - stage - 1)
        queue: deque = deque()
        for index in range(warmup):
            queue.append(self._forward_step(batch, index, size))
        for index in range(warmup, M):
            queue.append(self._forward_step(batch, index, size))
            self._backward_step(queue.popleft())
        while queue:
            self._backward_step(queue.popleft())


def clip_grad_norm_pipeline(model, ctx: PipelineContext, max_norm: float) -> float:
    """Global gradient norm across stages (every parameter lives on exactly one stage), applied identically."""
    square = torch.zeros((), dtype=torch.float64, device=ctx.device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            square = square + parameter.grad.detach().double().pow(2).sum()
    dist.all_reduce(square)
    total = square.sqrt()
    coefficient = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(coefficient.to(parameter.grad.dtype))
    return float(total)
