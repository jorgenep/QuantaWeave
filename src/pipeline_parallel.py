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

This is the GPipe schedule: all forwards, then all backwards. It has the usual pipeline bubble and holds M micro-batches of
activations per stage (use --activation-checkpointing to trade compute for that memory). Expert capacity is applied per
micro-batch. Not combinable with expert/tensor parallelism. Tested with CPU (Gloo) processes.
"""

import contextlib
import os
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
    def __init__(self, model, ctx: PipelineContext, microbatches: int, autocast: Callable = contextlib.nullcontext) -> None:
        if microbatches < 1:
            raise ValueError("microbatches must be positive")
        if model.pipeline is not ctx and model.pipeline != ctx:
            raise ValueError("the model was not built for this pipeline stage")
        self.model, self.ctx, self.microbatches, self.autocast = model, ctx, microbatches, autocast

    def train_step(self, batch: Tensor) -> dict:
        """Forward and backward over one batch [B, L+1]; gradients accumulate in the parameters."""
        model, ctx, M = self.model, self.ctx, self.microbatches
        if batch.size(0) % M:
            raise ValueError(f"batch size {batch.size(0)} must be divisible by --microbatches {M}")
        size = batch.size(0) // M
        total_layers = model.config.layers
        coef = model.router_aux_loss_coef
        width = model.config.hidden_size
        saved = []
        ce_total = torch.zeros((), dtype=torch.float64)
        aux_local, dropped_total, overflow_total = 0.0, 0, 0

        for index in range(M):
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
            aux_local += float(aux.detach()) / M
            dropped_total += int(torch.stack(dropped).sum())
            overflow_total += int(torch.stack(overflow).sum())
            if ctx.is_last:
                with self.autocast():
                    logits = model.head(hidden)
                cross_entropy = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), tokens[:, 1:].reshape(-1))
                loss = cross_entropy / M + aux_term
                ce_total += float(cross_entropy.detach()) / M
                saved.append((hidden_in, loss, None))
            else:
                dist.send(hidden.detach().float().contiguous(), dst=ctx.stage + 1)
                saved.append((hidden_in, hidden, aux_term))

        for index in reversed(range(M)):
            hidden_in, output, aux_term = saved[index]
            if ctx.is_last:
                output.backward()
            else:
                grad = torch.empty(output.shape, dtype=torch.float32, device=ctx.device)
                dist.recv(grad, src=ctx.stage + 1)
                torch.autograd.backward([output, aux_term], [grad.to(output.dtype), None])
            if not ctx.is_first:
                dist.send(hidden_in.grad.contiguous(), dst=ctx.stage - 1)

        totals = torch.tensor([float(ce_total), aux_local, float(dropped_total), float(overflow_total)], dtype=torch.float64, device=ctx.device)
        dist.all_reduce(totals)
        ce, aux_sum, dropped, overflow = totals.tolist()
        router_aux = aux_sum / total_layers
        return {"loss": ce + coef * router_aux, "ce_loss": ce, "router_aux_loss": router_aux,
                "dropped_routes": int(dropped), "overflow_routes": int(overflow)}


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
