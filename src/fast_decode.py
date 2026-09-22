"""Incremental decoding for QuantaWeave: a KV cache, one-token steps, and (on CUDA) a captured CUDA graph.

The ordinary forward pass re-reads the whole context for every generated token and, for a model this small, spends nearly
all its time launching kernels. FastDecoder computes one token at a time against cached keys and values, reads the model's
own weights (nothing is copied into a different format), and on CUDA records the whole one-token step as a CUDA graph so it
replays with a single launch. The step is first passed through torch.compile (a few seconds), which fuses its many tiny
kernels; if compilation fails the plain graph is used.

Inference differs from training in two deliberate ways:
  * no expert capacity: a token is never dropped from its experts (capacity is a training-time load-balancing device, and
    with it the answer for one token would depend on the tokens that come after it);
  * a context of ``max_sequence_length - 1`` tokens. Training computes its loss at positions 0 .. L-1 of an L+1 token window,
    so the output at the last position was never trained (its loss on real text is ~6 nats against ~0.5 elsewhere).

Sampling can run inside the graph too (``FastDecoder.stream``): the sampled token feeds the next step on the GPU and the host
only synchronises once per batch of ``BATCH`` tokens. Gumbel-max sampling uses uniform noise drawn outside the graph from an
ordinary seeded generator, so seeds stay reproducible; temperature / top-k / top-p apply as in ``filter_logits``. Stop
conditions are checked per batch (tokens after a stop are discarded), so streaming arrives in small bursts.

When the context fills up, the oldest half is dropped and the rest is re-read (positions are absolute, so cached keys cannot
just be shifted); that costs a short burst of steps once every half-context of generated text.

Handles the standard model, including quantized (int8/int4) and LoRA-adapted experts. Batch size 1.
"""

import contextlib
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from quantweave_moe_model import QuantaWeaveMoEForCausalLM


BATCH = 8                              # tokens generated per host synchronisation
DENSE_PREFILL_LIMIT = 4096             # experts x tokens up to which prefill evaluates every expert densely


def filter_logits(logits: Tensor, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """Top-k then nucleus (top-p) filtering of a 1-D logit vector; removed entries become -inf. Safe inside a CUDA graph."""
    if top_k and top_k < logits.numel():
        threshold = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probabilities = sorted_logits.softmax(dim=-1)
        remove = probabilities.cumsum(dim=-1) - probabilities > top_p          # keeps the token that crosses top_p
        logits = torch.empty_like(logits).scatter_(0, order, sorted_logits.masked_fill(remove, float("-inf")))
    return logits


@contextlib.contextmanager
def capture_mode():
    """The one autograd state the compiled step is warmed up and captured in. torch.compile guards on grad/inference mode, and a
    recompile during CUDA graph capture is an error, so callers running under inference_mode must not leak that in here."""
    with torch.inference_mode(False), torch.no_grad():
        yield


class Unsupported(Exception):
    """The model uses a layout FastDecoder does not handle (tensor-parallel attention, pipeline stages, ...)."""


def layer_weight(module: nn.Module) -> Tensor:
    """The effective dense weight of a linear-like module (plain, quantized, or LoRA-adapted)."""
    if isinstance(module, nn.Linear):
        return module.weight.detach()
    if hasattr(module, "merged_weight"):
        return module.merged_weight()
    if hasattr(module, "dequantize"):
        return module.dequantize()
    raise Unsupported(f"cannot read weights of {type(module).__name__}")


def precision_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "auto":
        precision = "bf16" if device.type in {"cuda", "xpu"} else "fp32"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


class _Layer:
    pass


class FastDecoder:
    def __init__(self, model: QuantaWeaveMoEForCausalLM, device: torch.device, precision: str = "auto", use_graph: Optional[bool] = None,
                 compile: Optional[bool] = None) -> None:
        if not isinstance(model, QuantaWeaveMoEForCausalLM) or model.expert_parallel is not None or model.pipeline is not None:
            raise Unsupported("only a whole single-process QuantaWeave model is supported")
        config = model.config
        self.blocks = list(model.blocks)
        self.device = device
        self.dtype = precision_dtype(device, precision)
        self.capacity = config.max_sequence_length - 1          # positions 0 .. L-1 are the ones training taught
        self.heads = config.attention_heads
        self.hidden = config.hidden_size
        self.head_dim = config.hidden_size // config.attention_heads
        self.vocab = config.vocab_size
        dtype = self.dtype
        with torch.no_grad():
            self.token_embedding = model.token_embedding.weight.detach().float()
            self.position_embedding = model.position_embedding.weight.detach().float()
            self.final_norm = (model.final_norm.weight.detach().float(), model.final_norm.bias.detach().float())
            self.lm_head = layer_weight(model.lm_head).to(dtype)
            self.layers: list[_Layer] = []
            for block in model.blocks:
                if not isinstance(block.attention, nn.MultiheadAttention):
                    raise Unsupported("tensor-parallel attention")
                moe = block.moe
                layer = _Layer()
                layer.norm1 = (block.attention_norm.weight.detach().float(), block.attention_norm.bias.detach().float())
                layer.norm2 = (block.moe_norm.weight.detach().float(), block.moe_norm.bias.detach().float())
                attention = block.attention
                layer.wqkv, layer.bqkv = attention.in_proj_weight.detach().to(dtype), attention.in_proj_bias.detach().to(dtype)
                layer.wo, layer.bo = attention.out_proj.weight.detach().to(dtype), attention.out_proj.bias.detach().to(dtype)
                layer.router = moe.router.weight.detach().float()
                layer.temperature = moe.router_temperature
                layer.bias = moe.expert_bias.detach().float().to(device) if moe.expert_bias is not None else None
                layer.top_k = moe.top_k
                layer.gate = torch.stack([layer_weight(e.gate) for e in moe.experts]).to(dtype)
                layer.up = torch.stack([layer_weight(e.up) for e in moe.experts]).to(dtype)
                layer.down = torch.stack([layer_weight(e.down) for e in moe.experts]).to(dtype)
                layer.keys = torch.zeros(self.heads, self.capacity, self.head_dim, dtype=dtype, device=device)
                layer.values = torch.zeros_like(layer.keys)
                self.layers.append(layer)
            self.positions = torch.arange(self.capacity, device=device)
        self.token_in = torch.zeros(1, dtype=torch.long, device=device)
        self.position_in = torch.zeros(1, dtype=torch.long, device=device)
        self.logits = torch.zeros(self.vocab, dtype=torch.float32, device=device)
        self.noise = torch.zeros(BATCH, self.vocab, dtype=torch.float32, device=device)     # uniform noise for Gumbel-max sampling
        self.sampled = torch.zeros(BATCH, dtype=torch.long, device=device)                  # tokens sampled by the last batch
        self.slot = torch.zeros(1, dtype=torch.long, device=device)                         # which noise row / output slot is next
        self.next_token = torch.zeros(1, dtype=torch.long, device=device)
        self.logit_mask = torch.zeros(self.vocab, dtype=torch.float32, device=device)       # -inf for ids that must never be sampled
        self._programs: dict = {}
        self.length = 0
        self.tokens: list[int] = []
        self.graph = None
        self.compiled = False
        self.compile_error: Optional[str] = None
        self._step_fn = self._step
        if use_graph is None:
            use_graph = device.type == "cuda"
        self.graph_enabled = use_graph
        if use_graph:
            with capture_mode():
                self._capture(device.type == "cuda" if compile is None else compile)

    # ---- the one-token step ------------------------------------------------------------------------------
    def _step(self) -> Tensor:
        dtype, heads, head_dim, hidden = self.dtype, self.heads, self.head_dim, self.hidden
        x = self.token_embedding[self.token_in] + self.position_embedding[self.position_in]        # [1, H] fp32
        visible = (self.positions <= self.position_in).view(1, 1, self.capacity)
        for layer in self.layers:
            normed = F.layer_norm(x, (hidden,), *layer.norm1).to(dtype)
            q, k, v = F.linear(normed, layer.wqkv, layer.bqkv).view(3, heads, head_dim).unbind(0)
            layer.keys.index_copy_(1, self.position_in, k.unsqueeze(1))
            layer.values.index_copy_(1, self.position_in, v.unsqueeze(1))
            attended = F.scaled_dot_product_attention(q.unsqueeze(1), layer.keys, layer.values, attn_mask=visible)
            x = x + F.linear(attended.reshape(1, hidden), layer.wo, layer.bo).float()

            normed = F.layer_norm(x, (hidden,), *layer.norm2)
            scores = F.linear(normed, layer.router) / layer.temperature                             # routing stays fp32
            probabilities = scores.softmax(dim=-1)
            chosen = (scores + layer.bias if layer.bias is not None else scores).topk(layer.top_k, dim=-1).indices
            weights = probabilities.gather(1, chosen)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            picked = chosen[0]
            inputs = normed[0].to(dtype)
            gate = torch.einsum("kfh,h->kf", layer.gate[picked], inputs)
            up = torch.einsum("kfh,h->kf", layer.up[picked], inputs)
            expert_out = torch.einsum("khf,kf->kh", layer.down[picked], F.silu(gate) * up)
            x = x + (expert_out.float() * weights[0][:, None]).sum(dim=0, keepdim=True)
        x = F.layer_norm(x, (hidden,), *self.final_norm)
        return F.linear(x.to(dtype), self.lm_head).float()[0]

    @torch.no_grad()
    def _capture(self, compile: bool) -> None:
        """Record the step as a CUDA graph. Everything it touches lives in preallocated buffers, and nothing in it syncs."""
        step = self._step
        if compile:
            try:
                candidate = torch.compile(self._step, fullgraph=True, dynamic=False)
                self._warm_up(candidate)                         # compiles here, so a failure is caught before recording
                step, self.compiled = candidate, True
            except Exception as error:                           # no compiler / unsupported op: the plain graph still works
                self.compile_error = f"{type(error).__name__}: {str(error)[:200]}"
        if not self.compiled:
            self._warm_up(step)
        self._step_fn = step
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.logits.copy_(step())
        self.graph = graph

    @staticmethod
    def _warm_up(step) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):                                   # kernels and the allocator must be warm before recording
                step()
        torch.cuda.current_stream().wait_stream(stream)

    @torch.no_grad()
    def _prefill(self, tokens: list[int]) -> None:
        """Forget everything and read ``tokens`` in one pass: attention over all of them at once, keys and values left in the cache.

        Same arithmetic as the one-token step (bf16/fp16 matmuls, fp32 norms and routing, no expert capacity), but a few
        milliseconds for the whole context instead of one launch-bound step per token."""
        self.reset()
        n = len(tokens)
        if n == 0:
            return
        hidden, heads = self.hidden, self.heads
        ids = torch.tensor(tokens, dtype=torch.long, device=self.device)
        x = self.token_embedding[ids] + self.position_embedding[:n]                              # [n, H] fp32
        # dense mode evaluates every expert and mixes by router weight: no capacity, no sort, no host sync. It does E/k times the
        # expert arithmetic, so it is only used while that stays cheap (small expert counts or short contexts)
        dense = self.blocks[0].moe.num_experts * n <= DENSE_PREFILL_LIMIT
        saved = [(block.moe, block.moe.drop_overflow_tokens, block.moe.static_dispatch) for block in self.blocks]
        for moe, _, _ in saved:
            moe.drop_overflow_tokens = False
            moe.static_dispatch = dense
        try:
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
                for block, layer in zip(self.blocks, self.layers):
                    normed = F.layer_norm(x, (hidden,), *layer.norm1).to(self.dtype)
                    q, k, v = (t.transpose(0, 1) for t in F.linear(normed, layer.wqkv, layer.bqkv).view(n, 3, heads, self.head_dim).unbind(1))
                    layer.keys[:, :n] = k
                    layer.values[:, :n] = v
                    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)          # [heads, n, head_dim]
                    x = x + F.linear(attended.transpose(0, 1).reshape(n, hidden), layer.wo, layer.bo).float()
                    x = x + block.moe(F.layer_norm(x, (hidden,), *layer.norm2).unsqueeze(0))[0][0].float()
        finally:
            for moe, enforce, static in saved:
                moe.drop_overflow_tokens, moe.static_dispatch = enforce, static
        self.length, self.tokens = n, list(tokens)

    @torch.no_grad()
    def _run(self, token: int) -> None:
        self.token_in.fill_(token)
        self.position_in.fill_(self.length)
        if self.graph is not None:
            self.graph.replay()
        else:
            self.logits.copy_(self._step())
        self.length += 1
        self.tokens.append(token)

    # ---- interface ------------------------------------------------------------------------------------------
    @property
    def uses_graph(self) -> bool:
        return self.graph is not None

    @property
    def description(self) -> str:
        if self.graph is None:
            return "KV cache"
        return "KV cache + CUDA graph" + (" + torch.compile" if self.compiled else "")

    def reset(self) -> None:
        self.length = 0
        self.tokens = []

    def feed(self, ids: list[int]) -> Tensor:
        """Forget everything, read ``ids`` (the last ``capacity`` of them) and return the logits for the next token."""
        window = ids[-self.capacity:]
        if not window:
            raise ValueError("nothing to decode from")
        self._prefill(window[:-1])
        self._run(window[-1])
        return self.logits

    def append(self, token: int) -> Tensor:
        """Add one token and return the logits for the one after it. The returned tensor is reused: consume it before the next call."""
        if self.length >= self.capacity:
            self._prefill(self.tokens[-(self.capacity // 2):])
        self._run(token)
        return self.logits

    # ---- sampling inside the step -------------------------------------------------------------------------------
    def _tail(self, logits: Tensor, temperature: float, top_k: int, top_p: float) -> None:
        """Choose the next token on the device and feed it (and the next position) to the following step."""
        logits = logits + self.logit_mask
        if temperature <= 0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            uniform = self.noise[self.slot][0].clamp(1e-20, 1.0 - 1e-7)
            token = (filter_logits(logits / temperature, top_k, top_p) - torch.log(-torch.log(uniform))).argmax(dim=-1, keepdim=True)
        self.sampled.index_copy_(0, self.slot, token)
        self.next_token.copy_(token)
        self.token_in.copy_(token)
        self.position_in.add_(1)
        self.slot.copy_((self.slot + 1) % BATCH)

    @torch.no_grad()
    def _program(self, key: tuple):
        """One step that also samples, for these sampling settings: a captured graph on CUDA, otherwise a plain function."""
        if key in self._programs:
            return self._programs[key]
        temperature, top_k, top_p = key
        if self.graph is None:
            def run() -> None:
                self._tail(self._step(), temperature, top_k, top_p)
        else:
            if len(self._programs) >= 6:
                self._programs.pop(next(iter(self._programs)))
            graph = torch.cuda.CUDAGraph()
            with capture_mode(), torch.cuda.graph(graph):
                self._tail(self._step_fn(), temperature, top_k, top_p)
            run = graph.replay
        self._programs[key] = run
        return run

    def warm(self, temperature: float, top_k: int = 0, top_p: float = 1.0, duration: int = 400) -> None:
        """Prepare these sampling settings now: capture the program and run a short throwaway generation, so the one-time costs
        (graph capture, lazy CUDA kernel loading, a laptop GPU's clocks ramping up from idle) are paid at start-up rather than by
        the first real request. ``duration`` is how many tokens of throwaway generation to run on a GPU."""
        self._program((float(temperature), int(top_k), float(top_p)))
        for _ in self.stream([0, 0], duration if self.device.type == "cuda" else BATCH + 2, temperature=temperature, top_k=top_k, top_p=top_p):
            pass
        self.reset()

    @torch.no_grad()
    def stream(self, ids: list[int], max_new: int, *, temperature: float, top_k: int = 0, top_p: float = 1.0, valid_ids: Optional[int] = None,
               unk_id: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Iterator[int]:
        """Read ``ids`` and yield up to ``max_new`` sampled tokens.

        Ids at or above ``valid_ids`` and ``unk_id`` are never sampled. The generator (if any) supplies the sampling noise, so a
        given seed reproduces the same tokens. Tokens are produced ``BATCH`` at a time; closing the iterator early is fine."""
        if not ids:
            raise ValueError("nothing to decode from")
        program = self._program((float(temperature), int(top_k), float(top_p)))
        self.logit_mask.zero_()
        if valid_ids is not None and valid_ids < self.vocab:
            self.logit_mask[valid_ids:] = float("-inf")
        if unk_id is not None and 0 <= unk_id < self.vocab:
            self.logit_mask[unk_id] = float("-inf")

        def refill() -> None:
            if temperature > 0:
                self.noise.copy_(torch.rand(BATCH, self.vocab, generator=generator, device=self.device))
            self.slot.zero_()

        prompt = ids[-self.capacity:]
        self._prefill(prompt[:-1])
        refill()
        self.token_in.fill_(prompt[-1])
        self.position_in.fill_(self.length)
        program()                                                # reads the last prompt token, samples the first new one
        self.length += 1
        self.tokens.append(prompt[-1])
        pending = int(self.next_token)                           # the one host synchronisation before the first token
        produced = 0
        while True:
            yield pending
            produced += 1
            if produced >= max_new:
                return
            if self.length >= self.capacity:                     # no room to read `pending`: drop the oldest half and re-read the rest
                self._prefill(self.tokens[-(self.capacity // 2):])
                self.token_in.fill_(pending)
                self.position_in.fill_(self.length)
            count = min(BATCH, max_new - produced, self.capacity - self.length)
            refill()
            for _ in range(count):
                program()
            batch = self.sampled[:count].tolist()                # the one synchronisation for this batch
            self.tokens.append(pending)
            self.tokens.extend(batch[:-1])
            self.length += count
            for token in batch[:-1]:
                yield token
                produced += 1
                if produced >= max_new:
                    return
            pending = batch[-1]
