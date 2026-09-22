"""Serve a QuantaWeave checkpoint over HTTP.

  python src/serve_quantweave.py --latest --host 127.0.0.1 --port 8000
  curl -s localhost:8000/generate -d '{"prompt": "Once upon a time", "temperature": 0}' | python3 -m json.tool
  curl -s localhost:8000/chat -d '{"messages": [{"role": "user", "content": "hi"}]}' | python3 -m json.tool
  curl -N localhost:8000/generate/stream -d '{"prompt": "Once upon a time", "tokens": 200}'   # text/event-stream

One model, one process. Reuses fast_decode.py's KV-cache/CUDA-graph decoder (see chat_quantweave_moe.py) for real
per-token throughput, through the same ChatSession/Settings used by the chat tool, so a request behaves exactly
like the equivalent `chat_quantweave_moe.py -m ...` call. Requests are served one at a time: the decoder's KV cache
is mutable per-request state, so concurrent requests are serialized behind a lock rather than corrupting each
other; generation runs in a worker thread so the event loop can still accept and queue requests while one runs.
This is a single-model reference server, not a batching/multi-tenant inference engine — there is no continuous
batching and no multi-GPU sharding of one server.

Hardening (all opt-in, all off by default — see SECURITY.md before deploying this anywhere real):
  --api-key KEY            require `Authorization: Bearer KEY` or `X-API-Key: KEY` on every route but /health
  --rate-limit N            reject with 429 past N requests per --rate-limit-window seconds per caller
  --request-timeout SECS    give up and return 503 if a request waits this long merely to *start* (see below)
  --max-queue N              reject new requests with 429 once N are already waiting to start
  --moderation-blocklist F   reject prompts/messages matching a regex in F (one per line) with 422 — a basic
                              keyword guardrail, not a safety or alignment solution
  /metrics                   Prometheus text-exposition counters/gauges (also gated by --api-key if set)

What --request-timeout does NOT do: cancel a generation already in progress. Python cannot forcibly stop a
blocking computation running in a worker thread, and fast_decode.py's decode loop has no cooperative cancellation
point, so once a request has the lock it runs to completion (bounded by its own `tokens`/`max_new_tokens`)
regardless of the timeout. The timeout only bounds how long a *new* request waits in the queue to acquire the lock
before giving up — it stops other callers hanging forever behind one slow request, it does not free the server
from that slow request any sooner. Combine it with --max-queue for real backpressure.
"""

import argparse
import asyncio
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from chat_quantweave_moe import (
    DEFAULT_CHECKPOINT,
    ChatSession,
    Settings,
    load_for_chat,
    resolve_model,
)
from hardware import choose_precision, detect
from train_quantweave_moe import select_device


class GenerateRequest(BaseModel):
    prompt: str
    tokens: int = Field(200, ge=1, le=8192)
    temperature: float = Field(0.8, ge=0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    seed: Optional[int] = None
    stop: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    system: str = ""
    tokens: int = Field(200, ge=1, le=8192)
    temperature: float = Field(0.8, ge=0)
    top_k: int = Field(0, ge=0)
    top_p: float = Field(1.0, gt=0, le=1)
    seed: Optional[int] = None


logger = logging.getLogger("quantweave.serve")


class JSONLogFormatter(logging.Formatter):
    """One JSON object per line: {time, level, message, ...whatever the caller passed via `extra=`}. Meant for
    log aggregators (CloudWatch, Loki, etc.); `--log-format text` (the default) keeps plain human-readable lines
    for a terminal instead."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED:
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(log_format: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JSONLogFormatter() if log_format == "json" else logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


class RateLimiter:
    """Sliding-window rate limit: at most ``limit`` calls to ``allow(key)`` per ``window`` seconds, per key.

    In-memory and single-process, matching this server's own single-process design — it does not coordinate
    across multiple replicas of the server. ``limit=None`` disables it (``allow`` always returns True).
    """

    def __init__(self, limit: Optional[int], window: float = 60.0) -> None:
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.limit is None:
            return True
        now = time.monotonic()
        bucket = self.hits[key]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.limit:
            return False
        bucket.append(now)
        return True


class Moderation:
    """A best-effort regex blocklist checked against request text. Not a safety or alignment solution — a basic
    guardrail hook only (see SECURITY.md). Patterns are one plain regex per line (case-insensitive); blank lines
    and '#'-prefixed lines are ignored."""

    def __init__(self, patterns_file: Optional[Path]) -> None:
        self.patterns: list[re.Pattern] = []
        if patterns_file is not None:
            for line in Path(patterns_file).read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    self.patterns.append(re.compile(line, re.IGNORECASE))

    def blocked(self, text: str) -> Optional[str]:
        for pattern in self.patterns:
            if pattern.search(text):
                return pattern.pattern
        return None


class Metrics:
    """Hand-rolled Prometheus text-exposition counters/gauges — no extra dependency, the same habit this project
    already has of writing its own small format encoders (see routing_diagnostics.py's SVG helpers) rather than
    pulling in a client library for something this small."""

    def __init__(self) -> None:
        self.requests_total: dict[str, int] = defaultdict(int)          # "route:status" -> count
        self.errors_total: dict[str, int] = defaultdict(int)            # route -> count (status >= 400)
        self.request_seconds_sum: dict[str, float] = defaultdict(float)
        self.request_seconds_count: dict[str, int] = defaultdict(int)
        self.tokens_generated_total = 0
        self.requests_in_flight = 0

    def observe(self, route: str, status: int, seconds: float, tokens: int = 0) -> None:
        self.requests_total[f"{route}:{status}"] += 1
        if status >= 400:
            self.errors_total[route] += 1
        self.request_seconds_sum[route] += seconds
        self.request_seconds_count[route] += 1
        self.tokens_generated_total += tokens

    def render(self) -> str:
        lines = [
            "# HELP quantweave_requests_total Total requests by route and status.",
            "# TYPE quantweave_requests_total counter",
        ]
        for key, count in sorted(self.requests_total.items()):
            route, status = key.rsplit(":", 1)
            lines.append(f'quantweave_requests_total{{route="{route}",status="{status}"}} {count}')
        lines += [
            "# HELP quantweave_errors_total Total error responses (status >= 400) by route.",
            "# TYPE quantweave_errors_total counter",
        ]
        for route, count in sorted(self.errors_total.items()):
            lines.append(f'quantweave_errors_total{{route="{route}"}} {count}')
        lines += [
            "# HELP quantweave_request_seconds_sum Total request handling seconds by route.",
            "# TYPE quantweave_request_seconds_sum counter",
        ]
        for route in sorted(self.request_seconds_sum):
            lines.append(f'quantweave_request_seconds_sum{{route="{route}"}} {self.request_seconds_sum[route]:.6f}')
            lines.append(f'quantweave_request_seconds_count{{route="{route}"}} {self.request_seconds_count[route]}')
        lines += [
            "# HELP quantweave_tokens_generated_total Total tokens generated across all requests.",
            "# TYPE quantweave_tokens_generated_total counter",
            f"quantweave_tokens_generated_total {self.tokens_generated_total}",
            "# HELP quantweave_requests_in_flight Requests currently being handled.",
            "# TYPE quantweave_requests_in_flight gauge",
            f"quantweave_requests_in_flight {self.requests_in_flight}",
        ]
        return "\n".join(lines) + "\n"


class Server:
    """Holds the loaded model and serializes access to its stateful decoder across requests."""

    def __init__(
        self, checkpoint, adapter, device: torch.device, precision: str, use_cache: bool = True, *,
        api_key: Optional[str] = None, rate_limit: Optional[int] = None, rate_limit_window: float = 60.0,
        request_timeout: Optional[float] = None, max_queue: Optional[int] = None,
        moderation_patterns: Optional[Path] = None,
    ) -> None:
        self.checkpoint, self.adapter, self.device = checkpoint, adapter, device
        model, tokenizer = load_for_chat(checkpoint, adapter, 0, device)
        self.precision = choose_precision(detect(device.type), precision)
        self.complete_session = ChatSession(model, tokenizer, device, self.precision, Settings(mode="complete"), use_cache)
        self.chat_session = ChatSession(model, tokenizer, device, self.precision, Settings(mode="chat"), use_cache)
        self.lock = asyncio.Lock()
        self.started_at = time.time()
        self.requests_served = 0
        self.api_key = api_key
        self.rate_limiter = RateLimiter(rate_limit, rate_limit_window)
        self.request_timeout = request_timeout
        self.max_queue = max_queue
        self.waiting = 0
        self.moderation = Moderation(moderation_patterns)
        self.metrics = Metrics()

    def info(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint), "adapter": str(self.adapter) if self.adapter else None,
            "device": str(self.device), "precision": self.precision, "decoding": self.complete_session.cache_note,
            "tokenizer": self.complete_session.tokenizer.kind, "vocab_size": self.complete_session.tokenizer.vocab_size,
            "context_length": self.complete_session.model.config.max_sequence_length - 1,
            "uptime_seconds": time.time() - self.started_at, "requests_served": self.requests_served,
            "auth_required": self.api_key is not None, "rate_limit": self.rate_limiter.limit,
            "request_timeout": self.request_timeout, "max_queue": self.max_queue,
        }

    async def _acquire(self) -> None:
        """Wait for the generation lock, honouring --max-queue and --request-timeout (see module docstring for
        exactly what the timeout does and does not bound).

        ``max_queue`` only rejects a request that would actually have to *wait* (the lock is already held) — a
        request that finds the lock free is always accepted regardless of ``max_queue``, including ``max_queue=0``
        (reject anything that would have to queue, but never the one request that doesn't need to). Checking
        ``lock.locked()`` and then awaiting ``acquire()`` is not atomic, so two requests arriving in the same
        scheduling tick could in principle both see the lock free and both proceed — an acceptable soft limit for
        a best-effort, single-process backpressure control, not a hard guarantee.
        """
        if not self.lock.locked():
            await self.lock.acquire()
            return
        if self.max_queue is not None and self.waiting >= self.max_queue:
            raise HTTPException(429, f"too many requests already queued (max {self.max_queue})")
        self.waiting += 1
        try:
            if self.request_timeout is not None:
                try:
                    await asyncio.wait_for(self.lock.acquire(), timeout=self.request_timeout)
                except asyncio.TimeoutError:
                    raise HTTPException(
                        503, f"timed out after {self.request_timeout}s waiting to start "
                             "(the server may still be busy with an earlier request)"
                    ) from None
            else:
                await self.lock.acquire()
        finally:
            self.waiting -= 1

    async def generate(self, request: GenerateRequest) -> dict:
        blocked = self.moderation.blocked(request.prompt)
        if blocked is not None:
            raise HTTPException(422, f"prompt blocked by moderation pattern: {blocked}")
        await self._acquire()
        try:
            self.complete_session.settings = Settings(
                mode="complete", tokens=request.tokens, temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, seed=request.seed, stop=request.stop,
            )
            reply = await asyncio.to_thread(self.complete_session.send, request.prompt)
            self.requests_served += 1
        finally:
            self.lock.release()
        return reply_payload(reply)

    async def chat(self, request: ChatRequest) -> dict:
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(422, "messages must end with a 'user' turn")
        blocked = self.moderation.blocked(request.messages[-1].content)
        if blocked is not None:
            raise HTTPException(422, f"message blocked by moderation pattern: {blocked}")
        await self._acquire()
        try:
            self.chat_session.reset()
            self.chat_session.settings = Settings(
                mode="chat", tokens=request.tokens, temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, seed=request.seed, system=request.system,
            )
            for turn in request.messages[:-1]:
                if turn.role == "user":
                    self.chat_session.history.append((turn.content, ""))
                elif turn.role == "assistant" and self.chat_session.history:
                    user, _ = self.chat_session.history[-1]
                    self.chat_session.history[-1] = (user, turn.content)
            reply = await asyncio.to_thread(self.chat_session.send, request.messages[-1].content)
            self.requests_served += 1
        finally:
            self.lock.release()
        return reply_payload(reply)

    async def stream_generate(self, request: GenerateRequest):
        """Server-sent events: one ``data: {...}`` line per piece of text, then a final ``event: done`` line."""
        blocked = self.moderation.blocked(request.prompt)
        if blocked is not None:
            raise HTTPException(422, f"prompt blocked by moderation pattern: {blocked}")
        await self._acquire()
        try:
            self.complete_session.settings = Settings(
                mode="complete", tokens=request.tokens, temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, seed=request.seed, stop=request.stop,
            )
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def on_text(piece: str) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, piece)

            async def run() -> None:
                reply = await asyncio.to_thread(self.complete_session.send, request.prompt, on_text)
                loop.call_soon_threadsafe(queue.put_nowait, ("done", reply))   # sentinel carries the reply itself

            task = asyncio.ensure_future(run())
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, tuple) and item[0] == "done":
                        reply = item[1]
                        break
                    yield f"data: {json.dumps({'text': item})}\n\n"
                self.requests_served += 1
                yield f"event: done\ndata: {json.dumps(reply_payload(reply))}\n\n"
            finally:
                if not task.done():
                    task.cancel()
        finally:
            self.lock.release()


def reply_payload(reply) -> dict:
    return {
        "text": reply.response, "tokens": reply.tokens, "seconds": reply.seconds,
        "tokens_per_second": reply.tokens_per_second, "stop_reason": reply.stop_reason,
        "unknown_characters": reply.unknown_characters, "truncated_context": reply.truncated_context,
    }


def build_app(server: Server) -> FastAPI:
    app = FastAPI(title="QuantaWeave inference server")

    def require_auth(request: Request) -> None:
        if server.api_key is None:
            return
        header = request.headers.get("authorization", "")
        key = header[7:] if header.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if key != server.api_key:
            raise HTTPException(401, "invalid or missing API key (Authorization: Bearer <key> or X-API-Key: <key>)")

    def require_rate_limit(request: Request) -> None:
        client_key = request.headers.get("x-api-key") or (request.client.host if request.client else "unknown")
        if not server.rate_limiter.allow(client_key):
            raise HTTPException(429, "rate limit exceeded")

    async def timed(route: str, handler):
        """Runs `handler()`, records it in server.metrics, logs one structured line, and re-raises whatever it
        raised (as the caller sees)."""
        request_id = uuid.uuid4().hex[:12]
        server.metrics.requests_in_flight += 1
        started = time.perf_counter()
        status = 200
        try:
            result = await handler()
            return result
        except HTTPException as error:
            status = error.status_code
            raise
        except Exception:
            status = 500
            raise
        finally:
            seconds = time.perf_counter() - started
            server.metrics.requests_in_flight -= 1
            server.metrics.observe(route, status, seconds)
            level = logging.INFO if status < 400 else logging.WARNING
            logger.log(level, f"{route} {status} {seconds:.3f}s", extra={
                "request_id": request_id, "route": route, "status": status, "seconds": round(seconds, 4),
            })

    @app.get("/health")
    async def health():
        return {"status": "ok", **server.info()}

    @app.get("/metrics")
    async def metrics(_auth: None = Depends(require_auth)):
        return PlainTextResponse(server.metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/generate")
    async def generate(request: GenerateRequest, _auth: None = Depends(require_auth), _rate: None = Depends(require_rate_limit)):
        async def handler():
            try:
                return await server.generate(request)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

        return await timed("/generate", handler)

    @app.post("/chat")
    async def chat(request: ChatRequest, _auth: None = Depends(require_auth), _rate: None = Depends(require_rate_limit)):
        async def handler():
            try:
                return await server.chat(request)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

        return await timed("/chat", handler)

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest, _auth: None = Depends(require_auth), _rate: None = Depends(require_rate_limit)):
        return StreamingResponse(server.stream_generate(request), media_type="text/event-stream")

    app.state.server = server
    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help=f"a checkpoint directory (default {DEFAULT_CHECKPOINT})")
    parser.add_argument("--run", help="an archived run: its epoch id under artifacts/runs, or the folder path")
    parser.add_argument("--latest", action="store_true", help="the newest archived run that contains a model")
    parser.add_argument("--adapter", type=Path, help="a QLoRA adapter directory")
    parser.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--no-cache", action="store_true", help="plain forward pass per token instead of the KV cache / CUDA graph")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    hardening = parser.add_argument_group("hardening (see SECURITY.md)")
    hardening.add_argument("--api-key", help="require this key via Authorization: Bearer <key> or X-API-Key: <key> on every route but /health")
    hardening.add_argument("--rate-limit", type=int, help="reject with 429 past this many requests per --rate-limit-window seconds per caller")
    hardening.add_argument("--rate-limit-window", type=float, default=60.0)
    hardening.add_argument("--request-timeout", type=float, help="give up and return 503 if a request waits this long merely to start (see module docstring)")
    hardening.add_argument("--max-queue", type=int, help="reject new requests with 429 once this many are already waiting to start")
    hardening.add_argument("--moderation-blocklist", type=Path, help="a file of regex patterns (one per line); a matching prompt/message is rejected with 422")
    hardening.add_argument("--log-format", choices=("text", "json"), default="text", help="'json' for one structured log line per request (log aggregators); 'text' for a terminal")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_logging(args.log_format)
    checkpoint, adapter = resolve_model(args.checkpoint, args.adapter, args.run, args.latest, args.runs_dir)
    device = select_device(args.device)
    server = Server(
        checkpoint, adapter, device, args.precision, use_cache=not args.no_cache,
        api_key=args.api_key, rate_limit=args.rate_limit, rate_limit_window=args.rate_limit_window,
        request_timeout=args.request_timeout, max_queue=args.max_queue, moderation_patterns=args.moderation_blocklist,
    )
    print(f"model: {checkpoint}" + (f" + adapter {adapter}" if adapter else "") + f"  device: {device}  decoding: {server.complete_session.cache_note}")
    if server.api_key is None:
        print("WARNING: no --api-key set, this server accepts unauthenticated requests from anyone who can reach it. "
              "See SECURITY.md before exposing it beyond localhost.")
    app = build_app(server)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
