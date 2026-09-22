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
batching, no multi-GPU sharding of one server, and no auth beyond what you put in front of it (a reverse proxy).
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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


class Server:
    """Holds the loaded model and serializes access to its stateful decoder across requests."""

    def __init__(self, checkpoint, adapter, device: torch.device, precision: str, use_cache: bool = True) -> None:
        self.checkpoint, self.adapter, self.device = checkpoint, adapter, device
        model, tokenizer = load_for_chat(checkpoint, adapter, 0, device)
        self.precision = choose_precision(detect(device.type), precision)
        self.complete_session = ChatSession(model, tokenizer, device, self.precision, Settings(mode="complete"), use_cache)
        self.chat_session = ChatSession(model, tokenizer, device, self.precision, Settings(mode="chat"), use_cache)
        self.lock = asyncio.Lock()
        self.started_at = time.time()
        self.requests_served = 0

    def info(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint), "adapter": str(self.adapter) if self.adapter else None,
            "device": str(self.device), "precision": self.precision, "decoding": self.complete_session.cache_note,
            "tokenizer": self.complete_session.tokenizer.kind, "vocab_size": self.complete_session.tokenizer.vocab_size,
            "context_length": self.complete_session.model.config.max_sequence_length - 1,
            "uptime_seconds": time.time() - self.started_at, "requests_served": self.requests_served,
        }

    async def generate(self, request: GenerateRequest) -> dict:
        async with self.lock:
            self.complete_session.settings = Settings(
                mode="complete", tokens=request.tokens, temperature=request.temperature, top_k=request.top_k,
                top_p=request.top_p, seed=request.seed, stop=request.stop,
            )
            reply = await asyncio.to_thread(self.complete_session.send, request.prompt)
            self.requests_served += 1
        return reply_payload(reply)

    async def chat(self, request: ChatRequest) -> dict:
        if not request.messages or request.messages[-1].role != "user":
            raise HTTPException(422, "messages must end with a 'user' turn")
        async with self.lock:
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
        return reply_payload(reply)

    async def stream_generate(self, request: GenerateRequest):
        """Server-sent events: one ``data: {...}`` line per piece of text, then a final ``event: done`` line."""
        async with self.lock:
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


def reply_payload(reply) -> dict:
    return {
        "text": reply.response, "tokens": reply.tokens, "seconds": reply.seconds,
        "tokens_per_second": reply.tokens_per_second, "stop_reason": reply.stop_reason,
        "unknown_characters": reply.unknown_characters, "truncated_context": reply.truncated_context,
    }


def build_app(server: Server) -> FastAPI:
    app = FastAPI(title="QuantaWeave inference server")

    @app.get("/health")
    async def health():
        return {"status": "ok", **server.info()}

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        try:
            return await server.generate(request)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/chat")
    async def chat(request: ChatRequest):
        try:
            return await server.chat(request)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest):
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint, adapter = resolve_model(args.checkpoint, args.adapter, args.run, args.latest, args.runs_dir)
    device = select_device(args.device)
    server = Server(checkpoint, adapter, device, args.precision, use_cache=not args.no_cache)
    print(f"model: {checkpoint}" + (f" + adapter {adapter}" if adapter else "") + f"  device: {device}  decoding: {server.complete_session.cache_note}")
    app = build_app(server)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
