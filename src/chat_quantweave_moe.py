"""Send messages to a QuantaWeave model: an interactive chat, one-off messages, or a batch of test messages.

  # interactive (multi-turn); the newest archived run, or --checkpoint DIR / --run EPOCH
  python src/chat_quantweave_moe.py --latest

  # one-off and repeated messages
  python src/chat_quantweave_moe.py --checkpoint artifacts/outputs/quantweave-moe-out --message "Once upon a time"

  # a batch of test messages; a .txt file (one per line, # comments) or .jsonl with optional expectations
  python src/chat_quantweave_moe.py --latest --messages-file tests.jsonl --json

  # piped messages
  printf 'Once upon a time\\nThe quick brown fox\\n' | python src/chat_quantweave_moe.py --latest

Two modes. ``complete`` (default) sends each message as the start of a text and shows the continuation: that is what a
base model, which is all the standard training produces, actually does. ``chat`` wraps the conversation as
"User: ... / Assistant:" turns and stops when the model starts a new "User:" line. Only a model trained on dialogue will
answer sensibly in chat mode; a base model will play along with the format at best.

Sampling is restricted to the tokens the tokenizer really has (a character model only ever saw about 40 characters, so the
rest of its output range is untrained noise) and never emits <unk>. Temperature 0 is greedy and fully deterministic;
--seed makes sampling reproducible. In the interactive session, /help lists the commands (/set, /reset, /mode, /save, ...).

Decoding uses a KV cache and, on CUDA, a captured CUDA graph (see fast_decode.py), typically 20-40x faster than re-reading the
whole context for every token; --no-cache uses the plain forward pass instead. Both use a context of max_sequence_length - 1
tokens, because the model's output at its very last position was never trained. Inference does not apply expert capacity.
"""

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch

from data_pipeline import load_tokenizer
from fast_decode import FastDecoder, Unsupported, filter_logits
from hardware import choose_precision, detect
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM
from train_quantweave_moe import autocast_context, select_device

RUNS_DIR = Path("artifacts/runs")
DEFAULT_CHECKPOINT = Path("artifacts/outputs/quantweave-moe-out")
CHAT_STOPS = ("\nUser:", "\nSystem:")


# ---- sampling ----------------------------------------------------------------------------------------------
def choose_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, generator: Optional[torch.Generator]) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probabilities = filter_logits(logits / temperature, top_k, top_p).softmax(dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))


# ---- settings, results ----------------------------------------------------------------------------------------
@dataclass
class Settings:
    mode: str = "complete"
    tokens: int = 200
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: list[str] = field(default_factory=list)
    system: str = ""

    def validate(self) -> None:
        if self.mode not in {"complete", "chat"}:
            raise ValueError("mode must be complete or chat")
        if self.tokens < 1 or self.temperature < 0 or self.top_k < 0 or not 0 < self.top_p <= 1:
            raise ValueError("tokens must be >= 1, temperature >= 0, top_k >= 0, and top_p in (0, 1]")


@dataclass
class Reply:
    message: str
    prompt: str
    response: str
    tokens: int
    seconds: float
    stop_reason: str                      # eos | stop | length
    unknown_characters: list[str] = field(default_factory=list)
    truncated_context: bool = False

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


class ChatSession:
    def __init__(self, model, tokenizer, device: torch.device, precision: str, settings: Settings, use_cache: bool = True) -> None:
        settings.validate()
        self.model, self.tokenizer, self.device, self.precision, self.settings = model, tokenizer, device, precision, settings
        self.history: list[tuple[str, str]] = []                 # (user message, assistant reply) pairs
        self.valid_ids = tokenizer.vocab_size
        model.eval()
        self.decoder: Optional[FastDecoder] = None
        self.cache_note = "no cache (full forward pass per token)"
        if use_cache:
            try:
                self.decoder = FastDecoder(model, device, precision)
                self.decoder.warm(settings.temperature, settings.top_k, settings.top_p)
                self.cache_note = self.decoder.description
            except Unsupported as error:
                self.cache_note = f"no cache ({error})"

    def reset(self) -> None:
        self.history.clear()

    def format_prompt(self, message: str) -> str:
        if self.settings.mode == "complete":
            return message
        parts = [f"System: {self.settings.system}\n"] if self.settings.system else []
        parts += [f"User: {user}\nAssistant: {reply}\n" for user, reply in self.history]
        parts.append(f"User: {message}\nAssistant:")
        return "".join(parts)

    def unknown_characters(self, text: str) -> list[str]:
        vocab = getattr(self.tokenizer, "vocab", None)
        if self.tokenizer.kind != "char" or vocab is None:
            return []
        return sorted({character for character in text if character not in vocab})

    def _plain_tokens(self, ids: list[int], generated: list[int], generator, usable: int):
        """Token source without a cache: a full forward pass over the last ``usable`` tokens for every new token."""
        settings, tokenizer = self.settings, self.tokenizer
        while True:
            window = torch.tensor([(ids + generated)[-usable:]], dtype=torch.long, device=self.device)
            with autocast_context(self.device, self.precision):
                logits = self.model(window)["logits"][0, -1].float()
            logits[self.valid_ids:] = float("-inf")                # ids the tokenizer never had are untrained noise
            if tokenizer.unk_id is not None and tokenizer.unk_id < self.valid_ids:
                logits[tokenizer.unk_id] = float("-inf")
            yield choose_token(logits, settings.temperature, settings.top_k, settings.top_p, generator)

    @torch.inference_mode()
    def send(self, message: str, on_text: Optional[Callable[[str], None]] = None) -> Reply:
        """Generate a reply. ``on_text`` receives the reply as it is produced (never a partial stop string)."""
        settings, tokenizer, decoder = self.settings, self.tokenizer, self.decoder
        prompt = self.format_prompt(message)
        stops = [s for s in ([*CHAT_STOPS] if settings.mode == "chat" else []) + list(settings.stop) if s]
        ids = tokenizer.encode(prompt) or [tokenizer.eos_id]
        # the last position of a training window is never trained, so the usable context is one shorter than the window
        usable = decoder.capacity if decoder is not None else self.model.config.max_sequence_length - 1
        generator = None
        if settings.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        holdback = max((len(s) for s in stops), default=1) - 1
        generated: list[int] = []
        text, emitted, reason = "", 0, "length"
        incremental = tokenizer.kind == "char"                     # a character's text never depends on its neighbours
        started = time.perf_counter()
        if decoder is not None:
            source = decoder.stream(ids, settings.tokens, temperature=settings.temperature, top_k=settings.top_k, top_p=settings.top_p,
                                    valid_ids=self.valid_ids, unk_id=tokenizer.unk_id, generator=generator)
        else:
            source = self._plain_tokens(ids, generated, generator, usable)
        for step, token in enumerate(source):
            if step >= settings.tokens:
                break
            if token == tokenizer.eos_id:
                reason = "eos"
                break
            generated.append(token)
            text = text + tokenizer.inverse.get(token, "?") if incremental else tokenizer.decode(generated)
            hits = [text.find(s) for s in stops if s in text]
            if hits:
                text, reason = text[: min(hits)], "stop"
                break
            if on_text is not None:
                safe = max(emitted, len(text) - holdback)
                if safe > emitted:
                    on_text(text[emitted:safe])
                    emitted = safe
        source.close()
        if on_text is not None and len(text) > emitted:
            on_text(text[emitted:])
        seconds = time.perf_counter() - started
        response = text.strip() if settings.mode == "chat" else text
        if settings.mode == "chat":
            self.history.append((message, response))
        return Reply(message, prompt, response, len(generated), seconds, reason, self.unknown_characters(prompt), len(ids) > usable)


# ---- choosing and loading a model ------------------------------------------------------------------------------
def run_folders(root: Path = RUNS_DIR) -> list[Path]:
    """Archived runs, oldest first; folders are named <epoch> or <epoch>-<n>."""
    if not root.is_dir():
        return []
    found = []
    for path in root.iterdir():
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", path.name)
        if path.is_dir() and match:
            found.append(((int(match.group(1)), int(match.group(2) or 0)), path))
    return [path for _, path in sorted(found)]


def usable_model(run: Path) -> Optional[tuple[Path, Optional[Path]]]:
    """(checkpoint dir, adapter dir) for an archived run, or None if it holds no model."""
    if (run / "model" / "model.pt").exists():
        return run / "model", None
    if (run / "merged" / "model.pt").exists():
        return run / "merged", None
    if (run / "adapter" / "adapter.pt").exists():
        meta = json.loads((run / "adapter" / "finetune_metadata.json").read_text())
        return Path(meta["base_checkpoint"]), run / "adapter"
    return None


def resolve_model(checkpoint: Optional[Path], adapter: Optional[Path], run: Optional[str], latest: bool, root: Path = RUNS_DIR) -> tuple[Path, Optional[Path]]:
    if checkpoint is not None:
        return checkpoint, adapter
    if run is not None:
        folder = Path(run) if Path(run).is_dir() else root / run
        found = usable_model(folder) if folder.is_dir() else None
        if found is None:
            raise FileNotFoundError(f"no model found in run '{run}' (looked in {folder})")
        return found
    if latest:
        for folder in reversed(run_folders(root)):
            found = usable_model(folder)
            if found is not None:
                return found
        raise FileNotFoundError(f"no archived run with a model under {root}")
    return DEFAULT_CHECKPOINT, adapter


def load_for_chat(checkpoint: Path, adapter: Optional[Path], quantize: int, device: torch.device):
    """(model, tokenizer). An adapter is applied over its (quantized) base; --quantize compresses the experts in memory."""
    if adapter is not None:
        from finetune_quantweave_moe import load_finetuned

        model = load_finetuned(checkpoint, adapter, device)
    else:
        config = QuantaWeaveConfig(**json.loads((checkpoint / "config.json").read_text()))
        model = QuantaWeaveMoEForCausalLM(config).to(device)
        model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device, weights_only=False)["model"])
        if quantize:
            from quantization import quantize_model

            quantize_model(model, quantize)
    return model.eval(), load_tokenizer(checkpoint)


# ---- test messages -----------------------------------------------------------------------------------------------
def read_messages(path: Path) -> list[dict]:
    """.jsonl: {"message": ..., optional "expect": str|list, "tokens", "temperature", "mode"}. Anything else: one message per line."""
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if path.suffix == ".jsonl":
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: not valid JSON ({error})") from error
            message = item.get("message", item.get("prompt"))
            if message is None:
                raise ValueError(f"{path}:{number}: needs a 'message' field")
            item["message"] = message
        else:
            item = {"message": line}
        expect = item.get("expect")
        item["expect"] = [expect] if isinstance(expect, str) else list(expect or [])
        items.append(item)
    return items


def run_batch(session: ChatSession, items: Sequence[dict], conversation: bool, emit: Callable[[dict], None]) -> list[dict]:
    """Send each message; ``emit`` receives one result dict per message. Per-message settings override the session's for that message."""
    results = []
    base = asdict(session.settings)
    for item in items:
        if not conversation:
            session.reset()
        overrides = {k: item[k] for k in ("mode", "tokens", "temperature", "top_k", "top_p", "seed") if k in item}
        session.settings = Settings(**{**base, **overrides})
        try:
            reply = session.send(item["message"])
        finally:
            session.settings = Settings(**base)
        failures = [needle for needle in item["expect"] if needle not in reply.response]
        result = {**asdict(reply), "tokens_per_second": reply.tokens_per_second, "expect": item["expect"],
                  "passed": None if not item["expect"] else not failures, "missing": failures}
        results.append(result)
        emit(result)
    return results


# ---- interactive session -----------------------------------------------------------------------------------------
HELP = """commands:
  /help                  this text
  /reset                 forget the conversation (chat mode)
  /mode complete|chat    switch modes
  /set NAME VALUE        temperature, top_k, top_p, tokens, seed (a number or "none"), system (text)
  /show                  current settings
  /history               the conversation so far
  /save FILE             write the conversation as JSON lines
  /quit                  leave (also Ctrl-D)
anything else is sent to the model."""


def apply_setting(settings: Settings, name: str, value: str) -> None:
    """Change one setting; a value that fails validation is rejected and leaves the settings untouched."""
    kinds = {"temperature": float, "top_p": float, "top_k": int, "tokens": int}
    candidate = Settings(**asdict(settings))
    if name == "seed":
        candidate.seed = None if value.lower() == "none" else int(value)
    elif name == "system":
        candidate.system = value
    elif name in kinds:
        setattr(candidate, name, kinds[name](value))
    else:
        raise ValueError(f"unknown setting '{name}' (try: temperature, top_k, top_p, tokens, seed, system)")
    candidate.validate()
    setattr(settings, name, getattr(candidate, name))


def repl(session: ChatSession, out=None, ask: Optional[Callable[[str], str]] = None, transcript: Optional[Callable[[dict], None]] = None,
         show_stats: bool = True) -> None:
    out = sys.stdout if out is None else out
    ask = input if ask is None else ask                       # looked up now, so it can be replaced (and tested)
    try:
        import readline  # noqa: F401  (line editing and history where available)
    except ImportError:
        pass
    print(f"QuantaWeave chat ({session.settings.mode} mode). /help for commands, /quit to leave.", file=out)
    while True:
        try:
            line = ask("you> ").rstrip("\n")
        except EOFError:
            print(file=out)
            return
        if not line.strip():
            continue
        if line.startswith("/"):
            command, _, rest = line[1:].partition(" ")
            try:
                if command in {"quit", "exit"}:
                    return
                elif command == "help":
                    print(HELP, file=out)
                elif command == "reset":
                    session.reset()
                    print("conversation cleared", file=out)
                elif command == "mode":
                    apply_mode = rest.strip()
                    Settings(**{**asdict(session.settings), "mode": apply_mode}).validate()
                    session.settings.mode = apply_mode
                    session.reset()
                    print(f"mode: {apply_mode}", file=out)
                elif command == "set":
                    name, _, value = rest.strip().partition(" ")
                    apply_setting(session.settings, name, value.strip())
                    print(f"{name} = {getattr(session.settings, name)!r}", file=out)
                elif command == "show":
                    print(json.dumps(asdict(session.settings)), file=out)
                elif command == "history":
                    for user, reply in session.history:
                        print(f"you: {user}\nmodel: {reply}", file=out)
                elif command == "save":
                    Path(rest.strip()).write_text("".join(json.dumps({"user": u, "assistant": a}) + "\n" for u, a in session.history))
                    print(f"saved {len(session.history)} turns to {rest.strip()}", file=out)
                else:
                    print(f"unknown command /{command}; /help lists them", file=out)
            except (ValueError, OSError) as error:
                print(f"error: {error}", file=out)
            continue
        if session.settings.mode == "complete":
            out.write(line)
        else:
            out.write("model> ")
        out.flush()
        reply = session.send(line, on_text=lambda piece: (out.write(piece), out.flush()))
        print(file=out)
        if reply.unknown_characters:
            print(f"  note: not in this model's vocabulary and replaced by <unk>: {' '.join(map(repr, reply.unknown_characters))}", file=out)
        if show_stats:
            print(f"  [{reply.tokens} tokens, {reply.tokens_per_second:.0f} tok/s, stopped: {reply.stop_reason}]", file=out)
        if transcript is not None:
            transcript({**asdict(reply), "tokens_per_second": reply.tokens_per_second, "settings": asdict(session.settings)})


class _NullStream:
    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


# ---- command line --------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    model = parser.add_argument_group("which model")
    model.add_argument("--checkpoint", type=Path, help=f"a checkpoint directory (default {DEFAULT_CHECKPOINT})")
    model.add_argument("--run", help="an archived run: its epoch id under artifacts/runs, or the folder path")
    model.add_argument("--latest", action="store_true", help="the newest archived run that contains a model")
    model.add_argument("--adapter", type=Path, help="a QLoRA adapter directory (applied over --checkpoint)")
    model.add_argument("--quantize", type=int, choices=(0, 4, 8), default=0, help="quantize the experts in memory to test the compressed model")
    model.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    send = parser.add_argument_group("sending messages")
    send.add_argument("-m", "--message", action="append", default=[], help="send this message (repeatable)")
    send.add_argument("--messages-file", type=Path, help=".txt (one message per line) or .jsonl ({message, expect, ...} per line)")
    send.add_argument("--conversation", action="store_true", help="in batch mode keep the history between messages (chat mode)")
    send.add_argument("--json", action="store_true", help="print one JSON object per reply instead of text")
    send.add_argument("--transcript", type=Path, help="append every exchange to this JSONL file")
    send.add_argument("--interactive", action="store_true", help="always run the interactive session, reading lines and /commands from stdin (even a pipe)")
    send.add_argument("--quiet", action="store_true", help="no statistics or notes")
    gen = parser.add_argument_group("generation")
    gen.add_argument("--mode", choices=("complete", "chat"), default="complete")
    gen.add_argument("--system", default="", help="chat mode: text placed before the conversation")
    gen.add_argument("--tokens", type=int, default=200, help="maximum new tokens per reply")
    gen.add_argument("--temperature", type=float, default=0.8, help="0 = greedy and deterministic")
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument("--seed", type=int, help="make sampling reproducible")
    gen.add_argument("--stop", action="append", default=[], help="stop when this text appears (repeatable)")
    gen.add_argument("--no-cache", action="store_true", help="plain forward pass per token instead of the KV cache / CUDA graph (much slower)")
    gen.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    gen.add_argument("--device", choices=("auto", "cuda", "rocm", "xpu", "cpu"), default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = Settings(mode=args.mode, tokens=args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed,
                        stop=list(args.stop), system=args.system)
    try:
        settings.validate()
        checkpoint, adapter = resolve_model(args.checkpoint, args.adapter, args.run, args.latest, args.runs_dir)
        items = [{"message": m, "expect": []} for m in args.message]
        if args.messages_file is not None:
            items += read_messages(args.messages_file)
        piped = not items and not args.interactive and not sys.stdin.isatty()
        if args.interactive and items:
            parser.error("--interactive cannot be combined with --message/--messages-file")
        if piped:
            items = [{"message": line.rstrip("\n"), "expect": []} for line in sys.stdin if line.strip()]
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    device = select_device(args.device)
    model, tokenizer = load_for_chat(checkpoint, adapter, args.quantize, device)
    precision = choose_precision(detect(device.type), args.precision)
    session = ChatSession(model, tokenizer, device, precision, settings, use_cache=not args.no_cache)
    status = sys.stderr if not args.quiet else _NullStream()
    print(f"model: {checkpoint}" + (f" + adapter {adapter}" if adapter else "") + f"  device: {device}  precision: {precision}"
          + (f"  experts quantized to int{args.quantize}" if args.quantize and adapter is None else "") + f"  decoding: {session.cache_note}", file=status)

    log = None
    if args.transcript is not None:
        args.transcript.parent.mkdir(parents=True, exist_ok=True)
        log = args.transcript.open("a", encoding="utf-8")
    write = (lambda entry: (log.write(json.dumps(entry) + "\n"), log.flush())) if log else None
    try:
        if not items:
            repl(session, transcript=write, show_stats=not args.quiet)
            return 0

        def emit(result: dict) -> None:
            if write:
                write({**result, "settings": asdict(session.settings)})
            if args.json:
                print(json.dumps(result), flush=True)
                return
            print(f"> {result['message']}")
            print(result["response"])
            if not args.quiet:
                if result["unknown_characters"]:
                    print(f"  note: replaced by <unk>: {' '.join(map(repr, result['unknown_characters']))}", file=status)
                verdict = "" if result["passed"] is None else ("  PASS" if result["passed"] else f"  FAIL (missing {result['missing']})")
                print(f"  [{result['tokens']} tokens, {result['tokens_per_second']:.0f} tok/s, stopped: {result['stop_reason']}]{verdict}", file=status)

        results = run_batch(session, items, args.conversation, emit)
    finally:
        if log:
            log.close()
    failed = [r for r in results if r["passed"] is False]
    checked = [r for r in results if r["passed"] is not None]
    if checked and not args.quiet:
        print(f"{len(checked) - len(failed)}/{len(checked)} checks passed", file=status)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
