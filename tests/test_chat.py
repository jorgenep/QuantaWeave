import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import chat_quantweave_moe as chat
import data_pipeline as dp
import generate_quantweave_moe as gen
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM

CPU = torch.device("cpu")
TOKENIZER = dp.CharTokenizer.build(["abcdefghijklmnopqrstuvwxyz \nUser:EN!."], 64)        # ids: 0 <unk>, 1 <eos>, then characters
VOCAB_SIZE = 96                                                                             # model outputs more ids than the tokenizer has


def ids_of(text: str) -> list[int]:
    return TOKENIZER.encode(text)


class ScriptedModel(torch.nn.Module):
    """Emits a fixed sequence of token ids, one per forward call, and records the context windows it was given."""

    def __init__(self, script: list[int], context: int = 32, strongest_invalid: bool = False):
        super().__init__()
        self.script, self.calls, self.windows = script, 0, []
        self.config = SimpleNamespace(max_sequence_length=context)
        self.strongest_invalid = strongest_invalid
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        self.windows.append(ids.size(1))
        logits = torch.zeros(1, ids.size(1), VOCAB_SIZE)
        wanted = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        logits[0, -1, wanted] = 50.0
        if self.strongest_invalid:
            logits[0, -1, VOCAB_SIZE - 1] = 60.0                       # an id the tokenizer does not have, ranked first
        return {"logits": logits}


def session(script, **settings) -> chat.ChatSession:
    values = dict(temperature=0.0, tokens=20)
    values.update(settings)
    return chat.ChatSession(ScriptedModel(script, context=values.pop("context", 32), strongest_invalid=values.pop("invalid", False)), TOKENIZER,
                            CPU, "fp32", chat.Settings(**values))


# ---- sampling -----------------------------------------------------------------------------------------------------
def test_top_k_and_top_p_filtering():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    assert torch.isfinite(chat.filter_logits(logits, top_k=2)).tolist() == [True, True, False, False]
    assert torch.isfinite(chat.filter_logits(logits, top_p=0.7)).tolist() == [True, True, False, False]     # keeps the token that crosses 0.7
    assert torch.isfinite(chat.filter_logits(logits, top_p=0.4)).tolist() == [True, False, False, False]
    assert torch.isfinite(chat.filter_logits(logits, top_p=1.0)).all()
    assert torch.isfinite(chat.filter_logits(logits, top_k=3, top_p=0.7)).tolist() == [True, True, False, False]
    assert torch.equal(chat.filter_logits(logits, top_k=10), logits)                                            # k larger than the vocabulary
    shuffled = torch.log(torch.tensor([0.05, 0.5, 0.15, 0.3]))                                                  # order must not matter
    assert torch.isfinite(chat.filter_logits(shuffled, top_p=0.7)).tolist() == [False, True, False, True]


def test_greedy_is_argmax_and_sampling_is_seedable_and_follows_the_distribution():
    logits = torch.log(torch.tensor([0.1, 0.6, 0.3]))
    assert chat.choose_token(logits, 0.0, 0, 1.0, None) == 1
    assert chat.choose_token(logits, 1.0, 1, 1.0, None) == 1                                                    # top_k=1 is greedy
    draws = lambda seed: [chat.choose_token(logits, 1.0, 0, 1.0, torch.Generator().manual_seed(seed + i)) for i in range(400)]  # noqa: E731
    assert draws(0) == draws(0) and draws(0) != draws(1000)
    share = sum(d == 1 for d in draws(0)) / 400
    assert 0.5 < share < 0.7
    low = [chat.choose_token(logits, 0.05, 0, 1.0, torch.Generator().manual_seed(i)) for i in range(50)]
    assert set(low) == {1}                                                                                       # a cold temperature is near-greedy


# ---- generation loop ----------------------------------------------------------------------------------------------
def test_eos_length_and_token_accounting():
    reply = session([*ids_of("ab"), TOKENIZER.eos_id]).send("hello")
    assert (reply.response, reply.stop_reason, reply.tokens) == ("ab", "eos", 2)
    reply = session(ids_of("abcdef")[:1] * 5, tokens=3).send("hello")
    assert (reply.response, reply.stop_reason, reply.tokens) == ("aaa", "length", 3) and reply.tokens_per_second > 0


def test_sampling_never_picks_ids_the_tokenizer_does_not_have_or_unk():
    s = session(ids_of("b") * 6, tokens=6, invalid=True)          # id 95 has the top logit every step
    assert s.send("hi").response == "bbbbbb"
    s = session([TOKENIZER.unk_id] * 3 + ids_of("c") * 3, tokens=3)
    assert "<unk>" not in s.send("hi").response


def test_stop_strings_truncate_and_streaming_never_leaks_a_partial_stop():
    s = session(ids_of("hi\nUser: x"), mode="chat", tokens=12)
    pieces = []
    reply = s.send("q", on_text=pieces.append)
    assert (reply.response, reply.stop_reason) == ("hi", "stop") and "".join(pieces).strip() == "hi"
    streamed = ""
    for piece in pieces:
        streamed += piece
        assert "\n" not in streamed, "a partial stop string must be held back"

    custom = session(ids_of("xEN!abc"), stop=["EN!!"], tokens=7)
    pieces = []
    reply = custom.send("q", on_text=pieces.append)
    assert reply.response == "xEN!abc" and "".join(pieces) == "xEN!abc" and reply.stop_reason == "length"    # "EN!" was held back, then released
    stopped = session(ids_of("xxEN!!z"), stop=["EN!!"], tokens=7).send("q")
    assert (stopped.response, stopped.stop_reason) == ("xx", "stop")


def test_long_prompts_are_windowed_to_the_models_context_and_empty_prompts_work():
    s = session(ids_of("ab"), context=8, tokens=2)
    reply = s.send("a" * 40)
    assert reply.truncated_context and max(s.model.windows) <= 8 and reply.response == "ab"
    empty = session(ids_of("ab"), tokens=2)
    assert empty.send("").response == "ab"


def test_unknown_characters_are_reported_for_char_models_only():
    s = session(ids_of("a"))
    assert s.send("héllo ☕").unknown_characters == ["é", "☕"]
    assert s.unknown_characters("plain text") == []


# ---- chat mode -------------------------------------------------------------------------------------------------------
def test_chat_template_history_and_reset():
    s = session(ids_of("ok"), mode="chat", system="Be brief.", tokens=2)
    assert s.format_prompt("hi") == "System: Be brief.\nUser: hi\nAssistant:"
    s.send("first")
    assert s.history == [("first", "ok")]
    assert s.format_prompt("second") == "System: Be brief.\nUser: first\nAssistant: ok\nUser: second\nAssistant:"
    s.reset()
    assert s.history == [] and session(ids_of("ok")).format_prompt("raw") == "raw"                      # complete mode sends the text as is
    complete = session(ids_of("ok"), tokens=2)
    complete.send("x")
    assert complete.history == []                                                                          # complete mode is stateless


def test_settings_validation():
    for bad in (dict(mode="shout"), dict(tokens=0), dict(temperature=-1), dict(top_p=0.0), dict(top_p=1.5), dict(top_k=-1)):
        with pytest.raises(ValueError):
            chat.Settings(**bad).validate()
    chat.Settings(temperature=0.0, top_p=1.0).validate()


# ---- choosing a model ---------------------------------------------------------------------------------------------------
def make_run(root: Path, name: str, kind: str = "train") -> Path:
    run = root / name
    (run / {"train": "model", "merged": "merged", "adapter": "adapter"}[kind]).mkdir(parents=True)
    if kind == "train":
        (run / "model" / "model.pt").write_bytes(b"x")
    elif kind == "merged":
        (run / "merged" / "model.pt").write_bytes(b"x")
    else:
        (run / "adapter" / "adapter.pt").write_bytes(b"x")
        (run / "adapter" / "finetune_metadata.json").write_text(json.dumps({"base_checkpoint": "/base/ckpt"}))
    return run


def test_run_folders_sort_by_epoch_numerically_and_latest_skips_runs_without_models(tmp_path):
    root = tmp_path / "runs"
    for name in ("9", "10", "10-2", "10-1", "notarun"):
        make_run(root, name)
    (root / "readme.txt").write_text("x")
    assert [p.name for p in chat.run_folders(root)] == ["9", "10", "10-1", "10-2"]                        # numeric, not lexicographic
    (root / "11").mkdir()                                                                                # newest run has no model
    assert chat.resolve_model(None, None, None, True, root)[0] == root / "10-2" / "model"
    assert chat.resolve_model(None, None, "9", False, root) == (root / "9" / "model", None)
    assert chat.resolve_model(None, None, str(root / "10"), False, root)[0] == root / "10" / "model"
    make_run(root, "12", "adapter")
    assert chat.resolve_model(None, None, None, True, root) == (Path("/base/ckpt"), root / "12" / "adapter")
    make_run(root, "13", "merged")
    assert chat.resolve_model(None, None, "13", False, root) == (root / "13" / "merged", None)
    assert chat.resolve_model(Path("explicit"), Path("ad"), "9", True, root) == (Path("explicit"), Path("ad"))   # --checkpoint wins
    assert chat.resolve_model(None, None, None, False, root)[0] == chat.DEFAULT_CHECKPOINT
    with pytest.raises(FileNotFoundError, match="no model found"):
        chat.resolve_model(None, None, "11", False, root)
    with pytest.raises(FileNotFoundError, match="no archived run"):
        chat.resolve_model(None, None, None, True, tmp_path / "empty")


# ---- test-message files -------------------------------------------------------------------------------------------------
def test_reading_message_files(tmp_path):
    txt = tmp_path / "m.txt"
    txt.write_text("first\n\n# a comment\nsecond message\n")
    assert [i["message"] for i in chat.read_messages(txt)] == ["first", "second message"]
    jsonl = tmp_path / "m.jsonl"
    jsonl.write_text('# header\n{"message": "a", "expect": "x"}\n{"prompt": "b", "expect": ["y", "z"], "tokens": 5}\n{"message": "c"}\n')
    items = chat.read_messages(jsonl)
    assert [i["message"] for i in items] == ["a", "b", "c"] and items[0]["expect"] == ["x"] and items[1]["expect"] == ["y", "z"] and items[2]["expect"] == []
    assert items[1]["tokens"] == 5
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"message": "ok"}\n{not json}\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        chat.read_messages(bad)
    bad.write_text('{"nothing": 1}\n')
    with pytest.raises(ValueError, match="needs a 'message'"):
        chat.read_messages(bad)


def test_batch_runs_apply_and_restore_per_message_overrides_and_check_expectations():
    s = session(ids_of("hello"), tokens=5, mode="complete")
    seen = []
    results = chat.run_batch(s, [
        {"message": "a", "expect": ["hel"]},
        {"message": "b", "expect": ["nope"], "tokens": 2},
        {"message": "c", "expect": [], "mode": "chat"},
    ], conversation=False, emit=seen.append)
    assert [r["passed"] for r in results] == [True, False, None] and results[1]["missing"] == ["nope"] and results[1]["tokens"] == 2
    assert s.settings.tokens == 5 and s.settings.mode == "complete"                                        # overrides did not stick
    assert seen == results and results[2]["prompt"].startswith("User: c")
    s.settings.mode = "chat"
    s.reset()
    chat.run_batch(s, [{"message": "one", "expect": []}, {"message": "two", "expect": []}], conversation=True, emit=lambda r: None)
    assert [u for u, _ in s.history] == ["one", "two"]                                                    # --conversation keeps history
    chat.run_batch(s, [{"message": "three", "expect": []}], conversation=False, emit=lambda r: None)
    assert [u for u, _ in s.history] == ["three"]                                                         # otherwise each message starts fresh


# ---- the interactive session -----------------------------------------------------------------------------------------------
def scripted_input(lines):
    queue = iter(lines)

    def ask(prompt):
        try:
            return next(queue)
        except StopIteration:
            raise EOFError

    return ask


def test_interactive_commands(tmp_path):
    s = session(ids_of("k"), tokens=2, temperature=0.8)                                                  # every reply is "kk"
    out = io.StringIO()
    saved = tmp_path / "chat.jsonl"
    logged = []
    chat.repl(s, out=out, ask=scripted_input([
        "hello there", "/set temperature 0", "/set top_p 0.5", "/set seed 3", "/set seed none", "/show", "/mode chat", "how are you", "/history",
        f"/save {saved}", "/reset", "/history", "/set bogus 1", "/set temperature -3", "/mode shout", "/nope", "", "/help", "/quit", "never reached"]),
        transcript=logged.append)
    text = out.getvalue()
    assert "hello therekk" in text and "[2 tokens" in text                                                # complete mode echoes the prompt then streams the reply
    assert "temperature = 0.0" in text and "top_p = 0.5" in text and "seed = 3" in text and "seed = None" in text
    assert '"temperature": 0.0' in text and "mode: chat" in text and "model> kk" in text
    assert "you: how are you\nmodel: kk" in text and "conversation cleared" in text
    assert "unknown setting 'bogus'" in text and "temperature >= 0" in text and "mode must be" in text and "unknown command /nope" in text and "commands:" in text
    assert [json.loads(l) for l in saved.read_text().splitlines()] == [{"user": "how are you", "assistant": "kk"}]
    assert len(logged) == 2 and logged[0]["settings"]["mode"] == "complete" and logged[1]["settings"]["mode"] == "chat"
    assert s.settings.temperature == 0.0 and s.settings.mode == "chat"

    eof_out = io.StringIO()
    chat.repl(session(ids_of("ok"), tokens=2), out=eof_out, ask=scripted_input(["hi"]))                  # Ctrl-D ends the session cleanly
    assert eof_out.getvalue().endswith("\n")


def test_interactive_session_notes_unknown_characters():
    out = io.StringIO()
    chat.repl(session(ids_of("ok"), tokens=2), out=out, ask=scripted_input(["café", "/quit"]), show_stats=False)
    assert "replaced by <unk>: 'é'" in out.getvalue() and "tok/s" not in out.getvalue()


# ---- the command line, against a real (tiny) trained checkpoint ---------------------------------------------------------------
@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("chat")
    data = root / "d.jsonl"
    data.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60)))
    trainer.run_training(trainer.default_args(data=[data], steps=25, batch_size=4, sequence_length=32, examples=60, hidden_size=32, layers=2, ffn_size=48,
                                              total_experts=4, active_experts=2, vocab_size=128, device="cpu", checkpoint_interval=0, lr=3e-3,
                                              capacity_factor=0, output=root / "model", checkpoint_dir=root / "ck"))
    return root / "model", data


def run_cli(capsys, *argv):
    code = chat.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_one_off_messages_are_deterministic_when_greedy_and_reproducible_when_seeded(checkpoint, capsys):
    model, _ = checkpoint
    common = ["--checkpoint", str(model), "--device", "cpu", "--tokens", "30"]
    code, out, err = run_cli(capsys, *common, "-m", "story 3: the quick", "--temperature", "0")
    assert code == 0 and out.startswith("> story 3: the quick\n") and "model:" in err and "tok/s" in err
    assert run_cli(capsys, *common, "-m", "story 3: the quick", "--temperature", "0")[1] == out
    a = run_cli(capsys, *common, "-m", "the", "--temperature", "1.0", "--seed", "5", "--top-k", "8")[1]
    assert run_cli(capsys, *common, "-m", "the", "--temperature", "1.0", "--seed", "5", "--top-k", "8")[1] == a
    assert run_cli(capsys, *common, "-m", "the", "--temperature", "1.0", "--seed", "6", "--top-k", "8")[1] != a
    for line in a.splitlines()[1:]:
        assert "�" not in line and "?" not in line                                                # only real characters are sampled (the corpus has no '?')


def test_json_output_transcript_and_quiet_mode(checkpoint, capsys, tmp_path):
    model, _ = checkpoint
    log = tmp_path / "t" / "transcript.jsonl"
    code, out, err = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "-m", "one café", "-m", "two", "--tokens", "8", "--temperature", "0",
                             "--json", "--transcript", str(log))
    rows = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and [r["message"] for r in rows] == ["one café", "two"] and rows[0]["unknown_characters"] == ["é"] and rows[0]["tokens"] == 8
    assert {"prompt", "response", "seconds", "stop_reason", "tokens_per_second", "passed"} <= set(rows[0])
    assert [json.loads(l)["message"] for l in log.read_text().splitlines()] == ["one café", "two"]
    quiet = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "-m", "two", "--tokens", "4", "--quiet")
    assert quiet[2] == "" and quiet[1].startswith("> two")


def test_expectations_set_the_exit_code(checkpoint, capsys, tmp_path):
    model, data = checkpoint
    greedy = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "-m", "story 3: the quick", "--temperature", "0", "--tokens", "12", "--json")[1]
    fragment = json.loads(greedy.splitlines()[0])["response"][3:8]                    # text greedy decoding is known to produce
    tests = tmp_path / "t.jsonl"
    tests.write_text(json.dumps({"message": "story 3: the quick", "expect": [fragment], "temperature": 0, "tokens": 12}) + "\n"
                     + '{"message": "story 4:", "expect": ["zzzz"], "temperature": 0, "tokens": 6}\n')
    code, out, err = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "--messages-file", str(tests))
    assert code == 1 and "PASS" in err and "FAIL (missing ['zzzz'])" in err and "1/2 checks passed" in err
    tests.write_text(json.dumps({"message": "story 3: the quick", "expect": [fragment], "temperature": 0, "tokens": 12}) + "\n")
    assert run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "--messages-file", str(tests))[0] == 0
    plain = tmp_path / "t.txt"
    plain.write_text("# smoke\nstory 1:\nstory 2:\n")
    _, out, _ = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "--messages-file", str(plain), "--tokens", "4", "-m", "first arg")
    assert [l for l in out.splitlines() if l.startswith("> ")] == ["> first arg", "> story 1:", "> story 2:"]


def test_piped_messages_are_read_from_stdin(checkpoint, capsys, monkeypatch):
    model, _ = checkpoint
    monkeypatch.setattr(sys, "stdin", io.StringIO("story 1:\n\nstory 2:\n"))
    code, out, _ = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "--tokens", "4")
    assert code == 0 and [l for l in out.splitlines() if l.startswith("> ")] == ["> story 1:", "> story 2:"]


def test_interactive_mode_is_used_when_there_are_no_messages_and_stdin_is_a_terminal(checkpoint, capsys, monkeypatch):
    model, _ = checkpoint
    tty = io.StringIO()
    tty.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", tty)
    monkeypatch.setattr("builtins.input", scripted_input(["/show", "/quit"]))
    code, out, _ = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu")
    assert code == 0 and "QuantaWeave chat" in out and '"mode": "complete"' in out


def test_bad_options_and_missing_models_are_clean_errors(checkpoint, capsys, tmp_path):
    model, _ = checkpoint
    for argv in (["--checkpoint", str(model), "--top-p", "0", "-m", "x"], ["--checkpoint", str(model), "--tokens", "0", "-m", "x"],
                 ["--run", "does-not-exist", "--runs-dir", str(tmp_path)], ["--latest", "--runs-dir", str(tmp_path / "none")],
                 ["--checkpoint", str(model), "--messages-file", str(tmp_path / "missing.txt")]):
        with pytest.raises(SystemExit) as raised:
            chat.main([*argv, "--device", "cpu"])
        assert raised.value.code == 2
    capsys.readouterr()


def test_run_and_latest_selection_and_quantized_model(checkpoint, capsys, tmp_path):
    model, _ = checkpoint
    import shutil
    runs = tmp_path / "runs"
    (runs / "100").mkdir(parents=True)
    shutil.copytree(model, runs / "100" / "model")
    common = ["--runs-dir", str(runs), "--device", "cpu", "-m", "story 1:", "--tokens", "6", "--temperature", "0"]
    by_latest = run_cli(capsys, "--latest", *common)
    by_run = run_cli(capsys, "--run", "100", *common)
    assert by_latest[0] == 0 and by_latest[1] == by_run[1] and str(runs / "100" / "model") in by_latest[2]
    quantized = run_cli(capsys, "--checkpoint", str(model), "--quantize", "8", *common[2:])
    assert quantized[0] == 0 and "experts quantized to int8" in quantized[2] and quantized[1].startswith("> story 1:")


def test_a_qlora_adapter_can_be_chatted_with(checkpoint, capsys, tmp_path):
    import finetune_quantweave_moe as ft
    model, data = checkpoint
    ft.run_finetune(ft.build_parser().parse_args(["--checkpoint", str(model), "--data", str(data), "--output", str(tmp_path / "adapter"), "--steps", "3",
                                                  "--batch-size", "4", "--sequence-length", "16", "--examples", "20", "--device", "cpu", "--bits", "8",
                                                  "--no-archive"]))
    capsys.readouterr()
    code, out, err = run_cli(capsys, "--checkpoint", str(model), "--adapter", str(tmp_path / "adapter"), "--device", "cpu", "-m", "story 2:",
                             "--tokens", "6", "--temperature", "0")
    assert code == 0 and "+ adapter" in err and out.startswith("> story 2:")


def test_bpe_models_stream_and_decode_cleanly(tmp_path, capsys):
    pytest.importorskip("tokenizers")
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(60)))
    trainer.run_training(trainer.default_args(data=[data], steps=6, batch_size=4, sequence_length=32, examples=60, hidden_size=32, layers=1, ffn_size=48,
                                              total_experts=4, active_experts=2, device="cpu", checkpoint_interval=0, capacity_factor=0, tokenizer="bpe",
                                              tokenizer_path=tmp_path / "tok", vocab_size=300, output=tmp_path / "m", checkpoint_dir=tmp_path / "ck"))
    capsys.readouterr()
    code, out, _ = run_cli(capsys, "--checkpoint", str(tmp_path / "m"), "--device", "cpu", "-m", "story 3: the quick", "--tokens", "10", "--temperature", "0")
    assert code == 0 and out.startswith("> story 3: the quick")


def test_old_generator_no_longer_samples_untrained_ids(checkpoint):
    model_dir, _ = checkpoint
    config = QuantaWeaveConfig(**json.loads((model_dir / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(torch.load(model_dir / "model.pt", weights_only=False)["model"])
    model.eval()
    tokenizer = dp.load_tokenizer(model_dir)
    assert config.vocab_size > tokenizer.vocab_size                       # the situation that produced the '?' noise
    torch.manual_seed(0)
    text = gen.sample_text(model, tokenizer, "the", 150, 2.0, CPU)
    assert "?" not in text


def test_interactive_flag_runs_the_session_from_a_pipe_and_rejects_mixing_with_messages(checkpoint, capsys, monkeypatch):
    model, _ = checkpoint
    monkeypatch.setattr(sys, "stdin", io.StringIO("/set temperature 0\n/set tokens 5\nstory 1:\n/reset\n/quit\n"))
    code, out, _ = run_cli(capsys, "--checkpoint", str(model), "--device", "cpu", "--interactive")
    assert code == 0 and "temperature = 0.0" in out and "tokens = 5" in out and "story 1:" in out and "conversation cleared" in out
    with pytest.raises(SystemExit):
        chat.main(["--checkpoint", str(model), "--device", "cpu", "--interactive", "-m", "x"])
    capsys.readouterr()
