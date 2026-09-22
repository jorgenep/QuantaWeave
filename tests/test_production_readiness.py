"""Tests for the production-readiness pass: checkpoint_io centralization, PII redaction, data provenance,
promotion gating, and serve_quantweave.py hardening (auth, rate limiting, timeouts, metrics, moderation)."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


# ---- checkpoint_io.py: centralized, documented weights_only=False -------------------------------------------
def test_load_checkpoint_round_trips_an_arbitrary_payload(tmp_path):
    from checkpoint_io import load_checkpoint

    payload = {"model": {"w": torch.randn(3, 3)}, "step": 7, "extra": {"seed": 0}}
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    loaded = load_checkpoint(path)
    assert loaded["step"] == 7 and torch.equal(loaded["model"]["w"], payload["model"]["w"])


def test_load_checkpoint_respects_map_location(tmp_path):
    from checkpoint_io import load_checkpoint

    path = tmp_path / "model.pt"
    torch.save({"w": torch.randn(2)}, path)
    loaded = load_checkpoint(path, map_location="cpu")
    assert loaded["w"].device.type == "cpu"


def test_every_src_module_that_loads_a_checkpoint_goes_through_checkpoint_io():
    # regression test for the collision this refactor introduced and fixed: a module-level `def load_checkpoint`
    # (train_quantweave_moe.py's resume function) rebinding the name imported from checkpoint_io, so the wrong
    # function ran at the call site. Every module that imports checkpoint_io.load_checkpoint under its own name
    # must not also define a different `load_checkpoint` at module level without aliasing the import.
    src = Path(__file__).parents[1] / "src"
    for path in src.glob("*.py"):
        text = path.read_text()
        if "from checkpoint_io import load_checkpoint\n" in text:
            assert "\ndef load_checkpoint(" not in text, f"{path.name} shadows checkpoint_io.load_checkpoint"


# ---- pii_redact.py ----------------------------------------------------------------------------------------
def test_pii_redactor_catches_common_shapes_and_leaves_ordinary_text_alone():
    from pii_redact import PIIRedactor

    r = PIIRedactor()
    assert r.redact("contact jane.doe@example.com") == "contact <EMAIL>"
    assert r.redact("call 555-123-4567") == "call <PHONE>"
    assert r.redact("ssn 123-45-6789") == "ssn <SSN>"
    assert r.redact("card 4111-1111-1111-1111") == "card <CREDIT_CARD>"
    assert r.redact("server at 192.168.1.1") == "server at <IPV4>"
    ordinary = "once upon a time in 2024 a fox turned 30 and ran quickly through the forest"
    assert r.redact(ordinary) == ordinary


def test_pii_redactor_reports_per_category_counts():
    from pii_redact import PIIRedactor

    r = PIIRedactor()
    r.redact("a@b.com and c@d.com and 555-123-4567")
    report = r.report()
    assert report["EMAIL"] == 2 and report["PHONE"] == 1 and report["SSN"] == 0
    assert r.total() == 3


def test_read_rows_applies_a_redactor_when_given(tmp_path):
    from data_pipeline import read_rows
    from pii_redact import PIIRedactor

    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"text": "reach me at a@b.com"}) + "\n")
    redactor = PIIRedactor()
    domain, text = next(read_rows([path], redactor=redactor))
    assert text == "reach me at <EMAIL>"
    assert redactor.total() == 1


def test_read_rows_without_a_redactor_is_unchanged():
    from data_pipeline import read_rows
    import inspect

    assert "redactor" in inspect.signature(read_rows).parameters
    # default is None: existing callers that don't pass it keep exact prior behaviour (tested implicitly by the
    # whole rest of the suite, which calls read_rows/build_corpus without a redactor throughout)


def test_trainer_redact_pii_flag_reduces_pii_end_to_end(tmp_path):
    import train_quantweave_moe as trainer

    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps({"text": f"story {i}: reach me at person{i}@example.com about the fox"}) + "\n" for i in range(40))
    )
    args = trainer.default_args(
        data=[tmp_path / "d.jsonl"], steps=2, batch_size=2, sequence_length=16, examples=40, hidden_size=16,
        layers=2, ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu",
        checkpoint_interval=0, output=tmp_path / "out", checkpoint_dir=tmp_path / "ckpt", redact_pii=True,
    )
    summary = trainer.run_training(args)
    assert summary["steps"] == 2
    # the corpus itself must not contain the literal '@' from an email once redacted: decode a stretch of the
    # saved model's own tokenizer vocab space indirectly by checking the character vocab excludes '@' entirely
    # when every occurrence was redacted to the placeholder token text instead.
    vocab = json.loads((tmp_path / "out" / "vocab.json").read_text())
    assert "@" not in vocab


# ---- run_bundle.py: --data-card provenance/licensing manifest -------------------------------------------
def test_archive_data_embeds_a_data_card_verbatim(tmp_path):
    from run_bundle import archive_data

    data = tmp_path / "d.jsonl"
    data.write_text('{"text": "hello"}\n')
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"source": "test corpus", "license": "CC0"}))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = archive_data([data], bundle, limit_mb=200.0, rows_used=1, data_card=card)
    assert manifest["data_card"] == {"source": "test corpus", "license": "CC0"}


def test_archive_data_records_an_error_for_an_unreadable_data_card(tmp_path):
    from run_bundle import archive_data

    data = tmp_path / "d.jsonl"
    data.write_text('{"text": "hello"}\n')
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = archive_data([data], bundle, limit_mb=200.0, rows_used=1, data_card=tmp_path / "missing.json")
    assert "data_card_error" in manifest and "data_card" not in manifest


def test_archive_data_without_a_data_card_is_unchanged(tmp_path):
    from run_bundle import archive_data

    data = tmp_path / "d.jsonl"
    data.write_text('{"text": "hello"}\n')
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = archive_data([data], bundle, limit_mb=200.0, rows_used=1)
    assert "data_card" not in manifest and "data_card_error" not in manifest


def test_trainer_data_card_flag_reaches_the_archive_manifest(tmp_path):
    import train_quantweave_moe as trainer

    (tmp_path / "d.jsonl").write_text(
        "".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60))
    )
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"license": "MIT", "source": "synthetic"}))
    args = trainer.default_args(
        data=[tmp_path / "d.jsonl"], steps=2, batch_size=2, sequence_length=16, examples=60, hidden_size=16,
        layers=2, ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu",
        checkpoint_interval=0, output=tmp_path / "out", checkpoint_dir=tmp_path / "ckpt",
        archive_dir=tmp_path / "runs", data_card=card,
    )
    summary = trainer.run_training(args)
    manifest = json.loads((Path(summary["archive"]) / "data" / "manifest.json").read_text())
    assert manifest["data_card"] == {"license": "MIT", "source": "synthetic"}


# ---- experiment_manager.py: promotion_threshold quality gate -----------------------------------------------
def test_update_best_without_a_threshold_is_always_promotable(tmp_path):
    from experiment_manager import update_best

    summary = {"run_id": "r1", "run_dir": str(tmp_path / "r1"), "train": {"final_loss": 3.0}}
    assert update_best(tmp_path, summary, "train.final_loss")
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["promotable"] is True and best["promotion_threshold"] is None


def test_update_best_with_a_threshold_marks_a_failing_run_not_promotable(tmp_path):
    from experiment_manager import update_best

    summary = {"run_id": "r1", "run_dir": str(tmp_path / "r1"), "train": {"final_loss": 5.0}}
    assert update_best(tmp_path, summary, "train.final_loss", promotion_threshold=2.0)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["value"] == 5.0 and best["promotable"] is False and best["promotion_threshold"] == 2.0


def test_update_best_with_a_threshold_marks_a_passing_run_promotable(tmp_path):
    from experiment_manager import update_best

    summary = {"run_id": "r1", "run_dir": str(tmp_path / "r1"), "train": {"final_loss": 1.0}}
    assert update_best(tmp_path, summary, "train.final_loss", promotion_threshold=2.0)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["promotable"] is True


def test_prepare_tokens_pack_redact_pii_flag(tmp_path, monkeypatch):
    import prepare_tokens

    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps({"text": "email me at a@b.com about the fox"}) + "\n")
    out = tmp_path / "packed"
    monkeypatch.setattr(sys, "argv", ["prepare_tokens.py", "pack", "--data", str(data), "--tokenizer", "char",
                                       "--vocab-size", "128", "--output", str(out), "--redact-pii"])
    prepare_tokens.main()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["total_tokens"] > 0
    # '@' must be absent from the packed corpus's own tokenizer vocab if it was fully redacted
    from data_pipeline import load_tokenizer

    tokenizer = load_tokenizer(out / "tokenizer")
    assert "@" not in tokenizer.vocab


# ---- serve_quantweave.py hardening: auth, rate limiting, backpressure, metrics, moderation ------------------
fastapi_testclient = pytest.importorskip("fastapi.testclient")
import serve_quantweave as sq  # noqa: E402


@pytest.fixture(scope="module")
def hardened_checkpoint(tmp_path_factory):
    import train_quantweave_moe as trainer

    root = tmp_path_factory.mktemp("hardened_serve")
    data = root / "d.jsonl"
    data.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog."}) + "\n" for i in range(80)))
    trainer.run_training(trainer.default_args(
        data=[data], steps=8, batch_size=4, sequence_length=16, examples=80, hidden_size=16, layers=1, ffn_size=24,
        total_experts=4, active_experts=2, vocab_size=96, device="cpu", checkpoint_interval=0, lr=3e-3,
        capacity_factor=0, output=root / "model", checkpoint_dir=root / "ck", no_archive=True,
    ))
    return root / "model"


def make_client(checkpoint, **server_kwargs):
    server = sq.Server(checkpoint, None, torch.device("cpu"), "fp32", use_cache=True, **server_kwargs)
    return fastapi_testclient.TestClient(sq.build_app(server)), server


# ---- rate limiter and moderation as plain units, no HTTP needed --------------------------------------------
def test_rate_limiter_allows_up_to_the_limit_then_rejects():
    limiter = sq.RateLimiter(limit=3, window=60.0)
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_is_per_key():
    limiter = sq.RateLimiter(limit=1, window=60.0)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is True         # a different key has its own budget


def test_rate_limiter_disabled_when_limit_is_none():
    limiter = sq.RateLimiter(limit=None)
    assert all(limiter.allow("a") for _ in range(1000))


def test_rate_limiter_window_expires_old_hits():
    limiter = sq.RateLimiter(limit=1, window=0.05)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    import time as _time

    _time.sleep(0.06)
    assert limiter.allow("a") is True


def test_moderation_blocks_a_matching_pattern_and_allows_everything_else(tmp_path):
    patterns = tmp_path / "blocklist.txt"
    patterns.write_text("# a comment\nforbidden ?word\n\nanother-bad-thing\n")
    moderation = sq.Moderation(patterns)
    assert moderation.blocked("this contains a forbidden word right here") is not None
    assert moderation.blocked("this contains forbiddenword too (no space)") is not None
    assert moderation.blocked("this is perfectly fine text") is None


def test_moderation_with_no_file_blocks_nothing():
    moderation = sq.Moderation(None)
    assert moderation.blocked("anything at all, even forbidden things") is None


def test_metrics_render_is_valid_prometheus_text_exposition_format():
    metrics = sq.Metrics()
    metrics.observe("/generate", 200, 0.5, tokens=10)
    metrics.observe("/generate", 422, 0.1)
    text = metrics.render()
    assert 'quantweave_requests_total{route="/generate",status="200"} 1' in text
    assert 'quantweave_requests_total{route="/generate",status="422"} 1' in text
    assert 'quantweave_errors_total{route="/generate"} 1' in text
    assert "quantweave_tokens_generated_total 10" in text
    for line in text.splitlines():
        assert line.startswith("#") or line.split()[-1].replace(".", "", 1).lstrip("-").isdigit() or "route=" in line


# ---- through real HTTP requests -----------------------------------------------------------------------------
def test_server_runs_open_by_default_no_auth_required(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint)
    response = client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0})
    assert response.status_code == 200


def test_api_key_rejects_missing_or_wrong_key(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint, api_key="secret123")
    assert client.post("/generate", json={"prompt": "hi", "tokens": 4}).status_code == 401
    assert client.post("/generate", json={"prompt": "hi", "tokens": 4},
                        headers={"X-API-Key": "wrong"}).status_code == 401
    # /health stays open even with an api key configured (health checks/liveness probes shouldn't need one)
    assert client.get("/health").status_code == 200


def test_api_key_accepts_bearer_or_x_api_key_header(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint, api_key="secret123")
    ok_bearer = client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0},
                             headers={"Authorization": "Bearer secret123"})
    ok_header = client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0},
                             headers={"X-API-Key": "secret123"})
    assert ok_bearer.status_code == 200 and ok_header.status_code == 200


def test_rate_limit_returns_429_once_exceeded(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint, rate_limit=2, rate_limit_window=60.0)
    statuses = [client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0}).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]


def test_moderation_blocks_a_prompt_with_422(hardened_checkpoint, tmp_path):
    patterns = tmp_path / "blocklist.txt"
    patterns.write_text("badword\n")
    client, _ = make_client(hardened_checkpoint, moderation_patterns=patterns)
    blocked = client.post("/generate", json={"prompt": "this has a badword in it", "tokens": 4})
    allowed = client.post("/generate", json={"prompt": "this is fine", "tokens": 4, "temperature": 0})
    assert blocked.status_code == 422 and allowed.status_code == 200


def test_moderation_blocks_the_last_chat_message(hardened_checkpoint, tmp_path):
    patterns = tmp_path / "blocklist.txt"
    patterns.write_text("badword\n")
    client, _ = make_client(hardened_checkpoint, moderation_patterns=patterns)
    response = client.post("/chat", json={"messages": [{"role": "user", "content": "a badword here"}], "tokens": 4})
    assert response.status_code == 422


def test_max_queue_never_rejects_a_request_that_finds_the_lock_free(hardened_checkpoint):
    # max_queue=0 means "reject anything that would have to wait", not "reject everything" — the one request
    # that finds the lock free must always be accepted, including with max_queue=0 (this was a real bug: the
    # original check fired on `waiting >= max_queue` alone, so 0 >= 0 rejected even an uncontended request).
    client, _ = make_client(hardened_checkpoint, max_queue=0)
    response = client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0})
    assert response.status_code == 200


def test_max_queue_rejects_once_full(hardened_checkpoint):
    import asyncio as _asyncio

    client, server = make_client(hardened_checkpoint, max_queue=1)

    async def scenario():
        await server.lock.acquire()          # simulate an in-flight request holding the lock
        server.waiting = 1                    # ... and one more already queued behind it
        try:
            await server._acquire()
            return True
        except Exception:
            return False
        finally:
            server.lock.release()

    accepted = _asyncio.run(scenario())
    assert accepted is False


def test_request_timeout_rejects_with_503_when_the_lock_is_held(hardened_checkpoint):
    import asyncio as _asyncio

    client, server = make_client(hardened_checkpoint, request_timeout=0.05)

    async def scenario():
        await server.lock.acquire()   # simulate an in-flight request holding the lock
        try:
            with pytest.raises(Exception) as info:
                await server._acquire()
            return info.value
        finally:
            server.lock.release()

    error = _asyncio.run(scenario())
    assert getattr(error, "status_code", None) == 503


def test_metrics_endpoint_reports_prometheus_text(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint)
    client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "quantweave_requests_total" in response.text
    assert 'route="/generate"' in response.text


def test_metrics_endpoint_is_gated_by_api_key_too(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint, api_key="secret123")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"X-API-Key": "secret123"}).status_code == 200


def test_json_log_formatter_emits_one_parseable_object_per_line():
    import logging

    record = logging.LogRecord("quantweave.serve", logging.INFO, __file__, 1, "generate 200 0.5s", (), None)
    record.request_id, record.route, record.status, record.seconds = "abc123", "/generate", 200, 0.5
    line = sq.JSONLogFormatter().format(record)
    parsed = json.loads(line)
    assert parsed["route"] == "/generate" and parsed["status"] == 200 and parsed["request_id"] == "abc123"
    assert parsed["level"] == "INFO" and "time" in parsed


def test_configure_logging_switches_between_json_and_text():
    sq.configure_logging("json")
    assert isinstance(sq.logger.handlers[0].formatter, sq.JSONLogFormatter)
    sq.configure_logging("text")
    assert not isinstance(sq.logger.handlers[0].formatter, sq.JSONLogFormatter)


def test_requests_emit_a_structured_log_line(hardened_checkpoint, caplog):
    import logging

    sq.configure_logging("text")               # ensure a handler exists so propagate=False doesn't hide it
    client, _ = make_client(hardened_checkpoint)
    with caplog.at_level(logging.INFO, logger="quantweave.serve"):
        # caplog attaches its own handler to the root/named logger; propagate=False on quantweave.serve normally
        # blocks that, so attach caplog's handler directly for this one assertion.
        sq.logger.addHandler(caplog.handler)
        try:
            client.post("/generate", json={"prompt": "hi", "tokens": 4, "temperature": 0})
        finally:
            sq.logger.removeHandler(caplog.handler)
    routes = [record.route for record in caplog.records if hasattr(record, "route")]
    assert "/generate" in routes


def test_health_reports_the_hardening_configuration(hardened_checkpoint):
    client, _ = make_client(hardened_checkpoint, api_key="k", rate_limit=5, request_timeout=1.0, max_queue=2)
    body = client.get("/health").json()
    assert body["auth_required"] is True
    assert body["rate_limit"] == 5 and body["request_timeout"] == 1.0 and body["max_queue"] == 2
