import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import chat_quantweave_moe as chat
import data_pipeline as dp
import fast_decode as fd
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM

CPU = torch.device("cpu")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def make_model(layers=3, experts=8, top_k=2, context=24, capacity=0.0, temperature=1.0, seed=0) -> QuantaWeaveMoEForCausalLM:
    torch.manual_seed(seed)
    config = QuantaWeaveConfig(vocab_size=64, hidden_size=32, layers=layers, ffn_size=48, num_experts=experts, top_k=top_k, attention_heads=4,
                               max_sequence_length=context, capacity_factor=capacity, router_temperature=temperature)
    model = QuantaWeaveMoEForCausalLM(config).eval()
    model.set_routing_controls(drop_overflow_tokens=False)       # inference never drops tokens; the reference must not either
    return model


def full_logits(model, tokens: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor([tokens]))["logits"][0]


def decoder(model, **kwargs) -> fd.FastDecoder:
    return fd.FastDecoder(model, CPU, "fp32", **kwargs)


# ---- numerical agreement with the real forward pass -----------------------------------------------------------------
@pytest.mark.parametrize("top_k,temperature,bias", [(1, 1.0, False), (2, 1.0, False), (2, 0.7, True), (3, 2.0, False)])
def test_stepping_reproduces_the_full_forward_pass_at_every_position(top_k, temperature, bias):
    model = make_model(top_k=top_k, temperature=temperature)
    if bias:
        for moe in model.moes():
            moe.expert_bias = torch.randn(moe.num_experts)
    tokens = torch.randint(0, 64, (23,)).tolist()
    reference = full_logits(model, tokens)
    dec = decoder(model)
    logits = dec.feed(tokens[:1])
    assert torch.allclose(logits, reference[0], atol=1e-5)
    for position in range(1, 23):
        logits = dec.append(tokens[position])
        assert torch.allclose(logits, reference[position], atol=1e-5), position


def test_batched_prefill_matches_the_full_forward_and_leaves_the_same_cache_as_stepping():
    model = make_model()
    tokens = torch.randint(0, 64, (17,)).tolist()
    stepped = decoder(model)
    for token in tokens:
        stepped._run(token)
    prefilled = decoder(model)
    logits = prefilled.feed(tokens)
    assert torch.allclose(logits, full_logits(model, tokens)[-1], atol=1e-5)
    for a, b in zip(stepped.layers, prefilled.layers):
        assert torch.allclose(a.keys[:, :17], b.keys[:, :17], atol=1e-5) and torch.allclose(a.values[:, :17], b.values[:, :17], atol=1e-5)
    assert prefilled.length == 17 and prefilled.tokens == tokens


def test_dense_and_sorted_prefill_agree(monkeypatch):
    model = make_model(experts=8)
    tokens = torch.randint(0, 64, (12,)).tolist()
    caches = []
    for limit in (0, 10**9):
        monkeypatch.setattr(fd, "DENSE_PREFILL_LIMIT", limit)
        dec = decoder(model)
        dec._prefill(tokens)
        caches.append([layer.keys[:, :12].clone() for layer in dec.layers])
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*caches))
    assert all(not moe.static_dispatch and not moe.drop_overflow_tokens for moe in model.moes())      # the model is left as it was


def test_prefill_restores_the_models_routing_settings():
    model = make_model()
    for moe in model.moes():
        moe.drop_overflow_tokens = True
    decoder(model)._prefill([1, 2, 3])
    assert all(moe.drop_overflow_tokens and not moe.static_dispatch for moe in model.moes())


# ---- the context window ------------------------------------------------------------------------------------------------------
def test_context_is_one_shorter_than_the_training_window_because_the_last_position_was_never_trained():
    model = make_model(context=24)
    dec = decoder(model)
    assert dec.capacity == 23
    tokens = list(range(1, 40))
    dec.feed(tokens)
    assert dec.tokens == tokens[-23:] and dec.length == 23                                            # only the last 23 tokens are read


def test_the_last_window_position_really_is_untrained_and_the_reason_for_the_shorter_context(tmp_path):
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60)))
    trainer.run_training(trainer.default_args(data=[data], steps=150, batch_size=8, sequence_length=24, examples=60, hidden_size=32, layers=2, ffn_size=48,
                                              total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0, lr=3e-3,
                                              capacity_factor=0, output=tmp_path / "m", checkpoint_dir=tmp_path / "c"))
    config = QuantaWeaveConfig(**json.loads((tmp_path / "m" / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config).eval()
    model.load_state_dict(torch.load(tmp_path / "m" / "model.pt", weights_only=False)["model"])
    tokenizer = dp.load_tokenizer(tmp_path / "m")
    stream = []
    for i in range(30, 60):
        stream += tokenizer.encode(f"story {i}: the quick brown fox jumps over the lazy dog, again and again.") + [tokenizer.eos_id]
    width = config.max_sequence_length
    windows = torch.tensor(stream[: (len(stream) // (width + 1)) * (width + 1)]).view(-1, width + 1)
    with torch.no_grad():
        logits = model(windows[:, :width])["logits"]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), windows[:, 1:].reshape(-1), reduction="none").view(-1, width).mean(0)
    assert loss[-1] > 1.3 * loss[:-1].mean() and loss[-1] > loss[:-1].mean() + 0.3, (float(loss[-1]), float(loss[:-1].mean()))   # the untrained slot is far worse


def test_the_plain_chat_path_also_stays_inside_the_trained_context():
    from test_chat import ScriptedModel, session, ids_of
    s = session(ids_of("ab"), context=8, tokens=2)
    s.send("a" * 40)
    assert max(s.model.windows) <= 7                                                                    # never the full 8-token window


# ---- sliding ------------------------------------------------------------------------------------------------------------------
def test_when_the_context_fills_the_oldest_half_is_dropped_and_the_rest_is_reread():
    model = make_model(context=13)                                                                      # capacity 12
    dec = decoder(model)
    tokens = torch.randint(0, 64, (30,)).tolist()
    dec.feed(tokens[:1])
    seen = tokens[:1]
    for token in tokens[1:]:
        logits = dec.append(token)
        if len(seen) >= 12:
            seen = seen[-6:]                                                                              # half of the capacity is kept
        seen = seen + [token]
        assert dec.tokens == seen and dec.length == len(seen) and dec.length <= 12
        assert torch.allclose(logits, full_logits(model, seen)[-1], atol=1e-5)                          # exactly what a fresh read of the kept tokens gives


# ---- streaming with sampling inside the step ---------------------------------------------------------------------------------------
def host_driven_greedy(model, prompt: list[int], count: int) -> list[int]:
    dec = decoder(model)
    logits = dec.feed(prompt)
    out = []
    for _ in range(count):
        token = int(logits.argmax())
        out.append(token)
        logits = dec.append(token)
    return out


@pytest.mark.parametrize("count", [1, 7, 8, 9, 20, 61])
def test_greedy_stream_equals_the_host_driven_loop_including_across_context_slides(count):
    model = make_model(context=13)                                                                      # capacity 12: 61 tokens slide several times
    prompt = torch.randint(2, 64, (5,)).tolist()
    streamed = list(decoder(model).stream(prompt, count, temperature=0.0))
    assert streamed == host_driven_greedy(model, prompt, count) and len(streamed) == count


def test_long_prompts_and_single_token_prompts_stream():
    model = make_model(context=13)
    assert len(list(decoder(model).stream(list(range(2, 50)), 5, temperature=0.0))) == 5              # prompt longer than the context
    assert len(list(decoder(model).stream([7], 5, temperature=0.0))) == 5
    with pytest.raises(ValueError):
        list(decoder(model).stream([], 5, temperature=0.0))


def test_sampling_is_reproducible_per_seed_and_respects_top_k_top_p_and_the_vocabulary_mask():
    model = make_model(context=30)
    dec = decoder(model)
    run = lambda seed, **kw: list(dec.stream([3, 4, 5], 30, temperature=1.0, generator=torch.Generator().manual_seed(seed), **kw))  # noqa: E731
    assert run(1) == run(1) and run(1) != run(2)
    greedy = list(dec.stream([3, 4, 5], 30, temperature=0.0))
    assert run(5, top_k=1) == greedy                                                                     # top_k=1 is greedy
    assert run(5, top_p=1e-6) == greedy                                                                  # a vanishing nucleus is greedy
    masked = run(3, valid_ids=20, unk_id=0)
    assert all(0 < t < 20 for t in masked)                                                               # ids >= 20 and <unk> never appear
    assert all(0 <= t < 64 for t in run(4))


def test_first_token_frequencies_follow_the_softmax_distribution():
    model = make_model(context=16)
    dec = decoder(model)
    prompt = [3, 4, 5, 6]
    probabilities = torch.softmax(full_logits(model, prompt)[-1], dim=-1)
    generator = torch.Generator().manual_seed(0)
    counts = torch.zeros(64)
    trials = 1500
    for _ in range(trials):
        counts[next(iter(dec.stream(prompt, 1, temperature=1.0, generator=generator)))] += 1
    assert (counts / trials - probabilities).abs().max() < 0.05
    top = int(probabilities.argmax())
    assert counts[top] / trials == pytest.approx(float(probabilities[top]), abs=0.05)


def test_temperature_reshapes_and_early_close_leaves_the_decoder_reusable():
    model = make_model(context=30)
    dec = decoder(model)
    cold = [list(dec.stream([3, 4], 12, temperature=1e-3, generator=torch.Generator().manual_seed(s))) for s in range(4)]
    assert all(c == cold[0] for c in cold)                                                                # a cold temperature is (near) greedy
    generator = dec.stream([3, 4], 50, temperature=0.0)
    first = [next(generator) for _ in range(3)]
    generator.close()                                                                                     # abandoned mid-batch
    assert list(dec.stream([3, 4], 5, temperature=0.0)) == host_driven_greedy(model, [3, 4], 5) and first == host_driven_greedy(model, [3, 4], 3)


def test_a_captured_program_per_sampling_setting_is_cached_and_bounded():
    dec = decoder(make_model())
    for k in range(9):
        list(dec.stream([2, 3], 1, temperature=0.5, top_k=k))
    assert len(dec._programs) <= 9                                                                        # eager programs are plain closures
    list(dec.stream([2, 3], 1, temperature=0.5, top_k=3))
    assert (0.5, 3, 1.0) in dec._programs


# ---- other model kinds ---------------------------------------------------------------------------------------------------------------
def test_quantized_and_lora_models_decode_like_their_own_forward_pass():
    from lora import add_lora
    from quantization import quantize_model
    model = make_model(top_k=2)
    quantize_model(model, 8)
    tokens = torch.randint(0, 64, (14,)).tolist()
    assert torch.allclose(decoder(model).feed(tokens), full_logits(model, tokens)[-1], atol=1e-4)
    add_lora(model, rank=4, alpha=8)
    for module in model.modules():
        if isinstance(module, torch.nn.Module) and hasattr(module, "lora_b"):
            module.lora_b.data.normal_(0, 0.1)
    assert torch.allclose(decoder(model).feed(tokens), full_logits(model, tokens)[-1], atol=1e-4)      # the adapter is folded in


def test_unsupported_layouts_are_refused_cleanly():
    with pytest.raises(fd.Unsupported):
        fd.FastDecoder(torch.nn.Linear(2, 2), CPU)
    import expert_parallel as ep
    ctx = ep.ExpertParallelContext(0, 1, CPU, None, False, 0, 2, None)
    tensor_parallel = QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(vocab_size=8, hidden_size=16, layers=1, ffn_size=8, num_experts=2, top_k=1,
                                                                  attention_heads=2, max_sequence_length=8), expert_parallel=ctx)
    with pytest.raises(fd.Unsupported):
        fd.FastDecoder(tensor_parallel, CPU)
    session = chat.ChatSession(tensor_parallel.eval(), dp.CharTokenizer.build(["ab"], 8), CPU, "fp32", chat.Settings())
    assert session.decoder is None and "no cache" in session.cache_note                                # the chat tool falls back instead of failing


# ---- through the chat tool --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fd")
    data = root / "d.jsonl"
    data.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again and again."}) + "\n" for i in range(60)))
    trainer.run_training(trainer.default_args(data=[data], steps=30, batch_size=4, sequence_length=64, examples=60, hidden_size=32, layers=2, ffn_size=48,
                                              total_experts=4, active_experts=2, vocab_size=96, device="cpu", checkpoint_interval=0, lr=3e-3,
                                              capacity_factor=0, output=root / "model", checkpoint_dir=root / "ck"))
    return root / "model"


def test_cached_and_plain_chat_paths_give_the_same_greedy_text(trained):
    model, tokenizer = chat.load_for_chat(trained, None, 0, CPU)
    settings = dict(temperature=0.0, tokens=40)
    cached = chat.ChatSession(model, tokenizer, CPU, "fp32", chat.Settings(**settings), use_cache=True)
    plain = chat.ChatSession(model, tokenizer, CPU, "fp32", chat.Settings(**settings), use_cache=False)
    assert cached.decoder is not None and plain.decoder is None and "KV cache" in cached.cache_note
    for message in ("story 3: the quick", "the lazy dog", "story 12:"):
        assert cached.send(message).response == plain.send(message).response


def test_chat_streams_the_same_text_it_returns_and_reports_stops(trained):
    model, tokenizer = chat.load_for_chat(trained, None, 0, CPU)
    session = chat.ChatSession(model, tokenizer, CPU, "fp32", chat.Settings(temperature=0.0, tokens=30, stop=["fox"]))
    pieces = []
    reply = session.send("story 3: the quick brown ", on_text=pieces.append)
    assert "".join(pieces) == reply.response and "fox" not in reply.response
    if reply.stop_reason == "stop":
        assert session.send("story 3: the quick brown ").response == reply.response


def test_long_generation_keeps_going_past_the_context_window(trained):
    model, tokenizer = chat.load_for_chat(trained, None, 0, CPU)
    session = chat.ChatSession(model, tokenizer, CPU, "fp32", chat.Settings(temperature=0.8, tokens=300, seed=1))
    tokenizer_eos, tokenizer.eos_id = tokenizer.eos_id, -1                                                # do not stop early
    try:
        reply = session.send("story 1: the quick")
    finally:
        tokenizer.eos_id = tokenizer_eos
    assert reply.tokens == 300 and reply.stop_reason == "length" and session.decoder.length <= session.decoder.capacity


def test_no_cache_flag_and_status_line(trained, capsys):
    assert chat.main(["--checkpoint", str(trained), "--device", "cpu", "-m", "story 3:", "--tokens", "5", "--temperature", "0"]) == 0
    assert "decoding: KV cache" in capsys.readouterr().err
    assert chat.main(["--checkpoint", str(trained), "--device", "cpu", "-m", "story 3:", "--tokens", "5", "--temperature", "0", "--no-cache"]) == 0
    captured = capsys.readouterr()
    assert "decoding: no cache" in captured.err


# ---- CUDA graph and compile -----------------------------------------------------------------------------------------------------------------
@cuda_only
def test_cuda_graph_and_eager_decoding_agree_and_compile_failure_falls_back(monkeypatch):
    dev = torch.device("cuda")
    model = make_model(layers=2, experts=8, context=40).to(dev)
    prompt = torch.randint(2, 64, (6,)).tolist()
    eager = list(fd.FastDecoder(model, dev, "bf16", use_graph=False).stream(prompt, 40, temperature=0.0))
    graph_plain = fd.FastDecoder(model, dev, "bf16", use_graph=True, compile=False)
    assert graph_plain.uses_graph and not graph_plain.compiled and graph_plain.description == "KV cache + CUDA graph"
    assert list(graph_plain.stream(prompt, 40, temperature=0.0)) == eager                              # capturing changes nothing
    assert list(graph_plain.stream(prompt, 40, temperature=0.0)) == eager                              # and replays are repeatable
    sampled = lambda seed: list(graph_plain.stream(prompt, 40, temperature=0.9, top_k=20, generator=torch.Generator(device=dev).manual_seed(seed)))  # noqa: E731
    assert sampled(1) == sampled(1) and sampled(1) != sampled(2)

    compiled = fd.FastDecoder(model, dev, "bf16", use_graph=True, compile=True)
    assert compiled.compiled and "torch.compile" in compiled.description
    agree = sum(a == b for a, b in zip(list(compiled.stream(prompt, 40, temperature=0.0)), eager)) / 40
    assert agree >= 0.9                                                                                   # fused bf16 kernels round slightly differently

    monkeypatch.setattr(torch, "compile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no compiler here")))
    fallback = fd.FastDecoder(model, dev, "bf16", use_graph=True, compile=True)
    assert not fallback.compiled and "no compiler" in fallback.compile_error and fallback.uses_graph
    assert list(fallback.stream(prompt, 40, temperature=0.0)) == eager


@cuda_only
def test_decoder_works_when_called_under_inference_mode_after_being_built_outside_it():
    dev = torch.device("cuda")
    model = make_model(layers=2, context=40).to(dev)
    dec = fd.FastDecoder(model, dev, "bf16", compile=True)
    with torch.inference_mode():
        first = list(dec.stream([2, 3, 4], 10, temperature=0.7, generator=torch.Generator(device=dev).manual_seed(0)))
        second = list(dec.stream([2, 3, 4], 10, temperature=0.0, top_k=5))                             # a new sampling setting captures a new graph in here
    assert len(first) == 10 and len(second) == 10
