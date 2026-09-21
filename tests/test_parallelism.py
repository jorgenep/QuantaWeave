import copy
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import expert_parallel as ep
import pipeline_parallel as pl
import tensor_parallel as tp
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


# ---- process helpers ---------------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch(worker, world: int, *args) -> None:
    mp.spawn(worker, args=(world, free_port(), *args), nprocs=world, join=True)


def join(rank: int, world: int, port: int, tensor_parallel: int = 1) -> ep.ExpertParallelContext:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    return ep.init_expert_parallel("cpu", tensor_parallel)


def make_config(layers=2, capacity_factor=0.0, heads=4, ffn=24, experts=4, coef=0.01) -> QuantaWeaveConfig:
    return QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=layers, ffn_size=ffn, num_experts=experts, top_k=2, attention_heads=heads,
                             max_sequence_length=12, capacity_factor=capacity_factor, min_expert_capacity=1, router_aux_loss_coef=coef)


def load_from_full(model, full: dict, ctx: ep.ExpertParallelContext) -> None:
    """Fill an expert/tensor-parallel model with the slices of a full model's weights that belong to this rank."""
    per_rank = model.moes()[0].num_local_experts
    state = {}
    for key in model.state_dict():
        full_key = key
        match = ep.EXPERT_KEY.search(key)
        if match:
            full_key = ep.EXPERT_KEY.sub(lambda m: f"{m.group(1)}{ctx.rank * per_rank + int(m.group(2))}{m.group(3)}", key, count=1)
        state[key] = tp.shard_tensor(full_key, full[full_key], ctx.tp_rank, ctx.tp_size)
    model.load_state_dict(state)


def expected_local(name: str, tensor: torch.Tensor, ctx: ep.ExpertParallelContext, per_rank: int) -> torch.Tensor:
    full_name = name
    match = ep.EXPERT_KEY.search(name)
    if match:
        full_name = ep.EXPERT_KEY.sub(lambda m: f"{m.group(1)}{ctx.rank * per_rank + int(m.group(2))}{m.group(3)}", name, count=1)
    return full_name


# ---- tensor parallel: slicing and attention ------------------------------------------------------------
@pytest.mark.parametrize("tp_size", [2, 4])
def test_tensor_parallel_state_slicing_roundtrips(tp_size):
    torch.manual_seed(0)
    model = QuantaWeaveMoEForCausalLM(make_config(heads=4, ffn=24))
    full = model.state_dict()
    shards = [tp.shard_state_dict(full, r, tp_size) for r in range(tp_size)]
    merged = tp.merge_state_dicts(shards)
    for key, tensor in full.items():
        assert torch.equal(merged[key], tensor), key
    assert shards[0]["blocks.0.attention.in_proj_weight"].shape[0] == full["blocks.0.attention.in_proj_weight"].shape[0] // tp_size
    assert shards[0]["blocks.0.moe.experts.1.down.weight"].shape[1] == 24 // tp_size
    assert torch.equal(shards[1]["blocks.0.moe_norm.weight"], full["blocks.0.moe_norm.weight"])       # replicated tensors untouched
    assert tp.is_tp_sharded("blocks.0.attention.out_proj.weight") and not tp.is_tp_sharded("blocks.0.attention.out_proj.bias")
    assert tp.is_tp_sharded_shared("blocks.3.attention.in_proj_bias") and not tp.is_tp_sharded_shared("blocks.0.moe.experts.0.gate.weight")


def attention_worker(rank, world, port):
    ctx = join(rank, world, port, tensor_parallel=world)
    torch.manual_seed(0)
    reference = torch.nn.MultiheadAttention(16, 4, batch_first=True)
    x_ref = torch.randn(2, 8, 16, requires_grad=True)
    mask = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    expected, _ = reference(x_ref, x_ref, x_ref, attn_mask=mask, need_weights=False)
    expected.square().sum().backward()

    attention = tp.TensorParallelAttention(16, 4, ctx.tp_size, ctx.tp_group)
    prefixed = {"blocks.0.attention." + k: v for k, v in reference.state_dict().items()}
    attention.load_state_dict({k[len("blocks.0.attention."):]: tp.shard_tensor(k, v, ctx.tp_rank, ctx.tp_size) for k, v in prefixed.items()})
    x = x_ref.detach().clone().requires_grad_(True)
    out, _ = attention(x, x, x)
    assert torch.allclose(out, expected, atol=1e-5), "attention output"
    out.square().sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-4), "input gradient (needs the all-reduce in copy_to_tp's backward)"
    for name, param in attention.named_parameters():
        want = tp.shard_tensor("blocks.0.attention." + name, dict(reference.named_parameters())[name].grad, ctx.tp_rank, ctx.tp_size)
        assert torch.allclose(param.grad, want, atol=1e-4), name
    with pytest.raises(ValueError, match="divisible"):
        tp.TensorParallelAttention(16, 4, 3, None)
    ep.finish(ctx)


def test_tensor_parallel_attention_matches_multihead_attention():
    launch(attention_worker, 2)


# ---- the whole model under expert x tensor parallelism -------------------------------------------------
def full_model_worker(rank, world, port, tensor_parallel, capacity_factor):
    ctx = join(rank, world, port, tensor_parallel)
    config = make_config(capacity_factor=capacity_factor)
    torch.manual_seed(0)
    reference = QuantaWeaveMoEForCausalLM(config)
    torch.manual_seed(7)
    batches = torch.randint(0, 32, (ctx.world_size, 3, 10))
    model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
    load_from_full(model, reference.state_dict(), ctx)

    out = model(batches[ctx.rank], labels=batches[ctx.rank])
    ref_outputs = [reference(batches[r], labels=batches[r]) for r in range(ctx.world_size)]
    mine = ref_outputs[ctx.rank]
    assert torch.allclose(out["logits"], mine["logits"], atol=1e-4), "logits"
    assert torch.allclose(out["loss"], mine["loss"], atol=1e-5), "loss"
    assert out["overflow_routes"].item() == mine["overflow_routes"].item()
    if capacity_factor:
        assert sum(o["overflow_routes"].item() for o in ref_outputs) > 0

    out["loss"].backward()
    sum(o["loss"] for o in ref_outputs).backward()
    per_rank = model.moes()[0].num_local_experts
    ep.sync_gradients(model, ctx)
    scale = 1.0 / ctx.world_size                       # everything is a mean over the expert-parallel group's batches
    reference_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        full_name = expected_local(name, param, ctx, per_rank)
        want = tp.shard_tensor(full_name, reference_params[full_name].grad, ctx.tp_rank, ctx.tp_size) * scale
        assert torch.allclose(param.grad, want, atol=1e-4, rtol=1e-3), f"gradient of {name}"

    expected_norm = torch.sqrt(sum((p.grad * scale).pow(2).sum() for p in reference.parameters())).item()
    norm = ep.clip_grad_norm_parallel(model, ctx, max_norm=1e-3)
    assert abs(norm - expected_norm) < 1e-4 * expected_norm, (norm, expected_norm)
    coefficient = 1e-3 / (expected_norm + 1e-6)
    name, param = next((n, p) for n, p in model.named_parameters() if n.endswith("final_norm.weight"))
    want = reference_params[name].grad * scale * coefficient
    assert torch.allclose(param.grad, want, atol=1e-6, rtol=1e-3), "clipped gradient"
    ep.finish(ctx)


@pytest.mark.parametrize("world,tensor_parallel,capacity_factor", [(2, 1, 0.0), (2, 2, 0.0), (4, 2, 0.6)])
def test_model_matches_single_device_under_expert_and_tensor_parallelism(world, tensor_parallel, capacity_factor):
    launch(full_model_worker, world, tensor_parallel, capacity_factor)


def test_tensor_parallel_needs_divisible_shapes():
    ctx = ep.ExpertParallelContext(0, 1, torch.device("cpu"), None, False, 0, 3, None)
    with pytest.raises(ValueError, match="divisible"):
        QuantaWeaveMoEForCausalLM(make_config(heads=4, ffn=24), expert_parallel=ctx)
    ctx2 = ep.ExpertParallelContext(0, 1, torch.device("cpu"), None, False, 0, 2, None)
    with pytest.raises(ValueError, match="ffn_size"):
        QuantaWeaveMoEForCausalLM(make_config(heads=4, ffn=25), expert_parallel=ctx2)


# ---- training end to end: replicas stay identical, consolidation inverts the sharding ------------------------
def write_corpus(path: Path) -> None:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(60)))


def training_worker(rank, world, port, tmp, name, overrides):
    import train_quantweave_moe as trainer
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    tmp = Path(tmp)
    values = dict(data=[tmp / "d.jsonl"], steps=6, batch_size=2, sequence_length=16, examples=60, hidden_size=16, layers=2, ffn_size=24,
                  total_experts=4, active_experts=2, vocab_size=64, device="cpu", checkpoint_interval=0,
                  output=tmp / name / "out", checkpoint_dir=tmp / name / "ckpt", metrics_file=tmp / name / "metrics.jsonl")
    values.update(overrides)
    trainer.run_training(trainer.default_args(**values))


def load_shards(directory: Path, world: int) -> list[dict]:
    return [torch.load(ep.shard_path(directory, r), weights_only=False) for r in range(world)]


def max_difference(a: dict, b: dict, keys) -> float:
    return max((a[k] - b[k]).abs().max().item() for k in keys) if keys else 0.0


@pytest.mark.parametrize("world,overrides", [
    (2, dict(expert_parallel=True)),
    (2, dict(expert_parallel=True, shard_optimizer=True)),
    (4, dict(expert_parallel=True, tensor_parallel=2)),
    (4, dict(expert_parallel=True, tensor_parallel=2, shard_optimizer=True, straggler_routing=True, straggler_interval=2)),
    (2, dict(tensor_parallel=2)),
])
def test_replicated_weights_stay_identical_and_consolidation_inverts_sharding(tmp_path, world, overrides):
    write_corpus(tmp_path / "d.jsonl")
    # a huge aux-loss weight makes gradient norms >> 1, so clipping is active on every step
    launch(training_worker, world, str(tmp_path), "run", {**overrides, "router_aux_coef": 300.0})
    tensor_parallel = overrides.get("tensor_parallel", 1)
    shards = load_shards(tmp_path / "run" / "out" / "shards", world)
    layout = [s["layout"] for s in shards]
    assert [(l["ep_rank"], l["tp_rank"]) for l in layout] == [(r // tensor_parallel, r % tensor_parallel) for r in range(world)]

    names = list(shards[0]["model"])
    replicated = [n for n in names if not ep.is_expert_parameter(n) and not tp.is_tp_sharded_shared(n)]
    sharded_shared = [n for n in names if tp.is_tp_sharded_shared(n)]
    for other in shards[1:]:                                                        # identical across the whole world
        assert max_difference(shards[0]["model"], other["model"], replicated) == 0.0, "replicated tensors diverged"
    for shard in shards:                                                            # identical across the expert group of one tp column
        first = next(s for s in shards if s["layout"]["tp_rank"] == shard["layout"]["tp_rank"])
        assert max_difference(first["model"], shard["model"], sharded_shared) == 0.0, "tensor-parallel slices diverged across data-parallel ranks"

    merged = torch.load(tmp_path / "run" / "out" / "model.pt", weights_only=False)["model"]
    config = QuantaWeaveConfig(**json.loads((tmp_path / "run" / "out" / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(merged)                                                   # a normal, single-device model
    per_rank = config.num_experts // (world // tensor_parallel)
    for shard in shards:
        lay = shard["layout"]
        for key, tensor in shard["model"].items():
            full_key = ep.EXPERT_KEY.sub(lambda m: f"{m.group(1)}{lay['ep_rank'] * per_rank + int(m.group(2))}{m.group(3)}", key, count=1)
            assert torch.equal(tp.shard_tensor(full_key, merged[full_key], lay["tp_rank"], lay["tp_size"]), tensor), key
    metrics = [json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 6 and all(m["grad_norm"] > 1.0 for m in metrics), "clipping should have been active"
    assert all(torch.isfinite(torch.tensor(m["loss"])) for m in metrics)


def test_sharded_and_unsharded_optimizers_train_identically_and_the_shard_is_smaller(tmp_path):
    write_corpus(tmp_path / "d.jsonl")
    launch(training_worker, 2, str(tmp_path), "plain", dict(expert_parallel=True, steps=5))
    launch(training_worker, 2, str(tmp_path), "zero", dict(expert_parallel=True, steps=5, shard_optimizer=True))
    a = torch.load(tmp_path / "plain" / "out" / "model.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "zero" / "out" / "model.pt", weights_only=False)["model"]
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def zero_state_worker(rank, world, port):
    ctx = join(rank, world, port)
    from sharded_optimizer import ShardedAdamW
    config = make_config(layers=2)
    torch.manual_seed(0)
    reference = QuantaWeaveMoEForCausalLM(config)
    batches = torch.randint(0, 32, (world, 3, 10))
    models, optimizers = [], []
    for sharded in (False, True):
        model = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
        load_from_full(model, reference.state_dict(), ctx)
        optimizers.append(ShardedAdamW(model, ctx, lr=1e-2) if sharded else torch.optim.AdamW(model.parameters(), lr=1e-2))
        models.append(model)
    for step in range(4):
        for model, optimizer, sharded in zip(models, optimizers, (False, True)):
            model(batches[ctx.rank], labels=batches[ctx.rank])["loss"].backward()
            ep.sync_gradients(model, ctx, optimizer if sharded else None)
            ep.clip_grad_norm_parallel(model, ctx, 1.0, sharded)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    for (name, a), (_, b) in zip(models[0].named_parameters(), models[1].named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), f"{name} differs between sharded and plain optimizers"

    plain_elements = sum(t.numel() for state in optimizers[0].state.values() for t in state.values() if torch.is_tensor(t) and t.dim() > 0)
    sharded_elements = optimizers[1].state_numel()
    replicated_elements = sum(p.numel() for n, p in models[1].named_parameters() if not ep.is_expert_parameter(n))
    assert 0 < sharded_elements < plain_elements
    assert plain_elements - sharded_elements >= replicated_elements * 2 * 0.25            # about half the replicated moments are gone

    state = copy.deepcopy(optimizers[1].state_dict())              # state_dict() aliases live tensors; a checkpoint serialises immediately
    fresh = QuantaWeaveMoEForCausalLM(config, expert_parallel=ctx)
    fresh.load_state_dict(models[1].state_dict())
    resumed = ShardedAdamW(fresh, ctx, lr=1e-2)
    resumed.load_state_dict(state)
    for model, optimizer in ((models[1], optimizers[1]), (fresh, resumed)):
        model(batches[ctx.rank], labels=batches[ctx.rank])["loss"].backward()
        ep.sync_gradients(model, ctx, optimizer)
        ep.clip_grad_norm_parallel(model, ctx, 1.0, True)
        optimizer.step()
    for (name, a), (_, b) in zip(models[1].named_parameters(), fresh.named_parameters()):
        assert torch.allclose(a, b, atol=1e-7), name
    bad = dict(state, owners=[0] * len(state["owners"]))
    if len(set(optimizers[1].owner)) > 1:
        with pytest.raises(ValueError, match="ownership"):
            resumed.load_state_dict(bad)
    ep.finish(ctx)


def test_sharded_optimizer_matches_adamw_holds_less_state_and_resumes():
    launch(zero_state_worker, 2)


# ---- straggler-aware routing -----------------------------------------------------------------------------
class _Shim:
    def __init__(self, moes):
        self._moes = moes

    def moes(self):
        return self._moes


def straggler_worker(rank, world, port):
    ctx = join(rank, world, port)
    config = make_config(layers=1, experts=8)
    torch.manual_seed(0)                                   # identical routers on every rank
    layer = ep.ExpertParallelMoE(config, ctx)
    layer.simulated_cost_per_row = 4e-3 if rank == 1 else 1e-3      # rank 1 is 4x slower per row
    shim = _Shim([layer])
    torch.manual_seed(100 + rank)
    inputs = torch.randn(1, 96, 16)

    def run(strength: float, iterations: int = 25):
        layer.expert_bias = None
        tracker = ep.DeviceLoadTracker(shim, ctx, strength=strength, interval=1, rate=0.3)
        history = []
        for step in range(1, iterations + 1):
            layer(inputs)
            history.append(tracker.update(step))
        return tracker, history

    _, baseline = run(0.0)
    tracker, steered = run(1.0)
    base_imbalance = sum(m["time_imbalance"] for m in baseline[-8:]) / 8
    steered_imbalance = sum(m["time_imbalance"] for m in steered[-8:]) / 8
    assert base_imbalance > 1.3, base_imbalance                       # the slow rank really is the bottleneck without steering
    assert steered_imbalance < 0.85 * base_imbalance, (base_imbalance, steered_imbalance)
    rows_slow = sum(m["device_rows"][1] for m in steered[-8:])
    rows_fast = sum(m["device_rows"][0] for m in steered[-8:])
    assert rows_slow < 0.6 * rows_fast                                # traffic moved off the slow rank
    assert tracker.rank_bias[0, 1] < 0 < tracker.rank_bias[0, 0]

    gathered = [torch.zeros_like(tracker.rank_bias) for _ in range(world)]
    dist.all_gather(gathered, tracker.rank_bias)
    assert all(torch.equal(g, gathered[0]) for g in gathered), "every rank must compute the same bias"
    assert layer.expert_bias.shape == (8,) and layer.expert_bias[:4].mean() > layer.expert_bias[4:].mean()

    restored = ep.DeviceLoadTracker(shim, ctx, strength=1.0)
    layer.expert_bias = None
    restored.load_state_dict(json.loads(json.dumps(tracker.state_dict())))
    assert torch.equal(layer.expert_bias.cpu(), tracker.rank_bias[0].repeat_interleave(4).float())
    ep.finish(ctx)


def test_straggler_routing_moves_traffic_off_a_slow_rank_and_stays_consistent():
    launch(straggler_worker, 2)


def test_expert_bias_steers_selection_but_not_mixing_weights():
    from quantweave_moe_model import TopKMoE
    torch.manual_seed(0)
    moe = TopKMoE(make_config(experts=4))
    hidden = torch.randn(1, 40, 16)
    baseline = moe(hidden)[0]
    moe.expert_bias = torch.zeros(4)
    assert torch.allclose(moe(hidden)[0], baseline)                   # a zero bias changes nothing
    moe.expert_bias = torch.tensor([-100.0, 0.0, 0.0, 0.0])
    moe(hidden)
    assert (moe.last_top_indices == 0).sum() == 0                     # the penalised expert is never chosen


# ---- pipeline parallelism ---------------------------------------------------------------------------------
def test_layer_partitioning():
    assert pl.partition_layers(8, 4) == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert pl.partition_layers(5, 3) == [(0, 2), (2, 4), (4, 5)]
    assert pl.partition_layers(2, 2) == [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match="cannot split"):
        pl.partition_layers(2, 3)


def stage_state(reference_state: dict, ctx: pl.PipelineContext, model) -> dict:
    state = {}
    for key in model.state_dict():
        match = ep.BLOCK_KEY.match(key)
        full_key = f"blocks.{ctx.layer_start + int(match.group(1))}." + key[match.end():] if match else key
        state[key] = reference_state[full_key]
    return state


def pipeline_worker(rank, world, port, layers, microbatches, coef):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    ctx = pl.init_pipeline_parallel("cpu", layers)
    config = make_config(layers=layers, coef=coef)
    torch.manual_seed(0)
    reference = QuantaWeaveMoEForCausalLM(config)
    model = QuantaWeaveMoEForCausalLM(config, pipeline=ctx)
    model.load_state_dict(stage_state(reference.state_dict(), ctx, model))
    assert (model.token_embedding is not None) == ctx.is_first and (model.lm_head is not None) == ctx.is_last
    assert len(model.blocks) == ctx.layer_end - ctx.layer_start

    torch.manual_seed(9)
    batch = torch.randint(0, 32, (4, 10))
    result = pl.PipelineEngine(model, ctx, microbatches).train_step(batch)
    expected = reference(batch, labels=batch)
    assert abs(result["loss"] - expected["loss"].item()) < 1e-5, (result["loss"], expected["loss"].item())
    if microbatches == 1:
        assert abs(result["router_aux_loss"] - expected["router_aux_loss"].item()) < 1e-5
    expected["loss"].backward()
    reference_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        match = ep.BLOCK_KEY.match(name)
        full = f"blocks.{ctx.layer_start + int(match.group(1))}." + name[match.end():] if match else name
        assert torch.allclose(param.grad, reference_params[full].grad, atol=1e-5, rtol=1e-3), f"gradient of {full}"

    expected_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in reference.parameters())).item()
    norm = pl.clip_grad_norm_pipeline(model, ctx, max_norm=1e-3)
    assert abs(norm - expected_norm) < 1e-4 * expected_norm
    with pytest.raises(ValueError, match="divisible"):
        pl.PipelineEngine(model, ctx, 3).train_step(batch)
    ep.finish(ctx)


@pytest.mark.parametrize("world,layers,microbatches,coef", [(2, 4, 1, 0.05), (2, 4, 2, 0.0), (3, 5, 2, 0.0)])
def test_pipeline_matches_single_device_loss_gradients_and_global_norm(world, layers, microbatches, coef):
    # With several micro-batches the balance loss is computed per micro-batch (a nonlinear statistic), so exact
    # equality with the whole-batch loss is only expected for one micro-batch or a zero aux weight.
    launch(pipeline_worker, world, layers, microbatches, coef)


def test_pipeline_training_consolidates_and_resumes_exactly(tmp_path):
    write_corpus(tmp_path / "d.jsonl")
    common = dict(pipeline_parallel=True, microbatches=2, batch_size=4, layers=3, checkpoint_interval=2, plateau_patience=1, controller_interval=2,
                  lr_decay="cosine", warmup_steps=1, schedule_steps=4)
    launch(training_worker, 3, str(tmp_path), "straight", {**common, "steps": 4})
    out = tmp_path / "straight" / "out"
    assert (out / "shards" / "model.rank2.pt").exists()
    config = QuantaWeaveConfig(**json.loads((out / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=False)["model"])          # 3 stages merged into one model
    assert model(torch.randint(0, 40, (2, 8)))["logits"].isfinite().all()
    metrics = [json.loads(line) for line in (tmp_path / "straight" / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 4 and all(torch.isfinite(torch.tensor(m["loss"])) for m in metrics)

    launch(training_worker, 3, str(tmp_path), "split", {**common, "steps": 2})
    launch(training_worker, 3, str(tmp_path), "split", {**common, "steps": 4, "output": tmp_path / "split" / "out2"})
    a = torch.load(out / "model.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "out2" / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_parallel_option_validation():
    import train_quantweave_moe as trainer
    base = dict(device="cpu", checkpoint_interval=0)
    for overrides, message in [
        (dict(pipeline_parallel=True, expert_parallel=True), "cannot be combined"),
        (dict(shard_optimizer=True), "need --expert-parallel"),
        (dict(pipeline_parallel=True, gradient_accumulation_steps=2), "micro-batches replace"),
        (dict(pipeline_parallel=True, microbatches=3, batch_size=4), "must divide"),
        (dict(expert_parallel=True, precision="fp16"), "fp16"),
        (dict(pipeline_parallel=True, diagnostics_interval=5), "diagnostics"),
        (dict(expert_parallel=True, aux_adapt=True), "not supported"),
        (dict(tensor_parallel=0), "positive"),
    ]:
        with pytest.raises(ValueError, match=message):
            trainer.run_training(trainer.default_args(**base, **overrides))
    with pytest.raises(RuntimeError, match="torchrun"):
        for name in ("RANK", "WORLD_SIZE"):
            os.environ.pop(name, None)
        pl.init_pipeline_parallel("cpu", 4)
