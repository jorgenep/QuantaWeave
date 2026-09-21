import json
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import expert_parallel as ep
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM, TopKMoE


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch(worker, world: int, *args) -> None:
    mp.spawn(worker, args=(world, free_port(), *args), nprocs=world, join=True)


def join_group(rank: int, world: int, port: int) -> ep.ExpertParallelContext:
    import os
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    return ep.init_expert_parallel("cpu")


def moe_config(capacity_factor: float) -> QuantaWeaveConfig:
    return QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=1, ffn_size=24, num_experts=4, top_k=2, attention_heads=2,
                             max_sequence_length=16, capacity_factor=capacity_factor, min_expert_capacity=1)


def parity_worker(rank: int, world: int, port: int, capacity_factor: float) -> None:
    ctx = join_group(rank, world, port)
    config = moe_config(capacity_factor)
    torch.manual_seed(0)                                   # identical reference on every rank
    reference = TopKMoE(config)
    torch.manual_seed(100)
    inputs = torch.randn(world, 3, 8, 16)                  # each rank's local batch
    sharded = ep.ExpertParallelMoE(config, ctx)
    sharded.router.load_state_dict(reference.router.state_dict())
    per_rank = config.num_experts // world
    for local, expert in enumerate(sharded.experts):
        expert.load_state_dict(reference.experts[rank * per_rank + local].state_dict())

    out, balance, dropped, overflow, _ = sharded(inputs[rank])
    expected = [reference(inputs[r]) for r in range(world)]
    assert torch.allclose(out, expected[rank][0], atol=1e-5), "forward output differs from single device"
    assert overflow.item() == expected[rank][3].item()
    if capacity_factor:
        assert any(e[3].item() > 0 for e in expected), "the test should exercise capacity overflow"

    out.square().sum().backward()
    sum(e[0].square().sum() for e in expected).backward()
    for local, expert in enumerate(sharded.experts):
        for (name, param), (_, ref_param) in zip(expert.named_parameters(), reference.experts[rank * per_rank + local].named_parameters()):
            assert torch.allclose(param.grad, ref_param.grad, atol=1e-4), f"expert grad {name}"
    router_grad = sharded.router.weight.grad.clone()
    dist.all_reduce(router_grad)
    assert torch.allclose(router_grad, reference.router.weight.grad, atol=1e-4), "router grad"

    # sync_gradients addresses parameters by ".moe.experts." names, hence the shim
    before = [p.grad.clone() for p in sharded.experts.parameters()]
    ep.sync_gradients(_Shim(sharded), ctx)
    for grad, param in zip(before, sharded.experts.parameters()):
        assert torch.allclose(param.grad, grad / world)
    assert torch.allclose(sharded.router.weight.grad, reference.router.weight.grad / world, atol=1e-4)
    ep.finish(ctx)


class _Shim(torch.nn.Module):
    """Wraps one MoE so parameter names look like a full model's (blocks.0.moe.experts...)."""

    def __init__(self, moe):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.blocks[0].moe = moe


@pytest.mark.parametrize("capacity_factor", [0, 0.5])
def test_expert_parallel_matches_single_device_forward_and_backward(capacity_factor):
    launch(parity_worker, 2, capacity_factor)


def train_worker(rank: int, world: int, port: int, tmp: str, steps: int, name: str, result_dir: str) -> None:
    import os
    import train_quantweave_moe as trainer
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    tmp_path = Path(tmp)
    args = trainer.default_args(
        data=[tmp_path / "d.jsonl"], steps=steps, batch_size=2, sequence_length=16, examples=60, hidden_size=16, layers=2,
        ffn_size=24, total_experts=4, active_experts=2, vocab_size=64, device="cpu", expert_parallel=True,
        checkpoint_interval=2, output=tmp_path / name / "out", checkpoint_dir=tmp_path / name / "ckpt",
        metrics_file=tmp_path / name / "metrics.jsonl", capacity_adapt=True, controller_interval=2, plateau_patience=1,
    )
    summary = trainer.run_training(args)
    if rank == 0:
        (Path(result_dir) / f"{name}.json").write_text(json.dumps({"final_loss": summary["final_loss"], "steps": summary["steps"]}))


def write_corpus(path: Path) -> None:
    path.write_text("".join(json.dumps({"text": f"story {i}: the quick brown fox jumps over the lazy dog, again."}) + "\n" for i in range(60)))


def test_expert_parallel_training_consolidates_resumes_and_matches_a_straight_run(tmp_path):
    write_corpus(tmp_path / "d.jsonl")
    launch(train_worker, 2, str(tmp_path), 4, "straight", str(tmp_path))
    out = tmp_path / "straight" / "out"
    assert (out / "model.pt").exists() and (out / "shards" / "model.rank1.pt").exists()

    # the consolidated checkpoint is an ordinary single-device checkpoint
    config = QuantaWeaveConfig(**json.loads((out / "config.json").read_text()))
    model = QuantaWeaveMoEForCausalLM(config)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=False)["model"])
    logits = model(torch.randint(0, 40, (2, 8)))["logits"]
    assert logits.isfinite().all()
    assert len({k for k in model.state_dict() if ".moe.experts." in k}) == 2 * 4 * 3          # all 4 experts x 2 layers present

    # 2 steps, then resume to 4 from the shard checkpoint: identical to the uninterrupted run
    launch(train_worker, 2, str(tmp_path), 2, "split", str(tmp_path))
    launch(train_worker, 2, str(tmp_path), 4, "split", str(tmp_path))
    a = torch.load(out / "model.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "out" / "model.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key
    lines = [json.loads(x) for x in (tmp_path / "straight" / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and all(torch.isfinite(torch.tensor(l["loss"])) for l in lines)


def test_expert_count_must_divide_and_torchrun_env_is_required(monkeypatch):
    for name in ("RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        ep.init_expert_parallel("cpu")
    with pytest.raises(ValueError, match="divisible"):
        ep.ExpertParallelMoE(moe_config(0), ep.ExpertParallelContext(0, 3, torch.device("cpu")))


def test_consolidation_renumbers_experts_and_validates_shards(tmp_path):
    def fake_shard(rank):
        return {"model": {"blocks.0.moe.router.weight": torch.ones(1), "blocks.0.moe.experts.0.gate.weight": torch.full((1,), rank * 10.0),
                          "blocks.0.moe.experts.1.gate.weight": torch.full((1,), rank * 10.0 + 1)},
                "world_size": 2, "rank": rank, "step": 7, "extra": {}}
    torch.save(fake_shard(0), tmp_path / "model.rank0.pt")
    with pytest.raises(ValueError, match="expected 2 shards"):
        ep.consolidate_checkpoint(tmp_path, tmp_path / "out")
    torch.save(fake_shard(1), tmp_path / "model.rank1.pt")
    ep.consolidate_checkpoint(tmp_path, tmp_path / "out")
    merged = torch.load(tmp_path / "out" / "model.pt", weights_only=False)
    values = {k: v.item() for k, v in merged["model"].items() if "experts" in k}
    assert values == {"blocks.0.moe.experts.0.gate.weight": 0.0, "blocks.0.moe.experts.1.gate.weight": 1.0,
                      "blocks.0.moe.experts.2.gate.weight": 10.0, "blocks.0.moe.experts.3.gate.weight": 11.0}
    assert merged["step"] == 7
    with pytest.raises(FileNotFoundError):
        ep.consolidate_checkpoint(tmp_path / "nothing", tmp_path / "out2")
