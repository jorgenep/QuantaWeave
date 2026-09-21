import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import finetune_quantweave_moe as ft
import lora
import quantization as quant
import train_quantweave_moe as trainer
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM


def corpus(path: Path, template: str, rows: int = 60) -> Path:
    path.write_text("".join(json.dumps({"text": template.format(i=i)}) + "\n" for i in range(rows)))
    return path


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("qlora")
    old = corpus(root / "old.jsonl", "story {i}: the quick brown fox jumps over the lazy dog, again and again.")
    trainer.run_training(trainer.default_args(
        data=[old], steps=25, batch_size=4, sequence_length=32, examples=60, hidden_size=32, layers=2, ffn_size=64,
        total_experts=8, active_experts=2, vocab_size=96, device="cpu", checkpoint_interval=0, lr=3e-3, capacity_factor=0,
        output=root / "base", checkpoint_dir=root / "ckpt"))
    # a different domain: the base has to adapt, and every character exists in the base vocabulary
    new = corpus(root / "new.jsonl", "record {i}: ship the blue box to the old dock on the lazy river, please.")
    return root, new


def test_lora_layer_starts_as_a_no_op_trains_only_the_adapter_and_merges_exactly():
    torch.manual_seed(0)
    linear = torch.nn.Linear(12, 7, bias=False)
    layer = lora.LoRALinear(linear, rank=4, alpha=8)
    x = torch.randn(5, 12)
    assert torch.equal(layer(x), linear(x))                                   # B starts at zero
    layer.lora_b.data.normal_()
    layer(x).sum().backward()
    assert layer.lora_a.grad is not None and layer.lora_b.grad is not None
    assert layer.base.weight.grad is None or layer.base.weight.grad.abs().sum() > 0   # the base is frozen by add_lora, not by the layer
    assert torch.allclose(layer(x), F.linear(x, layer.merged_weight()), atol=1e-5)
    with pytest.raises(ValueError):
        lora.LoRALinear(linear, rank=0, alpha=1)

    q = quant.QuantizedLinear.from_linear(linear, 8)
    over_quantized = lora.LoRALinear(q, 4, 8)
    over_quantized.lora_b.data.normal_()
    assert torch.allclose(over_quantized(x), F.linear(x, over_quantized.merged_weight()), atol=1e-5)


def tiny_model(experts=4):
    return QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(vocab_size=32, hidden_size=16, layers=2, ffn_size=24, num_experts=experts, top_k=2,
                                                      attention_heads=2, max_sequence_length=16, capacity_factor=0))


def test_add_lora_freezes_the_base_and_reports_the_trainable_share():
    model = tiny_model()
    info = lora.add_lora(model, rank=4, alpha=8)
    assert info["wrapped_layers"] == 2 * 4 * 3
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert info["trainable_parameters"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert info["trainable_parameters"] < 0.7 * info["total_parameters"]
    with pytest.raises(ValueError, match="already"):
        lora.add_lora(model)

    tuned = tiny_model()
    info = lora.add_lora(tuned, 2, 4, include_lm_head=True, train_router=True, train_norms=True)
    names = set(info["extra_trainable"])
    assert "blocks.0.moe.router.weight" in names and "final_norm.weight" in names and "blocks.1.attention_norm.bias" in names
    assert isinstance(tuned.lm_head, lora.LoRALinear)
    assert all(not p.requires_grad for n, p in tuned.named_parameters() if "token_embedding" in n)


def args_for(base, new, output, **overrides):
    values = ["--checkpoint", str(base / "base"), "--data", str(new), "--output", str(output), "--steps", "40",
              "--batch-size", "8", "--sequence-length", "32", "--examples", "60", "--device", "cpu", "--lr", "3e-3", "--bits", "4",
              "--rank", "8"]
    for key, value in overrides.items():
        values += [f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)])
    return ft.build_parser().parse_args(values)


def test_qlora_adapts_a_quantized_model_without_touching_its_base(base, tmp_path):
    root, new = base
    metadata = ft.run_finetune(args_for(root, new, tmp_path / "adapter", merge_output=tmp_path / "merged", train_router=True))
    assert metadata["loss_after"] < metadata["loss_before"] - 0.3
    assert metadata["trainable_fraction"] < 0.6 and metadata["quantized_model_bytes"] < metadata["fp32_parameter_bytes"]
    assert metadata["quantization"]["bits"] == 4 and metadata["quantization"]["compression"] > 3
    assert (tmp_path / "adapter" / "adapter.pt").exists() and (tmp_path / "adapter" / "vocab.json").exists()

    # reloading reproduces the fine-tuned model, and the frozen quantized base is bit-identical to a fresh quantization
    model = ft.load_finetuned(root / "base", tmp_path / "adapter")
    fresh = ft.load_base(root / "base", torch.device("cpu"))
    quant.quantize_model(fresh, 4)
    for key, tensor in fresh.state_dict().items():
        if key.endswith(("qweight", "scales")):
            assert torch.equal(tensor, model.state_dict()[key.replace(".gate.", ".gate.base.").replace(".up.", ".up.base.").replace(".down.", ".down.base.")]), key
    tokenizer = ft.load_tokenizer(root / "base")
    dataset = ft.WindowDataset(ft.build_corpus([new], tokenizer, 60), 32)
    assert ft.evaluate(model, dataset, torch.device("cpu"), "fp32") == pytest.approx(metadata["loss_after"], rel=1e-4)

    # merged full-precision checkpoint ~ the adapted quantized model
    merged_config = QuantaWeaveConfig(**json.loads((tmp_path / "merged" / "config.json").read_text()))
    merged = QuantaWeaveMoEForCausalLM(merged_config)
    merged.load_state_dict(torch.load(tmp_path / "merged" / "model.pt", weights_only=False)["model"])
    ids = torch.stack([dataset[i] for i in range(4)])
    with torch.inference_mode():
        similarity = F.cosine_similarity(merged.eval()(ids)["logits"].flatten(), model(ids)["logits"].flatten(), dim=0)
    assert similarity > 0.999


def test_adapter_only_updates_and_fp32_lora_path(base, tmp_path):
    root, new = base
    metadata = ft.run_finetune(args_for(root, new, tmp_path / "fp", bits=0, steps=30))
    assert metadata["bits"] == 0 and metadata["quantization"] is None and metadata["loss_after"] < metadata["loss_before"]
    model = ft.load_finetuned(root / "base", tmp_path / "fp")
    assert not any(isinstance(m, quant.QuantizedLinear) for m in model.modules())
    with pytest.raises(ValueError, match="does not match"):
        wrong = tiny_model()
        lora.add_lora(wrong, 2, 4)
        lora.load_adapter(wrong, tmp_path / "fp" / "adapter.pt")


def test_validation_and_cli(base, tmp_path, monkeypatch, capsys):
    root, new = base
    with pytest.raises(ValueError, match="context"):
        ft.run_finetune(args_for(root, new, tmp_path / "x", sequence_length=200))
    monkeypatch.setattr(sys, "argv", ["ft", *[str(a) for a in ["--checkpoint", root / "base", "--data", new, "--output", tmp_path / "cli", "--steps", "3",
                                                            "--batch-size", "4", "--sequence-length", "32", "--device", "cpu", "--bits", "8"]]])
    ft.main()
    assert '"loss_after"' in capsys.readouterr().out


def test_8bit_optimizer_is_selected_and_reduces_moment_precision(base, tmp_path):
    bnb = pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            ft.build_optimizer("adamw8bit", [torch.nn.Parameter(torch.zeros(4))], 1e-3)
        pytest.skip("8-bit optimizers need CUDA")
    parameter = torch.nn.Parameter(torch.randn(128, 64, device="cuda"))           # > 4096 elements: bnb quantizes its state
    optimizer = ft.build_optimizer("adamw8bit", [parameter], 1e-3)
    parameter.sum().backward()
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["state1"].dtype == torch.uint8 and state["state2"].dtype == torch.uint8
    full = torch.optim.AdamW([torch.nn.Parameter(torch.randn(128, 64, device="cuda"))])
    with pytest.raises(ValueError):
        ft.build_optimizer("sgd", [parameter], 1e-3)

    root, new = base
    metadata = ft.run_finetune(args_for(root, new, tmp_path / "gpu8", device="cuda", optimizer="adamw8bit", steps=10, rank=64))
    assert torch.isfinite(torch.tensor(metadata["loss_after"])) and metadata["optimizer"] == "adamw8bit"
