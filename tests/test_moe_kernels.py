import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import moe_kernels as mk
from quantweave_moe_model import QuantaWeaveConfig, QuantaWeaveMoEForCausalLM, TopKMoE

cuda_only = pytest.mark.skipif(not mk.HAS_TRITON or not torch.cuda.is_available(), reason="needs CUDA and Triton")


def reference_matmul(x, weight, counts):
    outputs, start = [], 0
    for e, count in enumerate(counts):
        outputs.append(x[start:start + count] @ weight[e].t())
        start += count
    return torch.cat(outputs)


COUNTS = [0, 5, 37, 1, 0, 64, 33, 2]                    # empty experts, partial tiles, exact multiples


@cuda_only
@pytest.mark.parametrize("in_features,out_features", [(48, 80), (64, 64), (33, 17)])       # includes non-multiples of the block sizes
def test_grouped_matmul_matches_per_expert_matmuls_forward_and_backward(in_features, out_features):
    torch.manual_seed(0)
    x = torch.randn(sum(COUNTS), in_features, device="cuda", requires_grad=True)
    weight = torch.randn(len(COUNTS), out_features, in_features, device="cuda", requires_grad=True)
    plan = mk.GroupedPlan(COUNTS, x.device)
    y = mk.grouped_matmul(x, weight, plan)
    expected = reference_matmul(x, weight, COUNTS)
    assert torch.allclose(y, expected, atol=1e-4, rtol=1e-4)

    upstream = torch.randn_like(y)
    y.backward(upstream)
    grads = (x.grad.clone(), weight.grad.clone())
    x.grad = weight.grad = None
    expected.backward(upstream)
    assert torch.allclose(grads[0], x.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(grads[1], weight.grad, atol=1e-3, rtol=1e-4)
    for e, count in enumerate(COUNTS):
        if count == 0:
            assert grads[1][e].abs().sum() == 0            # experts with no rows get an exactly-zero gradient


@cuda_only
def test_grouped_matmul_bf16_is_close_and_all_empty_plan_is_safe():
    torch.manual_seed(1)
    x = torch.randn(sum(COUNTS), 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(len(COUNTS), 96, 64, device="cuda", dtype=torch.bfloat16)
    y = mk.grouped_matmul(x, weight, mk.GroupedPlan(COUNTS, x.device))
    expected = reference_matmul(x.float(), weight.float(), COUNTS)
    assert (y.float() - expected).abs().max() < 0.25 and y.dtype == torch.bfloat16
    empty = mk.grouped_matmul(torch.empty(0, 64, device="cuda"), weight.float(), mk.GroupedPlan([0] * len(COUNTS), x.device))
    assert empty.shape == (0, 96)


@cuda_only
def test_grouped_swiglu_matches_the_loop_over_real_experts():
    torch.manual_seed(2)
    config = QuantaWeaveConfig(vocab_size=8, hidden_size=48, layers=1, ffn_size=72, num_experts=len(COUNTS), top_k=1, attention_heads=1, max_sequence_length=8)
    moe = TopKMoE(config).cuda()
    x = torch.randn(sum(COUNTS), 48, device="cuda", requires_grad=True)
    out = mk.grouped_swiglu(x, moe.experts, COUNTS)
    out.square().sum().backward()
    grouped = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    x_grad = x.grad.clone()
    moe.zero_grad(); x.grad = None
    chunks, start = [], 0
    for e, count in enumerate(COUNTS):
        if count:
            chunks.append(moe.experts[e](x[start:start + count]))
        start += count
    expected = torch.cat(chunks)
    expected.square().sum().backward()
    assert torch.allclose(out, expected, atol=1e-4, rtol=1e-4) and torch.allclose(x_grad, x.grad, atol=1e-3, rtol=1e-3)
    for name, grad in grouped.items():
        reference = dict(moe.named_parameters())[name].grad
        if reference is not None:
            assert torch.allclose(grad, reference, atol=1e-3, rtol=1e-3), name


def make_moe(capacity_factor=0.0):
    config = QuantaWeaveConfig(vocab_size=8, hidden_size=64, layers=1, ffn_size=96, num_experts=16, top_k=2, attention_heads=1,
                               max_sequence_length=512, capacity_factor=capacity_factor, min_expert_capacity=1)
    return TopKMoE(config)


@cuda_only
@pytest.mark.parametrize("capacity_factor", [0.0, 0.7])
def test_moe_layer_gives_the_same_output_and_gradients_with_kernels_on_or_off(capacity_factor):
    torch.manual_seed(3)
    moe = make_moe(capacity_factor).cuda()
    x = torch.randn(4, 96, 64, device="cuda", requires_grad=True)
    results = []
    for use in (False, True):
        moe.use_triton_kernels = use
        moe.zero_grad(); x.grad = None
        out, balance, dropped, overflow, _ = moe(x)
        (out.square().sum() + balance).backward()
        results.append((out.detach(), overflow.item(), x.grad.clone(), {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}))
    (out_a, over_a, gx_a, gp_a), (out_b, over_b, gx_b, gp_b) = results
    assert over_a == over_b and torch.allclose(out_a, out_b, atol=1e-4, rtol=1e-4) and torch.allclose(gx_a, gx_b, atol=1e-3, rtol=1e-3)
    for name, grad in gp_a.items():
        assert torch.allclose(grad, gp_b[name], atol=1e-3, rtol=1e-3), name
    if capacity_factor:
        assert over_a > 0                                                     # the run exercised dropped routes


@cuda_only
def test_full_model_trains_under_bf16_autocast_with_kernels():
    torch.manual_seed(4)
    config = QuantaWeaveConfig(vocab_size=64, hidden_size=64, layers=2, ffn_size=96, num_experts=16, top_k=2, attention_heads=2, max_sequence_length=64, capacity_factor=0)
    model = QuantaWeaveMoEForCausalLM(config).cuda()
    assert model.set_moe_kernel("triton") == "triton"
    ids = torch.randint(0, 64, (4, 48), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_kernel = model(ids, labels=ids)["loss"]
    loss_kernel.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    model.set_moe_kernel("loop")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_loop = model(ids, labels=ids)["loss"]
    assert abs(loss_kernel.item() - loss_loop.item()) < 0.05


@cuda_only
def test_kernel_benchmark_reports_both_paths():
    result = mk.benchmark(experts=16, tokens=2048, hidden=64, ffn=128, repeats=3)
    assert result["loop_ms"] > 0 and result["triton_ms"] > 0 and result["speedup"] > 0


def test_kernel_selection_without_cuda_and_for_unsupported_experts():
    model = QuantaWeaveMoEForCausalLM(QuantaWeaveConfig(vocab_size=8, hidden_size=8, layers=1, ffn_size=8, num_experts=2, top_k=1, attention_heads=1, max_sequence_length=8))
    with pytest.raises(ValueError):
        model.set_moe_kernel("bogus")
    assert model.set_moe_kernel("auto") == "loop" and model.set_moe_kernel("loop") == "loop"          # CPU model: falls back
    with pytest.raises(RuntimeError, match="CUDA"):
        model.set_moe_kernel("triton")
    from quantization import quantize_model
    quantize_model(model, 8)
    assert not mk.experts_support_grouped(model.blocks[0].moe.experts)
    assert mk.experts_support_grouped(TopKMoE(model.config).experts)
    assert mk.GroupedPlan([3, 0, 40], torch.device("cpu")).tiles == 1 + 2 and mk.GroupedPlan([3, 0, 40], torch.device("cpu")).rows == 43
