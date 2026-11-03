"""Offline (CPU) tests for the Megatron LoRA wrapper — directive §10 prep.

These cover everything that does not need the training pipeline:
  1. name preservation (mbridge/convert_to_hf/state_dict compatibility)
  2. section boundaries for the fused QKV / FC1 layers
  3. zero-init delta == 0 (LoRA sync test 1 precondition)
  4. merge -> effective weight == W + (alpha/r) * B A, and EXACT restore
  5. lora_hidden really hides the adapters from named_parameters/state_dict
  6. freeze pattern '.*lora_.*' leaves only adapters trainable

The pipeline-level megatron<->sglang equality gate (LoRA sync tests 1/2 against
a live SGLang engine) is scripts/lora_sync_check_1step.sh — run it once compute
is available; PASS is required before launching the 100-step run.

Run:  python tests/test_lora_offline.py
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/workspace/cua/repos/opsd-slime")

os.environ["OPSD_LORA"] = "1"
os.environ["OPSD_LORA_R"] = "8"  # small for the test
os.environ["OPSD_LORA_ALPHA"] = "16"  # scaling 2.0

from methods.lora_megatron import (  # noqa: E402
    _LoRAFusedLinear,
    _sections_for,
    apply_lora,
    freeze_base_params,
    lora_hidden,
    merged_lora_sync,
)


class _FakeCfg:
    kv_channels = 2
    num_attention_heads = 4  # q = 8
    num_query_groups = 2  # k = v = 4
    ffn_hidden_size = 12


class _FakeLinear(nn.Module):
    def __init__(self, out_features, in_features=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.forward_run = 0
        self.config = None  # set by the parent layer (mirrors ColumnParallelLinear.config)

    def forward(self, x, *args, **kwargs):
        self.forward_run += 1
        return F.linear(x, self.weight, self.bias)


import torch.nn.functional as F  # noqa: E402


class _FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeCfg()
        self.linear_qkv = _FakeLinear(16)  # q 8 | k 4 | v 4
        self.linear_proj = _FakeLinear(8)
        self.linear_qkv.config = self.config
        self.linear_proj.config = self.config


class _FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeCfg()
        self.linear_fc1 = _FakeLinear(24)  # gate 12 | up 12
        self.linear_fc2 = _FakeLinear(16)
        self.linear_fc1.config = self.config
        self.linear_fc2.config = self.config


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeCfg()
        self.self_attention = _FakeAttn()
        self.mlp = _FakeMLP()


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([_FakeLayer() for _ in range(2)])


def test_1_name_and_sections() -> None:
    model = _FakeModel()
    info = apply_lora(model)
    assert info["n_wrapped"] == 8, info  # 2 layers x (qkv, proj, fc1, fc2)
    names = [n for n, _ in model.named_parameters()]
    # base names preserved (no wrapper module names injected)
    assert "decoder.0.self_attention.linear_qkv.weight" in names
    # adapter names carry lora_<hf> and live on the same FQN
    for want in [
        "decoder.0.self_attention.linear_qkv.lora_q_A",
        "decoder.0.self_attention.linear_qkv.lora_k_B",
        "decoder.0.self_attention.linear_qkv.lora_v_A",
        "decoder.0.self_attention.linear_proj.lora_o_B",
        "decoder.1.mlp.linear_fc1.lora_gate_A",
        "decoder.1.mlp.linear_fc1.lora_up_B",
        "decoder.1.mlp.linear_fc2.lora_down_A",
    ]:
        assert want in names, f"missing {want}"
    # section boundaries on the fused qkv
    w = model.decoder[0].self_attention.linear_qkv
    assert [(n, s, e) for n, s, e in w._sections] == [("q", 0, 8), ("k", 8, 12), ("v", 12, 16)]
    fc1 = model.decoder[0].mlp.linear_fc1
    assert [(n, s, e) for n, s, e in fc1._sections] == [("gate", 0, 12), ("up", 12, 24)]
    print("Test 1 (name preservation + sections): PASS")


def test_2_zero_init_and_merge_restore() -> None:
    model = _FakeModel()
    apply_lora(model)
    w = model.decoder[0].self_attention.linear_qkv
    x = torch.randn(3, 16)

    # zero-init B -> delta == 0 -> forward equals the inner layer exactly
    inner_out = w.inner(x)
    wrapper_out = w(x)
    assert torch.allclose(inner_out, wrapper_out, atol=0.0), "zero-init LoRA changed the forward"

    # merge == W + (alpha/r) * B A ; restore must be EXACT (clone-based)
    w0 = w.weight.detach().clone()
    with merged_lora_sync(model):
        # during the sync the wrapper is hidden and the weight carries the delta
        assert model.decoder[0].self_attention.linear_qkv is w.inner
        merged = w.weight.detach().clone()
        delta = torch.cat(
            [w.lora_q_B @ w.lora_q_A, w.lora_k_B @ w.lora_k_A, w.lora_v_B @ w.lora_v_A], dim=0
        ) * w.scaling
        expected = w0 + delta
        assert torch.allclose(merged, expected, atol=1e-6), (merged - expected).abs().max()
        # adapter names are gone while hidden
        hidden_names = [n for n, _ in model.named_parameters()]
        assert not any("lora_" in n for n in hidden_names)
    assert model.decoder[0].self_attention.linear_qkv is w, "wrapper not restored"
    assert torch.equal(w.weight.detach(), w0), "restore is not exact"
    print("Test 2 (zero-init + merge/restore exactness): PASS")


def test_3_perturbation_changes_effective_weight() -> None:
    """LoRA sync test 2 (offline half): a deterministic LoRA perturbation must
    change the merged effective weight by exactly scaling * B A."""
    torch.manual_seed(3)
    model = _FakeModel()
    apply_lora(model)
    w = model.decoder[1].mlp.linear_fc1
    with torch.no_grad():
        w.lora_gate_B.normal_(0.0, 0.05)  # deterministic-enough perturbation
    w0 = w.weight.detach().clone()
    with merged_lora_sync(model):
        merged = w.weight.detach().clone()
    delta_gate = (w.lora_gate_B @ w.lora_gate_A) * w.scaling
    assert torch.allclose(merged[:12] - w0[:12], delta_gate, atol=1e-6)
    assert (merged[:12] - w0[:12]).abs().max() > 0, "perturbation did not reach the effective weight"
    assert torch.equal(w.weight.detach(), w0)
    print("Test 3 (perturbation reaches effective weight, exact restore): PASS")


def test_4_freeze_pattern() -> None:
    model = _FakeModel()
    apply_lora(model)
    trainable = freeze_base_params(model)
    total = sum(p.numel() for p in model.parameters())
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    assert all(not rg for n, rg in flags.items() if "lora_" not in n), "base param left trainable"
    assert all(rg for n, rg in flags.items() if "lora_" in n), "adapter param frozen"
    assert 0 < trainable < total
    print(f"Test 4 (freeze pattern): PASS (trainable {trainable}/{total} = {100*trainable/total:.1f}%)")


if __name__ == "__main__":
    test_1_name_and_sections()
    test_2_zero_init_and_merge_restore()
    test_3_perturbation_changes_effective_weight()
    test_4_freeze_pattern()
    print("test_lora_offline: 4/4 PASS")
