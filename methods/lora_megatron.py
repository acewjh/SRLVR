"""Official-config LoRA for the Megatron student (Qwen3-1.7B, TP1, --transformer-impl local).

Aligns the harness with the official OPSD baseline PEFT settings
(repos/opsd-npu/src/smoke_correctness.py, config_resolved.yaml):
    r=64, lora_alpha=128, lora_dropout=0.0, bias untouched (peft default "none"),
    target_modules = q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.

Megatron-fused layers carry several HF modules each, so the HF-module LoRA is
realized as SECTION-WISE adapters on the fused weights (exactly equivalent to
PEFT's per-module parameterization):
    self_attention.linear_qkv  out=[Q(2048)|K(1024)|V(1024)]  -> q/k/v adapters
    self_attention.linear_proj out=[O(2048)]                  -> o adapter
    mlp.linear_fc1             out=[gate(6144)|up(6144)]      -> gate/up adapters
    mlp.linear_fc2             out=[down(2048)]               -> down adapter

Name preservation: the wrapper registers the inner layer's weight/bias as its
OWN parameters, so every downstream name-based consumer (mbridge load_weights,
megatron_to_hf convert_to_hf, checkpoint state dicts) keeps working unchanged.
Adapter parameters are named `lora_<hf>_A/B` so slime's
--only-train-params-name-list '.*lora_.*' freezes exactly the base model.

Two contexts drive the sync paths:
  lora_hidden(model)      - hide adapters (swap wrappers for their inner
                            layers): HF weight loading sees the pristine model.
  merged_lora_sync(model) - save base weights, add the merged effective delta
                            (W + (alpha/r) B A) into them, hide adapters, run
                            the existing update_weights conversion, then
                            restore exactly (cloned backup, no bf16 drift).
"""

import logging
import os
from contextlib import contextmanager

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods import tp_utils

logger = logging.getLogger(__name__)

_LORA_ATTR = "_opsd_lora_wrapper"
_PROVIDERS = "_opsd_lora_parents"


def lora_enabled() -> bool:
    return os.environ.get("OPSD_LORA", "0") == "1"


def lora_cfg() -> dict:
    return {
        "r": int(os.environ.get("OPSD_LORA_R", 64)),
        "alpha": float(os.environ.get("OPSD_LORA_ALPHA", 128)),
        "dropout": float(os.environ.get("OPSD_LORA_DROPOUT", 0.0)),
    }


def _sections_for(module, module_name: str) -> list[tuple[str, int, int]]:
    """(hf_target_name, out_start, out_end) sections of a fused Megatron layer."""
    cfg = None
    parent = module
    while cfg is None and parent is not None:
        cfg = getattr(parent, "config", None)
        parent = getattr(parent, "parent", None)
    head_dim = cfg.kv_channels if (cfg is not None and cfg.kv_channels) else 128
    nq = cfg.num_attention_heads if cfg is not None else 16
    nkv = cfg.num_query_groups if cfg is not None else 8
    ffn = cfg.ffn_hidden_size if cfg is not None else 6144
    if module_name.endswith("linear_qkv"):
        q = nq * head_dim
        kv = nkv * head_dim
        return [("q", 0, q), ("k", q, q + kv), ("v", q + kv, q + 2 * kv)]
    if module_name.endswith("linear_proj"):
        return [("o", 0, int(module.weight.shape[0]))]
    if module_name.endswith("linear_fc1"):
        return [("gate", 0, ffn), ("up", ffn, 2 * ffn)]
    if module_name.endswith("linear_fc2"):
        return [("down", 0, int(module.weight.shape[0]))]
    return []


class _LoRAFusedLinear(nn.Module):
    """Name-preserving wrapper adding section-wise LoRA to a Megatron linear."""

    def __init__(self, inner: nn.Module, module_name: str, r: int, alpha: float, dropout: float):
        super().__init__()
        assert dropout == 0.0, "official OPSD LoRA uses dropout=0.0; other values not wired"
        self._sections = _sections_for(inner, module_name)
        assert self._sections, f"no LoRA sections derived for {module_name}"
        self._inner_forward = inner.forward
        self.scaling = alpha / r
        # re-register the inner layer's parameters as OURS (name preservation);
        # `_inner` is kept via object.__setattr__ so it is NOT a child module
        object.__setattr__(self, "_inner_ref", inner)
        self.weight = inner.weight
        if getattr(inner, "bias", None) is not None:
            self.bias = inner.bias
        # TP layout.  Adapters are stored at their GLOBAL shape on every TP
        # rank (replicated) so parameter names/shapes/checkpoints and the
        # optimizer are identical across TP ranks.  Only the rank's local
        # output (column-parallel) or input (row-parallel) slice is used in the
        # forward; replicated-adapter gradients are SUM-reduced across TP.
        self._tp_size, self._tp_rank, self._tp_group = tp_utils.tp_info()
        self._column = not module_name.endswith(("linear_proj", "linear_fc2"))
        local_out, local_in = int(inner.weight.shape[0]), int(inner.weight.shape[1])
        if self._column:
            self._out_offset = self._tp_rank * local_out
            self._local_out = local_out
            in_full = local_in
            global_out = sum(e - st for _n, st, e in self._sections)
            if global_out != local_out * self._tp_size:
                raise RuntimeError(
                    f"LoRA TP layout mismatch for {module_name}: global sections {global_out} "
                    f"vs local_out {local_out} x tp {self._tp_size}"
                )
        else:
            self._in_offset = self._tp_rank * local_in
            self._local_in = local_in
            in_full = local_in * self._tp_size
            global_out = int(inner.weight.shape[0])
            if sum(e - st for _n, st, e in self._sections) != global_out:
                raise RuntimeError(f"LoRA row-parallel sections do not cover {module_name}")
        device, dtype = inner.weight.device, inner.weight.dtype
        for name, st, e in self._sections:
            a = torch.empty(r, in_full, device=device, dtype=dtype)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            b = torch.zeros(e - st, r, device=device, dtype=dtype)
            pa = nn.Parameter(a)
            pb = nn.Parameter(b)
            for param in (pa, pb):
                param._srlvr_tp_mode = "replica"
            self.register_parameter(f"lora_{name}_A", pa)
            self.register_parameter(f"lora_{name}_B", pb)
        # sync test 2 helper: deterministic perturbation of ONE adapter
        if os.environ.get("OPSD_LORA_PERTURB") == "1" and module_name.endswith("linear_qkv"):
            with torch.no_grad():
                g = torch.Generator(device="cpu")
                g.manual_seed(42)
                self.lora_q_B.data += torch.randn(
                    self.lora_q_B.shape, generator=g, device=device, dtype=torch.float32
                ).to(self.lora_q_B.dtype) * 0.02

    def forward(self, x, *args, **kwargs):
        out = self._inner_forward(x, *args, **kwargs)
        group = self._tp_group if self._tp_size > 1 else None
        deltas = []
        for name, _s, _e in self._sections:
            # Replicated adapters: local contributions differ per TP rank, so
            # the global parameter gradient is their SUM.
            a = tp_utils.grad_sum(getattr(self, f"lora_{name}_A"), group)
            b = tp_utils.grad_sum(getattr(self, f"lora_{name}_B"), group)
            if self._column:
                z = F.linear(tp_utils.grad_sum(x, group), a)
                deltas.append(F.linear(z, b) * self.scaling)
            else:
                a_local = a[:, self._in_offset:self._in_offset + self._local_in]
                z = F.linear(x, a_local)
                deltas.append(F.linear(z, b) * self.scaling)
        delta = torch.cat(deltas, dim=-1) if len(deltas) > 1 else deltas[0]
        if self._column:
            delta = delta[..., self._out_offset:self._out_offset + self._local_out]
        elif group is not None:
            # Row-parallel: the base layer all-reduces its output, so the LoRA
            # contribution (computed from the local input shard) must be summed
            # over TP as well.
            delta = tp_utils.tp_sum(delta, group, differentiable=True)
        if isinstance(out, tuple):
            return (out[0] + delta.to(out[0].dtype),) + tuple(out[1:])
        return out + delta.to(out.dtype)

    def __getattr__(self, name):
        if name.startswith("_") or name in ("weight", "bias") or name.startswith("lora_"):
            return nn.Module.__getattr__(self, name)
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_inner_ref"), name)

    # the merged-weight sync swaps this wrapper for its inner layer
    @property
    def inner(self):
        return object.__getattribute__(self, "_inner_ref")

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Base weight/bias via the inner layer's own (parallelism-aware) method;
        adapter params via megatron's make_sharded_tensors_for_checkpoint, which
        inherits the CALLER's sharded_offsets (the homogeneous-layer PP offset).
        Without the caller offsets, all 28 layers' adapters collapse onto one
        checkpoint key/offset and DCP's integrity validation flags duplicates.
        NOTE: must NOT delegate to sharded_state_dict_default for the whole
        wrapper — it re-dispatches to this method (hasattr) and recurses."""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        sd = dict(self.inner.sharded_state_dict(prefix, sharded_offsets, metadata))
        adapter_params = {
            name: param
            for name, param in self.named_parameters(recurse=False)
            if "lora_" in name
        }
        sd.update(
            make_sharded_tensors_for_checkpoint(
                adapter_params,
                prefix,
                {},
                sharded_offsets,
                tp_group=self._tp_group,
                dp_cp_group=metadata.get("dp_cp_group") if metadata else None,
            )
        )
        return sd



def _walk_wrappers(module: nn.Module, prefix: str = ""):
    """Yield (parent_module, attr_name, wrapper, full_name) for every LoRA wrapper."""
    direct = [(module, attr) for attr, ch in module.named_children() if isinstance(ch, _LoRAFusedLinear)]
    # a wrapper may also be assigned as a plain attribute (not a child); find via __dict__
    for attr, val in vars(module).items():
        if isinstance(val, _LoRAFusedLinear) and all(attr != a for a, _ in direct):
            direct.append((module, attr))
    for parent, attr in direct:
        w = getattr(parent, attr)
        yield parent, attr, w, f"{prefix}{attr}"
    for name, child in module.named_children():
        # skip children that are wrappers themselves (no nested targets inside)
        if isinstance(child, _LoRAFusedLinear):
            continue
        yield from _walk_wrappers(child, prefix=f"{prefix}{name}.")


def apply_lora(model: nn.Module) -> dict:
    """Wrap every target linear of the (GPTModel) with section-wise LoRA."""
    r, alpha, dropout = lora_cfg()["r"], lora_cfg()["alpha"], lora_cfg()["dropout"]
    n_wrapped = 0
    adapters = []
    for module_name, module in list(model.named_modules()):
        tail = module_name.rsplit(".", 1)[-1] if "." in module_name else module_name
        if tail not in ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"):
            continue
        if getattr(module, _LORA_ATTR, False):
            continue
        sections = _sections_for(module, module_name)
        if not sections:
            continue
        parent_name, _, attr = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = _LoRAFusedLinear(module, module_name, r, alpha, dropout)
        setattr(parent, attr, wrapper)
        setattr(wrapper, _LORA_ATTR, True)
        wrappers = getattr(model, _PROVIDERS, [])
        wrappers.append((parent, attr))
        object.__setattr__(model, _PROVIDERS, wrappers)
        for name, s, _e in sections:
            adapters.append(f"{module_name}.lora_{name}_A")
        n_wrapped += 1
    logger.info("[lora] wrapped %d fused layers (r=%d alpha=%g)", n_wrapped, r, alpha)
    return {"n_wrapped": n_wrapped, "adapter_names": adapters}


def freeze_base_params(model: nn.Module) -> int:
    """Freeze everything except `lora_` parameters (belt-and-braces alongside
    slime's --only-train-params-name-list). Returns the trainable count."""
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "lora_" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
    logger.info(
        "[lora] trainable %.2fM / total %.2fM params (%.3f%%)",
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / max(total, 1),
    )
    return trainable


def _lora_root(m):
    """Dive through DDP/Float16 wrapping to the chunk that holds the LoRA state."""
    cur, depth = m, 0
    while cur is not None and depth < 6:
        if getattr(cur, _PROVIDERS, None):
            return cur
        cur = getattr(cur, "module", None)
        depth += 1
    return None


def _chunks(model):
    """Normalize a model chunk or a Sequence[chunk] into the chunks holding LoRA."""
    seq = model if isinstance(model, (list, tuple)) else [model]
    roots = [_lora_root(m) for m in seq]
    return [r for r in roots if r is not None]


@contextmanager
def lora_hidden(model: nn.Module):
    """Hide wrappers (swap inner layers back in) so name-based consumers see the
    pristine model. Restores the wrappers on exit."""
    holders = [h for chunk in _chunks(model) for h in getattr(chunk, _PROVIDERS, [])]
    refs = [(parent, attr, getattr(parent, attr)) for parent, attr in holders]
    for parent, attr, w in refs:
        setattr(parent, attr, w.inner)
    try:
        yield
    finally:
        for parent, attr, w in refs:
            setattr(parent, attr, w)


@contextmanager
def merged_lora_sync(model: nn.Module):
    """Merge effective LoRA weights into the base parameters, hide the adapters,
    run the enclosing block (update_weights conversion), then restore exactly."""
    if not lora_enabled():
        yield
        return
    holders = [h for chunk in _chunks(model) for h in getattr(chunk, _PROVIDERS, [])]
    refs = [(parent, attr, getattr(parent, attr)) for parent, attr in holders]
    saved = []
    for parent, attr, w in refs:
        saved.append((w, w.weight.detach().clone()))
        with torch.no_grad():
            parts = []
            for name, _s, _e in w._sections:
                full = (getattr(w, f"lora_{name}_B").float()
                        @ getattr(w, f"lora_{name}_A").float()) * w.scaling
                parts.append(full)
            delta = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            if w._column:
                delta = delta[w._out_offset:w._out_offset + w._local_out, :]
            else:
                delta = delta[:, w._in_offset:w._in_offset + w._local_in]
            if delta.shape != w.weight.shape:
                raise RuntimeError(
                    f"merged LoRA delta {tuple(delta.shape)} != base weight {tuple(w.weight.shape)}"
                )
            w.weight.data += delta.to(w.weight.dtype)
    for parent, attr, w in refs:
        setattr(parent, attr, w.inner)
    try:
        yield
    finally:
        for parent, attr, w in refs:
            setattr(parent, attr, w)
        for w, saved_weight in saved:
            with torch.no_grad():
                w.weight.data.copy_(saved_weight)


def custom_model_provider(pre_process: bool = True, post_process: bool = True, vp_stage=None):
    """slime --custom-model-provider-path entry point: build the default model,
    optionally apply LoRA and freeze base weights before DDP/optimizer."""
    import argparse

    from slime.backends.megatron_utils.model_provider import get_model_provider_func

    args = None
    from megatron.training.global_vars import get_args

    args = get_args()
    sentinel = getattr(args, "custom_model_provider_path", None)
    try:
        args.custom_model_provider_path = None
        default_provider = get_model_provider_func(args)
    finally:
        args.custom_model_provider_path = sentinel

    has_vp = "vp_stage" in __import__("inspect").signature(default_provider).parameters
    model = (
        default_provider(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if has_vp
        else default_provider(pre_process=pre_process, post_process=post_process)
    )
    if not lora_enabled():
        logger.info("[lora] OPSD_LORA=0: full-model provider, no adapters or base freezing")
        return model
    info = apply_lora(model)
    # the bridge's AutoMapping machinery needs a parallelism type for the wrapper
    # class if it is ever visible to it (belt-and-braces; the sync/load paths hide
    # the wrappers). TP1 makes column/row mathematically identical.
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        AutoMapping.register_module_type("_LoRAFusedLinear", "column")
    except Exception as _e:
        logger.warning(f"[lora] AutoMapping registration skipped: {_e}")
    trainable = freeze_base_params(model)
    logger.info(
        "[lora] provider done: %d wrapped layers, %.2fM trainable params",
        info["n_wrapped"],
        trainable / 1e6,
    )
    return model
