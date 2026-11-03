"""Probe: what is the true mapping from DistributedOptimizer params to LoRA names?

The A0 sidecar has to write Adam moments keyed by something that survives a
restart. The launcher comment says the optimizer's param objects are "rebound to
its fp32 main-param buffers", i.e. `id(param)` is NOT a model param, so the
mapping must be established empirically rather than assumed. This script builds
the exact same model + LoRA + DistributedOptimizer as `repro_lora_save.py` and
reports which of the candidate identity routes actually holds:

  R1 `p.model_param` back-pointer (Megatron sets it for the non-distributed path)
  R2 `id(p)` directly a model parameter
  R3 positional zip against `requires_grad` params in `named_parameters()` order
  R4 shard-main-param groups vs param_groups lengths

Usage: torchrun --nproc_per_node=1 tests/probe_optim_names.py
"""

import json
import os
import sys

PROJ = "/workspace/cua/repos/opsd-slime"
BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
SAVE_DIR = f"{PROJ}/runs/probe_optim_names"

sys.path.insert(0, f"{PROJ}/frameworks/slime-ascend")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-LM")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-Bridge/src")
sys.path.insert(0, PROJ)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

os.environ["OPSD_LORA"] = "1"

import mindspeed.megatron_adaptor  # noqa: E402,F401
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.optimizer import get_megatron_optimizer  # noqa: E402
from megatron.training.arguments import parse_args, validate_args  # noqa: E402
from megatron.training.training import (  # noqa: E402
    get_megatron_optimizer_config,
    get_model,
)
from slime.backends.megatron_utils.arguments import set_default_megatron_args  # noqa: E402
from slime.backends.megatron_utils.initialize import init  # noqa: E402
from methods.lora_megatron import custom_model_provider  # noqa: E402

sys.argv = [
    "probe_optim_names",
    "--swiglu", "--num-layers", "2", "--hidden-size", "2048",
    "--ffn-hidden-size", "6144", "--num-attention-heads", "16",
    "--group-query-attention", "--num-query-groups", "8",
    "--use-rotary-position-embeddings", "--disable-bias-linear",
    "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000", "--vocab-size", "151936",
    "--padded-vocab-size", "151936", "--kv-channels", "128", "--qk-layernorm",
    "--transformer-impl", "local",
    "--attention-backend", "flash",
    "--use-flash-attn",
    "--attention-softmax-in-fp32",
    "--attention-dropout", "0.0",
    "--hidden-dropout", "0.0",
    "--no-gradient-accumulation-fusion",
    "--micro-batch-size", "1", "--global-batch-size", "4",
    "--lr", "5e-6", "--lr-decay-style", "constant",
    "--weight-decay", "0.0",
    "--use-distributed-optimizer",
    "--seq-length", "512", "--max-position-embeddings", "32768",
    "--save", SAVE_DIR, "--save-interval", "1000000", "--train-iters", "1",
    "--no-save-optim",
]
args = parse_args()
args.hf_checkpoint = MODEL
args.megatron_to_hf_mode = "raw"
args.qkv_format = "bshd"
args = set_default_megatron_args(args)
validate_args(args)

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="hccl")
torch.npu.set_device(torch.distributed.get_rank() % torch.npu.device_count())
init(args)

model = get_model(custom_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)

# ---- candidate R3: the ordered trainable-name list the optimizer should follow
named = [(n, p) for n, p in model[0].named_parameters()]
trainable_names = [n for n, p in named if p.requires_grad]
name_by_id = {id(p): n for n, p in named}

optimizer_config, overrides = get_megatron_optimizer_config(args)
optimizer = get_megatron_optimizer(optimizer_config, model, config_overrides=overrides)

groups = optimizer.param_groups
opt_params = [p for g in groups for p in g["params"]]

report = {
    "optimizer_class": type(optimizer).__name__,
    "n_param_groups": len(groups),
    "group_sizes": [len(g["params"]) for g in groups],
    "n_opt_params": len(opt_params),
    "n_model_params": len(named),
    "n_trainable_named": len(trainable_names),
    "trainable_names_head": trainable_names[:6],
}
print("[probe] " + json.dumps(report, indent=2), flush=True)

# R1: model_param back-pointer
r1_hits = sum(1 for p in opt_params if getattr(p, "model_param", None) is not None)
r1_named = 0
for p in opt_params:
    mp = getattr(p, "model_param", None)
    if mp is not None and id(mp) in name_by_id:
        r1_named += 1
# R2: identity
r2 = sum(1 for p in opt_params if id(p) in name_by_id)
# R3: positional
r3_names = []
ok_r3 = True
if len(opt_params) == len(trainable_names):
    r3_names = list(trainable_names)
else:
    ok_r3 = False

print(f"[probe] R1 model_param set on {r1_hits}/{len(opt_params)}; "
      f"resolvable to a model name on {r1_named}", flush=True)
print(f"[probe] R2 id() is a model param for {r2}/{len(opt_params)}", flush=True)
print(f"[probe] R3 positional zip usable: {ok_r3} "
      f"(opt={len(opt_params)} vs trainable_named={len(trainable_names)})", flush=True)

# R4: shard groups
for attr in ("model_float16_groups", "shard_fp32_from_float16_groups"):
    v = getattr(optimizer, attr, None)
    if v is None:
        print(f"[probe] R4 {attr}: ABSENT", flush=True)
    else:
        print(f"[probe] R4 {attr}: groups={len(v)} sizes={[len(g) for g in v]}", flush=True)

# Is the inner torch optimizer's state addressable by the same param objects?
inner = getattr(optimizer, "optimizer", None)
print(f"[probe] inner optimizer: {type(inner).__name__}, "
      f"state entries={len(getattr(inner, 'state', {}))}", flush=True)

# R3 is only usable if the multiset of flattened sizes agrees AND the per-index
# order agrees. Print both lists in full so the answer is decided by data.
name_sizes = [p.numel() for _, p in named if p.requires_grad]
opt_sizes = [p.numel() for p in opt_params]
print(f"[probe] multiset sizes equal: {sorted(name_sizes) == sorted(opt_sizes)}", flush=True)
mismatch = [(i, trainable_names[i], name_sizes[i], opt_sizes[i])
            for i in range(min(len(name_sizes), len(opt_sizes)))
            if name_sizes[i] != opt_sizes[i]]
print(f"[probe] per-index size mismatches: {len(mismatch)}/{len(opt_sizes)}", flush=True)
for i, n, a, b in mismatch[:10]:
    print(f"[probe]   idx={i} {n.rsplit('.',1)[-1]} named={a} opt={b}", flush=True)

# Does an optimizer param share storage with some model param (i.e. is it a
# flattened view rather than an fp32 clone)? If yes, identity is recoverable
# exactly instead of positionally.
model_storage = {}
for n, p in named:
    try:
        model_storage.setdefault(p.untyped_storage().data_ptr(), []).append(n)
    except Exception:
        pass
sharing = 0
for p in opt_params:
    try:
        if p.untyped_storage().data_ptr() in model_storage:
            sharing += 1
    except Exception:
        pass
print(f"[probe] optimizer params sharing storage with a model param: "
      f"{sharing}/{len(opt_params)}", flush=True)

# What would the sidecar actually need to store?
if inner is not None and len(inner.state) > 0:
    k = next(iter(inner.state))
    st = inner.state[k]
    print(f"[probe] example state keys={sorted(st.keys())} "
          f"types={ {kk: type(vv).__name__ for kk, vv in st.items()} }", flush=True)
else:
    print("[probe] inner optimizer state is EMPTY before the first step "
          "(expected: Adam state is created lazily)", flush=True)

print("[probe] DONE", flush=True)
