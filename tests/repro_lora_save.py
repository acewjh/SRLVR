"""Offline reproduction of the LoRA checkpoint-save failure (1 NPU, minutes).

Builds the real Qwen3-1.7B Megatron model, applies the harness LoRA, freezes the
base, constructs the same DistributedOptimizer the actor uses, and calls
save_checkpoint. The forward is never run, so the known standalone-forward
breakage is irrelevant here — save is what we are testing.

Usage: torchrun --nproc_per_node=1 tests/repro_lora_save.py
"""

import os
import sys

PROJ = "/workspace/cua/repos/opsd-slime"
BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
SAVE_DIR = f"{PROJ}/runs/_lora_save_repro"

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
from megatron.training.checkpointing import save_checkpoint  # noqa: E402
from megatron.training.training import (  # noqa: E402
    get_megatron_optimizer_config,
    get_model,
)
from slime.backends.megatron_utils.arguments import set_default_megatron_args  # noqa: E402
from slime.backends.megatron_utils.initialize import init  # noqa: E402
from slime.backends.megatron_utils.model_provider import get_model_provider_func  # noqa: E402
from methods.lora_megatron import apply_lora, custom_model_provider  # noqa: E402

sys.argv = [
    "repro_save",
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

# replicate the actor's provider flow: default provider + LoRA + freeze
sentinel = getattr(args, "custom_model_provider_path", None)
args.custom_model_provider_path = "methods.lora_megatron.custom_model_provider"
model = get_model(custom_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
args.custom_model_provider_path = sentinel

optimizer_config, overrides = get_megatron_optimizer_config(args)
optimizer = get_megatron_optimizer(optimizer_config, model, config_overrides=overrides)

n_opt = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
print(f"[repro] optimizer params: {n_opt/1e6:.2f}M over {len(optimizer.param_groups)} groups", flush=True)

# which params does the optimizer know?
opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
model_params = {id(p) for _, p in model[0].named_parameters()}
print(f"[repro] opt_params subset of model params: {opt_params <= model_params}", flush=True)
only_in_opt = [n for n, p in model[0].named_parameters() if id(p) in opt_params]
print(f"[repro] trainable (first 6): {only_in_opt[:6]}", flush=True)

print("[repro] finding duplicate sharded keys ...", flush=True)
import collections
sd = model[0].sharded_state_dict()
def _keys(x, out):
    if isinstance(x, dict):
        for v in x.values(): _keys(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x: _keys(v, out)
    else:
        out.append(str(getattr(x, "key", None)))
ks = []; _keys(sd, ks)
dups = {k: c for k, c in collections.Counter(ks).items() if c > 1}
print(f"[repro] total keys={len(ks)} duplicates={len(dups)}", flush=True)
for k in list(dups)[:8]:
    print(f"[repro] DUP: {k}", flush=True)

print("[repro] calling save_checkpoint ...", flush=True)
save_checkpoint(
    24,
    model,
    optimizer,
    None,
    num_floating_point_operations_so_far=0,
)
print("[repro] SAVE OK", flush=True)
