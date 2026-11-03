"""End-to-end A0 gate on real NPUs: does a resume really restore Adam?

The CPU gates in `tests/test_optim_sidecar.py` use a fake optimizer. This script
uses the *real* one -- the same Qwen3-1.7B Megatron model, the same harness LoRA,
the same `DistributedOptimizer` the actor builds -- and answers the only question
that matters for A1:

    a fresh optimizer + sidecar takes the SAME next step as the optimizer that
    wrote it.

A forward pass is not needed (and is separately broken standalone): the inner
torch optimizer is stepped with deterministic synthetic gradients, which is the
same object and the same shard layout the training loop updates.

Run: torchrun --nproc_per_node=2 tests/verify_optim_sidecar_npu.py
(2 ranks on purpose: the optimizer state is DP-sharded, so 1 rank would not
exercise the per-rank sidecar files.)
"""

import os
import pathlib
import shutil
import sys

PROJ = "/workspace/cua/repos/opsd-slime"
BASE = "/workspace/cua"
MODEL = f"{BASE}/checkpoints/Qwen3-1.7B"
SCRATCH = f"{PROJ}/runs/_optim_sidecar_verify"

sys.path.insert(0, f"{PROJ}/frameworks/slime-ascend")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-LM")
sys.path.insert(0, f"{PROJ}/frameworks/Megatron-Bridge/src")
sys.path.insert(0, PROJ)

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

os.environ["OPSD_LORA"] = "1"
os.environ["OPSD_OPTIM_SIDECAR"] = "1"

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
from methods import optim_sidecar as sc  # noqa: E402
from methods.lora_megatron import custom_model_provider  # noqa: E402

ITER = 24

sys.argv = [
    "verify_optim_sidecar",
    "--swiglu", "--num-layers", "2", "--hidden-size", "2048",
    "--ffn-hidden-size", "6144", "--num-attention-heads", "16",
    "--group-query-attention", "--num-query-groups", "8",
    "--use-rotary-position-embeddings", "--disable-bias-linear",
    "--normalization", "RMSNorm", "--norm-epsilon", "1e-6",
    "--rotary-base", "1000000", "--vocab-size", "151936",
    "--padded-vocab-size", "151936", "--kv-channels", "128", "--qk-layernorm",
    "--transformer-impl", "local", "--attention-backend", "flash", "--use-flash-attn",
    "--attention-softmax-in-fp32", "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
    "--no-gradient-accumulation-fusion",
    "--micro-batch-size", "1", "--global-batch-size", "4",
    "--lr", "5e-6", "--lr-decay-style", "constant", "--weight-decay", "0.0",
    "--use-distributed-optimizer",
    "--seq-length", "512", "--max-position-embeddings", "32768",
    "--save", SCRATCH, "--save-interval", "1000000", "--train-iters", "1",
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
rank = torch.distributed.get_rank()
torch.npu.set_device(rank % torch.npu.device_count())
init(args)

if rank == 0:
    shutil.rmtree(SCRATCH, ignore_errors=True)

model = get_model(custom_model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=True)
optimizer_config, overrides = get_megatron_optimizer_config(args)
opt1 = get_megatron_optimizer(optimizer_config, model, config_overrides=overrides)
n_params = sum(len(g["params"]) for g in opt1.param_groups)
print(f"[verify] rank={rank} optimizer params={n_params}", flush=True)


def inject(opt, seed_base: int) -> None:
    """Deterministic synthetic gradients on the optimizer's own parameters."""
    for gi, g in enumerate(opt.param_groups):
        for pi, p in enumerate(g["params"]):
            torch.manual_seed(seed_base + gi * 1000 + pi)
            p.grad = torch.randn_like(p) * 1e-3


def inner_step(opt) -> None:
    opt.optimizer.step()


# --- 1. put real, non-trivial state into opt1 ------------------------------
for step in range(2):
    inject(opt1, 7)
    inner_step(opt1)
inner_state = opt1.optimizer.state
print(f"[verify] rank={rank} opt1 state entries={len(inner_state)}", flush=True)
assert len(inner_state) == n_params, f"expected {n_params} state entries, got {len(inner_state)}"

flat1 = [p for g in opt1.param_groups for p in g["params"]]
print(f"[verify] rank={rank} opt1 state keys={sorted(inner_state[flat1[0]].keys())}", flush=True)
weights_at_save = [p.detach().clone() for p in flat1]
moments_at_save = [
    (inner_state[p]["exp_avg"].detach().clone(), inner_state[p]["exp_avg_sq"].detach().clone())
    for g in opt1.param_groups
    for p in g["params"]
]

# --- 2. write the sidecar the way the save hook does -----------------------
sc.dump(opt1, SCRATCH, ITER)
tracker = pathlib.Path(SCRATCH) / sc.TRACKER_NAME
tracker.parent.mkdir(parents=True, exist_ok=True)
tracker.write_text(str(ITER))
path = sc.sidecar_path(SCRATCH, ITER, rank=rank)
assert path.is_file(), f"no sidecar written at {path}"
print(f"[verify] rank={rank} sidecar={path.name} bytes={path.stat().st_size}", flush=True)

# --- 3. a FRESH optimizer, exactly as a resume builds it -------------------
opt2 = get_megatron_optimizer(optimizer_config, model, config_overrides=overrides)
assert len(opt2.optimizer.state) == 0, "fresh optimizer should start with no Adam state"
restored = sc.restore_from_dir(opt2, SCRATCH)
print(f"[verify] rank={rank} restored={restored}/{n_params}", flush=True)
assert restored == n_params, f"restored {restored}, expected {n_params}"

# every restored tensor finite, and identical to what was saved
flat2 = [p for g in opt2.param_groups for p in g["params"]]
for idx, p in enumerate(flat2):
    st = opt2.optimizer.state[p]
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.isfinite(st[key]).all(), f"rank={rank} non-finite {key} at {idx}"
    ea, eas = moments_at_save[idx]
    assert torch.equal(st["exp_avg"].cpu(), ea.cpu()), f"rank={rank} exp_avg mismatch at {idx}"
    assert torch.equal(st["exp_avg_sq"].cpu(), eas.cpu()), f"rank={rank} exp_avg_sq mismatch at {idx}"
    assert sc._step_to_state(st.get("step", 0))[1] == sc._step_to_state(
        inner_state[flat1[idx]].get("step", 0)
    )[1], f"rank={rank} step mismatch at {idx}"
print(f"[verify] rank={rank} all restored moments finite and bit-identical", flush=True)

# --- 4. the actual claim: same next step -----------------------------------
for p, w in zip(flat2, weights_at_save):
    p.data.copy_(w)                      # both optimizers start from the same weights
for p, w in zip(flat1, weights_at_save):
    p.data.copy_(w)

inject(opt1, 11)
inject(opt2, 11)
inner_step(opt1)
inner_step(opt2)

worst = 0.0
for p, q in zip(flat1, flat2):
    worst = max(worst, float((p - q).abs().max().item()))
print(f"[verify] rank={rank} max|continuous - resumed| after the next step = {worst:.3e}", flush=True)
assert worst == 0.0, f"rank={rank} resumed step diverged by {worst}"

print(f"[verify] rank={rank} PASS", flush=True)
