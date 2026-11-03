#!/bin/bash
# ENV_READY hard gate for the opsd-slime harness (15 checks).
# Usage: bash scripts/check_env.sh [NPU_RANGE]
#   NPU_RANGE defaults to 0,1,2,3,4,5,6,7 (check 6 uses the first 2 NPUs).
# Writes reports/environment.txt, reports/pip_freeze.txt, reports/env_check.log
# and prints ENV_READY=PASS only when every check passes.

set -u

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
NPU_RANGE="${1:-0,1,2,3,4,5,6,7}"
# cua-rl = the slime env on the current machines; slime-npu = legacy naming
CONDA_ENV="${CONDA_ENV:-cua-rl}"
conda activate "$CONDA_ENV" 2>/dev/null || conda activate slime-npu
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-8.5.1/set_env.sh}"
NODE_IFNAME="${NODE_IFNAME:-eth0}"

source /opt/conda/etc/profile.d/conda.sh
source "$CANN_ENV"

export ASCEND_RT_VISIBLE_DEVICES="$NPU_RANGE"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_SOCKET_IFNAME="$NODE_IFNAME"
export GLOO_SOCKET_IFNAME="$NODE_IFNAME"
export TP_SOCKET_IFNAME="$NODE_IFNAME"
# PYTHONPATH must point at THIS project's frameworks (never another project's).
# compat must be present so compat/sitecustomize.py auto-loads (it shadows
# foreign PEP660 editable finders and backports torch-2.1 APIs).
export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ/frameworks/Megatron-Bridge/src:$PROJ/frameworks/mbridge:$PROJ/frameworks/MindSpeed:$PROJ/frameworks/sglang/python:$PROJ/frameworks/slime-ascend:$PROJ:$PYTHONPATH"

mkdir -p "$PROJ/reports" "$PROJ/logs"
LOG=$PROJ/reports/env_check.log
: > "$LOG"

pass=0
fail=0
check() {  # check <id> <description> <command...>
  local id="$1"; shift
  local desc="$1"; shift
  echo "=== [$id] $desc" | tee -a "$LOG"
  if "$@" >> "$LOG" 2>&1; then
    echo "PASS [$id] $desc"
    pass=$((pass+1))
  else
    echo "FAIL [$id] $desc  (see $LOG)"
    fail=$((fail+1))
  fi
}

py() { python -c "$1"; }

check 1 "torch import + version"          py 'import torch; print("torch", torch.__version__)'
check 2 "torch_npu import + version"      py 'import torch, torch_npu; print("torch_npu", torch_npu.__version__)'
check 3 "8 NPUs visible"                  py 'import torch, torch_npu; n=torch.npu.device_count(); print("device_count:", n); assert n >= 8, n'
check 4 "single-NPU fp32 matmul"          py '
import torch, torch_npu
a = torch.randn(512, 512, device="npu:0"); b = torch.randn(512, 512, device="npu:0")
c = a @ b
assert torch.isfinite(c).all().item(); print("matmul ok, sum:", float(c.sum()))'
check 5 "single-NPU bf16 matmul"          py '
import torch, torch_npu
a = torch.randn(512, 512, device="npu:0", dtype=torch.bfloat16); b = torch.randn(512, 512, device="npu:0", dtype=torch.bfloat16)
c = a @ b
assert torch.isfinite(c.float()).all().item(); print("bf16 matmul ok")'
check 6 "HCCL 2-card allreduce"           torchrun --nproc_per_node=2 --master_port=29733 "$PROJ/scripts/_hccl_check.py"
check 7 "transformers import"             py 'import transformers; print("transformers", transformers.__version__)'
check 8 "slime import (this project)"     py 'import slime; print("slime:", slime.__file__)'
check 9 "ray import"                      py 'import ray; print("ray", ray.__version__)'
check 10 "megatron.core import"           py 'import megatron.core; print("megatron.core:", megatron.core.__file__)'
check 11 "mbridge import"                 py 'import mbridge; print("mbridge:", mbridge.__file__)'
check 12 "Megatron-Bridge import"         py 'import megatron.bridge; print("megatron.bridge:", megatron.bridge.__file__)'
check 13 "sglang import (this project)"   py 'import sglang; print("sglang:", sglang.__file__)'
check 14 "NPU fused attention (npu_fusion_attention)" py '
import torch, torch_npu
from torch_npu import npu_fusion_attention
b, h, s, d = 1, 2, 128, 64
q = torch.randn(b, h, s, d, device="npu:0", dtype=torch.bfloat16)
k = torch.randn(b, h, s, d, device="npu:0", dtype=torch.bfloat16)
v = torch.randn(b, h, s, d, device="npu:0", dtype=torch.bfloat16)
ret = npu_fusion_attention(q, k, v, head_num=h, input_layout="BNSD", scale=1.0 / d**0.5)
out = ret[0] if isinstance(ret, tuple) else ret
assert torch.isfinite(out.float()).all().item(); print("npu_fusion_attention ok", tuple(out.shape))'
check 15 "Qwen3-1.7B tokenizer + config"  py '
from transformers import AutoConfig, AutoTokenizer
p = "/workspace/cua/checkpoints/Qwen3-1.7B"
cfg = AutoConfig.from_pretrained(p); tok = AutoTokenizer.from_pretrained(p)
print("model_type:", cfg.model_type, "vocab:", cfg.vocab_size, "hidden:", cfg.hidden_size)
assert cfg.vocab_size == 151936, cfg.vocab_size
ids = tok.apply_chat_template([{"role":"user","content":"hi"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
print("template head:", ids[:60].replace(chr(10), " "))'

# import-path audit: fail the gate if any framework resolves outside this project
check 16 "framework import paths resolve inside opsd-slime" py '
import sglang, slime, megatron.core, mbridge, megatron.bridge
proj = "/workspace/cua/repos/opsd-slime"
for name, mod in [("sglang", sglang), ("slime", slime), ("megatron.core", megatron.core), ("mbridge", mbridge), ("megatron.bridge", megatron.bridge)]:
    f = mod.__file__
    print(f"{name:14s} {f}")
    assert proj in f, f"{name} resolves outside {proj}: {f}"
import methods.opsd_loss
print("methods.opsd_loss import ok")'

echo ""
echo "passed=$pass failed=$fail"
if [ "$fail" -eq 0 ]; then
  echo "ENV_READY=PASS"
else
  echo "ENV_READY=FAIL"
fi

# ---- reports ----
{
  date
  echo "conda_env: $CONDA_ENV"
  echo "python: $(python --version 2>&1)"
  echo "CANN: $(cat /usr/local/Ascend/cann-8.5.1/latest 2>/dev/null || ls -d /usr/local/Ascend/cann-*)"
  python - <<'PY'
import importlib
for name in ["torch", "torch_npu", "transformers", "slime", "sglang", "ray", "mbridge", "megatron", "megatron.bridge", "datasets", "peft", "deepspeed", "methods"]:
    try:
        m = importlib.import_module(name)
        f = getattr(m, "__file__", "?")
        v = getattr(m, "__version__", "?")
        print(f"{name:16s} version={v!s:16s} path={f}")
    except Exception as e:
        print(f"{name:16s} IMPORT-ERROR: {e}")
PY
} > "$PROJ/reports/environment.txt" 2>&1

pip freeze > "$PROJ/reports/pip_freeze.txt" 2>&1
echo "reports written: environment.txt pip_freeze.txt env_check.log"
