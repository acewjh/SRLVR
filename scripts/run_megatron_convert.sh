#!/bin/bash
# Phase 4 step 1: HF -> Megatron(torch_dist) checkpoint conversion for Qwen3-1.7B.
# Uses slime's convert_hf_to_torch_dist.py + the vendored qwen3-1.7B model args.
# Output: $PROJ/runs/_init_megatron_qwen3-1p7b/  (HF checkpoint is never modified)

set -euo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
MODEL=$BASE/checkpoints/Qwen3-1.7B
OUT=$PROJ/runs/_init_megatron_qwen3-1p7b
NPU_ID="${NPU_ID:-0}"

source /opt/conda/etc/profile.d/conda.sh
conda activate slime-npu
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ/frameworks/Megatron-Bridge/src:$PROJ/frameworks/sglang/python:$PROJ/frameworks/slime-ascend:$PROJ"

mkdir -p "$OUT" "$PROJ/logs"

source "$PROJ/frameworks/slime-ascend/scripts/models/qwen3-1.7B.sh"

torchrun --nproc_per_node=1 --master_port=29811 \
  "$PROJ/frameworks/slime-ascend/tools/convert_hf_to_torch_dist.py" \
  --hf-checkpoint "$MODEL" \
  --save "$OUT" \
  --padded-vocab-size 151936 \
  "${MODEL_ARGS[@]}" \
  2>&1 | tee "$PROJ/logs/megatron_convert.log"
echo "conversion exit: $?"
ls -la "$OUT"
