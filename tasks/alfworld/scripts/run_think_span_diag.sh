#!/usr/bin/env bash
# Think-span HPI diagnostic: does e_t carry signal on the <think> span?
#
# Reuses tasks.alfworld.diagnostic (frozen, no training, no re-sampling). The
# question is narrow: the v2 pilot measured e_t ~ 0 on the ACTION span. If the
# teacher's response to the future trace exists but is expressed in <think>,
# e_T_think should be materially larger than e_T_action, and the paired
# |e_T|/|e_S| ratio should be > 1 on the think span (i.e. the teacher reacts to
# THIS future, not merely to "more text here").
#
# Usage: bash run_think_span_diag.sh <dump.pt> <checkpoint> <out.jsonl> [gpu]
set -uo pipefail

DUMP="${1:?usage: run_think_span_diag.sh <dump.pt> <checkpoint> <out.jsonl> [gpu]}"
CKPT="${2:?checkpoint}"
OUT="${3:?out jsonl}"
GPU="${4:-0}"

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$GPU"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export ALFWORLD_OVERLAY_SITE="${ALFWORLD_OVERLAY_SITE:-$BASE/envs/alfworld_overlay/lib/python3.10/site-packages}"
export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ/frameworks/Megatron-Bridge/src:$PROJ/frameworks/mbridge:$PROJ/frameworks/MindSpeed:$PROJ/frameworks/sglang/python:$PROJ/frameworks/slime-ascend:$PROJ:$PYTHONPATH"
export WANDB_MODE=disabled WANDB_DISABLED=true

cd "$PROJ" || exit 1
exec 9>&-  # never hold the orchestrator lock fd

python -m tasks.alfworld.diagnostic \
  --dump "$DUMP" \
  --checkpoint "$CKPT" \
  --out "$OUT" \
  --conditions C,T,S,P
