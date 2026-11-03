#!/usr/bin/env bash
# One independently identifiable run under a parameter group. The checkpoint
# directory contains only checkpoints; rollout dumps and eval are siblings.
set -euo pipefail
ARM="${1:?usage: run_alfworld_8b_fast.sh <grpo|gspo> [run-stamp]}"
case "$ARM" in grpo|gspo) ;; *) exit 2 ;; esac
PROJ=/workspace/cua/repos/opsd-slime
. "$PROJ/tasks/alfworld/scripts/_8b_fast_rollout.sh"
export MAX_TURNS="${MAX_TURNS:-50}" HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}" SEQ_LENGTH="${SEQ_LENGTH:-32768}"
if [ "$ARM" = "gspo" ]; then
  export EPS_CLIP="${EPS_CLIP:-0.0003}" EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.0004}"
  export LR="${LR:-1e-6}"
else
  export EPS_CLIP="${EPS_CLIP:-0.2}" EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
  export LR="${LR:-1e-6}"
fi
export SAVE_TRAIN_STATE=1
STAMP="${2:-$(date '+%Y%m%d_%H%M%S')}"
GROUP="s42x${TOTAL_ROLLOUTS:-200}-a${ACTOR_GPUS}r${ROLLOUT_GPUS}-${ALFWORLD_THINKING_STYLE}-cap${MAX_RESPONSE}-kl${KL_COEF}-frontier-k${ALFWORLD_V2_HSI_CANDIDATE_K}-cache${SGLANG_PREFIX_CACHE}"
if [ "$ARM" = "gspo" ]; then
  GROUP="${GROUP}-clip${EPS_CLIP}_${EPS_CLIP_HIGH}-lr${LR}"
fi
export RUN_NAME="alfworld-qwen3-8b-${ARM}-${GROUP}-${STAMP}"
ROOT="$PROJ/runs/alfworld-qwen3-8b-$ARM/$GROUP/$STAMP"
mkdir -p "$ROOT"
export RUN_DIR="$ROOT/checkpoints" DEBUG_ROLLOUT_DIR="$ROOT/rollouts"
export LOG="$ROOT/train.log" STATE_LOG="$ROOT/supervisor.log"
export EVAL_SAVE_DIR="$ROOT/eval/cp${TOTAL_ROLLOUTS:-200}"
export EVAL_LOG="$ROOT/eval.log"
if [ ! -f "$ROOT/config.env" ]; then
  for key in ACTOR_GPUS ROLLOUT_GPUS SGLANG_PREFIX_CACHE ALFWORLD_V2_HSI_CANDIDATE_K \
    ALFWORLD_THINKING_STYLE KL_COEF EPS_CLIP EPS_CLIP_HIGH LR \
    MAX_TURNS HISTORY_LENGTH MAX_RESPONSE SEQ_LENGTH; do
    printf 'export %s=%q\n' "$key" "${!key}"
  done > "$ROOT/config.env"
  printf 'export SAVE_TRAIN_STATE=1\n' >> "$ROOT/config.env"
fi
printf '[fast-run] root=%s host=%s\n' "$ROOT" "$(hostname)"
exec bash "$PROJ/tasks/alfworld/scripts/supervise_alfworld_8b_segments.sh" "$RUN_NAME" "$ARM"
