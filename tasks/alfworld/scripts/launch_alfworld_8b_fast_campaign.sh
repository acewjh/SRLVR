#!/usr/bin/env bash
# Two fresh-base rollouts test sampling, training, and a full-state checkpoint.
# Only after their quality/speed gate passes do we start the independent 200 run.
set -euo pipefail
ARM="${1:?usage: launch_alfworld_8b_fast_campaign.sh <grpo|gspo> <stamp>}"
STAMP="${2:?run stamp required}"
PROJ=/workspace/cua/repos/opsd-slime
. "$PROJ/tasks/alfworld/scripts/_8b_fast_rollout.sh"
ROOT="$PROJ/runs/alfworld-qwen3-8b-$ARM"
GROUP="s42x2-a${ACTOR_GPUS}r${ROLLOUT_GPUS}-${ALFWORLD_THINKING_STYLE}${ALFWORLD_THINKING_TOKENS:-}-k${ALFWORLD_V2_HSI_CANDIDATE_K}-cache${SGLANG_PREFIX_CACHE}"
PREFLIGHT="$ROOT/$GROUP/${STAMP}-preflight"
cd "$PROJ"
TOTAL_ROLLOUTS=2 SEGMENT_ROLLOUTS=2 SKIP_FINAL_EVAL=1 MAX_SEGMENT_RETRIES=1 \
  bash tasks/alfworld/scripts/run_alfworld_8b_fast.sh "$ARM" "${STAMP}-preflight"
PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}" \
  /opt/conda/envs/cua-rl/bin/python tasks/alfworld/scripts/check_fast_rollout.py \
  "$PREFLIGHT/train.log" "$ARM" | tee "$PREFLIGHT/quality_speed_gate.json"
printf '[fast-campaign] preflight passed; starting independent 200-rollout run from Qwen3-8B\n'
unset SKIP_FINAL_EVAL RESUME_FROM RESUME_WEIGHTS_ONLY
export TOTAL_ROLLOUTS=200 SEGMENT_ROLLOUTS=50
exec bash tasks/alfworld/scripts/run_alfworld_8b_fast.sh "$ARM" "$STAMP"
