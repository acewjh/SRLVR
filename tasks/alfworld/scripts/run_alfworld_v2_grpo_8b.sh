#!/bin/bash
# Vanilla GRPO on Qwen3-8B, per-turn ALFWorld v2 layout.
# Only the backbone differs from run_alfworld_v2_grpo_4b.sh -- geometry via
# _8b_backbone.sh, everything about the method frozen in _common_v2.sh.
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_grpo_8b.sh
. "$(dirname "$0")/_8b_backbone.sh"
. "$(dirname "$0")/_8b_fast_rollout.sh"
export ARM=grpo RUN_NAME="${RUN_NAME:-alfworld_v2_grpo_8b}"
# 8B campaign protocol: 200 optimizer rollouts, checkpoints every 50.  The
# bshd training path keeps its validated fixed micro-batch layout. Clip-higher
# is the small DAPO optimization we
# retain for this long run; override EPS_CLIP_HIGH=0.2 for the symmetric native
# PPO clip if a strict control is desired.
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
# Use all eight cards, with the sampling-heavy 2:6 split by default.
export PP="${PP:-1}"
export ALFWORLD_V2_PROTOCOL="${ALFWORLD_V2_PROTOCOL:-1}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export GBS_OVERRIDE="${GBS_OVERRIDE:-32}"
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"
exec bash "$(dirname "$0")/_common_v2.sh"
