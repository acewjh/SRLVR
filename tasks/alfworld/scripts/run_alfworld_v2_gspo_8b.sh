#!/bin/bash
# Native GSPO on Qwen3-8B, with the same ALFWorld protocol as the 8B GRPO arm.
# Use GSPO's native narrow sequence-ratio window and conservative learning rate.
# _common_v2.sh routes ARM=gspo to slime's sequence-level surrogate.
. "$(dirname "$0")/_8b_backbone.sh"
. "$(dirname "$0")/_8b_fast_rollout.sh"
export ARM=gspo RUN_NAME="${RUN_NAME:-alfworld_v2_gspo_8b}"
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
export EPS_CLIP="${EPS_CLIP:-0.0003}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.0004}"
export LR="${LR:-1e-6}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"
exec bash "$(dirname "$0")/_common_v2.sh"
