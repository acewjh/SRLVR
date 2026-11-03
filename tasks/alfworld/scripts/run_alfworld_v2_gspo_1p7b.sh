#!/bin/bash
# ALFWorld GSPO on Qwen3-1.7B with a narrow sequence-ratio clipping window.
export ARM=gspo
export ALFWORLD_V2_PROTOCOL=1
export EPS_CLIP="${EPS_CLIP:-0.0003}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.0004}"
export LR="${LR:-1e-6}"
export RUN_NAME="${RUN_NAME:-alfworld_v2_gspo_1p7b}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
exec bash "$(dirname "$0")/_common_v2.sh"
