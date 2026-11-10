#!/usr/bin/env bash
# Supervised 100-rollout Qwen3-8B GRPO run for one 8x32GB 910B4 node.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PP="${PP:-1}"
export TP="${TP:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
export COLOCATE="${COLOCATE:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export TOTAL_ROLLOUTS="${TOTAL_ROLLOUTS:-100}"
export SEGMENT_ROLLOUTS="${SEGMENT_ROLLOUTS:-50}"
export SAVE_TRAIN_STATE=1

RUN_NAME="${RUN_NAME:-alfworld-qwen3-8b-grpo-32g-$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/supervise_alfworld_8b_segments.sh" "$RUN_NAME" grpo
