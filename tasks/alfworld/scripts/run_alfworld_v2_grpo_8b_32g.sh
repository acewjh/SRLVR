#!/bin/bash
# Qwen3-8B GRPO layout for one 8x32GB Ascend 910B4 node.
# Keep the 64GB launcher defaults intact; this wrapper changes placement only.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PP="${PP:-1}"
export TP="${TP:-4}"
export ACTOR_GPUS="${ACTOR_GPUS:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
export COLOCATE="${COLOCATE:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export RUN_NAME="${RUN_NAME:-alfworld-qwen3-8b-grpo-32g-$(date +%Y%m%d_%H%M%S)}"
exec bash "$SCRIPT_DIR/run_alfworld_v2_grpo_8b.sh" "$@"
