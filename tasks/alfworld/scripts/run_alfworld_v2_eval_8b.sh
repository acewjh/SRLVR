#!/bin/bash
# Eval-only for a Qwen3-8B v2 checkpoint (Base or a GRPO LoRA).
# Same contract as run_alfworld_v2_eval_4b.sh: the backbone must be the one the
# checkpoint was trained with, or the numbers are silently from a different model.
# Usage: EVAL_LOAD=<ckpt dir> RUN_NAME=<tag> bash .../run_alfworld_v2_eval_8b.sh
. "$(dirname "$0")/_8b_backbone.sh"
export ARM=eval RUN_NAME="${RUN_NAME:-alfworld_v2_eval_8b}"
export PP="${PP:-1}"
# 32 GB-card 8B eval layout, same as the 8B fast-rollout training arm.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
exec bash "$(dirname "$0")/_common_v2.sh"
