#!/bin/bash
# Qwen3-4B GRPO campaign using the finalized 8B ALFWorld protocol.
. "$(dirname "$0")/_4b_backbone.sh"
# Memory-tuned campaign arm (cap=128): keep its measured 2 actor + 6 rollout
# non-colocated layout instead of the shared 8+8 colocated default.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
export ARM=grpo RUN_NAME="${RUN_NAME:-alfworld_v2_grpo_4b_200}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export GBS_OVERRIDE="${GBS_OVERRIDE:-32}"
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
# Full recomputation fits long sequences on 32 GB cards. The provider keeps
# frozen embeddings connected to all trainable adapters during recomputation.
export MODEL_PROVIDER_PATH="${MODEL_PROVIDER_PATH:-tasks.alfworld.memory_provider.custom_model_provider}"
export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-36}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-128}"
export RECOMPUTE_LOSS="${RECOMPUTE_LOSS:-1}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"
exec bash "$(dirname "$0")/_common_v2.sh"
