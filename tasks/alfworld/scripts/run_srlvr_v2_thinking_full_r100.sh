#!/usr/bin/env bash
set -euo pipefail

# Paired full-future SRLVR-v2 run: preserve the existing implementation and
# interaction path; this wrapper only freezes this run's requested settings.
BASE=/workspace/cua
export MODEL="${MODEL:-$BASE/checkpoints/Qwen3-1.7B}"
export ARM=srlvr_v2
export RUN_NAME=SRLVR-v2-thinking-full-r100
export NUM_ROLLOUT=100
export SAVE_INTERVAL=25
export ROLLOUT_BS=32
export SEQ_LENGTH=32768
export MAX_RESPONSE=4096
export ALFWORLD_THINKING_STYLE=brief
export SRLVR_GAMMA=0.99
export FUTURE_GAP=1
export SRLVR_COMPACT_FUTURE=0
export SRLVR_AUTO_BENCHMARK="${SRLVR_AUTO_BENCHMARK:-1}"
export OPSD_LORA_R=128 OPSD_LORA_ALPHA=128 OPSD_LORA_DROPOUT=0.0
export LR=1e-6 KL_COEF=0.001
export ACTOR_GPUS=2 ROLLOUT_GPUS=6 SEED=42 MAX_TURNS=50 HISTORY_LENGTH=0
export ALFWORLD_TASK_SAMPLING=frontier SGLANG_PREFIX_CACHE=1 SGLANG_DTYPE=bfloat16
exec bash "$(dirname "$0")/run_alfworld_srlvr_v2_1p7b.sh"
