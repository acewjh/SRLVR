#!/usr/bin/env bash
# Formal full-RL training entry for the two-node Qwen3-1.7B campaign.
#
# The protocol below is the revised one from SRLVR_REVISION_NOTE.md (2026-10-27),
# with the two experiment arms this campaign runs:
#
#   ARM=srlvr_v4   SRLVR-v4 privileged-teacher distillation
#   ARM=grpo       stock trajectory-balanced GRPO
#
# Shared protocol (identical on both arms, both nodes):
#   * full RL            OPSD_LORA=0 -- no adapters, every weight trains
#   * training  TP=2     ACTOR_GPUS=8, DP=4
#   * inference TP=1     ROLLOUT_GPUS_PER_ENGINE=1 -> 8 engines, colocated
#   * NPU graph          SGLANG_ENABLE_NPU_GRAPH default on, capture batch <= 8
#   * thinking enabled   ALFWORLD_THINKING_STYLE=brief, NO thinking token budget
#   * response cap       4096 per turn (the cap is the whole budget)
#   * four-family mix    32 positions/batch (v4) or 4 slots x 8 (GRPO)
#   * 4 calibration      SRLVR_CALIBRATION=4 (v4 only; ignored by GRPO)
#   * 50 batches         save_interval 25 -> cp25 and cp50
#
# Usage:
#   bash run_alfworld_full_rl_1p7b.sh <run-name> [resume-from]
#
# The run name is allocated by the caller through scripts/run_names.py, so a run
# is born conforming to <task>-<model>-<method>[-<variant>]-<stamp>.
set -eo pipefail

ARM_SEL="${ARM:?ARM must be srlvr_v4 or grpo}"
case "$ARM_SEL" in
  srlvr_v4|grpo) ;;
  *) echo "unsupported ARM=$ARM_SEL (want srlvr_v4|grpo)" >&2; exit 2 ;;
esac

PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
export ARM="$ARM_SEL" RUN_NAME="${1:?usage: $0 <run-name> [resume-from]}"

# ---- campaign protocol -------------------------------------------------------
export NUM_ROLLOUT=50
export SAVE_INTERVAL=25
export SEED=42
export MAX_TURNS=50
export HISTORY_LENGTH=0
export MAX_RESPONSE=4096
export SEQ_LENGTH=32768
# Thinking is on, but there is no separate thinking-token budget: the prompt
# never carries a "use up to N tokens" instruction, and MAX_RESPONSE is the only
# limit. Do NOT export ALFWORLD_THINKING_TOKENS here.
unset ALFWORLD_THINKING_TOKENS
export ALFWORLD_THINKING_STYLE=brief
export ALFWORLD_ACTION_HARNESS=command-normalization-v2-form
export ALFWORLD_SHOW_ADMISSIBLE=1
export ALFWORLD_ENV_ACTORS=1
export ALFWORLD_ENV_WORKERS=32

# ---- topology: train TP2 / infer TP1, colocated, NPU graph -------------------
export OPSD_LORA=0                       # full RL -- never inherited as 1
export ACTOR_GPUS=8
export ROLLOUT_GPUS=8
export TP=2
export PP=1
export COLOCATE=1
export ROLLOUT_GPUS_PER_ENGINE=1
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}"
export SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"
export SGLANG_PREFIX_CACHE=1
export SGLANG_MEM_FRACTION_STATIC=0.5
# KV pool reservation, not a context budget (context stays 32768). 65536 since
# 2026-10-28: at 98304 the pool cannot be re-mapped after a weight sync while the
# colocated full-RL trainer is resident (`aclrtMallocPhysical failed` on both
# SRLVR arms); observed utilisation never exceeded 0.31.
export SGLANG_MAX_TOTAL_TOKENS=65536
export SGLANG_MAX_RUNNING_REQUESTS=8
export SGLANG_SERVER_CONCURRENCY=8
export SGLANG_DTYPE=bfloat16
export RECOMPUTE_METHOD=uniform
export RECOMPUTE_NUM_LAYERS=1
# Full RL trains every weight, so the optimizer states are the dominant memory
# term and CPU offload is the lever the revision note leaves to the memory
# budget. It is off here: the 1.7B Adam state fits next to a colocated TP2
# actor on this card, and offload adds a D2H stream to every step.
export OPTIMIZER_CPU_OFFLOAD=0
export OOM_SEQUENCE_PARALLEL=0

# ---- shared training knobs ---------------------------------------------------
export LR=1e-6
export KL_COEF=0.001
export EPS_CLIP=0.2
export EPS_CLIP_HIGH=0.2
export CLIP_GRAD=1.0
export SAVE_TRAIN_STATE=1
export SRLVR_MODE=empirical
export SRLVR_SAMPLING=cdf

# ---- arm-specific -----------------------------------------------------------
if [ "$ARM" = srlvr_v4 ]; then
  export ROLLOUT_BS="${ROLLOUT_BS:-32}"
  export SRLVR_CALIBRATION=4
  # The gradient-safe provider is load-bearing with full recomputation: the
  # frozen-embedding variant silently produced grad_norm=0 on this stack.
  export MODEL_PROVIDER_PATH="${MODEL_PROVIDER_PATH:-tasks.alfworld.memory_provider.custom_model_provider}"
  # Selective response-only lm-head projection: required for TP2 (the loss needs
  # the hidden states to run the exact sharded full-vocabulary reduction) and it
  # is what keeps 32k-token complete-history prompts off the [S, V] path.
  export OPSD_SRLVR_SELECTIVE_LOGITS=1
  export OPSD_SRLVR_SELECTIVE_CHUNK=128
  export SRLVR_V4_COMPACT_FUTURE=1
  # Overridable: the formal campaign runs plain KL (beta=0), the JSD arm runs
  # beta=0.5. Hard-coding 0.0 here silently swallowed the experiment's flag.
  export SRLVR_V4_BETA="${SRLVR_V4_BETA:-0.0}"
  export SRLVR_V4_TOKEN_CLIP=0.05
  export SRLVR_V4_TEACHER_ATTN=sdpa
  export SRLVR_V4_TEACHER_CACHE=8
  export SRLVR_V4_TOKEN_CHUNK=128
  export SRLVR_V4_HINDSIGHT_NO_THINK=1
  export SRLVR_AUTO_BENCHMARK=0
else
  # GRPO: 4 task slots x G=8 == 32 trajectories per batch, same budget.
  export ROLLOUT_BS="${ROLLOUT_BS:-4}"
  export G="${G:-8}"
  # Full RL with reentrant recomputation needs the embedding gradient hook too;
  # without it every checkpointed layer is disconnected and grad_norm stays 0
  # (measured on the 10-27 LoRA arm as well).
  export MODEL_PROVIDER_PATH="${MODEL_PROVIDER_PATH:-tasks.alfworld.memory_provider.custom_model_provider}"
  unset OPSD_SRLVR_SELECTIVE_LOGITS OPSD_SRLVR_SELECTIVE_CHUNK
  unset SRLVR_VARIANT SRLVR_V4_COMPACT_FUTURE SRLVR_CALIBRATION SRLVR_GAMMA
fi

[ -n "${2:-}" ] && export RESUME_FROM="$2"
unset RUN_ROOT SAVE_DIR LOG DEBUG_ROLLOUT_DIR
exec bash "$PROJ/tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh"
