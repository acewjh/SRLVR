#!/bin/bash
# Qwen3-8B SRLVR-v2 on one 8x32GB Ascend node.
# TP4 keeps the actor on four cards (DP1); the other four cards host two TP2
# rollout engines. This is the 32GB geometry validated by the TP4 smoke run.
set -euo pipefail

# Select the 32GB geometry before sourcing backbone defaults, which otherwise
# set TP=2 and the generic 2+6 placement.
export TP="${TP_32G:-4}"
export PP="${PP_32G:-1}"
export ACTOR_GPUS="${ACTOR_GPUS_32G:-4}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS_32G:-4}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE_32G:-4}"
export COLOCATE="${COLOCATE_32G:-1}"

. "$(dirname "$0")/_8b_backbone.sh"
. "$(dirname "$0")/_8b_fast_rollout.sh"

export ARM=srlvr_v2
export OPSD_LORA=0
export G=1
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export SRLVR_CALIBRATION="${SRLVR_CALIBRATION:-4}"
export SRLVR_GAMMA="${SRLVR_GAMMA:-0.99}"
export SRLVR_COMPACT_FUTURE="${SRLVR_COMPACT_FUTURE:-0}"
export SRLVR_PROBE_ACCUM="${SRLVR_PROBE_ACCUM:-cpu}"
export SRLVR_ADVANTAGE_MODE="${SRLVR_ADVANTAGE_MODE:-batch}"
export SRLVR_AUTO_BENCHMARK=0
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.42}"
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-65536}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
# NPU graph OFF by default for this arm: the graph-replay path leaks host RSS
# at ~13 GiB/h per scheduler (measured 2026-11-09); eager decode leaks <1 GiB/h
# and the rollout wall-clock was unchanged (986 s vs 1009 s). Override with
# SGLANG_ENABLE_NPU_GRAPH=1 to restore the old behavior (then keep the
# 25-rollout segment supervisor, or the node will OOM after ~28 rollouts).
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-0}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"
export RUN_NAME="${RUN_NAME:-alfworld-qwen3-8b-srlvr-v2-32g-tp4-$(date +%Y%m%d_%H%M%S)}"

exec bash "$(dirname "$0")/_common_v2.sh" "$@"
