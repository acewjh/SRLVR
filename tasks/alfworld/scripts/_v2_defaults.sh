#!/bin/bash
# Size-independent ALFWorld v2 defaults -- the ONE shared recipe.
#
# Sourced by _common_v2.sh (every ALFWorld launch) and by the 1.7B entry before
# it builds RUN_NAME. Per-size files (`_1p7b`/`_4b`/`_8b_backbone.sh`) carry only
# the model geometry and the parallelism/memory values the hardware forces; every
# protocol value below is inherited by all sizes, so changing it here changes
# 1.7B, 4B and 8B together.
#
# Placement default mirrors the validated 1.7B campaign: all eight cards are the
# actor and the same eight cards host the colocated rollout engines (TP=2 ->
# actor DP=4). A size that cannot fit this layout MUST override it in its
# backend file with a measured reason (8B on 32 GB cards is the known case);
# don't silently fork the protocol.
#
# Every assignment is `${VAR:-default}` so an explicit environment value or a
# per-size override always wins.

# ---- protocol / task ---------------------------------------------------------
export ALFWORLD_V2_PROTOCOL="${ALFWORLD_V2_PROTOCOL:-1}"
export ALFWORLD_ACTION_HARNESS="${ALFWORLD_ACTION_HARNESS:-command-normalization-v2-form}"
export ALFWORLD_SHOW_ADMISSIBLE="${ALFWORLD_SHOW_ADMISSIBLE:-1}"
export ALFWORLD_ENV_ACTORS="${ALFWORLD_ENV_ACTORS:-1}"
export ALFWORLD_ENV_WORKERS="${ALFWORLD_ENV_WORKERS:-32}"
export ALFWORLD_THINKING_STYLE="${ALFWORLD_THINKING_STYLE:-brief}"
export ALFWORLD_TASK_SAMPLING="${ALFWORLD_TASK_SAMPLING:-frontier}"

# ---- rollout / step budget ---------------------------------------------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export G="${G:-8}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export GBS_OVERRIDE="${GBS_OVERRIDE:-32}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}" # zero means preserve complete history
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export FUTURE_GAP="${FUTURE_GAP:-1}"
export SEED="${SEED:-42}"

# ---- actor/rollout placement (32 GB-card default) ----------------------------
export ACTOR_GPUS="${ACTOR_GPUS:-8}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
export COLOCATE="${COLOCATE:-1}"

# ---- optimizer / loss --------------------------------------------------------
export KL_COEF="${KL_COEF:-0.001}"
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"
export OPSD_LORA_R="${OPSD_LORA_R:-256}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-256}"
export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0}"
export OPSD_LORA="${OPSD_LORA:-0}"
export LR="${LR:-1e-6}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.2}"
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"
export OPSD_SRLVR="${OPSD_SRLVR:-0}"
export OPSD_EXPERIMENT_METRICS="${OPSD_EXPERIMENT_METRICS:-1}"
# Calibration-probe execution only (never estimator identity, config or resume
# state). `auto` chooses DP-position parallel on a supported runtime, otherwise
# serial; explicit `parallel` rejects unsupported runtimes. Preserve the original
# precision, kernel policy and per-trajectory accumulation order. The 4B TP2/DP4
# full-step trial measured 3.47--3.67x calibration speedup with small NPU variation.
# `SRLVR_PROBE_BENCH=1` compares both paths on the same frozen batch.
export SRLVR_PROBE_MODE="${SRLVR_PROBE_MODE:-auto}"

# ---- serving ----------------------------------------------------------------
# NPU graph is ON for every model size. It is a shared serving default, not a
# hardware-specific knob: the 2026-10-30 on-card 4B TP=2 trial passed the
# initial weight sync, a 32-trajectory rollout and the optimizer step with
# graph captures enabled, so no size needs to turn it off. Debugging can still
# disable it with an explicit SGLANG_ENABLE_NPU_GRAPH=0 (the `${VAR:-1}` form
# below preserves that override through every source order).
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}"
export SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"
export SGLANG_PREFIX_CACHE="${SGLANG_PREFIX_CACHE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.42}"
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-65536}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
export SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
export RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
