#!/usr/bin/env bash
# SRLVR-v4, Qwen3-1.7B, SRLVR_REVISION_NOTE.md (2026-10-27 revision).
#
# One trajectory-level privileged teacher view (complete compact trace + outcome)
# is distilled into every turn's causal prompt.  The teacher is the frozen
# initial policy; the student is the LoRA actor.
#
# Geometry follows the revision note: 8 actor + 8 rollout, COLOCATE=1, one TP1
# engine per card.  TP stays 1 (not the note's 2) because v4 compares full-vocab
# student/teacher logits; TP2 would need a TP-aware KL.
#
# MODEL_PROVIDER_PATH is load-bearing: the revision note's 2026-10-27 finding is
# that frozen embeddings + reentrant full recomputation silently disconnect the
# LoRA gradient (grad_norm=0, reference_kl~1e-12) unless the embedding output is
# marked requires_grad.  `tasks.alfworld.memory_provider` adds that hook.
set -euo pipefail

export ARM=srlvr_v4
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export SEED="${SEED:-42}"
export MODEL_PROVIDER_PATH="${MODEL_PROVIDER_PATH:-tasks.alfworld.memory_provider.custom_model_provider}"

# Current revision protocol (SRLVR_REVISION_NOTE.md §2/§3).
export ALFWORLD_THINKING_STYLE="${ALFWORLD_THINKING_STYLE:-brief}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH=0
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export ALFWORLD_ACTION_HARNESS="${ALFWORLD_ACTION_HARNESS:-command-normalization-v2-form}"
export SGLANG_PREFIX_CACHE=1
# 2026-10-27 action-list experiment: the shared prompt module exposes
# ALFWORLD_SHOW_ADMISSIBLE (default off so the external baseline is unchanged).
# This arm turns it on, so every turn's prompt carries the current admissible
# actions.  Eval, if any, must set the same variable.
export ALFWORLD_SHOW_ADMISSIBLE=1

# SRLVR-v4 objective knobs.
export SRLVR_V4_COMPACT_FUTURE="${SRLVR_V4_COMPACT_FUTURE:-1}"
export SRLVR_V4_BETA="${SRLVR_V4_BETA:-0.0}"
export SRLVR_V4_TOKEN_CLIP="${SRLVR_V4_TOKEN_CLIP:-0.05}"
export SRLVR_V4_TEACHER_ATTN="${SRLVR_V4_TEACHER_ATTN:-sdpa}"
export SRLVR_V4_TEACHER_CACHE="${SRLVR_V4_TEACHER_CACHE:-8}"
export SRLVR_V4_TOKEN_CHUNK="${SRLVR_V4_TOKEN_CHUNK:-128}"
# Response-only lm-head projection: the prompt rows never form [S, V] logits.
# Required for long complete-history prompts on 32 GB cards; the v4 loss
# projects the response positions itself.
export OPSD_SRLVR_SELECTIVE_LOGITS=1
export OPSD_SRLVR_SELECTIVE_CHUNK="${OPSD_SRLVR_SELECTIVE_CHUNK:-128}"

# Shared training defaults.
export OPSD_LORA_R="${OPSD_LORA_R:-128}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
export OPSD_LORA_DROPOUT=0.0
export LR="${LR:-1e-6}"
export KL_COEF="${KL_COEF:-0.001}"
export SAVE_TRAIN_STATE=1
# Only LoRA adapters are trained; CPU offload is unnecessary and this torch
# build cannot run fused AdamW on CPU.
export OPTIMIZER_CPU_OFFLOAD=0

# Revision-note geometry: train/infer share all 8 cards.  The v4 full-vocab loss
# still needs actor TP=1 (DP=8); each card runs one strict TP1 SGLang engine.
export TP=1 PP=1
export ACTOR_GPUS="${ACTOR_GPUS:-8}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-1}"
export COLOCATE="${COLOCATE:-1}"
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.42}"
# 65536, matching run_alfworld_v2_1p7b.sh: the KV pool is re-mapped after every
# weight sync while the colocated trainer is resident, so it must stay well below
# the 10.2-12 GB a full-RL actor step leaves free (aclrtMallocPhysical failure
# with the old 98304/0.5 pair, 2026-10-28). With f=0.42 the measured pool is
# 65536 tok / 7.02 GB, versus 98304 / 10.52 GB before; utilisation never exceeded
# ~0.31, so the cap is a reservation, not a context budget.
export SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-65536}"
export SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-8}"
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-8}"

# No automatic benchmark for v4 yet: the checkpoint format is the ordinary
# Megatron LoRA layout, and the eval driver expects the v2/v3 completion marker.
export SRLVR_AUTO_BENCHMARK=0

exec bash "$(dirname "$0")/run_alfworld_v2_1p7b.sh"
