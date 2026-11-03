#!/bin/bash
# SRLVR (v2/v3) on Qwen3-8B, per-turn ALFWorld v2 layout.
#
# Structure copied from run_alfworld_v2_grpo_8b.sh: geometry comes from
# _8b_backbone.sh, the 2:6 train/rollout split from _8b_fast_rollout.sh, and
# everything about the method is frozen in _common_v2.sh. Only the SRLVR knobs
# and the matched-budget batch geometry are set here.
#
# SRLVR values are the ones the colleague's 1.7B campaign is stable on
# (reports/srlvr_compact_1node_20261029/common.env + run_alfworld_srlvr_v2_1p7b.sh):
#   SRLVR_COMPACT_FUTURE=1          the requested delta vs. the shared default
#   SRLVR_AUTO_BENCHMARK=0          training-only; no chained benchmark
#   SGLANG_MEM_FRACTION_STATIC=0.42
#   OPSD_SRLVR_SELECTIVE_LOGITS=0   response-only projection OFF
#   SRLVR_GAMMA=0.99
#   SEED=42, NUM_ROLLOUT=50, SAVE_INTERVAL=25
#
# MATCHED BUDGET. SRLVR is G=1, so ROLLOUT_BS is the trajectory count directly;
# GRPO at 8B uses ROLLOUT_BS=4 x G=8 = 32. Both arms therefore see 32 real
# trajectories per optimizer step, and GBS is pinned at 32 rows in both.
#
# Usage:
#   ARM=srlvr_v2 bash tasks/alfworld/scripts/run_alfworld_v2_srlvr_8b.sh
#   ARM=srlvr_v3 NUM_ROLLOUT=2 RUN_NAME=smoke bash .../run_alfworld_v2_srlvr_8b.sh
set -euo pipefail

. "$(dirname "$0")/_8b_backbone.sh"
. "$(dirname "$0")/_8b_fast_rollout.sh"

export ARM="${ARM:-srlvr_v2}"
export RUN_NAME="${RUN_NAME:-alfworld_v2_${ARM}_8b}"

# ---- SRLVR knobs: the colleague's stable 1.7B values --------------------------
export SRLVR_COMPACT_FUTURE="${SRLVR_COMPACT_FUTURE:-1}"
export SRLVR_AUTO_BENCHMARK="${SRLVR_AUTO_BENCHMARK:-0}"
export SRLVR_GAMMA="${SRLVR_GAMMA:-0.99}"
export OPSD_SRLVR_SELECTIVE_LOGITS="${OPSD_SRLVR_SELECTIVE_LOGITS:-0}"
# 8B ADAPTATION, not the colleague's value. The 1.7B campaign (and
# _common_v2.sh's own default) uses 0.42, which is correct for a 1.7B engine
# (~3.4 GB of weights) and fatal for an 8B one: _8b_fast_rollout.sh leaves
# inference TP at 1, so every rollout card holds the FULL model, and Qwen3-8B in
# bf16 is ~16.4 GB against 29.49 GB usable -- 0.42 reserves only 12.4 GB.
# Measured (srlvr8b-smoke-01, 2026-10-29): every engine died with
#   RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
#   Current value: mem_fraction_static=0.42
# 0.85 reserves ~25 GB: the 16.4 GB of weights plus ~8.6 GB of KV pool.
# If this ever proves too tight, the alternative is inference TP=2
# (ROLLOUT_GPUS_PER_ENGINE=2, ~8.2 GB of weights per card) at the cost of
# running 3 engines instead of 6.
export SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
export SLIME_SGLANG_HTTP_TIMEOUT="${SLIME_SGLANG_HTTP_TIMEOUT:-2400}"
export SEED="${SEED:-42}"

# ---- horizon: the colleague's 1.7B campaign length ---------------------------
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

# ---- protocol, identical to run_alfworld_v2_grpo_8b.sh -----------------------
export PP="${PP:-1}"
export ALFWORLD_V2_PROTOCOL="${ALFWORLD_V2_PROTOCOL:-1}"
# Environment parallelism, matching the colleague's 4B ablations
# (ALFWORLD_ENV_ACTORS=1, ALFWORLD_ENV_WORKERS=32). Without these the rollout
# feeds SGLang ONE trajectory at a time: measured on this run, `#running-req`
# oscillated between 0 and 1 for the whole of rollout 0, and the cap-probe's
# own numbers show the same thing -- generation_seconds_sum 17891 s against
# episode_wall_seconds_max 2277 s, i.e. the 32 trajectories barely overlapped.
# SGLang was never the limit (max_running_requests=8); the environment pool was.
export ALFWORLD_ENV_ACTORS="${ALFWORLD_ENV_ACTORS:-1}"
export ALFWORLD_ENV_WORKERS="${ALFWORLD_ENV_WORKERS:-32}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
# Response cap. 4096, NOT the 128 that `run_alfworld_v2_grpo_8b.sh` carries.
#
# That 128 comes from an 8B launcher which has never actually been run (the
# colleague's tree holds no 8B run at all), while his STABLE 1.7B campaign --
# the one this comparison has to be commensurable with -- uses
#   MAX_RESPONSE=4096  SEQ_LENGTH=32768
# Measured here (srlvr8b-v2-03, 2026-10-30) with the inherited 128:
#   rollout/truncated_ratio   = 1.0     every response hit the cap
#   rollout/action_valid_rate = 0.0     so no action could be parsed
#   rollout/success_rate      = 0.0     and the reward is therefore always 0
# With all rewards equal the advantage is identically 0 and training is a
# no-op that still reports a healthy grad_norm -- exactly the failure
# docs/归档_Qwen35_9B_SRLVR适配受阻 §7 records for Qwen3-8B on math at
# cap=1024. A bigger model thinks longer; a smaller model's cap is not
# portable to it, and `ALFWORLD_THINKING_STYLE=brief` still needs room for the
# thought plus the action.
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"
export SAVE_TRAIN_STATE="${SAVE_TRAIN_STATE:-1}"

# ---- matched 32-trajectory budget -------------------------------------------
# G=1 for SRLVR, so the batch IS the trajectory count. GRPO reaches the same 32
# via ROLLOUT_BS=4 x 8 generations.
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export GBS_OVERRIDE="${GBS_OVERRIDE:-32}"

# ---- 8B GEOMETRY: 4/4 split, TP=4, FULL PARAMETER ---------------------------
# `_8b_fast_rollout.sh` defaults to a 2:6 split, which was never actually run
# (the colleague's tree has no 8B run at all -- `runs/*8b*` is empty), and 2
# training cards cannot hold 8B under any sharding. The colleague's stable 1.7B
# campaign trains FULL-parameter (`OPSD_LORA=0`) on 8 colocated cards at TP=2,
# so full-parameter is also what keeps the 8B SRLVR-vs-GRPO comparison on the
# same methodological footing.
#
#   full-parameter 8B, per DP replica
#     weights bf16  8.2e9 x 2 B                = 16.4 GB   on card
#     grads   bf16                             = 16.4 GB   on card
#     AdamW   fp32 m+v  8.2e9 x 2 x 4 B        = 65.6 GB   -> CPU offload
#                                     on card total 32.8 GB, split by TP
#
#     TP=1  32.8 GB/card  -> OOM (measured, srlvr8b-smoke-03: "Tried to
#                            allocate 15.26 GiB" while loading weights)
#     TP=2   8.2 GB weights + 8.2 GB grads per card  <-- chosen
#     TP=4  would halve that again, but SRLVR REFUSES it:
#             ValueError: SRLVR supports tensor_model_parallel_size in {1,2}
#           (measured in DRY_RUN; the estimator's logit projection assumes that
#           layout). TP=2 is also what the colleague's 1.7B campaign uses.
#
# With 4 training cards at TP=2 the actor runs 2 data-parallel replicas. Per
# card: 8.2 GB weights + 8.2 GB grads + 8.2 GB for the frozen KL reference
# (KL_COEF=0.001 keeps a second copy of the model) = 24.6 GB, leaving ~4.9 GB of
# the 29.49 GB for activations -- which is ample at cap=128, where each turn
# emits a single action.
#
# OPTIMIZER_CPU_OFFLOAD is already 1 by default in _common_v2.sh (line 467), so
# the 65.6 GB of optimizer state never lands on the cards.
#
# 4 rollout cards instead of 6 costs roughly a third of the generation
# throughput; that is the price of training full-parameter rather than LoRA.
# NOT colocated: ACTOR_GPUS + ROLLOUT_GPUS == 8 exactly, so each side owns its
# cards and the weight sync never competes with the KV pool for contiguous
# memory (the failure that blocked the Qwen3.5-9B colocated arm).
#
# All four are assigned UNCONDITIONALLY. `_8b_fast_rollout.sh` and
# `_runtime_defaults.sh` are sourced ABOVE and already give each of these a
# value, so `${VAR:-...}` would silently keep theirs -- exactly how an earlier
# attempt ended up running with lora=0 while TP=2 looked correct.
export ACTOR_GPUS=4
export ROLLOUT_GPUS=4
# TP: 4 is what makes full-parameter 8B fit (the two flat DDP buffers halve from
# 15.26 GiB each to ~7.6 GiB), but SRLVR's validate_runtime only ALLOWS the
# widths listed in SRLVR_ALLOWED_TP, whose default is "1,2". So the two travel
# together: ask for TP=4 only when the caller has widened the allow-list.
#
# Verified before widening (2026-10-30, /tmp/tp_equiv.py): sharded vocabulary
# scoring at 2/4/8 shards matches dense TP=1 with dloss 3.7e-08, dmu 1.8e-07 and
# hidden-state gradient relative error ~7e-07. The gradient check is the one that
# matters -- tp_utils.py records a case where value matched dense to 4e-16 while
# the gradient was off by 49% and grad_norm still looked healthy. Structurally
# v2/v3 are safe anyway: every vocabulary-sharded reduction in srlvr_hsi.py is
# under @torch.no_grad() and produces only the scalar advantage coefficient,
# while actor_loss is a plain PPO token surrogate that does no sharded
# reduction. That 49% case was SRLVR-v4, which has a trainable teacher path.
# TP=4. The colleague's own code no longer caps it: as of his 2026-10-30 15:41
# update `validate_runtime` only requires `tp >= 1` plus "actor GPUs divisible
# by TP", so the earlier SRLVR_ALLOWED_TP shim is gone. His 4B ablations run
# TP=2, but 8B cannot: Megatron's DDP allocates a flat param buffer AND a flat
# grad buffer, 15.26 GiB each at TP=2, and 15.26 + 15.26 > 29.49 GiB per card
# (measured, srlvr8b-v2-01). TP=4 halves both to ~7.6 GiB.
# Equivalence verified before adopting it: sharded vocabulary scoring at 2/4/8
# shards matches dense TP=1 with hidden-state gradient relative error ~7e-07.
export TP=4
export PP="${PP:-1}"
export OPSD_LORA=0
export OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-1}"

exec bash "$(dirname "$0")/_common_v2.sh"
