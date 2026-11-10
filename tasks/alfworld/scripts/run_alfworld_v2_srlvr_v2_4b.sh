#!/bin/bash
# SRLVR-v2 on Qwen3-4B -- thin entry over the shared size-independent recipe.
#
# Protocol, placement, budget, optimizer, LoRA ranks and serving defaults all
# come from tasks/alfworld/scripts/_v2_defaults.sh; the 4B geometry and the
# hardware-forced TP=2 / 0.30 / 32768 live in _4b_backbone.sh (NPU graph is the
# shared default: on for every size, explicit =0 still wins).
# This wrapper adds ONLY the algorithm-dictated values:
#   * G=1 and ROLLOUT_BS=32 (one trajectory per task position, 32 tasks per
#     rollout => the same 32-trajectory global batch as GRPO's 4 x 8);
#   * SRLVR_CALIBRATION=4 and the frozen v2 gamma/compact defaults;
#   * full RL (never LoRA for this A/B) and the fp32 probe CPU staging.
# The only per-run difference between the two arms is SRLVR_ADVANTAGE_MODE.
# Placement (ACTOR_GPUS=8, ROLLOUT_GPUS=8, COLOCATE=1), serving and budget are
# deliberately NOT repeated here: a local copy would silently fork the shared
# recipe the next time it changes.
#
# SRLVR_PROBE_ACCUM=cpu keeps the two calibration gradient vectors (Z and Y) in
# host memory, keeping two full FP32 vectors outside the model/activation pool.
# Device staging is available as an explicit override; its peak with the full
# training/serving placement must be measured before changing this default.
#
# Usage: SRLVR_ADVANTAGE_MODE=history|batch RUN_NAME=... bash run_alfworld_v2_srlvr_v2_4b.sh
. "$(dirname "$0")/_4b_backbone.sh"
export ARM=srlvr_v2
export OPSD_LORA=0                           # experiment invariant: full RL
export G=1                                   # registry also dictates G=1 for this arm
export ROLLOUT_BS="${ROLLOUT_BS:-32}"        # 32 x G=1 = 32-trajectory batch
export SRLVR_CALIBRATION="${SRLVR_CALIBRATION:-4}"
export SRLVR_GAMMA="${SRLVR_GAMMA:-0.99}"
export SRLVR_COMPACT_FUTURE="${SRLVR_COMPACT_FUTURE:-0}"
export SRLVR_PROBE_ACCUM="${SRLVR_PROBE_ACCUM:-cpu}"
export SRLVR_ADVANTAGE_MODE="${SRLVR_ADVANTAGE_MODE:-batch}"
export SRLVR_AUTO_BENCHMARK=0                # this wrapper calls _common_v2 directly
export RUN_NAME="${RUN_NAME:-alfworld-qwen3-4b-srlvr_v2-$SRLVR_ADVANTAGE_MODE-s${SEED:-42}-r${NUM_ROLLOUT:-100}-n32}"
exec bash "$(dirname "$0")/_common_v2.sh"
