#!/bin/bash
# Candidate-Centered HSI on Qwen3-4B (Node0 of the 2026-10-03 campaign).
#
# One rollout + one future-conditioned frozen-teacher forward + one policy train
# step. No matched/control teacher, no old-actor replay, no full-vocab credit.
# See tasks/alfworld/candidate_center.py for the estimator and _common_v2.sh's
# `hsi_candidate` arm for the engine/sampler conventions.
#
# Usage: RUN_NAME=... NUM_ROLLOUT=... bash run_alfworld_v2_hsi_candidate_4b.sh
. "$(dirname "$0")/_4b_backbone.sh"
# Frozen-teacher arm: preserve its measured non-colocated 2+6 layout.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
export ARM=hsi_candidate RUN_NAME="${RUN_NAME:-alfworld_v2_hsi_candidate_4b}"
exec bash "$(dirname "$0")/_common_v2.sh"
