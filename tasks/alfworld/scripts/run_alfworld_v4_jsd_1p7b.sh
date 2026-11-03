#!/usr/bin/env bash
# SRLVR-v4 with generalized JSD + randomized hindsight withholding (one node).
#
# Two changes against the plain v4 arm, both aimed at the same weakness observed
# on 2026-10-28 (action-valid 0.36 -> 0.07, success 0, reference_KL 500x the LoRA
# run):
#
#  1. JSD instead of KL. beta=0.5 is the symmetric Jensen-Shannon divergence
#     between the teacher and the student. Plain KL (beta=0) is mode-seeking and
#     saturates at the teacher's sharp tokens; JSD is bounded and mass-covering,
#     so a student that cannot yet match the teacher is not punished without
#     limit. SRLVR_V4_BETA already parameterises this path
#     (methods/srlvr_v4.py::generalized_jsd_per_token) and is covered by the TP=2
#     gradient-equivalence test for beta in {0, 0.5, 1}.
#
#  2. Randomized hindsight withholding. The privileged block shows the WHOLE
#     episode, so the teacher can in principle recite turn t+1 while the student
#     is scored on turn t -- the outcome and later actions leak the answer.
#     SRLVR_V4_HINDSIGHT_DROP_TURNS=k withholds k randomly chosen transitions
#     (seeded per task|episode|rollout, so it is reproducible and resume-safe).
#     The trace keeps its original indices, so a withheld turn is a visible gap.
#
# Everything else matches the formal campaign: full RL, train TP=2, inference
# TP=1, NPU graph, thinking without a token budget, cap 4096, four-family mix,
# 32 trajectories/batch, 50 batches, save every 25.
set -eo pipefail

export ARM=srlvr_v4
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export SEED="${SEED:-42}"

# --- the two experimental knobs ----------------------------------------------
export SRLVR_V4_BETA="${SRLVR_V4_BETA:-0.5}"
export SRLVR_V4_HINDSIGHT_DROP_TURNS="${SRLVR_V4_HINDSIGHT_DROP_TURNS:-3}"
# The non-compact renderer cannot drop transitions; fail closed instead of
# silently training on a full trace.
export SRLVR_V4_COMPACT_FUTURE=1

exec bash "$(dirname "$0")/run_alfworld_full_rl_1p7b.sh" "$@"
