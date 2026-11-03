#!/bin/bash
# HSI V3 (the 2026-10-04 protocol) on Qwen3-8B -- Node0 of the V4 campaign, cell 4.
#
# Built ahead of its launch per BG_TASK_PROMPT_V4.md section 3 ("开工前先建"): it is
# additive and mirrors run_alfworld_v2_hsi_4b.sh for the arm and
# run_alfworld_v2_grpo_8b.sh for the backbone, so no shared file changes.
#
# The estimator is the shared `math_opsd/hsi2_jsd.py` (SIGN = lagged historical
# reward baseline A_seq_i = R_i - b_k; WHERE = exact normalized JSD between the
# normal-prefix view p and the hindsight view q, both scored by the SAME
# theta_old), selected by _common_v2.sh's `hsi_v3` arm -- the exact branch the 4B
# ALFWorld HSI run `alfworld-qwen3-4b-hsi-s42-20261004_224349` used, so the two
# sizes stay the same method with only the backbone varying.
#
# Only the backbone differs from the 4B HSI arm: geometry via _8b_backbone.sh
# (TP1 x PP1 x DP4 -- read that file for why the 64 GB node does NOT inherit the
# math card's PP4), everything about the method frozen in _common_v2.sh.
#
# Usage: RUN_NAME=... NUM_ROLLOUT=... bash run_alfworld_v2_hsi_8b.sh
. "$(dirname "$0")/_8b_backbone.sh"
# ARM defaults to `hsi_v3` but must yield to a caller that exported a different
# arm: launch_c64.sh's routing passes the arm through this same script, and a hard
# assignment would silently run the wrong estimator.
export ARM="${ARM:-hsi_v3}" RUN_NAME="${RUN_NAME:-alfworld_v2_hsi_8b}"
exec bash "$(dirname "$0")/_common_v2.sh"
