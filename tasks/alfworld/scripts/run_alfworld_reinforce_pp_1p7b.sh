#!/usr/bin/env bash
# REINFORCE++ arm (one node), matched to the GRPO arm on everything else.
#
# REINFORCE++ ("REINFORCE++: A Simple and Efficient Approach for Aligning Large
# Language Models", 2501.03262) keeps PPO's clipped surrogate but drops the
# critic: the baseline comes from the BATCH, and advantages are whitened over the
# batch rather than within a sampled group.
#
# Here that is `ALFWORLD_V2_ADVANTAGE_MODE=batch`, implemented in
# tasks/alfworld/train_data.py: the per-trajectory reward is standardised over
# every trajectory of the step (4 slots x G=8 = 32) instead of within each group
# of 8. The rollout, the clipped surrogate (eps 0.2), the initial-model KL
# (low_var_kl, 0.001) and the trajectory-balanced reduction are byte-identical to
# the GRPO arm, so the ONLY variable between the two arms is the advantage
# estimator -- which is what makes the comparison readable.
#
# Motivation: GRPO zeroes any group whose 8 samples agree, and early on most
# groups are all-fail, so a large share of steps carry no signal. Pooling the
# baseline over 32 trajectories keeps those tasks informative.
set -eo pipefail

export ARM=grpo
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export G="${G:-8}"
export SEED="${SEED:-42}"

export ALFWORLD_V2_ADVANTAGE_MODE=batch

exec bash "$(dirname "$0")/run_alfworld_full_rl_1p7b.sh" "$@"
