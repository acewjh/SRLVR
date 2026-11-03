#!/bin/bash
# ALFWorld GiGPO on Qwen3-1.7B -- the "group-in-group" arm.
#
# This is the matched control of `run_alfworld_v2_1p7b.sh ARM=grpo`: same model,
# schedule, seed, sampler, LoRA/full-RL geometry, turn budget, response cap,
# curriculum and evaluation. ONLY the advantage estimator differs --
#   A_{i,t} = GRPO episode advantage + w * anchor-state step advantage
# computed inside each task group's G=8 trajectories (arXiv:2505.10978).
#
# Usage: RUN_NAME=... NUM_ROLLOUT=... bash run_alfworld_v2_gigpo_1p7b.sh
export ARM=gigpo
exec bash "$(dirname "$0")/run_alfworld_v2_1p7b.sh"
