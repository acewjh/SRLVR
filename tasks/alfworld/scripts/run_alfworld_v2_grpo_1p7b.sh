#!/bin/bash
# ALFWorld GRPO on Qwen3-1.7B -- Node1 of the 1.7B matrix (campaign_1p7b).
#
# ADDITIVE file (created 2026-10-05). Twin of run_alfworld_v2_hsi_1p7b.sh: it
# sources no backbone file because _common_v2.sh's own defaults are already the
# validated Qwen3-1.7B geometry (_common_v2.sh:28-35).
#
# Arm is the stock GRPO arm of _common_v2.sh (`grpo` branch): stock advantage
# estimator, no privileged teacher, no postprocess, no HPI/HSI weighting. The
# point of this pair is that ONLY the estimator differs between Node0 and Node1 --
# same model, same schedule, same seed 42, same sampler, same LoRA, same lr /
# clip_grad, same verifier, same task limits, same eval protocol.
#
# Task protocol fixed by BG_TASK_PROMPT_V4.md / CAMPAIGN_QUEUE_1p7b.md:
#   MAX_TURNS=50  HISTORY_LENGTH=0  MAX_RESPONSE=128  SEQ_LENGTH=32768
# NUM_ROLLOUT=200 with save_interval=50 -> cp50/100/150/200.
#
# Usage: RUN_NAME=... bash run_alfworld_v2_grpo_1p7b.sh
export ARM="${ARM:-grpo}"
# Full-softmax sampler (T=1.0 / top_p=1.0 / top_k=-1) -- the sampler half of the
# "only the estimator differs" contract with the HSI cell. Without it the shared
# launcher falls through to the 2026-10-03 LEGACY sampler (T=0.6 / top_p 0.95 /
# top_k 20); see _common_v2.sh:362-390. The 4B arms inherit it from
# launch_c64.sh; this wrapper calls _common_v2.sh directly, so it must set it.
export ALFWORLD_V2_PROTOCOL=1
export RUN_NAME="${RUN_NAME:-alfworld_v2_grpo_1p7b}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-200}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
exec bash "$(dirname "$0")/_common_v2.sh"
