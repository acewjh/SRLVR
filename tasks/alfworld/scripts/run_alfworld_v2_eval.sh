#!/bin/bash
# Eval-only for a v2 checkpoint (Base / GRPO / HSI). Uses the v2 rollout so the
# prompts and per-turn layout match training exactly.
#
# LoRA NOTE: a LoRA checkpoint MUST be evaluated with the LoRA model provider
# (already in _common_v2.sh). Evaluating a LoRA dir without it silently yields a
# plain model whose numbers are indistinguishable from Base.
#
# Usage: EVAL_LOAD=<ckpt dir> RUN_NAME=<tag> bash tasks/alfworld/scripts/run_alfworld_v2_eval.sh
export ARM=eval RUN_NAME="${RUN_NAME:-alfworld_v2_eval}"
exec bash "$(dirname "$0")/_common_v2.sh"
