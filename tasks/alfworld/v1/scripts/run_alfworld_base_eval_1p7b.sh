#!/bin/bash
# Arm A -- BASE evaluation of Qwen3-1.7B on ALFWorld (no training).
# Runs the eval splits (valid_seen + valid_unseen) through the same rollout used
# by every other arm, with no optimizer step, and dumps per-episode trajectories
# for tasks/alfworld/v1/report_eval.py.
#
# Point LOAD_AT at a trained checkpoint dir to evaluate a trained arm instead.
#
# Usage: bash tasks/alfworld/v1/scripts/run_alfworld_base_eval_1p7b.sh
export METHOD=base_eval RUN_NAME="${RUN_NAME:-alfworld_base_eval_1p7b}"
exec bash "$(dirname "$0")/_common.sh"
