#!/bin/bash
# v2 Arm B on Qwen3-4B -- vanilla GRPO on the per-turn layout, matched to
# run_alfworld_v2_hpi_future_4b.sh in everything except METHOD.
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh
. "$(dirname "$0")/_4b_backbone.sh"
export ARM=grpo RUN_NAME="${RUN_NAME:-alfworld_v2_grpo_4b}"
exec bash "$(dirname "$0")/_common_v2.sh"
