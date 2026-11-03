#!/bin/bash
# v2 Arm D on Qwen3-4B -- Future-HSI, method identical to the 1.7B arm.
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_hpi_future_4b.sh
. "$(dirname "$0")/_4b_backbone.sh"
# Teacher-bearing retired arm: preserve its measured non-colocated 2+6 layout.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
export ARM=hpi_future RUN_NAME="${RUN_NAME:-alfworld_v2_hpi_4b}"
exec bash "$(dirname "$0")/_common_v2.sh"
