#!/bin/bash
# v2 Arm B -- vanilla GRPO on the per-turn layout. Reference for Future-HPI.
# Only difference from run_alfworld_v2_hpi_future.sh: no Full/Control scorer, no
# HPI weights (fit_turn_weights stays 1.0).
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_grpo.sh
export ARM=grpo RUN_NAME="${RUN_NAME:-alfworld_v2_grpo}"
exec bash "$(dirname "$0")/_common_v2.sh"
