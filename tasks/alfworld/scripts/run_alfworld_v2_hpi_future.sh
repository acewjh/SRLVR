#!/bin/bash
# v2 Arm D -- Future-HPI: the same GRPO objective, token credit re-allocated by the
# self-normalised Full-vs-Control action density ratio.
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_hpi_future.sh
export ARM=hpi_future RUN_NAME="${RUN_NAME:-alfworld_hpi_v2_future}"
exec bash "$(dirname "$0")/_common_v2.sh"
