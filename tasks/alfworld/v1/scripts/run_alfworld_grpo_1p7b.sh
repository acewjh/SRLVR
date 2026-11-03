#!/bin/bash
# Arm B -- stock GRPO on ALFWorld. Reference arm for the HPI comparison:
# G=8, terminal-success reward, slime's own advantage estimator and policy loss.
# The only difference from the HPI arms is the loss: no privileged teacher.
#
# Usage: bash tasks/alfworld/v1/scripts/run_alfworld_grpo_1p7b.sh
export METHOD=grpo RUN_NAME="${RUN_NAME:-alfworld_grpo_1p7b}"
exec bash "$(dirname "$0")/_common.sh"
