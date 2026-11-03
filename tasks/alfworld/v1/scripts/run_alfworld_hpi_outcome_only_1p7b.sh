#!/bin/bash
# Arm C -- HPI with OUTCOME-ONLY privilege.
# The frozen initial policy re-scores each turn's own tokens conditioned on the
# terminal outcome; w = 1 + alpha*tanh(|e|/scale) re-weights the GRPO token loss.
# Identical to Arm D in every respect except that the privileged block carries no
# future interaction trace.
#
# Usage: ALFWORLD_HPI_SCALE=<calibrated> bash tasks/alfworld/v1/scripts/run_alfworld_hpi_outcome_only_1p7b.sh
export METHOD=hpi_outcome_only RUN_NAME="${RUN_NAME:-alfworld_hpi_outcome_only_1p7b}"
exec bash "$(dirname "$0")/_common.sh"
