#!/bin/bash
# Arm D -- HPI with OUTCOME + DELAYED FUTURE TRACE privilege.
# The only intended difference from Arm C: the privileged block additionally
# contains the future environment interaction trace (observations + executed
# actions, past the --hpi-future-gap window, future <think> stripped).
#
# Usage: ALFWORLD_HPI_SCALE=<calibrated> bash tasks/alfworld/v1/scripts/run_alfworld_hpi_future_1p7b.sh
export METHOD=hpi_future RUN_NAME="${RUN_NAME:-alfworld_hpi_future_1p7b}"
exec bash "$(dirname "$0")/_common.sh"
