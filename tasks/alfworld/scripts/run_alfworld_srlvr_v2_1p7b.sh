#!/usr/bin/env bash
set -euo pipefail
export ARM=srlvr_v2
export NUM_ROLLOUT="${NUM_ROLLOUT:-25}"
export SRLVR_AUTO_BENCHMARK="${SRLVR_AUTO_BENCHMARK:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export SRLVR_GAMMA="${SRLVR_GAMMA:-0.99}"
exec bash "$(dirname "$0")/run_alfworld_v2_1p7b.sh"
