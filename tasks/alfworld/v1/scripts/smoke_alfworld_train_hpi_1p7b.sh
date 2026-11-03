#!/bin/bash
# ============================================================
# Gate D -- ONE HPI (future_outcome) training step on ALFWorld.
#
# Pass conditions (all must hold; check with check_alfworld_gates.sh):
#   * teacher-forced target token ids == student generated token ids (the loss
#     raises AssertionError otherwise -- it is not a warning)
#   * privilege isolation assertions PASS
#   * finite HPI-weighted loss, finite GRPO/base loss, finite combined loss
#   * reference sanity: the control view is the student's own prompt, so
#     alpha=0 must reproduce the GRPO loss (checked separately by the CPU gate)
#   * weight sync completed
#
# ALFWORLD_HPI_SCALE: Gate D is also the CALIBRATION run. Run it once with a
# placeholder scale, read hpi/abs_e_p75 from the log, then set ALFWORLD_HPI_SCALE
# to that value for the real run (convention inherited from math_opsd: scale = p75
# of |e|, e.g. 0.045126 for the math task).
#
# Usage: ALFWORLD_HPI_SCALE=1.0 bash tasks/alfworld/v1/scripts/smoke_alfworld_train_hpi_1p7b.sh
# ============================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ=/workspace/cua/repos/opsd-slime

export METHOD="${METHOD:-hpi_future}"
export RUN_NAME=alfworld_smoke_train_${METHOD}_1p7b
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BS="${ROLLOUT_BS:-1}"
export MAX_TURNS="${MAX_TURNS:-8}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export ALFWORLD_HPI_SCALE="${ALFWORLD_HPI_SCALE:-1.0}"
export SAVE_DIR="${SAVE_DIR:-$PROJ/runs/alfworld_smoke_train_${METHOD}_1p7b}"
export LOG="$PROJ/logs/alfworld_smoke_train_${METHOD}_1p7b.log"
: > "$LOG"
bash "$HERE/_common.sh"
echo "[gate-d] exit=$? log=$LOG"
echo "[gate-d] calibration: read hpi/abs_e_p75 from the log and use it as ALFWORLD_HPI_SCALE"
grep -o "hpi/abs_e_p75[^,}]*" "$LOG" 2>/dev/null | tail -3
