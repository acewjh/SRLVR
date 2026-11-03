#!/bin/bash
# ============================================================
# Gate C -- ONE GRPO training step on ALFWorld.
#
# Pass conditions (all must hold; check with check_alfworld_gates.sh):
#   * finite loss, finite grad norm
#   * finite reward statistics with non-zero variance inside the group
#   * weight sync completed (rollout weights == training weights)
#   * loss_mask/turn_ids contract held through convert (it raises otherwise)
#
# Deliberately tiny: 1 task x G=8 x 1 step.
#
# Usage: bash tasks/alfworld/v1/scripts/smoke_alfworld_train_grpo_1p7b.sh
# ============================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ=/workspace/cua/repos/opsd-slime

export METHOD=grpo
export RUN_NAME=alfworld_smoke_train_grpo_1p7b
export NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export MAX_RESPONSE="${MAX_RESPONSE:-640}"
export MAX_TURNS="${MAX_TURNS:-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"        # never checkpoint a smoke
export SAVE_DIR="${SAVE_DIR:-$PROJ/runs/alfworld_smoke_train_grpo_1p7b}"
export LOG="${LOG:-$PROJ/logs/alfworld_smoke_train_grpo_1p7b.log}"
: > "$LOG"
bash "$HERE/_common.sh"
echo "[gate-c] exit=$? log=$LOG"
