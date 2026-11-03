#!/bin/bash
# ============================================================
# Gate A + Gate B -- inference only, no training step.
#
# Gate A: ONE ALFWorld episode end to end. Checks that SGLang serves the
#         checkpoint, the Qwen3 chat template is applied with thinking ON, the
#         <think>/<action> parser recovers commands, the environment advances and
#         the episode runs multiple turns without token corruption.
# Gate B: the 12-game smoke split (one game per task category per eval split).
#         Records success / turns / invalid actions / generated tokens / wall time.
#
# Prerequisite: the NPUs must be genuinely idle. This script does NOT check that
# for you -- verify with npu-smi and ps before running, per the harness rules.
#
# Usage: bash tasks/alfworld/v1/scripts/smoke_alfworld_rollout_1p7b.sh
# ============================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJ=/workspace/cua/repos/opsd-slime

export METHOD=base_eval
export RUN_NAME=alfworld_smoke_rollout_1p7b
export ALFWORLD_EVAL_CONFIG="$PROJ/tasks/alfworld/v1/configs/eval_alfworld_smoke.json"
export N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
# Gate A is the same run: the first episode is the single-episode check.
export SAVE_DIR="${SAVE_DIR:-$PROJ/runs/alfworld_smoke_rollout_1p7b}"
export LOG="$PROJ/logs/alfworld_smoke_rollout_1p7b.log"
export MAX_TURNS="${MAX_TURNS:-15}"

: > "$LOG"
START=$(date +%s)
bash "$HERE/_common.sh"
RC=$?
END=$(date +%s)
echo "[gate-a/b] exit=$RC wall=$((END-START))s log=$LOG"
exit $RC
