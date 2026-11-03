#!/bin/bash
# Remaining eval work, parked 2026-09-22 on the user's instruction so that the
# HPI-v1.5 campaign can take the 8 NPUs. Nothing here was deleted -- every queue
# and artifact stays on disk; this script just re-drives them in order.
#
# Content (all idempotent -- each cell skips when its dump already exists):
#
#   1. GRPO formal, cp75 x {aime24, aime25, hmmt25}
#      NOT RUN NOW, deliberately: the GRPO cp75/cp100 checkpoints come from a
#      reset-Adam resume after the 23:39 stall, so they carry a scientific
#      confound. Only the clean cp25 arm was completed (3/3 benchmarks).
#   2. Stage B -- HPI-v1's four pending formal cells
#   3. Stage C -- GT-OPSD cp75 x 3
#
# Usage: nohup bash scripts/run_remaining_evals.sh > /dev/null 2>&1 &
set -uo pipefail
P=/workspace/cua/repos/opsd-slime
LOG=$P/logs/remaining_evals.log
LOCK=/tmp/opsd_remaining_evals.lock

exec 7>"$LOCK"
if ! flock -n 7; then echo "[remaining] another runner holds $LOCK -- exiting" >> "$LOG"; exit 1; fi

echo "[remaining] start $(date '+%F %H:%M:%S')" >> "$LOG"

echo "[remaining] (1) GRPO cp75 formal -- see header: deliberately parked" >> "$LOG"
# To run it, uncomment. It is safe/idempotent; the confound is scientific, not technical.
#   bash $P/scripts/eval_queue_grpo.sh

echo "[remaining] (2) Stage B -- HPI-v1 pending formal cells" >> "$LOG"
bash $P/scripts/resume_hpi_v1_eval.sh
echo "[remaining] Stage B rc=$? $(date '+%F %H:%M:%S')" >> "$LOG"

echo "[remaining] (3) Stage C -- GT-OPSD cp75 x 3" >> "$LOG"
bash $P/scripts/eval_gtopsd_cp75.sh
echo "[remaining] Stage C rc=$? $(date '+%F %H:%M:%S')" >> "$LOG"

echo "[remaining] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
