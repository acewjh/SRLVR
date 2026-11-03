#!/bin/bash
# A2: the three missing GRPO formal cells -- cp75 x {aime24, aime25, hmmt25}.
#
# Only cp25 was ever completed formally (3/3 benchmarks, 2026-09-22). This adds
# the cp75 row at the frozen house protocol (K=12, 30 problems, 38912 max tokens,
# all 8 NPUs). Cells with a dump are skipped, so re-running is safe.
#
# ⚠️  SCIENTIFIC CAVEAT -- stated here rather than discovered later.
# `scripts/run_remaining_evals.sh` (written 2026-09-22) parks this work with the
# note that the GRPO cp75/cp100 checkpoints "come from a reset-Adam resume after
# the 23:39 stall, so they carry a scientific confound". That is the same
# --no-save-optim resume defect A0 fixed for the vNext arms, and it cannot be
# repaired retroactively: `runs/math-qwen3-1p7b-grpo-20260921_224239/iter_0000074` already exists.
# So these numbers are reported as the GRPO cp75 *row*, and any comparison that
# depends on GRPO having had a continuous optimizer schedule is flagged
# unresolved. A clean GRPO rerun is possible now that the sidecar exists
# (`bash scripts/resume_training.sh grpo` into a fresh dir), but that is a new
# training run and is NOT silently folded in here.
#
# Usage: nohup bash scripts/eval_queue_grpo_cp75.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
EVAL_CELL_LOG_TAG=grpo75

P=/workspace/cua/repos/opsd-slime
RUN_DIR=$P/runs/math-qwen3-1p7b-grpo-20260921_224239
LOG=$P/logs/eval_queue_grpo_cp75.log
LOCK=/tmp/opsd_eval_grpo_cp75.lock

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu

export EVAL_LORA=1
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-$OPSD_EVAL_MEM_FRACTION}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-$OPSD_EVAL_CHUNKED_PREFILL}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[grpo75] another queue holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

[ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ] || {
  echo "[grpo75] FATAL: no GRPO checkpoints" >> "$LOG"; exit 1; }

echo "[grpo75] start $(date '+%F %H:%M:%S')" >> "$LOG"
run_eval_cell cp75 aime24 eval_aime24_dev.json 12 grpo 2
run_eval_cell cp75 aime25 eval_aime25_dev.json 12 grpo 3
run_eval_cell cp75 hmmt25 eval_hmmt25_dev.json 12 grpo 3
echo "[grpo75] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
