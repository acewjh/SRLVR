#!/bin/bash
# Resume ONLY the unfinished HPI-v1 formal eval cells.
#
# HPI-v1 eval state as of 2026-09-20 16:50 (frozen here for the record):
#
#   COMPLETED (dumps on disk, never recomputed):
#     hpisel_cpcp25_aime24   04:27   47.50%  Avg@12
#     hpisel_cpcp50_aime24   07:17   50.00%
#     hpisel_cpcp75_aime24   10:04   54.44%
#     hpisel_cpcp100_aime24  12:53   48.61%
#     selector: top-2 = cp75, cp50  (reports/hpi_selector.json)
#
#   PENDING (this script runs them):
#     hpi_cp75_aime25, hpi_cp75_hmmt25, hpi_cp50_aime25, hpi_cp50_hmmt25
#
# Cells are skipped whenever their dump already exists, so this is safe to run
# repeatedly and will never duplicate a completed eval.
#
# Usage: nohup bash scripts/resume_hpi_v1_eval.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"  # all-NPU eval default; aborts if the budget is short

P=/workspace/cua/repos/opsd-slime
RUN_DIR=$P/runs/math-qwen3-1p7b-rewardcv-hpi-20260919_201825
LOG=$P/logs/eval_queue_hpi.log
LOCK=/tmp/opsd_eval_hpi_v1_resume.lock

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu

export EVAL_LORA=1
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.78}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-1024}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[hpi-v1-resume] another queue holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

cleanup_orphans() {
  pkill -9 -f 'sglang[:]:sched' 2>/dev/null
  pkill -9 -f 'sglang[:]:detok' 2>/dev/null
  pkill -9 -f 'sglang[.]launch_serve' 2>/dev/null
  sleep 5
  ray stop --force >/dev/null 2>&1
  sleep 3
}

run_cell() { # <cp> <dataset> <k> <prefix> <conc>
  local cp=$1 ds=$2 k=$3 prefix=$4 conc=$5
  local iter=$(( ${cp#cp} - 1 ))
  local tag="${prefix}_${cp}_${ds}"
  local staging=$P/runs/eval_staging/${prefix}_${cp}_${iter}
  # an already-completed cell under either tag spelling is reused, never rerun
  for cand in "${tag}" "${prefix}_cp${cp}_${ds}" "${prefix}sel_${cp}_${ds}" "${prefix}sel_cp${cp}_${ds}"; do
    if [ -s "$P/runs/dev_eval/${cand}_eval_0.pt" ]; then
      echo "[hpi-v1-resume] $cand already has a dump -- skipping" >> "$LOG"
      return 0
    fi
  done
  echo "[hpi-v1-resume] === $tag K=$k conc=$conc ($(date '+%F %H:%M:%S')) ===" >> "$LOG"
  bash $P/scripts/make_eval_staging.sh "$RUN_DIR" "$iter" "$staging" >> "$LOG" 2>&1
  EVAL_K="$k" EVAL_CONCURRENCY="$conc" EVAL_CONFIG="$P/configs/eval_${ds}_dev.json" \
    bash $P/train_scripts/run_dev_eval.sh "$staging" "$tag" >> "$LOG" 2>&1
  echo "[hpi-v1-resume] rc=$? for $tag" >> "$LOG"
  cleanup_orphans
}

echo "[hpi-v1-resume] start $(date '+%F %H:%M:%S')" >> "$LOG"

# top-2 from reports/hpi_selector.json; AIME24 is already covered by the
# selector dumps and is skipped by the alias loop above.
for cp in cp75 cp50; do
  run_cell "$cp" aime25 12 hpi 3
  run_cell "$cp" hmmt25 12 hpi 3
done

echo "[hpi-v1-resume] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
