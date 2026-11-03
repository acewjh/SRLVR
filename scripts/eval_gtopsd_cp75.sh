#!/bin/bash
# Stage C: fill in the missing GT-OPSD eval — cp75 x {aime24, aime25, hmmt25} at the
# formal K=12 / 38912 protocol.
#
# GT-OPSD cp25/cp50/cp100 x 3 datasets already exist (baseline node). cp75 only ever had
# a "cheap" AIME24 K=4 cell, so the 75-step row was never like-for-like comparable with
# our own 75-step checkpoints. This closes that gap.
#
# The checkpoint is an already-merged HF directory (no LoRA adapters), so this uses the
# HF-cell path: EVAL_LORA=0. Cells are skipped when their dump already exists.
#
# Usage: nohup bash scripts/eval_gtopsd_cp75.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"  # all-NPU eval default
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

P=/workspace/cua/repos/opsd-slime
HF_DIR=/workspace/cua/runs/gt_opsd_cp75_merged
LOG=$P/logs/eval_gtopsd_cp75.log
LOCK=/tmp/opsd_eval_gtopsd75.lock

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu

export EVAL_LORA=0                       # merged HF dir: no adapters to build
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.82}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-2048}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[gtopsd75] another queue holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

echo "[gtopsd75] start $(date '+%F %H:%M:%S')" >> "$LOG"
[ -f "$HF_DIR/config.json" ] || { echo "[gtopsd75] FATAL: no HF model at $HF_DIR" >> "$LOG"; exit 1; }

for ds in aime24 aime25 hmmt25; do
  tag="gtopsd75_${ds}"
  dump=$P/runs/dev_eval/${tag}_eval_0.pt
  if [ -s "$dump" ]; then
    echo "[gtopsd75] $tag already has a dump -- skipping" >> "$LOG"; continue
  fi
  conc=3
  echo "[gtopsd75] === $tag K=12 conc=$conc ($(date '+%F %H:%M:%S')) ===" >> "$LOG"
  EVAL_K=12 EVAL_CONCURRENCY="$conc" EVAL_CONFIG="$P/configs/eval_${ds}_dev.json" \
    bash $P/train_scripts/run_dev_eval.sh "$HF_DIR" "$tag" >> "$LOG" 2>&1
  rc=$?
  echo "[gtopsd75] rc=$rc for $tag" >> "$LOG"
  if [ "$rc" = "0" ] && [ -s "$dump" ]; then
    n=$(count_samples "$dump")
    if [ "$n" != "360" ]; then
      echo "[gtopsd75] FATAL: $tag produced $n samples, expected 360 (30 x K=12)" >> "$LOG"
      cleanup_orphans; exit 1
    fi
    echo "[gtopsd75] $tag verified: $n samples" >> "$LOG"
  fi
  cleanup_orphans
done

echo "[gtopsd75] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
