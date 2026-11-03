#!/bin/bash
# Formal eval queue for Reward-CV vNext (Reward_CV_vNext_Spec.md §8/§9).
#
#   cp100 x {aime24, aime25, hmmt25} for BOTH arms, at the frozen house protocol.
#   cp100 is PRE-REGISTERED (spec §8), so there is no cheap selector phase.
#
# Ordering is fixed in advance to form PAIRED results as early as possible: within
# each dataset, the new method runs first and then its reward-only control, so a
# single benchmark's pair is complete before moving on. Only completed pairs are
# interpreted; a lone cell is never reported as a conclusion (spec §10).
#
# Protocol is unchanged and NOT tunable here: K=12, 30 problems, thinking=True,
# T=1.0, top_p=0.95, top_k=-1, min_p=0, presence_penalty=0, max_new_tokens=38912,
# context 40960, repaired official grader, all 8 NPUs.
#
# Every cell is hard-verified to 30*K samples; a mismatch aborts the queue.
# Cells with an existing dump are skipped, so this is restartable.
#
# Usage: nohup bash scripts/eval_queue_cv_vnext.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"  # all-NPU eval default
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
EVAL_CELL_LOG_TAG=cvnext-eval

P=/workspace/cua/repos/opsd-slime
LOG=${CVVNEXT_EVAL_LOG:-$P/logs/eval_queue_cv_vnext.log}
LOCK=${CVVNEXT_EVAL_LOCK:-/tmp/opsd_eval_cv_vnext.lock}

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu

export EVAL_LORA=1
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-$OPSD_EVAL_MEM_FRACTION}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-$OPSD_EVAL_CHUNKED_PREFILL}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[cvnext-eval] another queue holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

# Overridable so the A1 clean rerun reuses this exact, frozen protocol against
# its own run directories instead of forking a second copy of the queue.
CV_RUN=${CV_RUN:-$P/runs/math-qwen3-1p7b-rewardcv-vnext-20260923_130456}
RO_RUN=${RO_RUN:-$P/runs/math-qwen3-1p7b-rewardonly-vnext-20260923_184341}
CV_PREFIX=${CV_PREFIX:-cvvnext}
RO_PREFIX=${RO_PREFIX:-cvonly}

echo "[cvnext-eval] queue start $(date '+%F %H:%M:%S')" >> "$LOG"
for rd in "$CV_RUN" "$RO_RUN"; do
  [ -s "$rd/latest_checkpointed_iteration.txt" ] || {
    echo "[cvnext-eval] FATAL: no checkpoints in $rd" >> "$LOG"; exit 1; }
done

# Paired order: aime24 pair, then aime25 pair, then hmmt25 pair.
# concurrency matches the v1.5 formal cells (2/3/3) -- this stack's batch geometry
# has changed results before, so it is inherited rather than re-tuned.
RUN_DIR="$CV_RUN" run_eval_cell cp100 aime24 eval_aime24_dev.json 12 "$CV_PREFIX" 2
RUN_DIR="$RO_RUN" run_eval_cell cp100 aime24 eval_aime24_dev.json 12 "$RO_PREFIX" 2
RUN_DIR="$CV_RUN" run_eval_cell cp100 aime25 eval_aime25_dev.json 12 "$CV_PREFIX" 3
RUN_DIR="$RO_RUN" run_eval_cell cp100 aime25 eval_aime25_dev.json 12 "$RO_PREFIX" 3
RUN_DIR="$CV_RUN" run_eval_cell cp100 hmmt25 eval_hmmt25_dev.json 12 "$CV_PREFIX" 3
RUN_DIR="$RO_RUN" run_eval_cell cp100 hmmt25 eval_hmmt25_dev.json 12 "$RO_PREFIX" 3

echo "[cvnext-eval] queue DONE $(date '+%F %H:%M:%S')" >> "$LOG"
