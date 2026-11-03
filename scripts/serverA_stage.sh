#!/bin/bash
# One stage of the Server A queue. Every stage is idempotent and skips work that
# is already on disk, so re-running the queue after any interruption is safe.
#
# Keeping stages behind this dispatcher (instead of inlining them in the queue)
# means the queue loop only ever reads `serverA_stages.list` -- new stages can be
# appended while the queue is running without editing a script bash is executing
# (bash reads a running script by byte offset; editing one mid-run is a known
# way to corrupt it).
#
# Usage: bash scripts/serverA_stage.sh <stage>
set -uo pipefail

P=/workspace/cua/repos/opsd-slime
LOG=$P/logs/serverA_queue.log
STAGE="${1:?usage: serverA_stage.sh <stage>}"

CV_CLEAN=$P/runs/math-qwen3-1p7b-rewardcv-vnext-clean-sidecar-20260924_024328
RO_CLEAN=$P/runs/math-qwen3-1p7b-rewardonly-vnext-clean-sidecar-20260924_042749

echo "[stage] BEGIN $STAGE $(date '+%F %H:%M:%S')" >> "$LOG"

# A training stage and an eval stage both want all 8 NPUs; the queue serializes
# them, and ray must be fully torn down between a training run and an eval or
# every cell hangs on the "active python is not the slime-npu env" gate.
case "$STAGE" in
  a1_arm1)
    bash $P/scripts/resume_training.sh cv_vnext_clean
    ;;
  a1_arm2)
    bash $P/scripts/resume_training.sh reward_only_vnext_clean
    ;;
  a1_eval)
    ray stop --force >/dev/null 2>&1; sleep 10
    CV_RUN="$CV_CLEAN" CV_PREFIX=cvvnext_clean \
    RO_RUN="$RO_CLEAN" RO_PREFIX=cvonly_clean \
    CVVNEXT_EVAL_LOG=$P/logs/eval_queue_cv_vnext_clean.log \
    CVVNEXT_EVAL_LOCK=/tmp/opsd_eval_cvnext_clean.lock \
      bash $P/scripts/eval_queue_cv_vnext.sh
    ;;
  # Retry pass for the A1 eval. Byte-identical to a1_eval on purpose: the eval
  # queue skips cells whose dump already exists, so this only re-runs the cells
  # that produced nothing. It exists because a cell can die mid-run (observed
  # 2026-09-24: `cvvnext_clean_cp100_aime25` lost its SGLangEngine actor to a
  # SYSTEM_ERROR at ~90 min, rc=1, no dump) and the queue's contract is to
  # WARN-and-continue rather than stall the whole run -- so without an explicit
  # second pass the missing cell would simply never be measured.
  a1_eval_retry)
    ray stop --force >/dev/null 2>&1; sleep 10
    CV_RUN="$CV_CLEAN" CV_PREFIX=cvvnext_clean \
    RO_RUN="$RO_CLEAN" RO_PREFIX=cvonly_clean \
    CVVNEXT_EVAL_LOG=$P/logs/eval_queue_cv_vnext_clean.log \
    CVVNEXT_EVAL_LOCK=/tmp/opsd_eval_cvnext_clean.lock \
      bash $P/scripts/eval_queue_cv_vnext.sh
    ;;
  a2_grpo_cp75)
    # RE-ENABLED BY USER DIRECTIVE 2026-09-25 ("GRPO cp75 ... 是需要放进论文里面的").
    # It had been cut on 2026-09-24 to save 7.5 h; the user now wants this row in
    # the paper, so it runs.
    #
    # CONFOUND, must travel with every number this produces: the GRPO cp75
    # checkpoint (`runs/math-qwen3-1p7b-grpo-20260921_224239/iter_0000074`) comes from a
    # reset-Adam resume after the 2026-09-21 23:39 stall -- `scripts/run_remaining_evals.sh`
    # documents it as "scientific confound". Only GRPO cp25 is a clean,
    # continuous arm. cp75 is therefore not like-for-like against a
    # continuously-trained arm and must be footnoted as such.
    ray stop --force >/dev/null 2>&1; sleep 10
    bash $P/scripts/eval_queue_grpo_cp75.sh
    ;;
  a3_opsd_unified)
    ray stop --force >/dev/null 2>&1; sleep 10
    bash $P/scripts/eval_opsd_unified.sh
    ;;
  *)
    echo "[stage] FATAL: unknown stage '$STAGE'" >> "$LOG"; exit 2 ;;
esac
rc=$?

echo "[stage] END $STAGE rc=$rc $(date '+%F %H:%M:%S')" >> "$LOG"
exit "$rc"
