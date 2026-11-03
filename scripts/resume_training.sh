#!/bin/bash
# Relaunch ONE training stage, continuing from its latest checkpoint when one
# exists and starting fresh when none does. This is the single training path for
# both the master queue and the wedge watchdog, so a watchdog-driven resume can
# never drift from the launch it is continuing.
#
# Usage: bash scripts/resume_training.sh <hpi_v2|grpo>
#
# The method configuration is re-exported here at exactly the frozen values:
#   hpi_v2 : reward_cv_hpi_v2, K=4, alpha=0.5, scale frozen from the v2 calibration,
#            clamp [0.70,1.30], lambda=1, seed 42, LoRA r64/a128, GBS 32
#   grpo   : stock GRPO, G=8, rollout-batch 4 -> GBS 32, seed 42, LoRA r64/a128
#
# A resumed tail is NOT a continuous continuation: the LoRA checkpoints are saved
# with --no-save-optim, so Adam restarts at the resume boundary. This is recorded
# in the final report and applies equally to the clean Reward-CV cp100 from 09-18.
set -uo pipefail
P=/workspace/cua/repos/opsd-slime
STAGE="${1:?usage: resume_training.sh <hpi_v2|hpi_v15|grpo>}"
LOG=$P/logs/resume_training.log
SCALE_FILE=/tmp/opsd_watchdog/hpi_v2_scale.env
FINAL_ITER=99   # save_interval 25 over 100 steps -> last checkpoint is iter_99 (cp100)

mkdir -p "$(dirname "$LOG")"

# A finished run must never be restarted from scratch, and must never be
# silently re-run: refuse loudly instead.
guard_finished() { # <run dir>
  local it
  it=$(cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")
  if [ -n "$it" ] && [ "$it" -ge "$FINAL_ITER" ]; then
    echo "[resume] $STAGE already complete (iter $it) -- nothing to do" >> "$LOG"
    exit 0
  fi
}

# Reward-CV vNext (Reward_CV_vNext_Spec.md). Two arms, identical except for the
# privileged feature and lambda; both share one run-stage body.
cv_vnext_common() { # <method> <run_dir> <log_tag>
  local method=$1 run_dir=$2 log_tag=$3
  guard_finished "$run_dir"
  export OPSD_METHOD="$method"
  export OPSD_SEED="${OPSD_SEED:-42}"
  export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
  export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"
  export OPSD_CV_VNEXT_BETA="${OPSD_CV_VNEXT_BETA:-0.9}"
  export OPSD_CV_VNEXT_B_INIT="${OPSD_CV_VNEXT_B_INIT:-0.5}"
  export OPSD_CV_VNEXT_LAMBDA_WARMUP="${OPSD_CV_VNEXT_LAMBDA_WARMUP:-4}"
  export OPSD_CV_VNEXT_LOG_EVERY="${OPSD_CV_VNEXT_LOG_EVERY:-10}"
  export OPSD_CV_VNEXT_CHUNK="${OPSD_CV_VNEXT_CHUNK:-128}"
  export OPSD_CV_VNEXT_STATE_DIR="$run_dir"
  export OPSD_CV_VNEXT_LOG="$run_dir/cv_vnext_trajectory.jsonl"
  export SAVE_DIR="$run_dir"
  export LOG_TAG="$log_tag"
  RESUME_ARGS=()
  if [ -s "$run_dir/latest_checkpointed_iteration.txt" ]; then
    RESUME_ARGS=(RESUME_FROM="$run_dir")
    echo "[resume] $STAGE CONTINUE from iter $(cat "$run_dir/latest_checkpointed_iteration.txt") $(date '+%F %H:%M:%S')" >> "$LOG"
  else
    echo "[resume] $STAGE FRESH start $(date '+%F %H:%M:%S')" >> "$LOG"
  fi
  env "${RESUME_ARGS[@]}" bash $P/train_scripts/run_math_opsd_slime_npu.sh >> "$LOG" 2>&1
  rc=$?
}

case "$STAGE" in
  cv_vnext)
    cv_vnext_common reward_cv_vnext "$P/runs/math-qwen3-1p7b-rewardcv-vnext-20260923_130456" cvvnext_100
    ;;
  reward_only_vnext)
    cv_vnext_common reward_only_vnext "$P/runs/math-qwen3-1p7b-rewardonly-vnext-20260923_184341" cvonly_100
    ;;
  # A1 clean rerun (2026-09-24). The first pair of arms resumed 3x and 1x
  # respectively and -- because --no-save-optim was in force at the time -- each
  # resume reset Adam, so the two are not the same experiment. These stages write
  # to fresh directories and run with OPSD_OPTIM_SIDECAR=1, which restores Adam
  # from `iter_N/optim_sidecar_rank*.pt`, so a resume no longer breaks the
  # pairing. The original runs are left untouched for the record.
  cv_vnext_clean)
    cv_vnext_common reward_cv_vnext "$P/runs/math-qwen3-1p7b-rewardcv-vnext-clean-sidecar-20260924_024328" cvvnext_clean_100
    ;;
  reward_only_vnext_clean)
    cv_vnext_common reward_only_vnext "$P/runs/math-qwen3-1p7b-rewardonly-vnext-clean-sidecar-20260924_042749" cvonly_clean_100
    ;;
  hpi_v15)
    # HPI-v1.5: v2 role-faithful replay + UNSIGNED |e| salience. Same frozen scale
    # as v2 -- the calibration is a property of the replay path, which is shared.
    RUN_DIR=$P/runs/math-qwen3-1p7b-rewardcv-hpi-v15-20260922_123443
    guard_finished "$RUN_DIR"
    [ -s "$SCALE_FILE" ] && . "$SCALE_FILE"
    if [ -z "${OPSD_HPI_SCALE:-}" ]; then
      echo "[resume] FATAL: OPSD_HPI_SCALE unset and $SCALE_FILE missing -- refusing to start" >> "$LOG"
      exit 1
    fi
    export OPSD_METHOD=reward_cv_hpi_v15
    export OPSD_HPI_ALPHA="${OPSD_HPI_ALPHA:-0.5}"
    export OPSD_HPI_K="${OPSD_HPI_K:-4}"
    export OPSD_HPI_LOG_EVERY="${OPSD_HPI_LOG_EVERY:-10}"
    export OPSD_CV_LAMBDA="${OPSD_CV_LAMBDA:-1.0}"
    export OPSD_SEED="${OPSD_SEED:-42}"
    export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
    export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"
    export SAVE_DIR="$RUN_DIR"
    export LOG_TAG=hpiv15_100
    RESUME_ARGS=()
    if [ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ]; then
      RESUME_ARGS=(RESUME_FROM="$RUN_DIR")
      echo "[resume] hpi_v15 CONTINUE from iter $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") scale=$OPSD_HPI_SCALE $(date '+%F %H:%M:%S')" >> "$LOG"
    else
      echo "[resume] hpi_v15 FRESH start scale=$OPSD_HPI_SCALE $(date '+%F %H:%M:%S')" >> "$LOG"
    fi
    env "${RESUME_ARGS[@]}" bash $P/train_scripts/run_math_opsd_slime_npu.sh >> "$LOG" 2>&1
    rc=$?
    ;;
  hpi_v2)
    RUN_DIR=$P/runs/math-qwen3-1p7b-rewardcv-hpi-v2-20260920_174246
    guard_finished "$RUN_DIR"
    # Frozen scale, written by the campaign that owns this run. Absent only if
    # something outside this repo's control cleared /tmp.
    [ -s "$SCALE_FILE" ] && . "$SCALE_FILE"
    if [ -z "${OPSD_HPI_SCALE:-}" ]; then
      echo "[resume] FATAL: OPSD_HPI_SCALE unset and $SCALE_FILE missing -- refusing to start" >> "$LOG"
      exit 1
    fi
    export OPSD_METHOD=reward_cv_hpi_v2
    export OPSD_HPI_ALPHA="${OPSD_HPI_ALPHA:-0.5}"
    export OPSD_HPI_K="${OPSD_HPI_K:-4}"
    export OPSD_HPI_LOG_EVERY="${OPSD_HPI_LOG_EVERY:-10}"
    export OPSD_CV_LAMBDA="${OPSD_CV_LAMBDA:-1.0}"
    export OPSD_SEED="${OPSD_SEED:-42}"
    export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
    export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"
    export SAVE_DIR="$RUN_DIR"
    export LOG_TAG=hpiv2_100
    RESUME_ARGS=()
    if [ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ]; then
      RESUME_ARGS=(RESUME_FROM="$RUN_DIR")
      echo "[resume] hpi_v2 CONTINUE from iter $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") scale=$OPSD_HPI_SCALE $(date '+%F %H:%M:%S')" >> "$LOG"
    else
      echo "[resume] hpi_v2 FRESH start scale=$OPSD_HPI_SCALE $(date '+%F %H:%M:%S')" >> "$LOG"
    fi
    env "${RESUME_ARGS[@]}" bash $P/train_scripts/run_math_opsd_slime_npu.sh >> "$LOG" 2>&1
    rc=$?
    ;;
  grpo)
    RUN_DIR=$P/runs/math-qwen3-1p7b-grpo-20260921_224239
    guard_finished "$RUN_DIR"
    export OPSD_LORA=1 OPSD_LORA_R=64 OPSD_LORA_ALPHA=128 OPSD_LORA_DROPOUT=0.0
    export OPSD_SEED="${OPSD_SEED:-42}"
    export SAVE_DIR="$RUN_DIR"
    if [ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ]; then
      echo "[resume] grpo CONTINUE from iter $(cat "$RUN_DIR/latest_checkpointed_iteration.txt") $(date '+%F %H:%M:%S')" >> "$LOG"
      RESUME_FROM="$RUN_DIR" bash $P/scripts/run_grpo.sh >> "$LOG" 2>&1
    else
      echo "[resume] grpo FRESH start $(date '+%F %H:%M:%S')" >> "$LOG"
      bash $P/scripts/run_grpo.sh >> "$LOG" 2>&1
    fi
    rc=$?
    ;;
  *)
    echo "[resume] FATAL: unknown stage '$STAGE'" >> "$LOG"; exit 1 ;;
esac
echo "[resume] $STAGE exited rc=$rc $(date '+%F %H:%M:%S')" >> "$LOG"
exit "$rc"
