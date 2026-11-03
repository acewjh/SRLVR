#!/usr/bin/env bash
# Single-instance supervisor for the SRLVR-v4 1.7B run.
#
# Restarts the training launcher from the latest complete Megatron checkpoint
# after a crash, and stops only when checkpoint `NUM_ROLLOUT-1` exists. The
# launcher's own `/tmp/opsd_<run>.lock` is the liveness signal: the training
# process inherits it, so a stale pid file cannot make a dead run look alive.
#
# Usage:
#   RUN_NAME=alfworld-qwen3-1p7b-srlvr_v4-s42-r50-n32-<stamp> \
#     setsid nohup bash tasks/alfworld/scripts/supervise_srlvr_v4.sh \
#     >> <run_root>/supervisor.out 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$HERE/../../.." && pwd)

export RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
export RUN_ROOT="${RUN_ROOT:-$PROJ/runs/$RUN_NAME}"
export SAVE_DIR="${SAVE_DIR:-$RUN_ROOT/checkpoints}"
export DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-$RUN_ROOT/rollouts}"
export LOG="${LOG:-$RUN_ROOT/train.log}"
# SRLVR_REVISION_NOTE.md (2026-10-27): 50 batches, 4096-token response cap,
# 8 actor + 8 rollout colocated, one TP1 engine per card.  The v4 full-vocab
# loss keeps actor TP=1; MODEL_PROVIDER_PATH is the gradient-safe provider
# (frozen embedding + reentrant recompute otherwise gives grad_norm=0).
export NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export ROLLOUT_BS="${ROLLOUT_BS:-32}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export ACTOR_GPUS="${ACTOR_GPUS:-8}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
export TP=1 PP=1 COLOCATE=1 ROLLOUT_GPUS_PER_ENGINE=1
export MODEL_PROVIDER_PATH="${MODEL_PROVIDER_PATH:-tasks.alfworld.memory_provider.custom_model_provider}"
# Action-list arm: shared prompt default stays off for the external baseline;
# this run explicitly turns the current-state admissible actions on.
export ALFWORLD_SHOW_ADMISSIBLE=1
export SGLANG_PREFIX_CACHE=1
export OPTIMIZER_CPU_OFFLOAD=0
export SRLVR_V4_COMPACT_FUTURE="${SRLVR_V4_COMPACT_FUTURE:-1}"
export OPSD_SRLVR_SELECTIVE_LOGITS="${OPSD_SRLVR_SELECTIVE_LOGITS:-1}"
export SRLVR_V4_TOKEN_CHUNK="${SRLVR_V4_TOKEN_CHUNK:-128}"
export OPSD_SRLVR_SELECTIVE_CHUNK="${OPSD_SRLVR_SELECTIVE_CHUNK:-128}"

SUP_LOG="$RUN_ROOT/supervisor.log"
STATUS="$RUN_ROOT/supervisor_status.txt"
TRAIN_LOCK="/tmp/opsd_${RUN_NAME}.lock"
mkdir -p "$RUN_ROOT" "$SAVE_DIR" "$DEBUG_ROLLOUT_DIR"

# One supervisor per run. fd 9 must NOT leak into the training child: a long
# training run would otherwise keep the supervisor lock forever.
exec 9>"$RUN_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "supervisor already running for $RUN_NAME" >> "$SUP_LOG"
  exit 0
fi

latest_checkpoint() {
  cat "$SAVE_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo -1
}

training_alive() {
  # The training launcher owns /tmp/opsd_<run>.lock; the child train.py inherits
  # it. A process-name check is the pre-lock fallback (Ray startup window).
  if ! flock -n "$TRAIN_LOCK" true 2>/dev/null; then
    return 0
  fi
  if pgrep -f "run_alfworld_v2_1p7b[.]sh" >/dev/null 2>&1; then
    return 0
  fi
  if pgrep -f "train[.]py .*--save ${SAVE_DIR}" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

run_training() {
  (
    # Never let the training child hold the supervisor's flock.
    exec 9>&-
    export ALLOW_EXISTING_RUN_ROOT=1
    if [ -s "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
      export RESUME_FROM="$SAVE_DIR"
    else
      unset RESUME_FROM
    fi
    bash "$HERE/run_alfworld_srlvr_v4_1p7b.sh"
  )
}

attempt=0
echo "=== supervisor start $(date '+%F %T') run=$RUN_NAME rounds=$NUM_ROLLOUT ===" >> "$SUP_LOG"
while :; do
  latest=$(latest_checkpoint)
  if [ "$latest" -ge $((NUM_ROLLOUT - 1)) ] 2>/dev/null; then
    echo "$(date '+%F %T') DONE latest=$latest" >> "$STATUS"
    echo "=== supervisor done $(date '+%F %T') latest=$latest ===" >> "$SUP_LOG"
    exit 0
  fi

  if training_alive; then
    echo "$(date '+%F %T') RUNNING latest=$latest attempt=$attempt" > "$STATUS"
    sleep 60
    continue
  fi

  attempt=$((attempt + 1))
  echo "$(date '+%F %T') START attempt=$attempt latest=$latest" >> "$STATUS"
  echo "=== attempt $attempt start $(date '+%F %T') latest=$latest ===" >> "$SUP_LOG"
  run_training >> "$SUP_LOG" 2>&1
  rc=$?
  latest_after=$(latest_checkpoint)
  echo "$(date '+%F %T') EXIT rc=$rc attempt=$attempt latest=$latest_after" >> "$STATUS"
  echo "=== attempt $attempt exit rc=$rc $(date '+%F %T') latest=$latest_after ===" >> "$SUP_LOG"

  if [ "$latest_after" -ge $((NUM_ROLLOUT - 1)) ] 2>/dev/null; then
    echo "$(date '+%F %T') DONE latest=$latest_after" >> "$STATUS"
    echo "=== supervisor done $(date '+%F %T') latest=$latest_after ===" >> "$SUP_LOG"
    exit 0
  fi
  if [ "$rc" -eq 0 ]; then
    sleep 15
  elif [ "$attempt" -lt 5 ]; then
    sleep 30
  else
    sleep 120
  fi
done
