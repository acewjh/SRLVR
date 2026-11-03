#!/usr/bin/env bash
# Keep a finished-training run's evaluation moving until both checkpoints are scored.
#
# The host silently reaps long-lived detached processes (this is what ended the
# SRLVR-v4 supervisor on 2026-10-28 at 19:27: the supervisor vanished, its
# launcher's EXIT trap ran `ray stop`, and nothing restarted it). A training run
# that has already produced TRAIN_OK must not lose its evaluation to that.
#
# Idempotent by construction: the supervisor skips an iteration whose
# EVAL_OK_cp<N> marker exists, so restarting it never re-scores a finished
# checkpoint. The watchdog also refuses to start a second copy while any eval or
# trainer process is alive, so an orphaned eval child is never doubled.
#
# Usage: RUN_NAME=<run> ARM=<grpo|srlvr_v4> setsid nohup bash keep_eval_alive.sh &
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$HERE/../../.." && pwd)
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
ARM_SEL="${ARM:?ARM must be set}"
RUN_ROOT="${RUN_ROOT:-$PROJ/runs/$RUN_NAME}"
KEEP_LOG="$RUN_ROOT/keeper.log"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/keeper.lock"
if ! flock -n 9; then
  echo "$(date '+%F %T') keeper already running" >> "$KEEP_LOG"
  exit 0
fi

say() { echo "[$(date '+%F %T')] $*" >> "$KEEP_LOG"; }

done_markers() {
  local n=0
  for m in "$RUN_ROOT"/EVAL_OK_cp*; do [ -e "$m" ] && n=$((n + 1)); done
  echo "$n"
}

busy() {
  # A supervisor holding its own lock is alive and needs nothing from us.
  if ! flock -n "$RUN_ROOT/supervisor.lock" true 2>/dev/null; then return 0; fi
  # An orphaned eval or trainer must be allowed to finish before we restart.
  pgrep -f "eval_alfworld_ckpt[.]sh" >/dev/null 2>&1 && return 0
  # Another node may be scoring an iteration for this run (EVAL_INFLIGHT_cpN);
  # starting our own copy would duplicate hours of GPU time.
  for f in "$RUN_ROOT"/EVAL_INFLIGHT_cp*; do [ -e "$f" ] && return 0; done
  pgrep -f "train[.]py .*--save $RUN_ROOT" >/dev/null 2>&1 && return 0
  return 1
}

say "keeper start arm=$ARM_SEL targets=$(ls "$RUN_ROOT"/EVAL_OK_cp* 2>/dev/null | wc -l) done"
while :; do
  if [ ! -e "$RUN_ROOT/TRAIN_OK" ]; then
    say "no TRAIN_OK yet; waiting"
    sleep "$SLEEP_SECONDS"
    continue
  fi
  # Two scored checkpoints (cp25, cp50) completes this job.
  if [ "$(done_markers)" -ge 2 ]; then
    say "both checkpoints scored; keeper exiting"
    exit 0
  fi
  if busy; then
    sleep "$SLEEP_SECONDS"
    continue
  fi
  say "supervisor is gone and nothing is running; relaunching"
  ( exec 9>&-
    cd "$PROJ" || exit 1
    ARM="$ARM_SEL" RUN_NAME="$RUN_NAME" setsid nohup \
      bash "$HERE/supervise_full_rl.sh" >> "$RUN_ROOT/supervisor.out" 2>&1 < /dev/null &
  )
  sleep "$SLEEP_SECONDS"
done
