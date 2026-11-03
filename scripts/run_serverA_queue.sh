#!/bin/bash
# Server A master queue: A1 (clean two-arm training + formal eval) -> A2 (GRPO
# cp75) -> A3 (GT-OPSD cp100 + Base on the unified evaluator).
#
# Design constraints this obeys (CLAUDE.md + RUN_NOTES.md):
#   * ONE launcher: a flock makes a second instance impossible, and every stage
#     is idempotent, so a restart resumes rather than redoing work.
#   * A failed stage STOPS the queue and writes a BLOCKED marker instead of
#     retrying forever. A cron heartbeat restarts the queue, but it will refuse
#     to start while BLOCKED exists -- the failure is surfaced, not buried.
#   * The stage list is re-read every pass, so stages can be appended at runtime.
#   * No stage preempts anything: before a stage runs, the queue waits for the
#     NPUs to be genuinely idle.
#
# Usage: nohup bash scripts/run_serverA_queue.sh > /dev/null 2>&1 &
set -uo pipefail

P=/workspace/cua/repos/opsd-slime
LOG=$P/logs/serverA_queue.log
LIST=$P/scripts/serverA_stages.list
LOCK=/tmp/opsd_serverA_queue.lock
BLOCKED=$P/logs/serverA_queue.BLOCKED

exec 8>"$LOCK"
if ! flock -n 8; then
  echo "[queue] another queue holds $LOCK -- exiting $(date '+%F %H:%M:%S')" >> "$LOG"
  exit 1
fi

if [ -f "$BLOCKED" ]; then
  echo "[queue] REFUSING to start: $BLOCKED exists ($(cat "$BLOCKED"))" >> "$LOG"
  exit 1
fi

mkdir -p "$P/logs"
echo "[queue] start $(date '+%F %H:%M:%S')" >> "$LOG"

# Anchored patterns: a bare substring would also match an unrelated shell whose
# command line merely mentions the name (the documented 11 h self-match stall).
busy() {
  pgrep -f 'slime-ascend/train[.]py' >/dev/null 2>&1 && return 0
  pgrep -f 'sglang[.]launch_serve'   >/dev/null 2>&1 && return 0
  return 1
}
# `npu-smi | grep -c 'No running processes found'` == 8 is the same idiom the
# wedge watchdog and the campaign scripts use.
idle_npus() {
  [ "$(npu-smi info 2>/dev/null | grep -c 'No running processes found')" -ge 8 ]
}
wait_for_idle() {
  local waited=0
  while busy || ! idle_npus; do
    sleep 60; waited=$((waited + 1))
    if [ $((waited % 15)) -eq 0 ]; then
      echo "[queue] waiting for idle NPUs (${waited}min) $(date '+%F %H:%M:%S')" >> "$LOG"
    fi
  done
  # two consecutive clean checks, 60 s apart
  sleep 60
}

PREV=""
while :; do
  CUR=$(grep -vE '^\s*(#|$)' "$LIST" 2>/dev/null | tr '\n' ' ')
  if [ "$CUR" = "$PREV" ] && [ -n "$PREV" ]; then
    echo "[queue] stage list unchanged and every stage ran -- DONE $(date '+%F %H:%M:%S')" >> "$LOG"
    break
  fi
  PREV="$CUR"

  while read -r stage; do
    case "$stage" in ''|'#'*) continue ;; esac
    wait_for_idle
    echo "[queue] running stage=$stage $(date '+%F %H:%M:%S')" >> "$LOG"
    bash $P/scripts/serverA_stage.sh "$stage"
    rc=$?
    if [ "$rc" != "0" ]; then
      echo "[queue] stage=$stage FAILED rc=$rc -- writing $BLOCKED and stopping" >> "$LOG"
      echo "$stage rc=$rc at $(date '+%F %H:%M:%S')" > "$BLOCKED"
      exit 1
    fi
    echo "[queue] stage=$stage ok $(date '+%F %H:%M:%S')" >> "$LOG"
  done < "$LIST"
done
