#!/bin/bash
# Server A heartbeat, independent of the Claude session.
#
# The host silently reaps long-lived detached processes (CLAUDE.md), and the
# in-harness cron store has already lost this job once (`.claude/scheduled_tasks.json`
# was rewritten to {"tasks": []} while the job was supposedly persisted). So the
# restart path is a plain detached shell loop: it holds no NPU and no repo lock,
# it only relaunches the queue when the queue is gone.
#
# Safety:
#   * `flock` on its own lock makes a second supervisor impossible.
#   * It will NOT relaunch while `serverA_queue.BLOCKED` exists -- a failed stage
#     is a decision point, not something to retry in a loop.
#   * Relaunching is harmless anyway: the queue is idempotent and its own flock
#     rejects a double start.
#
# Usage: nohup setsid bash scripts/serverA_supervisor.sh > /dev/null 2>&1 &
set -uo pipefail

P=/workspace/cua/repos/opsd-slime
LOG=$P/logs/serverA_supervisor.log
# v2: the v1 lock got POISONED by fd inheritance -- the supervisor's flock
# fd (7) was inherited by the queue it launched and by that queue's children,
# so the lock stayed held after the supervisor died and no new supervisor
# could ever start. Same failure the repo already hit on the all-stages lock.
LOCK=/tmp/opsd_serverA_supervisor.v2.lock
BLOCKED=$P/logs/serverA_queue.BLOCKED
COMPLETE=$P/logs/serverA_queue.COMPLETE
BEAT=$P/logs/serverA_supervisor.heartbeat
INTERVAL="${SERVERA_SUPERVISOR_INTERVAL:-600}"

mkdir -p "$P/logs"

exec 7>"$LOCK"
if ! flock -n 7; then
  echo "[supervisor] another supervisor holds $LOCK -- exiting $(date '+%F %H:%M:%S')" >> "$LOG"
  exit 1
fi

echo "[supervisor] start pid=$$ interval=${INTERVAL}s $(date '+%F %H:%M:%S')" >> "$LOG"

while :; do
  ts=$(date '+%F %H:%M:%S')
  date '+%s' > "$BEAT"

  if [ -f "$COMPLETE" ]; then
    # Queue finished every stage and exited 0; without this the supervisor
    # relaunches it forever (DONE -> absent -> relaunch -> DONE, all no-op).
    echo "[supervisor] $ts COMPLETE marker present -- standing down" >> "$LOG"
  elif [ -f "$BLOCKED" ]; then
    echo "[supervisor] $ts BLOCKED ($(cat "$BLOCKED")) -- not restarting" >> "$LOG"
  elif pgrep -f 'run_serverA_queue[.]sh' >/dev/null 2>&1; then
    echo "[supervisor] $ts queue alive" >> "$LOG"
  else
    echo "[supervisor] $ts queue ABSENT -- relaunching" >> "$LOG"
    # 7>&- : do NOT let the launched queue inherit the supervisor lock fd
    ( exec 7>&-; nohup setsid bash $P/scripts/run_serverA_queue.sh > /dev/null 2>&1 & )
    disown 2>/dev/null || true
    sleep 30
  fi

  sleep "$INTERVAL"
done
