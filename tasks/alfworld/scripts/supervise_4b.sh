#!/bin/bash
# Keep ONE ALFWorld 4B launcher alive across Claude-session exits.
#
# Why this exists: a cp25 arm runs ~3 h (HSI) / ~9 h (GRPO), and the previous
# session proved that `setsid nohup ... &` from a tool call is NOT enough -- the
# GRPO smoke was killed mid-save when the session went away. A supervisor that
# only relaunches is cheap; losing 9 h of rollout is not.
#
# It relaunches with RESUME_FROM once a checkpoint exists, so a mid-run kill
# continues from the last saved iteration instead of silently restarting at 0
# (the launcher itself refuses to touch a non-empty SAVE_DIR without RESUME_FROM).
#
# Single instance: flock on the supervisor's own lock. The launcher's own lock
# (`/tmp/opsd_<RUN_NAME>.lock`, per-node /tmp) is used as the liveness probe --
# `flock -n` succeeding means nobody holds it, i.e. the launcher is gone.
#
# Usage (on the node that will run the arm):
#   RUN_NAME=alfworld_v2_grpo_4b SAVE_DIR=... LOG=... \
#   LAUNCHER=tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh \
#   nohup setsid bash tasks/alfworld/scripts/supervise_4b.sh > /dev/null 2>&1 &
set -uo pipefail

PROJ=/workspace/cua/repos/opsd-slime
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
LAUNCHER="${LAUNCHER:?LAUNCHER required (path under $PROJ)}"
LOG="${LOG:-$PROJ/logs/${RUN_NAME}.log}"
SAVE_DIR="${SAVE_DIR:-$PROJ/runs/$RUN_NAME}"
INTERVAL="${SUPERVISE_INTERVAL:-120}"
MAX_RELAUNCH="${SUPERVISE_MAX_RELAUNCH:-5}"

SELF_LOCK="/tmp/opsd_supervise_${RUN_NAME}.lock"
RUN_LOCK="/tmp/opsd_${RUN_NAME}.lock"
SLOG="$PROJ/logs/${RUN_NAME}.supervisor.log"

log() { echo "[supervise $RUN_NAME] $(date '+%F %H:%M:%S') $*" >> "$SLOG"; }

exec 9>"$SELF_LOCK"
if ! flock -n 9; then log "another supervisor holds $SELF_LOCK -- exit"; exit 1; fi
# 9>&- keeps the launcher from inheriting this lock (the poisoning bug the repo
# already hit once: an inherited flock fd outlives its owner and wedges restart).

relaunch_count=0
log "start pid=$$ interval=${INTERVAL}s launcher=$LAUNCHER"

while :; do
  # Liveness: can we take the run lock? If yes, no launcher holds it.
  if ( exec 9>&-; exec 8>"$RUN_LOCK"; flock -n 8 ) 2>/dev/null; then
    # Dead. Only rc=0 means "finished" -- an earlier version stood down on ANY
    # exit marker, so the HSI arm's cap=1024 OOM (rc=1) silently ended the run
    # instead of being retried. A stale marker from a previous attempt is fine:
    # the decision keys off the *current* liveness probe, not the marker alone.
    last_exit=$(grep -o 'exited rc=[0-9]*' "$LOG" 2>/dev/null | tail -1)
    if [ "$last_exit" = "exited rc=0" ]; then
      log "clean exit ($last_exit) -- standing down"
      break
    fi
    if [ "$relaunch_count" -ge "$MAX_RELAUNCH" ]; then
      log "dead (last='${last_exit:-none}') and relaunch budget ($MAX_RELAUNCH) exhausted -- standing down"
      break
    fi
    relaunch_count=$(( relaunch_count + 1 ))
    # Empty means "no resume": the launcher treats an unset/empty RESUME_FROM as
    # a fresh start and applies its own refuse-to-clobber guard.
    RESUME_FROM_V=""
    if [ -s "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
      RESUME_FROM_V="$SAVE_DIR"
      log "checkpoint at iter $(cat "$SAVE_DIR/latest_checkpointed_iteration.txt") -- resuming"
    else
      log "no checkpoint yet -- restarting from scratch"
    fi
    ( exec 9>&-; cd "$PROJ" && \
        RESUME_FROM="$RESUME_FROM_V" RUN_NAME="$RUN_NAME" \
        SAVE_DIR="$SAVE_DIR" LOG="$LOG" \
        nohup setsid bash "$PROJ/$LAUNCHER" >> "/tmp/${RUN_NAME}.stdout.log" 2>&1 < /dev/null & )
    sleep 30
    continue
  fi

  sleep "$INTERVAL"
done
