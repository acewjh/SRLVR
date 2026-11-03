#!/bin/bash
# Wedge watchdog — turns the silent SGLang wedge into a self-healing resume.
#
# WHY THIS EXISTS
# ---------------
# Training on this node wedged silently three times (2026-09-18 2h47m, 09-19 2h54m,
# 09-20 2h55m): every process alive, NPU memory held, zero log output, no error.
# The recovery that works is a manual kill + `RESUME_FROM=<run dir>` relaunch, which
# needs a human (or Claude) awake. The machine now runs unattended for ~2 days, so
# the recovery is automated here instead.
#
# Root-cause work (2026-09-20) added bounded waits so the *common* stall should now
# surface as a retry or a raised error rather than a hang:
#   - slime/utils/http_utils.py: bounded client timeout (SLIME_HTTP_TIMEOUT_S)
#   - methods/math_rollout.py: _abort bounded + direct engine URLs, not the router
#   - update_weight_from_distributed.py: engines always resumed (try/finally)
# This watchdog is the backstop for the stalls those changes do not cover -- it is
# the only guard that catches *any* hang of unknown origin.
#
# CONTRACT
# --------
# It watches whatever the master queue points it at via $DIR/active.env:
#     WATCH_TAG=<label>
#     WATCH_LOG=<training log file>
#     WATCH_RUN_DIR=<save dir, used for the resume target>
#     WATCH_RESUME_CMD=<command that relaunches training with RESUME_FROM set>
# When active.env is absent the watchdog idles (evals run in the same window and
# must never be mistaken for a stalled training run).
#
# A wedge is: the training log has not been written for WATCH_STALE_MIN minutes
# WHILE a train.py process is still alive. Both conditions are required -- a
# finished or crashed run has no live process and is left alone.
#
# Usage: nohup bash scripts/wedge_watchdog.sh > /dev/null 2>&1 &
set -uo pipefail
P=/workspace/cua/repos/opsd-slime
DIR=/tmp/opsd_watchdog
STEP_FILE=$DIR/active.env
LOG=$P/logs/wedge_watchdog.log
LOCK=$DIR/watchdog.lock
POLL_S="${WATCH_POLL_S:-120}"
STALE_MIN="${WATCH_STALE_MIN:-20}"
MAX_RESTARTS="${WATCH_MAX_RESTARTS:-8}"

mkdir -p "$DIR"
exec 5>"$LOCK"
if ! flock -n 5; then
  echo "[watchdog] another watchdog holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

log() { echo "[watchdog] $(date '+%F %H:%M:%S') $*" >> "$LOG"; }

# Process liveness. Bracket the first character so the pattern cannot match this
# script's own command line (pkill/pgrep -f self-match killed a shell here once).
train_pids() { pgrep -f '[s]lime-ascend/train.py' 2>/dev/null; }

# Snapshot the evidence before destroying it, so a post-mortem is still possible.
dump_evidence() {
  local tag=$1 stamp dir
  stamp=$(date '+%Y%m%d_%H%M%S')
  dir=$P/logs/stall_${stamp}_${tag}
  mkdir -p "$dir"
  ps -eo pid,stat,pcpu,etime,comm --sort=-pcpu 2>/dev/null | head -30 > "$dir/ps.txt"
  npu-smi info > "$dir/npu_smi.txt" 2>&1
  tail -60 "$STEP_FILE" > "$dir/active_env.txt" 2>&1
  if [ -n "${WATCH_LOG:-}" ] && [ -f "$WATCH_LOG" ]; then
    tail -60 "$WATCH_LOG" > "$dir/last_60_lines.txt" 2>&1
    ls -la "$WATCH_LOG" > "$dir/log_stat.txt" 2>&1
  fi
  log "evidence dumped to $dir"
}

# Kill the training run and every SGLang/Ray process it owns, then wait for the
# NPUs to actually drain. Explicit PIDs for the trainer; bracketed patterns for
# the engines (these patterns must never appear in this script's own cmdline).
recover_clean() {
  local pid
  for pid in $(train_pids); do
    log "SIGTERM train.py pid=$pid"
    kill "$pid" 2>/dev/null
  done
  sleep 20
  for pid in $(train_pids); do
    log "SIGKILL train.py pid=$pid"
    kill -9 "$pid" 2>/dev/null
  done
  pkill -9 -f 'sglang[:]:sched'      2>/dev/null
  pkill -9 -f 'sglang[:]:detok'      2>/dev/null
  pkill -9 -f 'sglang[.]launch_serve' 2>/dev/null
  sleep 5
  ray stop --force >/dev/null 2>&1
  sleep 5
  # NPUs must be free before a relaunch, or the new engines fail to allocate.
  local busy
  for _ in $(seq 1 30); do
    busy=$(npu-smi info 2>/dev/null | grep -c 'No running processes found')
    [ "$busy" -ge 8 ] && break
    sleep 10
  done
  log "NPUs free: $busy/8"
}

log "watchdog start (poll=${POLL_S}s stale=${STALE_MIN}min max_restarts=$MAX_RESTARTS)"

restarts=0
while :; do
  sleep "$POLL_S"

  # No active training stage (eval running, or nothing scheduled): idle.
  [ -f "$STEP_FILE" ] || continue
  # shellcheck disable=SC1090
  . "$STEP_FILE"

  [ -n "${WATCH_LOG:-}" ] || continue
  [ -n "${WATCH_RESUME_CMD:-}" ] || continue

  now=$(date +%s)
  mtime=$(stat -c %Y "$WATCH_LOG" 2>/dev/null || echo 0)
  age_min=$(( (now - mtime) / 60 ))

  if [ -z "$(train_pids)" ]; then
    # Trainer exited on its own: finished, or died with a real error. Either way
    # the master queue owns the next move -- do not interfere, do not count a
    # restart.
    continue
  fi

  if [ "$age_min" -lt "$STALE_MIN" ]; then
    continue
  fi

  restarts=$((restarts + 1))
  log "WEDGE DETECTED tag=${WATCH_TAG:-?} log_stale=${age_min}min restarts=$restarts/$MAX_RESTARTS"
  dump_evidence "${WATCH_TAG:-unknown}"

  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    log "FATAL: exceeded WATCH_MAX_RESTARTS=$MAX_RESTARTS -- giving up, needs a human"
    exit 1
  fi

  recover_clean

  if [ ! -s "$WATCH_RUN_DIR/latest_checkpointed_iteration.txt" ]; then
    log "FATAL: no checkpoint in $WATCH_RUN_DIR -- cannot resume; needs a human"
    exit 1
  fi
  log "resuming from iter $(cat "$WATCH_RUN_DIR/latest_checkpointed_iteration.txt")"
  # Detach so the watchdog's own loop survives the child, and so the resume is not
  # a child of the loop (a relaunch must outlive this iteration).
  nohup bash -c "$WATCH_RESUME_CMD" >> "$LOG" 2>&1 &
  log "resume launched pid=$!"
  sleep 60
done
