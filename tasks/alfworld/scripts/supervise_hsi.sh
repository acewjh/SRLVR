#!/usr/bin/env bash
# Supervise ONE ALFWorld v2 run on this node: relaunch it after a crash OR a WEDGE.
#
# WHY A WEDGE IS THE IMPORTANT CASE. `collapse_guard.sh` only notices a trainer
# that is GONE. This stack's dominant failure is the opposite: the SGLang engines
# hard-deadlock and the trainer stays alive with zero log output for hours. On
# 2026-10-03 the shared-base run sat like that from 04:53 to release, and the
# guard reported it only at 06:52 -- ~2h of a 3.5-day run lost to a condition the
# documentation had already recorded ("日志静默 >30min 只在进程仍存活时作为
# secondary guard"). So: stale log + live trainer = wedge, and this script acts.
#
# ONE LAUNCHER, ALWAYS. `flock` on a per-run lock makes a second instance exit
# immediately, and the loop refuses to relaunch while a launcher's log is still
# growing -- a supervisor that double-launches is worse than no supervisor.
#
# RECOVERY SEMANTICS (the fix spec, section 11). Checkpoints are written with
# `--no-save-optim`, so a resumed run does NOT continue the optimizer trajectory
# and is not a strictly equivalent result:
#   * before cp25  -> always restart from cp0 (cheap, and keeps the arm clean)
#   * at/after cp25 -> resume so the run can still finish, but record
#     NON_CONTIGUOUS so the result can never be mistaken for a continuous one
# It never restarts because the method is doing badly -- only because the
# machinery stopped.
#
# Usage: supervise_hsi.sh <run-name> [arm] [num_rollout]
#
# The arm defaults to `hsi_v3` (our method). It used to default to `hsi`, which at
# the time resolved to the retired top-128 candidate estimator; that arm is gone
# (2026-10-15) and launch_c64.sh no longer carries an `hsi` case, so the default is
# spelled the way launch_c64.sh accepts it.
set -uo pipefail

RUN="${1:?usage: supervise_hsi.sh <run-name> [arm] [num_rollout]}"
ARM="${2:-hsi_v3}"
NUM_ROLLOUT_TARGET="${3:-100}"
R=/workspace/cua/repos/opsd-slime
LOG="$R/logs/$RUN.log"
RUN_DIR="$R/runs/$RUN"
STATE="/tmp/afw_sup_$RUN"
mkdir -p "$STATE"
SLOG="$STATE/supervisor.log"

#: Minutes without a log write, with the trainer still alive, that count as a wedge.
WEDGE_MIN="${WEDGE_MIN:-35}"
#: A launcher that writes nothing for this long is itself wedged.
LAUNCH_GRACE_MIN="${LAUNCH_GRACE_MIN:-30}"
CP_FIRST="${CP_FIRST:-25}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$SLOG"; }

exec 9>"$STATE/supervisor.lock"
if ! flock -n 9; then echo "another supervisor holds $STATE/supervisor.lock" >> "$SLOG"; exit 1; fi

trainer_pids() { ps -eo pid,cmd | grep -E "[t]rain[.]py.*--save $RUN_DIR" | awk '{print $1}'; }
tracker() { cat "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | head -1 | tr -dc '0-9'; }
log_age_min() { echo $(( ($(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0)) / 60 )); }

teardown() {
  # Kill by PID only, and only this run's trainer: a pattern kill here would take
  # out the other node's or another session's processes on the shared tree.
  for p in $(trainer_pids); do kill "$p" 2>/dev/null; done
  sleep 6
  for p in $(ps -eo pid,cmd | grep -E "[.]launch_serve|sglang[:]:sched|sglang[:]:detok" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
  done
  sleep 4
  ray stop --force >/dev/null 2>&1
  sleep 6
}

launch() { # $1 = mode: fresh | resume
  local mode="$1"
  if [ "$mode" = "resume" ]; then
    # slap a permanent marker on the run: this result is NOT a continuous 100-cycle arm
    touch "$RUN_DIR/NON_CONTIGUOUS"
    say "RESUME from iter $(tracker) -- marking NON_CONTIGUOUS"
  else
    say "FRESH start from cp0"
  fi
  cd "$R" || return 1
  RUN_NAME="$RUN" NUM_ROLLOUT="$NUM_ROLLOUT_TARGET" SAVE_INTERVAL=25 \
    setsid nohup bash tasks/alfworld/scripts/launch_c64.sh "$ARM" \
    >> "$STATE/launch_$mode.out" 2>&1 < /dev/null &
  local before
  before=$(log_age_min)
  for _ in $(seq 1 $((LAUNCH_GRACE_MIN + 10))); do
    sleep 60
    # The launcher is healthy once the log starts moving again.
    [ "$(log_age_min)" -lt 3 ] && { say "launcher is writing again"; return 0; }
  done
  say "WARNING: launcher produced no log activity within ${LAUNCH_GRACE_MIN}min"
}

say "supervising $RUN (arm=$ARM, target=$NUM_ROLLOUT_TARGET, wedge>${WEDGE_MIN}min)"
for _ in $(seq 1 100000); do
  sleep 120
  pids=$(trainer_pids)
  if [ -n "$pids" ]; then
    age=$(log_age_min)
    if [ "$age" -ge "$WEDGE_MIN" ]; then
      say "WEDGE: trainer alive ($pids) but log stale ${age}min -> teardown + relaunch"
      echo "$(date '+%F %T') wedge age=${age}min" >> "$STATE/WEDGES"
      teardown
      iter=$(tracker)
      if [ -n "$iter" ] && [ "$iter" -ge "$CP_FIRST" ]; then launch resume; else launch fresh; fi
      say "relaunched after wedge; waiting before resuming supervision"
      continue
    fi
    continue
  fi
  # No trainer: crashed, finished, or reclaimed by the host.
  iter=$(tracker)
  if [ -z "$iter" ]; then
    say "trainer gone, no checkpoint -> restart from cp0"
    launch fresh
  elif [ "$iter" -ge "$((NUM_ROLLOUT_TARGET - 1))" ]; then
    say "trainer gone at iter $iter (>= target-1): run COMPLETE"
    touch "$STATE/COMPLETE"
    exit 0
  elif [ "$iter" -lt "$CP_FIRST" ]; then
    say "trainer gone at iter $iter (< cp$CP_FIRST) -> restart from cp0"
    launch fresh
  else
    say "trainer gone at iter $iter (>= cp$CP_FIRST) -> resume (NON_CONTIGUOUS)"
    launch resume
  fi
done
