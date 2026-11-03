#!/usr/bin/env bash
# Supervise the Candidate-Centered HSI run on this node: relaunch after a crash
# OR a WEDGE (trainer alive, log stale -- this stack's dominant failure is the
# SGLang engines hard-deadlocking while the trainer stays up with no output).
#
# Same recovery contract as supervise_hsi.sh, with the campaign's save grid:
#   SAVE_INTERVAL=5 so cp10/cp25/cp50 all land (the report needs cp10 and cp50).
#   * before cp25 -> restart from cp0 (cheap; keeps the arm clean)
#   * at/after cp25 -> resume, but mark NON_CONTIGUOUS (no-save-optim resume)
# It never restarts because the method is doing badly -- only when machinery stops.
#
# Usage: supervise_candidate.sh <run-name> [arm] [num_rollout]
set -uo pipefail

RUN="${1:?usage: supervise_candidate.sh <run-name> [arm] [num_rollout]}"
ARM="${2:-hsi_candidate}"
NUM_ROLLOUT_TARGET="${3:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
R=/workspace/cua/repos/opsd-slime
LOG="$R/logs/$RUN.log"
RUN_DIR="$R/runs/$RUN"
STATE="/tmp/afw_sup_$RUN"
mkdir -p "$STATE"
SLOG="$STATE/supervisor.log"

WEDGE_MIN="${WEDGE_MIN:-35}"
LAUNCH_GRACE_MIN="${LAUNCH_GRACE_MIN:-30}"
CP_FIRST="${CP_FIRST:-25}"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$SLOG"; }

exec 9>"$STATE/supervisor.lock"
if ! flock -n 9; then echo "another supervisor holds $STATE/supervisor.lock" >> "$SLOG"; exit 1; fi

trainer_pids() { ps -eo pid,cmd | grep -E "[t]rain[.]py.*--save $RUN_DIR" | awk '{print $1}'; }
tracker() { cat "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | head -1 | tr -dc '0-9'; }
log_age_min() { echo $(( ($(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0)) / 60 )); }

teardown() {
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
  # RESUME_FROM is what actually makes a resume happen. `_common_v2.sh` reads it to
  # decide between `--load <checkpoint>` + `--no-load-optim --no-load-rng` and the
  # fresh path; without it the launcher falls through to
  # `elif [ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" ]` and dies with
  # "[v2] FATAL: <dir> already holds a run". That is how this supervisor spent
  # 21:51-22:00 relaunching into an immediate FATAL every three minutes on run
  # ...174810: the trainer had died at iter 39 and NOTHING could bring it back.
  #
  # Note what this resume is and is not: the campaign saves with
  # `--no-save-optim --no-save-rng`, so the checkpoint carries no Adam moments and
  # any resume restarts the optimizer. That is why the run is marked
  # NON_CONTIGUOUS -- it is a property of the harness, not of this restart.
  local resume_env=()
  if [ "$mode" = "resume" ]; then
    touch "$RUN_DIR/NON_CONTIGUOUS"
    resume_env=(RESUME_FROM="$RUN_DIR")
    say "RESUME from iter $(tracker) via RESUME_FROM=$RUN_DIR -- marking NON_CONTIGUOUS"
  else
    say "FRESH start from cp0"
  fi
  cd "$R" || return 1
  # 9>&- on the child: do not leak this supervisor's flock fd into the long-lived
  # launcher, or the lock stays held for the whole run (measured hazard).
  #
  # `env` rather than an assignment prefix: bash only treats a LITERAL `NAME=value`
  # word as an assignment prefix, so `${resume_env[@]+"${resume_env[@]}"}` before the
  # command is parsed as the command itself ("line 75: RESUME_FROM=...: command not
  # found") and the launcher then runs WITHOUT the resume variable. `env` takes the
  # expanded words as assignments, which is what this needs.
  env RUN_NAME="$RUN" NUM_ROLLOUT="$NUM_ROLLOUT_TARGET" SAVE_INTERVAL="$SAVE_INTERVAL" \
    ${resume_env[@]+"${resume_env[@]}"} \
    setsid nohup bash -c 'exec 9>&-; exec bash tasks/alfworld/scripts/launch_c64.sh "$0"' "$ARM" \
    >> "$STATE/launch_$mode.out" 2>&1 < /dev/null &
  local before
  before=$(log_age_min)
  for _ in $(seq 1 $((LAUNCH_GRACE_MIN + 10))); do
    sleep 60
    [ "$(log_age_min)" -lt 3 ] && { say "launcher is writing again"; return 0; }
  done
  say "WARNING: launcher produced no log activity within ${LAUNCH_GRACE_MIN}min"
}

say "supervising $RUN (arm=$ARM, target=$NUM_ROLLOUT_TARGET, save_every=$SAVE_INTERVAL, wedge>${WEDGE_MIN}min)"
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
