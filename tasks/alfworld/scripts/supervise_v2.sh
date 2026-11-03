#!/usr/bin/env bash
# Supervise one V2 arm: relaunch after a crash, and after a WEDGE (trainer alive,
# log stale -- this stack's dominant failure is the SGLang engines hard-deadlocking
# while the trainer stays up with no output).
#
# The 2026-10-04 protocol's restart contract, which differs from the retired
# supervisor's in three ways:
#
#   * SAVE_INTERVAL is 25 (not 5), so the first checkpoint is cp25 = iter_24. Any
#     crash before that is a restart from cp0 with nothing to resume from -- which
#     is why the cheap argv smoke runs BEFORE the real launch rather than after.
#   * A resume is only ever taken from a NEW-protocol checkpoint, and only after
#     `resume_guard.py` has verified that `trainer_state.json` is readable and the
#     adapter safetensors open. A half-written checkpoint resumed blindly is how a
#     run silently trains from the wrong weights.
#   * When the trainer exits at the target, this runs the Plan-B evaluation: the
#     trainer holds all 8 NPUs, so per-checkpoint eval DURING training is not
#     possible without stopping it. Task book section 13 forbids
#     `stop train -> eval -> resume train`, so cp25/50/75/100 are scored in
#     sequence AFTER the run finishes.
#
# Usage: supervise_v2.sh <run-name> <hsi_v2|grpo_v2> [method-slug] [num-rollout]
set -uo pipefail

RUN="${1:?usage: supervise_v2.sh <run-name> <hsi_v2|grpo_v2> [method-slug] [num-rollout]}"
ARM_CASE="${2:?usage: supervise_v2.sh <run-name> <hsi_v2|grpo_v2> [method-slug] [num-rollout]}"
METHOD_SLUG="${3:-$ARM_CASE}"
NUM_ROLLOUT_TARGET="${4:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
R=/workspace/cua/repos/opsd-slime
LOG="$R/logs/$RUN.log"
RUN_DIR="$R/runs/$RUN"
STATE="/tmp/afw_sup_$RUN"
mkdir -p "$STATE"
SLOG="$STATE/supervisor.log"

WEDGE_MIN="${WEDGE_MIN:-45}"
LAUNCH_GRACE_MIN="${LAUNCH_GRACE_MIN:-40}"
# Consecutive restarts that reach the launcher but never produce a new checkpoint.
# This stack's documented failure mode is an SGLang hard-deadlock every few hours
# (see the campaign notes); with a 25-cycle save grid, a crash that always lands
# before cp25 would restart forever and burn NPU-time silently. After this many
# fruitless restarts the supervisor stops and leaves a marker for a human/next
# cycle instead of looping.
MAX_FRUITLESS_RESTARTS="${MAX_FRUITLESS_RESTARTS:-5}"
# First checkpoint on this grid: the earliest iter a resume may use.
CP_FIRST="${CP_FIRST:-25}"
# 0-based rollout ids of the four checkpoints the campaign reports.
EVAL_ITERS=(${EVAL_ITERS:-24 49 74 99})

say() { echo "[$(date '+%F %T')] $*" | tee -a "$SLOG"; }

exec 9>"$STATE/supervisor.lock"
if ! flock -n 9; then echo "another supervisor holds $STATE/supervisor.lock" >> "$SLOG"; exit 1; fi

trainer_pids() { ps -eo pid,cmd | grep -E "[t]rain[.]py.*--save $RUN_DIR" | awk '{print $1}'; }
fruitless_restarts() { cat "$STATE/FRUITLESS" 2>/dev/null | tr -dc '0-9' || true; }
tracker() { head -1 "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9'; }
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

# Refuse to resume from a checkpoint that does not open. A truncated checkpoint is
# the one failure that resumes "successfully" and trains the wrong weights.
checkpoint_is_complete() {
  local iter="$1"
  [ -n "$iter" ] || return 1
  python "$R/tasks/alfworld/scripts/resume_guard.py" "$RUN_DIR" "$iter" >> "$SLOG" 2>&1
}

launch() { # $1 = mode: fresh | resume
  local mode="$1"
  # Fruitless-restart accounting: a restart is "fruitful" if the checkpoint
  # tracker advanced since the previous launch. Any restart before cp25 can never
  # advance it, so this fires exactly on the pathological loop.
  local before_iter
  before_iter=$(tracker)
  if [ -n "$before_iter" ] && [ "$before_iter" = "$(cat "$STATE/LAST_LAUNCH_ITER" 2>/dev/null)" ] && [ -e "$STATE/LAST_LAUNCH_ITER" ]; then
    echo $(( $(fruitless_restarts) + 1 )) > "$STATE/FRUITLESS"
  elif [ ! -e "$STATE/LAST_LAUNCH_ITER" ]; then
    echo 0 > "$STATE/FRUITLESS"
  else
    echo 0 > "$STATE/FRUITLESS"
  fi
  echo "${before_iter:-}" > "$STATE/LAST_LAUNCH_ITER"
  local fruitless
  fruitless=$(fruitless_restarts)
  if [ "${fruitless:-0}" -ge "$MAX_FRUITLESS_RESTARTS" ]; then
    say "CRASH LOOP: $fruitless consecutive restarts with no checkpoint progress -> STOPPING"
    touch "$STATE/CRASH_LOOP"
    echo "$(date '+%F %T') stopped after $fruitless fruitless restarts; tracker=${before_iter:-none}" >> "$STATE/WEDGES"
    exit 3
  fi
  local resume_env=()
  if [ "$mode" = "resume" ]; then
    touch "$RUN_DIR/NON_CONTIGUOUS"
    resume_env=(RESUME_FROM="$RUN_DIR")
    say "RESUME from iter $(tracker) via RESUME_FROM=$RUN_DIR -- marked NON_CONTIGUOUS"
  else
    say "FRESH start from cp0"
  fi
  cd "$R" || return 1
  # 9>&- on the child: do not leak this supervisor's flock fd into the long-lived
  # launcher, or the lock stays held for the whole run (measured hazard).
  env RUN_NAME="$RUN" NUM_ROLLOUT="$NUM_ROLLOUT_TARGET" SAVE_INTERVAL="$SAVE_INTERVAL" \
    ${resume_env[@]+"${resume_env[@]}"} \
    setsid nohup bash -c 'exec 9>&-; exec bash "$1" "$2"' _ "$R/tasks/alfworld/scripts/launch_c64.sh" "$ARM_CASE" \
    >> "$STATE/launch_$mode.out" 2>&1 < /dev/null &
  for _ in $(seq 1 $((LAUNCH_GRACE_MIN + 10))); do
    sleep 60
    [ "$(log_age_min)" -lt 3 ] && { say "launcher is writing again"; return 0; }
  done
  say "WARNING: launcher produced no log activity within ${LAUNCH_GRACE_MIN}min"
}

run_eval() {
  say "training finished -- Plan B evaluation of iters ${EVAL_ITERS[*]}"
  # Idempotent: EVAL_OK short-circuits, and the driver resumes at the first
  # checkpoint without a recorded eval. Never pauses training (none is running).
  setsid nohup bash -c \
    'exec 9>&-; bash "$6" "$0" "$1" "$2" "$3" "$4" "$5"' \
    "$RUN" "$METHOD_SLUG" "${EVAL_ITERS[@]}" "$R/tasks/alfworld/scripts/eval_arm_campaign.sh" \
    >> "$STATE/eval.out" 2>&1 < /dev/null &
  say "eval driver detached (see $STATE/eval.out)"
}

say "supervising $RUN (arm=$ARM_CASE, target=$NUM_ROLLOUT_TARGET, save_every=$SAVE_INTERVAL, wedge>${WEDGE_MIN}min)"
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
      if [ -n "$iter" ] && [ "$iter" -ge "$CP_FIRST" ] && checkpoint_is_complete "$iter"; then
        launch resume
      else
        launch fresh
      fi
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
    run_eval
    exit 0
  elif [ "$iter" -lt "$CP_FIRST" ]; then
    say "trainer gone at iter $iter (< cp$CP_FIRST) -> restart from cp0"
    launch fresh
  elif checkpoint_is_complete "$iter"; then
    say "trainer gone at iter $iter (>= cp$CP_FIRST) -> resume (NON_CONTIGUOUS)"
    launch resume
  else
    say "trainer gone at iter $iter but the checkpoint does not open -> restart from cp0"
    launch fresh
  fi
done
