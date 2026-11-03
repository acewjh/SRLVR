#!/usr/bin/env bash
# Single-instance supervisor: Qwen3-4B / ALFWorld, 100 optimizer steps.
#
# Launch (detached):
#   setsid nohup bash tasks/alfworld/scripts/supervise_grpo4b_100.sh \
#       --run-name <on-rule-name> --state-dir reports/grpo4b_100_<stamp> \
#       >/dev/null 2>&1 </dev/null &
#
# The default entry is the stock 4B GRPO launcher. `--entry <script>` selects a
# different arm (e.g. run_alfworld_v2_srlvr_v2_4b.sh); every other responsibility
# -- idle gate, resume, checkpoint pinning -- is arm-agnostic and unchanged.
#
# Responsibilities:
#   * wait until no foreign trainer/eval holds the eight NPUs;
#   * run the chosen 4B launcher (TP=2 default from _4b_backbone.sh) with
#     NUM_ROLLOUT=100 / SAVE_INTERVAL=25;
#   * resume from the newest complete checkpoint after any crash;
#   * while the launcher runs, hardlink-pin every COMPLETED checkpoint under
#     <state-dir>/ckpt_pin (the shared FS has external checkpoint pruning).
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
ENTRY=$PROJ/tasks/alfworld/scripts/run_alfworld_v2_grpo_4b.sh

NUM_ROLLOUT=100
SAVE_INTERVAL=25
RUN_NAME=""
STATE=""
IDLE_STABLE_SECONDS="${IDLE_STABLE_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-30}"
PIN_POLL_SECONDS="${PIN_POLL_SECONDS:-60}"

while [ $# -gt 0 ]; do
  case "$1" in
    --num-rollout) NUM_ROLLOUT="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --state-dir) STATE="$2"; shift 2 ;;
    --entry) ENTRY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$RUN_NAME" ] || { echo "--run-name is required" >&2; exit 2; }
[ -n "$STATE" ] || { echo "--state-dir is required" >&2; exit 2; }
[ -f "$ENTRY" ] || { echo "--entry not found: $ENTRY" >&2; exit 2; }
SAVE_DIR="$PROJ/runs/$RUN_NAME"
mkdir -p "$STATE" "$STATE/ckpt_pin"

exec 9>"$STATE/supervisor.lock"
if ! flock -n 9; then
  echo "[grpo4b-sup] another supervisor holds $STATE/supervisor.lock" >&2
  exit 0
fi
printf '%s\n' "$$" > "$STATE/supervisor.pid"
log() { echo "$(date '+%F %T') $*" >> "$STATE/supervisor.log"; }

npu_busy() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | grep -qE '^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|'
}
trainer_busy() { pgrep -f '[s]lime-ascend/train\.py' >/dev/null 2>&1; }

wait_idle() {
  local label="$1" since; since=$(date +%s)
  while :; do
    if trainer_busy || npu_busy; then
      since=$(date +%s)
      log "wait_idle($label): busy"
    else
      local now; now=$(date +%s)
      if [ $(( now - since )) -ge "$IDLE_STABLE_SECONDS" ]; then
        log "wait_idle($label): idle for $(( now - since ))s -> proceeding"
        return 0
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

latest_iteration() {
  local tracker="$SAVE_DIR/latest_checkpointed_iteration.txt"
  [ -s "$tracker" ] && cat "$tracker" || echo "-1"
}

# Pin only checkpoints the tracker says are COMPLETE. A directory without a
# matching tracker entry may be mid-write; hardlinking it would preserve a
# half-written checkpoint while the real one is replaced.
pin_completed() {
  local latest; latest=$(latest_iteration)
  [ "$latest" -ge 0 ] 2>/dev/null || return 0
  local i dir name
  for (( i=0; i<=latest; i++ )); do
    name=$(printf 'iter_%07d' "$i")
    dir="$SAVE_DIR/$name"
    if [ -d "$dir" ] && [ ! -d "$STATE/ckpt_pin/$name" ]; then
      if cp -al "$dir" "$STATE/ckpt_pin/$name" 2>>"$STATE/pin.log"; then
        log "pinned $name"
      else
        log "WARN: failed to pin $name (see pin.log)"
      fi
    fi
  done
  [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ] && \
    cp -f "$SAVE_DIR/latest_checkpointed_iteration.txt" "$STATE/ckpt_pin/latest_checkpointed_iteration.txt"
  # The curriculum snapshot is small and needed for a faithful resume.
  [ -d "$SAVE_DIR/curriculum" ] && cp -a "$SAVE_DIR/curriculum" "$STATE/ckpt_pin/" 2>/dev/null
  printf '%s\n' "$latest" > "$STATE/latest_iteration.txt"
}

final_iteration=$(( NUM_ROLLOUT - 1 ))
log "start host=$(hostname) run=$RUN_NAME entry=$(basename "$ENTRY") rounds=$NUM_ROLLOUT save_interval=$SAVE_INTERVAL save_dir=$SAVE_DIR"
wait_idle train

attempt=0
while :; do
  latest=$(latest_iteration)
  if [ "$latest" -ge "$final_iteration" ]; then
    log "training complete: latest=$latest final=$final_iteration"
    break
  fi
  attempt=$(( attempt + 1 ))
  if [ "$attempt" -gt 30 ]; then
    log "FATAL: 30 launch attempts without reaching iteration $final_iteration"
    exit 1
  fi
  # SAVE_DIR is exported so a launcher running from a frozen source overlay
  # still writes (and lets the supervisor pin) the SAME run directory.
  export NUM_ROLLOUT SAVE_INTERVAL RUN_NAME SAVE_DIR
  if [ -s "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    export RESUME_FROM="$SAVE_DIR"
  else
    unset RESUME_FROM
  fi
  log "launch attempt=$attempt latest=$latest resume=${RESUME_FROM:-none}"
  bash "$ENTRY" >> "$STATE/launcher.log" 2>&1 &
  child=$!
  while kill -0 "$child" 2>/dev/null; do
    pin_completed
    sleep "$PIN_POLL_SECONDS"
  done
  wait "$child"; rc=$?
  pin_completed
  after=$(latest_iteration)
  log "launcher exited rc=$rc latest=$after"
  if [ "$rc" = "0" ] && [ "$after" -ge "$final_iteration" ]; then
    break
  fi
  [ "$rc" = "0" ] && log "WARNING: launcher rc=0 but latest=$after < final=$final_iteration"
  sleep 120
done

pin_completed
printf '%s\n' "done $(date '+%F %T') latest=$(latest_iteration)" > "$STATE/DONE"
log "all phases complete"
