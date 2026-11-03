#!/usr/bin/env bash
# Single-instance supervisor for the Qwen3-1.7B ALFWorld GiGPO cell.
#
# Responsibilities, in order:
#   1. wait until the node's eight NPUs are genuinely idle (a foreign eval or
#      trainer may be running; starting on top of one would OOM both);
#   2. run `ARM=gigpo NUM_ROLLOUT=... run_alfworld_v2_1p7b.sh`, resuming from the
#      latest complete checkpoint after any crash;
#   3. after the final checkpoint, evaluate the requested iterations with the
#      same full-benchmark harness the matched GRPO cell used;
#   4. leave completion markers and a machine-readable status file.
#
# Usage:
#   setsid nohup bash tasks/alfworld/scripts/supervise_gigpo_1p7b.sh \
#       --num-rollout 50 --save-interval 25 --evals 24,49 >/dev/null 2>&1 &
#
# The supervisor is idempotent: an existing flock holder exits 0, and every
# phase checks its own completion marker.
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
ENTRY=$PROJ/tasks/alfworld/scripts/run_alfworld_v2_1p7b.sh
EVAL=$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh

NUM_ROLLOUT=50
SAVE_INTERVAL=25
EVALS="24,49"
MODEL_SLUG=qwen3-1p7b
BACKBONE=1p7b
IDLE_STABLE_SECONDS=300
POLL_SECONDS=30

while [ $# -gt 0 ]; do
  case "$1" in
    --num-rollout) NUM_ROLLOUT="$2"; shift 2 ;;
    --save-interval) SAVE_INTERVAL="$2"; shift 2 ;;
    --evals) EVALS="$2"; shift 2 ;;
    --run-name) RUN_NAME_OVERRIDE="$2"; shift 2 ;;
    --stamp) STAMP_OVERRIDE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

STAMP="${STAMP_OVERRIDE:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME_OVERRIDE:-alfworld-qwen3-1p7b-gigpo-full-s42-r${NUM_ROLLOUT}-n4-${STAMP}}"
ROOT="$PROJ/runs/$RUN_NAME"
STATE="$PROJ/reports/gigpo_1p7b_${STAMP}"
mkdir -p "$STATE"

exec 9>"$STATE/supervisor.lock"
if ! flock -n 9; then
  echo "[gigpo-sup] another supervisor holds $STATE/supervisor.lock" >&2
  exit 0
fi
printf '%s\n' "$$" > "$STATE/supervisor.pid"

log() { echo "$(date '+%F %T') $*" >> "$STATE/supervisor.log"; }

write_status() {
  /usr/bin/python3 - "$STATE/status.json" "$RUN_NAME" "$ROOT" "$NUM_ROLLOUT" "$*" <<'PY'
import json, os, sys, time
path, run, root, rounds, note = sys.argv[1:6]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
data.update(run_name=run, root=root, num_rollout=int(rounds), note=note,
            updated=time.strftime('%F %T'), pid=os.getpid(), host=os.uname().nodename)
with open(path + '.tmp', 'w') as handle:
    json.dump(data, handle, indent=2)
os.replace(path + '.tmp', path)
PY
}

npu_busy() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | grep -qE '^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|'
}

trainer_busy() {
  pgrep -f '[s]lime-ascend/train\.py' >/dev/null 2>&1
}

# Idle = no trainer process AND no NPU process rows, sustained for a window.
# The stability window is what prevents launching into the gap between a
# foreign eval's teardown and its supervisor's next start.
wait_idle() {
  local label="$1" since
  since=$(date +%s)
  while :; do
    if trainer_busy || npu_busy; then
      since=$(date +%s)
      log "wait_idle($label): busy (trainer=$(trainer_busy && echo yes || echo no) npu=$(npu_busy && echo yes || echo no))"
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
  local tracker="$ROOT/checkpoints/latest_checkpointed_iteration.txt"
  [ -s "$tracker" ] && cat "$tracker" || echo "-1"
}

final_iteration=$(( NUM_ROLLOUT - 1 ))
attempt=0
log "start host=$(hostname) run=$RUN_NAME rounds=$NUM_ROLLOUT save_interval=$SAVE_INTERVAL root=$ROOT"
write_status "waiting for idle"

wait_idle train

while :; do
  latest=$(latest_iteration)
  if [ "$latest" -ge "$final_iteration" ]; then
    log "training complete: latest_iteration=$latest final=$final_iteration"
    break
  fi
  attempt=$(( attempt + 1 ))
  if [ "$attempt" -gt 20 ]; then
    log "FATAL: 20 launch attempts without reaching iteration $final_iteration"
    write_status "failed: too many attempts (latest=$latest)"
    exit 1
  fi
  export ARM=gigpo NUM_ROLLOUT SAVE_INTERVAL RUN_NAME RUN_ROOT="$ROOT"
  export ALFWORLD_V2_GIGPO_STEP_W="${ALFWORLD_V2_GIGPO_STEP_W:-1.0}"
  export ALFWORLD_V2_GIGPO_GAMMA="${ALFWORLD_V2_GIGPO_GAMMA:-0.95}"
  export ALFWORLD_V2_GIGPO_MODE="${ALFWORLD_V2_GIGPO_MODE:-mean_std_norm}"
  if [ -s "$ROOT/checkpoints/latest_checkpointed_iteration.txt" ]; then
    export RESUME_FROM="$ROOT/checkpoints"
  else
    unset RESUME_FROM
    # A retry before the first checkpoint keeps the same run root/identity.
    [ -d "$ROOT" ] && export ALLOW_EXISTING_RUN_ROOT=1
  fi
  log "launch attempt=$attempt resume=${RESUME_FROM:-none} latest=$latest"
  write_status "training attempt=$attempt latest=$latest"
  bash "$ENTRY" >> "$STATE/launcher.log" 2>&1
  rc=$?
  after=$(latest_iteration)
  log "launcher exited rc=$rc latest=$after"
  if [ "$rc" = "0" ] && [ "$after" -ge "$final_iteration" ]; then
    break
  fi
  if [ "$rc" = "0" ]; then
    log "WARNING: launcher reported success but latest=$after < final=$final_iteration; restarting"
  fi
  sleep 120
done

write_status "training done; starting evals"
IFS=',' read -r -a eval_iters <<< "$EVALS"
for iter in "${eval_iters[@]+"${eval_iters[@]}"}"; do
  [ -n "$iter" ] || continue
  marker="$STATE/EVAL_OK_cp$(( iter + 1 ))"
  if [ -f "$marker" ]; then
    log "eval iter=$iter already complete"
    continue
  fi
  wait_idle "eval-cp$(( iter + 1 ))"
  eval_name=$(PYTHONPATH="$PROJ" /usr/bin/python3 "$PROJ/scripts/run_names.py" --allocate \
      --task alfworld --model "$MODEL_SLUG" --method gigpo \
      --variant "cp$(( iter + 1 ))-full-eval") || {
    log "FATAL: could not allocate an eval run name"
    write_status "eval name allocation failed"
    exit 1
  }
  log "eval iter=$iter run=$eval_name"
  write_status "eval cp$(( iter + 1 )) running"
  METHOD_SLUG=gigpo MODEL_SLUG="$MODEL_SLUG" BACKBONE="$BACKBONE" RUN_NAME="$eval_name" \
    bash "$EVAL" "$ROOT/checkpoints" "$iter" >> "$STATE/eval_cp$(( iter + 1 )).log" 2>&1
  rc=$?
  log "eval iter=$iter exited rc=$rc"
  if [ "$rc" = "0" ]; then
    printf '%s\n' "eval cp$(( iter + 1 )) rc=0 $(date '+%F %T')" > "$marker"
  else
    log "eval iter=$iter FAILED rc=$rc; not writing the marker"
  fi
done

write_status "done"
log "all phases complete"
