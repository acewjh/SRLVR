#!/bin/bash
# Supervise the OpenApps GUI GRPO run across SGLang engine wedges.
#
# The colocated engine occasionally dies (ACL/500) and the flush then never
# returns; no retry can recover a dead engine. Each segment runs a bounded
# number of rollouts, saves every SAVE_INTERVAL, and a crash resumes from the
# previous segment's run directory (weights + optimizer sidecar).
#
# Resume contract (measured):
#   * LOAD_PATH is the RUN DIR (it must contain latest_checkpointed_iteration.txt)
#     -- Megatron resolves the tracker and methods/optim_sidecar.py finds the
#     sidecar only on the run dir. Passing iter_N is treated as a fresh start.
#   * slime sets start_rollout_id = loaded_iter + 1 and loops over
#     range(start_rollout_id, num_rollout), so NUM_ROLLOUT must be the
#     CUMULATIVE total (start + this segment's rollouts), never just the
#     segment size.
#   * checkpoints are weights-only (--no-save-optim), so the launcher passes
#     --no-load-optim/--no-load-rng; Adam comes back from the sidecar.
#
# Env:
#   TARGET=100        total rollouts across segments
#   SEG_ROLLOUTS=25   rollouts per segment
#   SAVE_INTERVAL=25  checkpoint interval (first save at iter_24)
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${OPENAPPS_RUNS_ROOT:-$PROJ/runs}"
TARGET="${TARGET:-100}"
SEG_ROLLOUTS="${SEG_ROLLOUTS:-25}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
BASE_STAMP="${BASE_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUP_LOG="$PROJ/logs/gui_supervisor_${BASE_STAMP}.log"
MAX_SEGMENTS="${MAX_SEGMENTS:-40}"

mkdir -p "$PROJ/logs"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [supervisor] $*" | tee -a "$SUP_LOG"; }

run_start_rollout() {  # 0 for a fresh run, iter+1 for a resumed one
  local run_dir="$1"
  [ -f "$run_dir/latest_checkpointed_iteration.txt" ] || { echo 0; return; }
  local it
  it=$(tr -cd '0-9' < "$run_dir/latest_checkpointed_iteration.txt")
  [ -n "$it" ] || { echo 0; return; }
  echo $((it + 1))
}

# Cumulative progress lives in a supervisor-scoped file: scanning every
# gui-*.log would pick up unrelated historical runs (one already reached 99).
PROGRESS_FILE="$PROJ/logs/gui_supervisor_${BASE_STAMP}.progress"
segment_last_rollout() {  # highest rollout index in one segment log, or empty
  local log_file="$1"
  [ -f "$log_file" ] || return 0
  local idx
  idx=$(grep -a -o "rollout [0-9]* metrics" "$log_file" | tail -1 | grep -o "[0-9]*" | tail -1)
  [ -n "${idx:-}" ] && printf '%s' "$idx"
}

done_rollouts=$(cat "$PROGRESS_FILE" 2>/dev/null || echo 0)
# Optional: start from an existing run directory (validated weights + sidecar).
resume_run="${INITIAL_RESUME_RUN:-}"
if [ "$done_rollouts" -eq 0 ] && [ -n "$resume_run" ]; then
  done_rollouts=$(run_start_rollout "$resume_run")
  echo "$done_rollouts" > "$PROGRESS_FILE"
fi
attempt=0

while [ "$done_rollouts" -lt "$TARGET" ]; do
  attempt=$((attempt + 1))
  [ "$attempt" -gt "$MAX_SEGMENTS" ] && { log "FATAL: ${MAX_SEGMENTS} segments without reaching ${TARGET}"; exit 1; }

  if [ -n "$resume_run" ]; then
    start=$(run_start_rollout "$resume_run")
    RUN_NAME="gui-qwen25vl3b-grpo-r${attempt}-$(date +%Y%m%d_%H%M%S)"
  else
    start=0
    RUN_NAME="gui-qwen25vl3b-grpo-${BASE_STAMP}"
  fi

  # One segment must not overshoot the target.
  n=$SEG_ROLLOUTS
  # Align segment ends with the periodic checkpoint cadence after a smoke
  # resume (e.g. start=11 runs14 more rollouts to iter24).
  until_save=$((SAVE_INTERVAL - start % SAVE_INTERVAL))
  [ "$n" -gt "$until_save" ] && n=$until_save
  room=$((TARGET - start))
  [ "$n" -gt "$room" ] && n=$room
  [ "$n" -le 0 ] && { log "nothing left to run (start=${start}, target=${TARGET})"; break; }
  total=$((start + n))

  log "segment ${attempt}: start=${start}, n=${n}, NUM_ROLLOUT=${total}, run=${RUN_NAME}, load=${resume_run:-<base model>}"

  OPENAPPS_RUNS_ROOT="$RUNS_ROOT" \
  RUN_NAME="$RUN_NAME" \
  LOAD_PATH="${resume_run:-}" \
  NUM_ROLLOUT="$total" \
  ROLLOUT_BS="${ROLLOUT_BS:-2}" G="${G:-8}" GBS="${GBS:-16}" \
  SAVE_INTERVAL="$SAVE_INTERVAL" \
  COLOCATE="${COLOCATE:-0}" \
  OPENAPPS_POOL_SIZE="${OPENAPPS_POOL_SIZE:-16}" \
  OPSD_MAX_CONCURRENT_ENVS="${OPSD_MAX_CONCURRENT_ENVS:-64}" \
  OSWORLD_DEBUG_ACTIONS=1 \
    bash "$PROJ/scripts/run_openapps_gui_grpo.sh" >> "$SUP_LOG" 2>&1
  rc=$?

  last_idx=$(segment_last_rollout "$PROJ/logs/$RUN_NAME.log")
  # Generation may finish before its optimizer step or checkpoint. Only a
  # saved checkpoint counts as durable progress for restart/completion.
  if [ -f "$RUNS_ROOT/$RUN_NAME/latest_checkpointed_iteration.txt" ]; then
    resume_run="$RUNS_ROOT/$RUN_NAME"
    done_rollouts=$(run_start_rollout "$resume_run")
    echo "$done_rollouts" > "$PROGRESS_FILE"
  fi
  log "segment ${attempt} exited rc=${rc}, generated_last=${last_idx:-none}, saved_progress=${done_rollouts}/${TARGET}, resume_run=${resume_run:-none}"

  if [ "$done_rollouts" -ge "$TARGET" ] && [ "$rc" -eq 0 ]; then
    log "target reached"
    break
  fi
  if [ "$done_rollouts" -ge "$TARGET" ]; then
    log "FATAL: final checkpoint exists but launcher failed rc=${rc}; inspect before declaring success"
    exit "$rc"
  fi
  sleep 60
done

log "done: ${done_rollouts}/${TARGET} rollouts, last resume point ${resume_run:-none}"
exit 0
