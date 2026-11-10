#!/bin/bash
# Supervise an 8B SRLVR-v2 run across the known SGLang host-RSS leak.
#
# The SGLang schedulers grow host RSS every rollout; a single process OOMs at
# ~28 rollouts (2026-11-07: ~900 GB across 8 schedulers, Ray killed the actor).
# Keep NUM_ROLLOUT at the run's original value (the optimizer-param scheduler
# stores total_num_iters in the checkpoint and any other value fails the resume
# assert), but stop the process right after the checkpoint target lands so the
# next segment starts a fresh Ray and resets host memory.
#
# Usage: supervise_srlvr_8b_segments.sh <run-name>
#   Segment boundaries are the periodic checkpoints 49 and 74; the last segment
#   runs to completion (iter_99).
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
RUN="${1:?usage: supervise_srlvr_8b_segments.sh <run-name>}"
RUN_DIR="$PROJ/runs/$RUN"
LOG="$PROJ/logs/$RUN.log"
SUP_LOG="$PROJ/logs/$RUN.segment_supervisor.log"
LOCK="/tmp/opsd_${RUN}.segments.lock"
TOTAL=100
STOP_AFTER=(49 74)
MAX_RETRIES="${MAX_SEGMENT_RETRIES:-3}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another segment supervisor holds $LOCK" >> "$SUP_LOG"
  exit 1
fi

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SUP_LOG"; }
tracker() { head -1 "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9'; }

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl

launch_and_stop() {
  local stop_after="$1"   # 49/74, or 99 for the final segment
  local out="/tmp/${RUN}.segment_${stop_after}.out"
  say "launch stop_after=$stop_after from checkpoint=$(tracker)"
  setsid env \
    MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072 \
    SGLANG_MALLOC_TRIM_S="${SGLANG_MALLOC_TRIM_S:-}" \
    SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-}" \
    ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-}" \
    ACTOR_GPUS_32G=8 ROLLOUT_GPUS_32G=8 ROLLOUT_GPUS_PER_ENGINE_32G=4 COLOCATE_32G=1 \
    NUM_ROLLOUT="$TOTAL" SAVE_INTERVAL=25 \
    RUN_NAME="$RUN" SAVE_DIR="$RUN_DIR" RESUME_FROM="$RUN_DIR" LOG="$LOG" \
    bash "$PROJ/tasks/alfworld/scripts/run_alfworld_v2_srlvr_v2_8b_32g.sh" >> "$out" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    local last; last="$(tracker)"
    if [ "$stop_after" -lt 99 ] && [ "${last:--1}" -ge "$stop_after" ]; then
      say "checkpoint $last landed; stopping pid group $pid for a Ray memory reset"
      kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
      kill -KILL -"$pid" 2>/dev/null || true
      break
    fi
    sleep 60
  done
  wait "$pid" 2>/dev/null
  local rc=$?
  say "segment stop_after=$stop_after exited rc=$rc tracker=$(tracker)"
  ray stop --force >/dev/null 2>&1 || true
  sleep 15
  return "$rc"
}

for target in "${STOP_AFTER[@]}"; do
  last="$(tracker)"; last="${last:--1}"
  if [ "$last" -ge "$target" ]; then
    say "skip stop-after=$target (checkpoint $last already there)"
    continue
  fi
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    launch_and_stop "$target"
    last="$(tracker)"; last="${last:--1}"
    if [ "$last" -ge "$target" ]; then
      say "segment reached checkpoint $last"
      break
    fi
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      say "giving up at stop-after=$target after $attempt attempts"
      exit 1
    fi
    say "retry $attempt/$MAX_RETRIES after 120s"
    sleep 120
  done
done

last="$(tracker)"; last="${last:--1}"
if [ "$last" -lt 99 ]; then
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    launch_and_stop 99
    last="$(tracker)"; last="${last:--1}"
    [ "$last" -ge 99 ] && { say "final checkpoint $last"; break; }
    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      say "giving up on the final segment after $attempt attempts"
      exit 1
    fi
    sleep 120
  done
fi
say "all segments complete (latest=$(tracker))"
