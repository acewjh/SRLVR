#!/usr/bin/env bash
# Evaluate a training run's checkpoints on THIS node as soon as each one lands.
#
# The trainer owns one node; a second node is otherwise idle, so its cards are
# used to score checkpoints while training continues elsewhere. This is the same
# split the GRPO campaign used by hand on 2026-10-28 (cp25 on node1, cp50
# opportunistically on node0), turned into a watcher so it does not depend on
# someone noticing.
#
# Correctness comes from three rules:
#   * a checkpoint is only scored once full_rl_checkpoint_guard.py accepts it
#     (tracker match, readable common.pt, shards present and not still growing);
#   * at most one eval per node: an EVAL_INFLIGHT_cpN marker blocks a second copy,
#     which is what stopped two nodes duplicating 2 h of GPU time;
#   * EVAL_OK_cpN is written only after the driver verifies that the log really
#     contains `successfully loaded checkpoint ... iteration N`.
#
# The RUN_NAME handed to the eval driver is distinct from the training run, so
# its mirror and output directories cannot collide with the supervisor's own
# post-training evaluation; the EVAL_OK marker is what makes that supervisor
# skip the iteration.
#
# Usage:
#   RUN_NAME=<train run> ITERS="24 49" setsid nohup bash watch_and_eval_ckpts.sh &
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$HERE/../../.." && pwd)
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
ITERS="${ITERS:-24 49}"
RUN_ROOT="${RUN_ROOT:-$PROJ/runs/$RUN_NAME}"
POLL="${POLL:-300}"
WATCH_LOG="$RUN_ROOT/eval_watch.log"

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/eval_watch.lock"
if ! flock -n 9; then
  echo "$(date '+%F %T') watcher already running" >> "$WATCH_LOG"
  exit 0
fi

say() { echo "[$(date '+%F %T')] $*" >> "$WATCH_LOG"; }
say "watcher start iters=$ITERS poll=${POLL}s"

for iter in $ITERS; do
  cp=$((iter + 1))
  marker="$RUN_ROOT/EVAL_OK_cp$cp"
  inflight="$RUN_ROOT/EVAL_INFLIGHT_cp$cp"
  iter_dir="$RUN_ROOT/checkpoints/iter_$(printf '%07d' "$iter")"

  # Wait for THIS checkpoint, keeping the ordering of ITERS.
  while :; do
    if [ -e "$marker" ]; then
      say "cp$cp already scored; skipping"
      break
    fi
    if [ ! -d "$iter_dir" ]; then
      sleep "$POLL"
      continue
    fi
    if ! /opt/conda/envs/cua-rl/bin/python \
        "$PROJ/tasks/alfworld/scripts/full_rl_checkpoint_guard.py" \
        "$RUN_ROOT/checkpoints" "$iter" >> "$WATCH_LOG" 2>&1; then
      say "cp$cp present but not yet complete; waiting"
      sleep "$POLL"
      continue
    fi
    if [ -e "$inflight" ]; then
      say "cp$cp is in flight elsewhere; waiting"
      sleep "$POLL"
      continue
    fi
    say "cp$cp complete and unclaimed; starting eval"
    bash "$HERE/eval_grpo_cp50_node0.sh" "$RUN_NAME" "$iter" >> "$WATCH_LOG" 2>&1
    rc=$?
    say "cp$cp eval driver rc=$rc"
    # The driver writes EVAL_OK on success; on failure retry once more later
    # rather than spinning (the checkpoint cannot change any more).
    [ "$rc" -eq 0 ] && break
    sleep 1800
  done
done

say "watcher done"
