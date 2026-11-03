#!/usr/bin/env bash
# Extend the running 4B GRPO arm from 50 to 100 rollouts, then hand off.
#
# Why wait instead of relaunching: the arm is mid-flight on worker-0 (step ~31/50).
# Killing it now would throw away every step since checkpoint 24, and slime's
# `--num-rollout` is the run's TOTAL, not a delta -- so the extension must be a
# RESUME with NUM_ROLLOUT=100 from the checkpoint the first leg writes at 49.
# This script therefore waits for the first leg to exit on its own, then resumes
# into the SAME run directory (same identity, same logs) for steps 50..99.
#
# Usage (on worker-0, detached):
#   setsid nohup bash .../extend_grpo_to100.sh > /tmp/extend_grpo.out 2>&1 &
set -uo pipefail

R=/workspace/cua/repos/opsd-slime
RUN=afw64b_grpo                 # legacy name, frozen while the campaign is live
DIR="$R/runs/$RUN"
LOG_TAG="[extend-grpo]"

log() { echo "$LOG_TAG $(date '+%F %H:%M:%S') $*"; }

# 1. Wait for the first leg. Liveness is the launcher process, not log silence:
#    a 30-min gap in a long rollout is normal, a missing process is not.
log "waiting for the running leg to exit (run=$RUN)"
while pgrep -f '[c]ommon_v2.sh' > /dev/null 2>&1; do sleep 120; done
log "first leg gone; checking for a checkpoint"

for _ in $(seq 1 30); do
  [ -s "$DIR/latest_checkpointed_iteration.txt" ] && break
  sleep 60
done
if [ ! -s "$DIR/latest_checkpointed_iteration.txt" ]; then
  log "FATAL: no checkpoint in $DIR after the first leg exited; not resuming"
  exit 1
fi

ITER=$(cat "$DIR/latest_checkpointed_iteration.txt")
if [ "$ITER" -ge 99 ]; then
  log "already at iter $ITER (>=99) -- nothing to extend; standing down"
  exit 0
fi
log "resuming from iter $ITER with NUM_ROLLOUT=100 (target step 99)"

# 2. Resume. RUN_NAME must be passed explicitly or launch_c64.sh will allocate a
#    NEW name and start a fresh run instead of continuing this one.
export RUN_NAME="$RUN"
export RESUME_FROM="$DIR"
export NUM_ROLLOUT=100
export SAVE_INTERVAL=25
cd "$R" || exit 1
exec bash "$R/tasks/alfworld/scripts/launch_c64.sh" grpo
