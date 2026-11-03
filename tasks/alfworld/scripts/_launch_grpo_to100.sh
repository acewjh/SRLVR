#!/bin/bash
# GRPO arm continuation cp25 -> cp100 on worker-1. Same frozen schedule / GBS / turns
# as the HSI arm, so the two curves stay comparable.
BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
cd "$PROJ" || exit 1
export PYTHONPATH="$PROJ/frameworks/slime-ascend:$PROJ"
export ALFWORLD_HPI_DATA=$BASE/data/alfworld/json_2.1.1
export ARM=grpo
export RESUME_FROM=$PROJ/runs/alfworld_v2_grpo_cp25
export RUN_NAME=alfworld_v2_grpo_cp25
export SAVE_DIR=$PROJ/runs/alfworld_v2_grpo_cp25
export LOG=$PROJ/logs/alfworld_v2_grpo_cp25_to100.log
export NUM_ROLLOUT=100
setsid nohup bash "$PROJ/tasks/alfworld/scripts/run_alfworld_v2_grpo.sh" \
  >> "$PROJ/logs/grpo_to100_launch.log" 2>&1 < /dev/null &
echo "launched pid $!"
