#!/bin/bash
R=/workspace/cua/repos/opsd-slime
export METHOD_SLUG=grpo
for it in 24 49; do
  echo "===== grpo cp$it start $(date '+%F %H:%M:%S') ====="
  bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$R/runs/afw64b_grpo" "$it"
  echo "===== grpo cp$it done rc=$? $(date '+%F %H:%M:%S') ====="
  sleep 60
done
echo "ALL DONE $(date '+%F %H:%M:%S')"
