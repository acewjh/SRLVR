#!/bin/bash
R=/workspace/cua/repos/opsd-slime
for it in 24 49; do
  echo "===== hsi cp$it start $(date '+%F %H:%M:%S') ====="
  bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$R/runs/afw64b_hsi" "$it"
  echo "===== hsi cp$it done rc=$? $(date '+%F %H:%M:%S') ====="
  sleep 60
done
echo "ALL DONE $(date '+%F %H:%M:%S')"
