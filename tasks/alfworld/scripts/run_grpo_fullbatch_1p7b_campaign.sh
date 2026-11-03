#!/usr/bin/env bash
# Train on the idle node, then evaluate both retained checkpoints with dev64.
set -euo pipefail
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$PROJ"
TRAIN_RUN="${RUN_NAME:-alfworld-qwen3-1p7b-grpo-actions-lr1e-6-s42-r50-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="$TRAIN_RUN" NUM_ROLLOUT=50 SAVE_INTERVAL=25 \
  bash tasks/alfworld/scripts/run_alfworld_v2_grpo_fullbatch_1p7b.sh

for step in 25 50; do
  name=$(python3 scripts/run_names.py --allocate --task alfworld --model qwen3-1p7b \
    --method grpo --variant "actions-lr1e-6-s42-step${step}-dev64")
  env RUN_NAME="$name" METHOD_SLUG=grpo BACKBONE=1p7b MODEL_SLUG=qwen3-1p7b \
    OPSD_SRLVR=0 OPSD_LORA="${OPSD_LORA:-0}" \
    MAX_RESPONSE=128 MAX_TURNS=50 HISTORY_LENGTH=0 SEQ_LENGTH=32768 \
    G=1 ROLLOUT_BS=4 SEED=42 SGLANG_DTYPE=bfloat16 SGLANG_PREFIX_CACHE=1 \
    ALFWORLD_THINKING_STYLE="${ALFWORLD_THINKING_STYLE:-brief}" \
    EVAL_CONFIG="$PROJ/tasks/alfworld/v1/configs/eval_alfworld_dev64.json" \
    RESUME_WEIGHTS_ONLY=1 SAVE_TRAIN_STATE=0 \
    LOG="$PROJ/logs/$name.log" SAVE_DIR="$PROJ/runs/$name" \
    bash tasks/alfworld/scripts/eval_alfworld_ckpt.sh "$PROJ/runs/$TRAIN_RUN" "$((step-1))"
  printf '[grpo-fullbatch-campaign] step=%s evaluation=%s\n' "$step" "$name"
done
