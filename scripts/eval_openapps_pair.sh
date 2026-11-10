#!/bin/bash
# Run after the training supervisor exits; both evaluations publish actor weights.
# Usage: bash scripts/eval_openapps_pair.sh /absolute/checkpoint-run
set -euo pipefail
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ"
CHECKPOINT_RUN="${1:?Usage: eval_openapps_pair.sh /absolute/checkpoint-run}"
[ -s "$CHECKPOINT_RUN/latest_checkpointed_iteration.txt" ] || {
  echo "Checkpoint run must contain latest_checkpointed_iteration.txt" >&2; exit 1;
}
export MODEL_PATH="${MODEL_PATH:-$(cd "$PROJ/../.." && pwd)/checkpoints/Qwen2.5-VL-3B-Instruct}"
export COLOCATE=0 ACTOR_GPUS=4 ROLLOUT_GPUS=4 TP=2 ROLLOUT_GPUS_PER_ENGINE=2
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.4}"
export OPENAPPS_POOL_SIZE=16 OPSD_MAX_CONCURRENT_ENVS=16
export OSWORLD_TRAIN_VISION=1 OSWORLD_VCS=1 OSWORLD_IMAGES_TO_KEEP=3 OSWORLD_MIN_REMOVAL_THRESHOLD=1
export MAX_RESPONSE=24576 SEQ_LENGTH=8192 OSWORLD_MAX_TURNS=16 OSWORLD_MAX_TOKENS_PER_TURN=1024
export SGLANG_CONTEXT_LENGTH=16384 SGLANG_CHUNKED_PREFILL=4096 OSWORLD_FEEDBACK_PROFILE=legacy
export EVAL_ONLY=1 DEBUG_ROLLOUT_ONLY=0 ROLLOUT_SHUFFLE=0 G=1 ROLLOUT_BS=1 GBS=2
export OPENAPPS_EVAL_FUNCTION=methods.openapps_eval.generate_rollout
PAIR_STAMP="${PAIR_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BASE_RUN="gui-openapps-eval-base-$PAIR_STAMP"
CKPT_RUN="gui-openapps-eval-ckpt-$PAIR_STAMP"
mkdir -p "$PROJ/logs"
for name in "$BASE_RUN" "$CKPT_RUN"; do
  [ ! -e "$PROJ/runs/$name" ] && [ ! -e "${OPENAPPS_RUNS_ROOT:-$PROJ/runs}/$name" ] || {
    echo "Evaluation run already exists: $name; choose a new PAIR_STAMP" >&2; exit 1;
  }
done
RUN_NAME="$BASE_RUN" LOAD_PATH="$MODEL_PATH" bash scripts/run_openapps_gui_grpo.sh > "logs/$BASE_RUN.launcher.log" 2>&1
RUN_NAME="$CKPT_RUN" LOAD_PATH="$CHECKPOINT_RUN" bash scripts/run_openapps_gui_grpo.sh > "logs/$CKPT_RUN.launcher.log" 2>&1
python3 methods/openapps_eval_compare.py \
  "$PROJ/runs/$BASE_RUN/openapps_eval_results.json" \
  "$PROJ/runs/$CKPT_RUN/openapps_eval_results.json" \
  --output "$PROJ/logs/openapps_eval_comparison_$PAIR_STAMP.json"
