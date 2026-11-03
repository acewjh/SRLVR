#!/usr/bin/env bash
# Evaluate the GRPO cp50 checkpoint on the idle node while node1 finishes cp25.
#
# Why a separate entry: the training supervisor drives cp25 -> cp50 serially on
# one node, and each eval needs the whole machine. Node0 is free (the SRLVR-v4
# arm is paused), so cp50 is evaluated here in parallel to halve the wait.
#
# RUN_NAME is deliberately distinct from the training run so the mirror dir and
# the eval output dir cannot collide with a cp50 eval the node1 supervisor might
# also start; when this one succeeds it writes EVAL_OK_cp50 into the TRAINING run
# root, which is what makes that supervisor skip its own cp50.
#
# The protocol is byte-identical to the supervisor's eval block: same prompt
# settings, same TP (2, matching the saved shard set), same official splits.
# Engine count does not affect a result -- every episode carries its own sampling
# seed and keeps one engine for all its turns.
set -uo pipefail

PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
TRAIN_RUN="${1:?usage: $0 <train-run-name> <iter>}"
ITER="${2:?usage: $0 <train-run-name> <iter>}"

export RUN_NAME
RUN_NAME=$("/opt/conda/envs/cua-rl/bin/python" "$PROJ/scripts/run_names.py" --allocate \
            --task alfworld --model qwen3-1p7b --method grpo --variant "cp$((ITER + 1))n0")

export OPSD_LORA=0
export ALFWORLD_SHOW_ADMISSIBLE=1
export ALFWORLD_THINKING_STYLE=brief
unset ALFWORLD_THINKING_TOKENS
export MAX_TURNS=50 HISTORY_LENGTH=0 MAX_RESPONSE=4096 SEQ_LENGTH=32768
export TP=2 PP=1
export ROLLOUT_GPUS_PER_ENGINE=1
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}" SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"
export N_SAMPLES_PER_EVAL_PROMPT=1
export RESUME_WEIGHTS_ONLY=1 SAVE_TRAIN_STATE=0
export BACKBONE=1p7b METHOD_SLUG=grpo MODEL_SLUG=qwen3-1p7b
unset MODEL_PROVIDER_PATH OPSD_SRLVR_SELECTIVE_LOGITS OPSD_SRLVR_SELECTIVE_CHUNK

TRAIN_ROOT="$PROJ/runs/$TRAIN_RUN"
MARKER="$TRAIN_ROOT/EVAL_OK_cp$((ITER + 1))"
LOG="$TRAIN_ROOT/eval_cp$((ITER + 1))_node0.log"

# Advertise that this iteration is being scored HERE, so a keep-alive watchdog
# on the other node does not start a duplicate eval for it.
INFLIGHT="$TRAIN_ROOT/EVAL_INFLIGHT_cp$((ITER + 1))"
trap 'rm -f "$INFLIGHT"' EXIT
: > "$INFLIGHT"
echo "[eval-node0] start $(date '+%F %T') run=$RUN_NAME iter=$ITER" >> "$LOG"
bash "$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" \
     "$TRAIN_ROOT/checkpoints" "$ITER" >> "$LOG" 2>&1
rc=$?
echo "[eval-node0] launcher rc=$rc $(date '+%F %T')" >> "$LOG"

# eval_alfworld_ckpt.sh already refuses a run whose log lacks the
# "successfully loaded checkpoint ... iteration N" line, so rc=0 means the
# checkpoint really was the one scored.
if [ "$rc" -eq 0 ]; then
  printf 'cp%s iter=%s rc=0 finished=%s by=%s\n' \
    "$((ITER + 1))" "$ITER" "$(date '+%F %T')" "$(hostname)" > "$MARKER"
  echo "[eval-node0] marker written: $MARKER" >> "$LOG"
else
  echo "[eval-node0] NO marker (rc=$rc)" >> "$LOG"
fi
exit "$rc"
