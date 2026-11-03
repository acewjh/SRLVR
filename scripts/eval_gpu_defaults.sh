#!/bin/bash
# Canonical NPU budget for EVERY eval on this node. Source this from any script
# that launches an eval (dev eval, formal eval, eval queue, selector).
#
#   source .../scripts/eval_gpu_defaults.sh
#
# Policy (user directive 2026-09-20): an eval uses ALL visible NPUs. An
# under-provisioned eval fails loudly instead of quietly costing hours.
#
# Why this file exists: eval_queue_hpi.sh exported NUM_TRAIN_GPUS=1 but never
# NUM_ROLLOUT_GPUS, so train_scripts/run_dev_eval.sh fell back to its default of
# 4 -> 1 trainer + 4 engines = 5 of 8 NPUs, silently ~4x slower per cell. The
# historical eval_queue_fixedgrader.sh had set NUM_ROLLOUT_GPUS=7; the new queue
# dropped it and nothing complained. Keep the default and the guard together so
# a caller cannot recreate that state by omission.
#
# Escape hatch for a deliberately partial run: EVAL_ALLOW_PARTIAL_GPUS=1.

if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  TOTAL_NPUS=$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
else
  TOTAL_NPUS=$(npu-smi info -l 2>/dev/null | awk '/Total Count/{print $4; exit}')
fi
TOTAL_NPUS="${TOTAL_NPUS:-8}"

# The driver ranks drive the Megatron actor (restore + update_weights + RM);
# every remaining card contributes rollout throughput. The driver must be able to
# rebuild the checkpoint's parallelism, so callers that know the model geometry
# export EVAL_ACTOR_NPUS = TP x PP (1 for 1.7B/4B, 2 for the PP2 8B card) before
# sourcing this file. Default 1 preserves the historical behaviour.
: "${EVAL_ACTOR_NPUS:=1}"
: "${NUM_TRAIN_GPUS:=$EVAL_ACTOR_NPUS}"
: "${NUM_ROLLOUT_GPUS:=$(( TOTAL_NPUS - NUM_TRAIN_GPUS ))}"
: "${NUM_NODE_GPUS:=$TOTAL_NPUS}"

if [ "$(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS ))" -lt "$TOTAL_NPUS" ] \
   && [ "${EVAL_ALLOW_PARTIAL_GPUS:-0}" != "1" ]; then
  echo "[eval-gpu] FATAL: eval would use $(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS )) of $TOTAL_NPUS NPUs."
  echo "[eval-gpu]        Raise NUM_ROLLOUT_GPUS, or set EVAL_ALLOW_PARTIAL_GPUS=1 to do this on purpose."
  exit 1
fi

export NUM_TRAIN_GPUS NUM_ROLLOUT_GPUS NUM_NODE_GPUS TOTAL_NPUS
echo "[eval-gpu] NPU budget: train=$NUM_TRAIN_GPUS rollout=$NUM_ROLLOUT_GPUS total=$TOTAL_NPUS"
