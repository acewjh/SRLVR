#!/usr/bin/env bash
# Single-instance supervisor for one formal full-RL arm (SRLVR-v4 or GRPO).
#
# Responsibilities, in order:
#   1. keep the trainer alive until the last rollout (49) is checkpointed;
#   2. resume from the newest COMPLETE checkpoint, never from a partial one;
#   3. fail closed on the two silent failure modes this stack has actually hit:
#        * zero gradient from the first step (frozen-embedding/recompute break);
#        * a launcher that exits early without ever producing a checkpoint;
#   4. after training, run the formal evaluation of cp25 and cp50 and leave a
#      per-checkpoint completion marker, so the eval step is idempotent.
#
# Usage:
#   ARM=srlvr_v4 RUN_NAME=... setsid nohup bash supervise_full_rl.sh \
#     >> runs/$RUN_NAME/supervisor.out 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
PROJ=$(cd "$HERE/../../.." && pwd)
ARM_SEL="${ARM:?ARM must be srlvr_v4 or grpo}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
RUN_ROOT="${RUN_ROOT:-$PROJ/runs/$RUN_NAME}"
SAVE_DIR="$RUN_ROOT/checkpoints"
LOG="$RUN_ROOT/train.log"
SUP_LOG="$RUN_ROOT/supervisor.log"
STATUS="$RUN_ROOT/supervisor_status.txt"
TRAIN_LOCK="/tmp/opsd_${RUN_NAME}.lock"
NUM_ROLLOUT="${NUM_ROLLOUT:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
# Rollout ids that carry a checkpoint: interval boundaries plus the final batch.
EVAL_ITERS=(${EVAL_ITERS:-$((SAVE_INTERVAL - 1)) $((NUM_ROLLOUT - 1))})
# Consecutive restarts that never advance the checkpoint tracker.
MAX_FRUITLESS="${MAX_FRUITLESS:-4}"

mkdir -p "$RUN_ROOT" "$SAVE_DIR"

exec 9>"$RUN_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "supervisor already running for $RUN_NAME" >> "$SUP_LOG"
  exit 0
fi

say() { echo "[$(date '+%F %T')] $*" | tee -a "$SUP_LOG"; }
tracker() { cat "$SAVE_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9'; }
fruitless() { cat "$RUN_ROOT/.fruitless" 2>/dev/null | tr -dc '0-9'; }

training_alive() {
  # The launcher owns the run's lock; the child train.py inherits it. A live
  # holder means the run is alive even before train.py shows up in ps.
  if ! flock -n "$TRAIN_LOCK" true 2>/dev/null; then return 0; fi
  pgrep -f "run_alfworld_full_rl_1p7b[.]sh $RUN_NAME" >/dev/null 2>&1 && return 0
  pgrep -f "train[.]py .*--save $SAVE_DIR" >/dev/null 2>&1 && return 0
  return 1
}

checkpoint_is_complete() {
  local iter="$1"
  [ -n "$iter" ] && [ "$iter" -ge 0 ] || return 1
  # Full RL saves the Megatron torch-dist layout, not the HF/LoRA one that
  # resume_guard.py opens: common.pt + .metadata + __*.distcp shards.
  python "$PROJ/tasks/alfworld/scripts/full_rl_checkpoint_guard.py" "$SAVE_DIR" "$iter" >> "$SUP_LOG" 2>&1
}

# Can this run learn at all? The frozen-embedding/reentrant-recompute break on
# this stack is silent: the loss is finite but EVERY gradient is exactly 0, and
# the run then burns all 50 batches producing nothing (measured 2026-10-28 on
# the node-1 GRPO arm: 18/18 optimizer steps at train/grad_norm=0.0).
#
# A single zero step is NOT the failure: a batch whose trajectories are all
# failures has zero advantage everywhere and legitimately yields zero gradient
# (see the GRPO zero-gradient-vs-HSI note). So require a run of consecutive zero
# steps, and only conclude once the trainer has produced enough of them.
ZERO_STREAK_LIMIT="${ZERO_STREAK_LIMIT:-5}"

# Read the optimizer-step gradient history. Values are compared NUMERICALLY: a
# regex for "starts with 0" also matches 0.028, which would condemn a healthy
# run, and the logs carry ANSI colour codes that break naive pattern matching.
grad_stats() {
  python - "$LOG" <<'PY'
import re
import sys

values = []
try:
    with open(sys.argv[1], "rb") as handle:
        for raw in handle:
            for hit in re.finditer(rb"train/grad_norm'?:?\s*([0-9.eE+-]+)", raw):
                try:
                    values.append(float(hit.group(1)))
                except ValueError:
                    pass
except FileNotFoundError:
    pass
streak = 0
for value in reversed(values):
    if value == 0.0:
        streak += 1
    else:
        break
print(f"steps={len(values)} last_zero_run={streak} max={max(values) if values else 0.0}")
PY
}

gradient_gate() {
  local deadline=$(( $(date +%s) + ${GRAD_GATE_SECONDS:-7200} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if grep -aq "train/grad_norm" "$LOG" 2>/dev/null; then
      local steps streak max_grad
      eval "$(grad_stats)"
      if [ "$max_grad" != "0.0" ]; then
        say "gradient gate: PASS steps=$steps max_train_grad_norm=$max_grad"
        return 0
      fi
      if [ "$steps" -ge "$ZERO_STREAK_LIMIT" ] && [ "$streak" -ge "$ZERO_STREAK_LIMIT" ]; then
        say "gradient gate: FAIL steps=$steps last $streak consecutive zero, max=0.0"
        return 1
      fi
    fi
    # A trainer that exits is judged by the caller via its recorded steps.
    if ! pgrep -f "train[.]py .*--save $SAVE_DIR" >/dev/null 2>&1; then
      sleep 20
      pgrep -f "train[.]py .*--save $SAVE_DIR" >/dev/null 2>&1 || return 2
    fi
    sleep 60
  done
  return 3
}

run_training() {
  (
    exec 9>&-
    export ALLOW_EXISTING_RUN_ROOT=1
    local latest
    latest=$(tracker)
    if [ -n "$latest" ] && [ "$latest" -ge 0 ] && checkpoint_is_complete "$latest"; then
      export RESUME_FROM="$SAVE_DIR"
      say "resume from iter $latest (verified by full_rl_checkpoint_guard)"
    else
      unset RESUME_FROM
    fi
    # LAUNCHER_SH lets an experiment arm supply its own thin wrapper (extra
    # protocol flags) while reusing this supervisor's restart/scan logic.
    ARM="$ARM_SEL" bash "$HERE/${LAUNCHER_SH:-run_alfworld_full_rl_1p7b.sh}" "$RUN_NAME"
  )
}

attempt=0
say "=== supervisor start run=$RUN_NAME arm=$ARM_SEL rounds=$NUM_ROLLOUT eval_iters=${EVAL_ITERS[*]} ==="
while :; do
  latest=$(tracker)
  if [ -n "$latest" ] && [ "$latest" -ge $((NUM_ROLLOUT - 1)) ] 2>/dev/null; then
    say "training complete: latest=$latest"
    break
  fi

  if training_alive; then
    echo "$(date '+%F %T') RUNNING latest=${latest:--1} attempt=$attempt" > "$STATUS"
    sleep 60
    continue
  fi

  # Reached here: no trainer holds the lock.
  if [ "$attempt" -gt 0 ]; then
    if [ "${latest:--1}" = "$(cat "$RUN_ROOT/.last_launch_iter" 2>/dev/null || echo -1)" ]; then
      echo $(( $(fruitless) + 1 )) > "$RUN_ROOT/.fruitless"
    else
      echo 0 > "$RUN_ROOT/.fruitless"
    fi
    if [ "$(fruitless)" -ge "$MAX_FRUITLESS" ]; then
      say "CRASH LOOP: $(fruitless) restarts with no checkpoint progress -> STOPPING"
      touch "$RUN_ROOT/CRASH_LOOP"
      echo "$(date '+%F %T') STOPPED crash loop latest=${latest:--1}" > "$STATUS"
      exit 3
    fi
  fi
  echo "${latest:--1}" > "$RUN_ROOT/.last_launch_iter"

  attempt=$((attempt + 1))
  say "attempt $attempt start (resume iter ${latest:--1})"
  echo "$(date '+%F %T') START attempt=$attempt latest=${latest:--1}" > "$STATUS"
  run_training >> "$SUP_LOG" 2>&1 &
  train_pid=$!
  gradient_gate "$RUN_ROOT"
  gate_rc=$?
  wait "$train_pid"
  rc=$?
  latest_after=$(tracker)
  say "attempt $attempt exit rc=$rc latest=${latest_after:--1} grad_gate=$gate_rc"

  if [ "$gate_rc" -eq 1 ]; then
    say "FATAL: $ZERO_STREAK_LIMIT+ consecutive optimizer steps at train/grad_norm=0."
    say "FATAL: the run cannot learn; inspect the model provider / recompute setup."
    touch "$RUN_ROOT/ZERO_GRADIENT"
    echo "$(date '+%F %T') STOPPED zero-gradient run" > "$STATUS"
    exit 4
  fi

  if [ "$rc" -eq 0 ]; then
    sleep 10
  elif [ "$attempt" -lt 5 ]; then
    sleep 60
  else
    sleep 300
  fi
done

echo "$(date '+%F %T') TRAIN_DONE latest=$(tracker)" > "$STATUS"
touch "$RUN_ROOT/TRAIN_OK"

# ---- formal evaluation -------------------------------------------------------
# Sequential by necessity: the trainer holds all 8 cards, and the eval driver
# refuses to start while a trainer is alive. The yardstick is the same
# interaction protocol the model trained under (admissible actions shown,
# thinking enabled, 4096 cap, full history), on the official valid splits.
#
# TP must match the topology the checkpoint was SAVED with: these runs train at
# TP=2, and the eval entry point otherwise defaults to TP=1 (its own launcher,
# unlike the 1.7B training wrapper, does not set TP=2). Loading a TP2 shard set
# into a TP1 model is not a reshard this stack supports, so it is pinned here.
export ARM="$ARM_SEL" RUN_NAME
export OPSD_LORA=0
export ALFWORLD_SHOW_ADMISSIBLE=1
export ALFWORLD_THINKING_STYLE=brief
unset ALFWORLD_THINKING_TOKENS
export MAX_TURNS=50 HISTORY_LENGTH=0 MAX_RESPONSE=4096 SEQ_LENGTH=32768
export TP=2 PP=1
# The eval GPU split is NOT set here: _common_v2.sh's EVAL_ONLY block derives it
# from TP (actor = TP, rollout = the rest of the node). Hard-coding it at the
# call site is exactly how the 4/4 mistake survived -- see that block.
export ROLLOUT_GPUS_PER_ENGINE=1
export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}" SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}"
export N_SAMPLES_PER_EVAL_PROMPT=1
export RESUME_WEIGHTS_ONLY=1 SAVE_TRAIN_STATE=0
unset MODEL_PROVIDER_PATH OPSD_SRLVR_SELECTIVE_LOGITS OPSD_SRLVR_SELECTIVE_CHUNK

for iter in "${EVAL_ITERS[@]}"; do
  marker="$RUN_ROOT/EVAL_OK_cp$((iter + 1))"
  [ -e "$marker" ] && { say "eval cp$((iter + 1)) already done, skipping"; continue; }
  say "eval cp$((iter + 1)) (iter_$iter) starting"
  echo "$(date '+%F %T') EVAL src=cp$((iter + 1)) running" > "$STATUS"
  BACKBONE=1p7b METHOD_SLUG="$ARM_SEL" MODEL_SLUG=qwen3-1p7b \
    timeout 86400 bash "$HERE/eval_alfworld_ckpt.sh" "$SAVE_DIR" "$iter" >> "$SUP_LOG" 2>&1
  eval_rc=$?
  say "eval cp$((iter + 1)) exit rc=$eval_rc"
  if [ "$eval_rc" -eq 0 ]; then
    printf 'cp%s iter=%s rc=0 finished=%s\n' "$((iter + 1))" "$iter" \
      "$(date '+%F %T')" > "$marker"
  else
    printf 'cp%s iter=%s rc=%s finished=%s FAILED\n' "$((iter + 1))" "$iter" \
      "$eval_rc" "$(date '+%F %T')" >> "$SUP_LOG"
  fi
done

echo "$(date '+%F %T') DONE train+eval" > "$STATUS"
say "=== supervisor done $(date '+%F %T') ==="
