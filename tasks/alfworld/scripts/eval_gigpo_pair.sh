#!/usr/bin/env bash
# Evaluate GiGPO cp24 and its matched GRPO control under the TRAINING protocol.
#
# Why this exists instead of reusing the supervisor's eval call: the eval entry
# `run_alfworld_v2_eval.sh` does NOT inherit the 1.7B training wrapper's protocol
# env (`ALFWORLD_SHOW_ADMISSIBLE=1`, `command-normalization-v2-form`, TP=2 split).
# A first GiGPO eval launched that way showed admissible_actions unset and a
# 1-actor/7-engine split, i.e. different prompts from training; it was stopped and
# is recorded as VOID. This script reproduces the exact env the ALFWorld 1.7B
# campaign's pinned evaluations use (see
# reports/alfworld_1p7b_cp25_eval_20261028_190351/runner.py::env_for).
#
# Both checkpoints are evaluated serially, each under the identical protocol, so
# the only variable between the two numbers is the trained policy.
#
# Usage: setsid nohup bash eval_gigpo_pair.sh >/dev/null 2>&1 &
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
STATE="${STATE:-$PROJ/reports/gigpo_1p7b_20261029_020000}"
GIGPO_RUN=$PROJ/runs/alfworld-qwen3-1p7b-gigpo-full-s42-r25-n4-20261029_020000
GRPO_RUN=$PROJ/runs/alfworld-qwen3-1p7b-grpo-full-20261028_015327
ITER=24
POLICY_STABLE_SECONDS=300
POLL_SECONDS=30

mkdir -p "$STATE"
exec 9>"$STATE/eval_pair.lock"
if ! flock -n 9; then
  echo "[eval-pair] another eval-pair holds $STATE/eval_pair.lock" >&2
  exit 0
fi
printf '%s\n' "$$" > "$STATE/eval_pair.pid"
log() { echo "$(date '+%F %T') $*" >> "$STATE/eval_pair.log"; }

npu_busy() {
  npu-smi info 2>/dev/null | sed -n '/Process id/,$p' \
    | grep -qE '^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|'
}
trainer_busy() { pgrep -f '[s]lime-ascend/train\.py' >/dev/null 2>&1; }

wait_idle() {
  local label="$1" since; since=$(date +%s)
  while :; do
    if trainer_busy || npu_busy; then
      since=$(date +%s)
      log "wait_idle($label): busy (trainer=$(trainer_busy && echo yes || echo no) npu=$(npu_busy && echo yes || echo no))"
    else
      local now; now=$(date +%s)
      if [ $(( now - since )) -ge "$POLICY_STABLE_SECONDS" ]; then
        log "wait_idle($label): idle for $(( now - since ))s -> proceeding"
        return 0
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

run_one() {
  local arm="$1" src="$2" marker="$3"
  if [ -f "$marker" ]; then
    log "skip $arm: $marker exists"
    return 0
  fi
  if [ ! -d "$src/checkpoints/iter_$(printf '%07d' "$ITER")" ]; then
    log "FATAL: $arm checkpoint missing: $src/checkpoints/iter_$(printf '%07d' "$ITER")"
    return 1
  fi
  wait_idle "eval-$arm-cp25"
  local name
  name=$(PYTHONPATH="$PROJ" /usr/bin/python3 "$PROJ/scripts/run_names.py" --allocate \
    --task alfworld --model qwen3-1p7b --method "$arm" --variant "cp25-full-eval") || {
    log "FATAL: cannot allocate eval name for $arm"; return 1; }
  local out=$PROJ/runs/alfworld-qwen3-1p7b-$arm/$name
  local lg=$PROJ/logs/alfworld-qwen3-1p7b-$arm/$name.log
  mkdir -p "$(dirname "$out")" "$(dirname "$lg")"
  log "start arm=$arm src=$src iter=$ITER name=$name out=$out log=$lg"
  env \
    OPSD_CONDA_ENV=cua-rl BACKBONE=1p7b MODEL_SLUG=qwen3-1p7b METHOD_SLUG="$arm" \
    RUN_NAME="$name" SAVE_DIR="$out" LOG="$lg" DEBUG_ROLLOUT_DIR="$out/rollouts" \
    OPSD_LORA=0 TP=2 PP=1 COLOCATE=0 ACTOR_NODES=1 \
    ACTOR_GPUS=4 ROLLOUT_GPUS=4 ROLLOUT_GPUS_PER_ENGINE=1 \
    MAX_TURNS=50 MAX_RESPONSE=4096 SEQ_LENGTH=32768 HISTORY_LENGTH=0 SEED=42 KL_COEF=0.001 \
    ALFWORLD_THINKING_STYLE=brief \
    ALFWORLD_ACTION_HARNESS=command-normalization-v2-form \
    ALFWORLD_SHOW_ADMISSIBLE=1 \
    ALFWORLD_ENV_ACTORS=1 ALFWORLD_ENV_WORKERS=8 ALFWORLD_EVAL_CONCURRENCY=8 \
    ALFWORLD_OVERLAY_SITE="$BASE/envs/alfworld_overlay/lib/python3.10/site-packages" \
    ALFWORLD_HPI_DATA="$BASE/data/alfworld/json_2.1.1" \
    SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}" SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-8}" SGLANG_PREFIX_CACHE=1 \
    SGLANG_MAX_TOTAL_TOKENS=65536 SGLANG_MAX_RUNNING_REQUESTS=8 SGLANG_SERVER_CONCURRENCY=8 \
    SGLANG_MEM_FRACTION_STATIC=0.5 SGLANG_DTYPE=bfloat16 \
    N_SAMPLES_PER_EVAL_PROMPT=1 RESUME_WEIGHTS_ONLY=1 SAVE_TRAIN_STATE=0 \
    OPSD_EXPERIMENT_METRICS=1 EVAL_CONFIG="$PROJ/tasks/alfworld/v1/configs/eval_alfworld.json" \
    NO_PROXY='*' no_proxy='*' RAY_NUM_CPUS=16 RAY_OBJECT_STORE=8000000000 \
    WANDB_MODE=disabled WANDB_DISABLED=true \
    bash "$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$src/checkpoints" "$ITER" \
      >> "$STATE/eval_${arm}_cp25.log" 2>&1
  local rc=$?
  log "end arm=$arm rc=$rc log=$lg"
  if [ "$rc" = 0 ] && grep -aq "successfully loaded checkpoint.*iteration $ITER" "$lg"; then
    printf '%s\n' "arm=$arm iter=$ITER rc=0 name=$name $(date '+%F %T')" > "$marker"
    grep -a "\[v2-eval\]" "$lg" | tail -4 >> "$STATE/eval_${arm}_cp25.summary"
    return 0
  fi
  log "FATAL: $arm eval rc=$rc (no marker). See $lg"
  return 1
}

rc=0
run_one gigpo "$GIGPO_RUN" "$STATE/EVAL_OK_gigpo_cp25" || rc=1
run_one grpo "$GRPO_RUN" "$STATE/EVAL_OK_grpo_cp25" || rc=1
log "pair done rc=$rc"
exit "$rc"
