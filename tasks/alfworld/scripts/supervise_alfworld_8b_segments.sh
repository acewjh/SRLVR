#!/usr/bin/env bash
# Run an 8B ALFWorld arm in 50-step cells so each checkpoint also resets SGLang
# host memory. Once iter 199 is saved, evaluate valid_seen and valid_unseen.
set -uo pipefail

RUN="${1:?usage: supervise_alfworld_8b_segments.sh <run-name> <grpo|gspo>}"
ARM="${2:?usage: supervise_alfworld_8b_segments.sh <run-name> <grpo|gspo>}"
case "$ARM" in grpo|gspo) ;; *) echo "unsupported arm: $ARM" >&2; exit 2 ;; esac

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
. "$PROJ/tasks/alfworld/scripts/_8b_fast_rollout.sh"
RUN_DIR="${RUN_DIR:-$PROJ/runs/$RUN}"
LOG="${LOG:-$PROJ/logs/$RUN.log}"
STATE_LOG="${STATE_LOG:-$PROJ/logs/$RUN.segment_supervisor.log}"
LOCK="/tmp/opsd_${RUN}.segments.lock"
TOTAL="${TOTAL_ROLLOUTS:-200}"
SEGMENT="${SEGMENT_ROLLOUTS:-50}"
SAVE_INTERVAL="$SEGMENT"
PP="${PP:-1}"
MAX_RETRIES="${MAX_SEGMENT_RETRIES:-3}"
WEIGHTS_ONLY_BEFORE="${WEIGHTS_ONLY_BEFORE:-0}"
MARK_NON_CONTIGUOUS="${MARK_NON_CONTIGUOUS:-0}"
WRAPPER="$PROJ/tasks/alfworld/scripts/run_alfworld_v2_${ARM}_8b.sh"

mkdir -p "$RUN_DIR" "$(dirname "$STATE_LOG")"
exec 9>"$LOCK"
if ! flock -n 9; then echo "another segment supervisor holds $LOCK" >> "$STATE_LOG"; exit 1; fi

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATE_LOG"; }
tracker() { head -1 "$RUN_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -dc '0-9'; }

attempt=0
while :; do
  last="$(tracker)"
  last="${last:--1}"
  completed=$((last + 1))
  if [ "$completed" -ge "$TOTAL" ]; then
    break
  fi
  segment_end=$(( (completed / SEGMENT + 1) * SEGMENT ))
  [ "$segment_end" -gt "$TOTAL" ] && segment_end="$TOTAL"

  resume_args=()
  if [ "$last" -ge 0 ]; then
    if [ "$MARK_NON_CONTIGUOUS" = "1" ] && [ ! -e "$RUN_DIR/INITIAL_RESUME_STARTED" ]; then
      touch "$RUN_DIR/NON_CONTIGUOUS" "$RUN_DIR/INITIAL_RESUME_STARTED"
    fi
    resume_args+=(RESUME_FROM="$RUN_DIR")
    if [ "$WEIGHTS_ONLY_BEFORE" -gt 0 ] && [ "$last" -lt "$WEIGHTS_ONLY_BEFORE" ]; then
      resume_args+=(RESUME_WEIGHTS_ONLY=1)
    fi
  fi

  say "launch arm=$ARM last_iter=$last target=$segment_end rollout_gpus=$ROLLOUT_GPUS actor_gpus=$ACTOR_GPUS pp=$PP"
  cd "$PROJ" || exit 1
  setsid env RUN_NAME="$RUN" SAVE_DIR="$RUN_DIR" LOG="$LOG" \
    NUM_ROLLOUT="$segment_end" SAVE_INTERVAL="$SAVE_INTERVAL" \
    ROLLOUT_GPUS="$ROLLOUT_GPUS" ACTOR_GPUS="$ACTOR_GPUS" PP="$PP" SAVE_TRAIN_STATE=1 \
    "${resume_args[@]}" \
    bash "$WRAPPER" >> "$STATE_LOG" 2>&1 < /dev/null &
  child=$!
  wait "$child"
  rc=$?

  # Ray's normal teardown should stop the serving engines. Reap any leftovers
  # from this segment only before starting a fresh Ray session.
  kill -TERM -- "-$child" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$child" 2>/dev/null || true

  last="$(tracker)"
  last="${last:--1}"
  if [ "$last" -ge "$((segment_end - 1))" ]; then
    PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ${PYTHONPATH:+:$PYTHONPATH}" \
      /opt/conda/envs/cua-rl/bin/python \
      "$PROJ/tasks/alfworld/scripts/check_campaign_checkpoint.py" "$RUN_DIR" "$last" \
      >> "$STATE_LOG" 2>&1 || { say "checkpoint validation failed at iter=$last"; exit 1; }
    say "checkpoint iter=$last complete (launcher rc=$rc)"
    attempt=0
    continue
  fi

  attempt=$((attempt + 1))
  say "segment stopped rc=$rc at iter=$last, retry $attempt/$MAX_RETRIES"
  if [ "$attempt" -ge "$MAX_RETRIES" ]; then
    say "stopping after $attempt incomplete attempts; run remains resumable"
    exit 1
  fi
  sleep 15
done

if [ "$(tracker)" -lt "$((TOTAL - 1))" ]; then
  say "final checkpoint missing: tracker=$(tracker) target=$((TOTAL - 1))"
  exit 1
fi

if [ "${SKIP_FINAL_EVAL:-0}" = "1" ]; then
  say "preflight complete at iter $(tracker); final evaluation skipped for preflight"
  exit 0
fi

say "training complete at iter $(tracker); starting full ALFWorld evaluation"
cd "$PROJ" || exit 1
BACKBONE=8b MODEL_SLUG=qwen3-8b METHOD_SLUG="$ARM" PP="$PP" \
  SAVE_TRAIN_STATE=0 RESUME_WEIGHTS_ONLY=1 \
  RUN_NAME="${RUN}-eval-cp$TOTAL" SAVE_DIR="${EVAL_SAVE_DIR:-${RUN_DIR%/checkpoints}/eval/cp$TOTAL}" \
  LOG="${EVAL_LOG:-${LOG%.log}.eval.log}" \
  bash "$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$RUN_DIR" "$((TOTAL - 1))" \
  >> "$STATE_LOG" 2>&1
rc=$?
say "final evaluation exited rc=$rc"
[ "$rc" -eq 0 ] || exit "$rc"

# One node also measures the base model under the new prompt. Old long-thinking
# base scores cannot serve as the control for action-only trained policies.
if [ "$ARM" = "grpo" ] && [ "${EVAL_BASE_AFTER_TRAIN:-1}" = "1" ]; then
  say "evaluating Qwen3-8B base with the same action-only protocol"
  BACKBONE=8b MODEL_SLUG=qwen3-8b METHOD_SLUG=base PP="$PP" \
    SAVE_TRAIN_STATE=0 RESUME_WEIGHTS_ONLY=1 RUN_NAME="${RUN}-base-eval" \
    SAVE_DIR="${RUN_DIR%/checkpoints}/eval/base" LOG="${LOG%.log}.base-eval.log" \
    bash "$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" --base \
    "${MODEL:-$BASE/checkpoints/Qwen3-8B}" >> "$STATE_LOG" 2>&1
  rc=$?
  say "matching base evaluation exited rc=$rc"
fi
exit "$rc"
