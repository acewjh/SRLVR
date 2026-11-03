#!/bin/bash
# Stage D: STANDARD GRPO on the math task (directive: "接着跑标准的GRPO的效果").
#
# ONE launcher for every model size: MODEL=<slug> selects the card in
# configs/models/. Convention: scripts/run_grpo_<size>.sh is a 3-line shim.
#
# Standard GRPO = slime's stock advantage estimator + stock PPO-style policy loss,
# with G samples per prompt and the plain verifier reward. No privileged teacher, no
# custom loss, no HPI weighting.
#
#   rollout   math_opsd.grpo_rollout.generate_rollout   (G-aware; the OPSD rollout is G=1)
#   loss      policy_loss          (stock, --advantage-estimator grpo)
#   reward    math_opsd.rm_math_verify.rm_math_verify  (repaired official grader)
#   G = 8, rollout_batch_size = 4  ->  GBS 32   (same global batch as every other arm)
#
# Held identical to the Reward-CV / HPI arms so the comparison is clean:
#   LoRA r64/alpha128, GBS 32, seed 42, Adam, lr 5e-6 constant, max_grad_norm 0.1,
#   T=1.1 / top_p .95 / top_k 20, max_completion 1024, 100 steps, cp25/50/75/100,
#   bshd, repaired official grader -- and the SAME TP/PP as the model card, so a
#   size's two arms differ only in the method, never in the geometry.
#
# KL: --kl-coef 0 (slime default). Kept at 0 so the run needs no reference model and
# matches the KL-free objective of the other arms.
#
# Usage: nohup bash scripts/run_grpo.sh > /dev/null 2>&1 &
set -uo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../train_scripts/lib/model_env.sh
source "$P/train_scripts/lib/model_env.sh" || exit 1

BASE="$CUA_BASE"
MODEL="$MODEL_DIR"
DATA=$BASE/data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet
LOCK=/tmp/opsd_grpo.lock

# ---- topology + run identity ------------------------------------------------
# Identity is allocated before the lock and the resume guard, because both write
# into the run's log and the guard reads its SAVE_DIR.
G="${G:-8}"
ROLLOUT_BS="${ROLLOUT_BS:-4}"          # 4 prompts x G8 = GBS 32
GLOB_BS=$(( ROLLOUT_BS * G ))
MAX_RESPONSE="${MAX_RESPONSE:-1024}"
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
OPSD_SEED="${OPSD_SEED:-42}"
MBS="${MBS:-$MODEL_MBS_DEFAULT}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"
NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_GPUS:-$(( ${NUM_NODE_GPUS:-8} - NUM_TRAIN_GPUS ))}"

# <task>-<model>-grpo[-<variant>]-<stamp>, same rule and same implementation as
# every other run (scripts/run_names.py). The variant carries what differs from
# the model card's own recipe.
VARIANT=""
_add_variant() { VARIANT="${VARIANT:+$VARIANT-}$1"; }
[ "$MBS" != "$MODEL_MBS_DEFAULT" ] && _add_variant "mbs$MBS"
[ "$MAX_RESPONSE" != "1024" ] && _add_variant "c$MAX_RESPONSE"
[ "$NUM_ROLLOUT" != "100" ] && _add_variant "r$NUM_ROLLOUT"
VARIANT="${RUN_VARIANT:-$VARIANT}"

RESUME_FROM="${RESUME_FROM:-}"
if [ -n "$RESUME_FROM" ]; then
  # Same rule as the training launcher: a resumed segment continues in the
  # directory it resumed from, and only the log gets a segment marker.
  RUN_NAME="${RUN_NAME:-$(basename "$RESUME_FROM")}"
  SAVE_DIR="${SAVE_DIR:-$RESUME_FROM}"
  LOG="${LOG:-$P/logs/${RUN_NAME}.resume.$(date +%Y%m%d_%H%M%S).log}"
else
  RUN_NAME="${RUN_NAME:-$(allocate_run_name "grpo" "$VARIANT")}" || exit 1
  SAVE_DIR="${SAVE_DIR:-$P/runs/$RUN_NAME}"
  LOG="${LOG:-$P/logs/$RUN_NAME.log}"
fi
# Legacy log-name override; it no longer influences the run directory.
[ -n "${LOG_TAG:-}" ] && : "${LOG:=$P/logs/$LOG_TAG.log}"

exec 8>"$LOCK"
if ! flock -n 8; then echo "[grpo] another GRPO run holds $LOCK -- exiting" >&2; exit 1; fi
# RESUME_FROM=<save dir> continues an interrupted run instead of starting fresh.
# Same semantics (and the same reset-Adam caveat) as run_math_opsd_slime_npu.sh:
# the LoRA checkpoints are written with --no-save-optim, so a resumed tail is not
# a continuous continuation. The "refuse to overwrite" guard is kept for the
# fresh path so a finished run can never be clobbered by accident.
# (RESUME_FROM was read and consumed up in the run-identity block)
RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  [ -s "$RESUME_FROM/latest_checkpointed_iteration.txt" ] || {
    echo "[grpo] FATAL: RESUME_FROM=$RESUME_FROM has no latest_checkpointed_iteration.txt" >&2; exit 1; }
  RESUME_ARGS=(--no-load-optim --no-load-rng)
  echo "[grpo] RESUMING from $RESUME_FROM (iter $(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")) -- optimizer state absent, Adam restarts" >> "$LOG"
elif [ -e "$SAVE_DIR" ]; then
  echo "[grpo] FATAL: $SAVE_DIR already exists -- refusing to overwrite" >&2; exit 1
fi

source "$P/train_scripts/lib/npu_env.sh"
opsd_enter_env
opsd_export_npu_env "$P" "expandable_segments:True"

# standard GRPO: LoRA kept ON to match the other arms
export OPSD_LORA="${OPSD_LORA:-1}"
export OPSD_LORA_R="${OPSD_LORA_R:-64}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"
export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"

# (topology and run identity were fixed above, before the lock)

# world_size = TP x PP x DP, so DP is whatever is left of the actor NPUs; the
# Megatron invariant GBS % (MBS x DP) == 0 is checked here rather than being
# rediscovered as a cryptic megatron assert mid-run.
DP=$(( NUM_TRAIN_GPUS / ( MODEL_TP * MODEL_PP ) ))
if [ "$DP" -lt 1 ]; then
  echo "[grpo] FATAL: --actor-num-gpus-per-node $NUM_TRAIN_GPUS < TP x PP = $(( MODEL_TP * MODEL_PP )) for $MODEL_NAME" >> "$LOG"
  exit 1
fi
if [ "$(( GLOB_BS % ( MBS * DP ) ))" -ne 0 ]; then
  echo "[grpo] FATAL: GBS=$GLOB_BS is not a multiple of MBS x DP = $MBS x $DP = $(( MBS * DP ))" >> "$LOG"
  exit 1
fi

mkdir -p "$SAVE_DIR" "$P/logs"
echo "[grpo] run=$RUN_NAME (task=$MODEL_TASK model=$MODEL_SLUG method=grpo variant=${VARIANT:-none})" >> "$LOG"
echo "[grpo] model=$MODEL_NAME slug=$MODEL_SLUG dir=$MODEL" >> "$LOG"
echo "[grpo] parallel: tp=$MODEL_TP pp=$MODEL_PP dp=$DP (actor_npus=$NUM_TRAIN_GPUS) rollout_npus=$NUM_ROLLOUT_GPUS mbs=$MBS" >> "$LOG"
echo "[grpo] G=$G rollout_bs=$ROLLOUT_BS GBS=$GLOB_BS steps=$NUM_ROLLOUT seed=$OPSD_SEED save_dir=$SAVE_DIR" >> "$LOG"

if ! ray status 2>/dev/null | grep -q "1 node"; then
  ray stop --force 2>/dev/null || true
  ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats \
      --dashboard-host 0.0.0.0 --dashboard-port 8265 2>&1 | tail -2
fi
cleanup_ray() { ray stop --force 2>/dev/null || true; }
trap cleanup_ray EXIT

cd "$P"
python "$P/frameworks/slime-ascend/train.py" \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "$MODEL" \
   --load "${RESUME_FROM:-$MODEL}" \
   --save "$SAVE_DIR" \
   --save-interval "$SAVE_INTERVAL" \
   "${RESUME_ARGS[@]}" \
   --prompt-data "$DATA" \
   --input-key problem \
   --label-key solution \
   --rollout-function-path math_opsd.grpo_rollout.generate_rollout \
   --rollout-batch-size "$ROLLOUT_BS" \
   --over-sampling-batch-size "$ROLLOUT_BS" \
   --n-samples-per-prompt "$G" \
   --rollout-max-response-len "$MAX_RESPONSE" \
   --rollout-temperature 1.1 \
   --rollout-top-p 0.95 \
   --rollout-top-k 20 \
   --rollout-shuffle \
   --rollout-num-gpus "$NUM_ROLLOUT_GPUS" \
   --rollout-num-gpus-per-engine 1 \
   --sglang-attention-backend ascend \
   --sglang-mem-fraction-static "$MODEL_SGLANG_MEM_FRACTION" \
   --sglang-context-length 32768 \
   --sglang-chunked-prefill-size "$MODEL_SGLANG_CHUNKED_PREFILL" \
   --sglang-disable-radix-cache \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$NUM_TRAIN_GPUS" \
   --num-gpus-per-node "${NUM_NODE_GPUS:-8}" \
   --num-rollout "$NUM_ROLLOUT" \
   --transformer-impl local \
   --qkv-format bshd \
   --tensor-model-parallel-size "$MODEL_TP" \
   --pipeline-model-parallel-size "$MODEL_PP" \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --global-batch-size "$GLOB_BS" \
   --micro-batch-size "$MBS" \
   --calculate-per-token-loss \
   --loss-type policy_loss \
   --advantage-estimator grpo \
   --eps-clip 0.2 \
   --kl-coef 0 \
   --custom-rm-path math_opsd.rm_math_verify.rm_math_verify \
   --optimizer adam \
   --lr 5e-6 \
   --lr-decay-style constant \
   --weight-decay 0.0 \
   --adam-beta1 0.9 \
   --adam-beta2 0.999 \
   --clip-grad 0.1 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   --use-flash-attn \
   --no-gradient-accumulation-fusion \
   --log-interval 1 \
   --seed "$OPSD_SEED" \
   --custom-model-provider-path math_opsd.lora_megatron.custom_model_provider \
   --only-train-params-name-list '.*\.lora_.*' \
   --no-save-optim \
   --no-save-rng \
   >> "$LOG" 2>&1
echo "[grpo] exited rc=$? $(date '+%F %H:%M:%S')" >> "$LOG"
