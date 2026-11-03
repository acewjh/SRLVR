#!/bin/bash
# ============================================================
# Shared configuration for the four Qwen3-1.7B ALFWorld arms.
#
# Sourced by run_alfworld_*.sh. The point of this file is that the four arms are
# matched BY CONSTRUCTION, not by four copies of the same flags drifting apart:
# the ONLY thing a per-arm script may change is METHOD (and, transitively, the
# loss/hook selection that METHOD implies).
#
# Held identical across Base / GRPO / Outcome-only / Future-HPI:
#   model      Qwen3-1.7B, thinking ON (enable_thinking=True)
#   prompt     the same system + per-turn transcript + admissible-action block
#   sampling   T=0.6 / top_p 0.95 / top_k 20  (Qwen3 thinking recommendation)
#   rollout    G=8 rollouts per task, same task split, same max turns
#   layout     identical per-turn segment layout and loss mask
#   optimizer  LoRA r64/alpha128, Adam, lr 5e-6 constant, clip-grad 0.1
#   budget     150 steps, checkpoints at 25/50/75/100/150
#   eval       valid_seen + valid_unseen, terminal success reward
#
# LoRA (rather than full fine-tuning) mirrors the proven config for this stack in
# scripts/run_grpo.sh. It also matters for memory: the per-turn-segment layout is
# O(T^2) in context length, so anything that shrinks optimizer/activation state is
# load-bearing here.
#
# Usage: METHOD=<grpo|hpi_outcome_only|hpi_future> bash tasks/alfworld/v1/scripts/run_*.sh
# ============================================================
set -uo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
source "$PROJ/tasks/alfworld/scripts/_runtime_defaults.sh"
MODEL=$BASE/checkpoints/Qwen3-1.7B
ALFWORLD_DATA_DIR="${ALFWORLD_DATA_DIR:-$PROJ/tasks/alfworld/v1/configs/data}"

METHOD="${METHOD:?METHOD must be set by the calling run_alfworld_*.sh}"
RUN_NAME="${RUN_NAME:-alfworld_${METHOD}_1p7b}"

# ---- matched experiment knobs -------------------------------------------------
G="${G:-8}"                                  # rollouts per task (do NOT sweep)
ROLLOUT_BS="${ROLLOUT_BS:-2}"                # tasks per step -> GBS = ROLLOUT_BS * G
GLOB_BS=$(( ROLLOUT_BS * G ))
# Turn budget. Bounded by MEMORY, not by task needs: the LoRA lm_head is frozen but
# still materialises a full-vocab fp32 output for the whole padded sequence
# (`linear_with_frozen`), and bshd pads to the LONGEST sample in the batch, so the
# cost tracks turns x MAX_RESPONSE, not the mean. Measured on a 29.5 GB card with
# TP=1 + full activation recompute + loss recompute:
#   turns=4  cap=768  (resp mean 1610, max ~2.5k)  -> fits
#   turns=10 cap=1024 (resp mean 3437, max 5148)   -> OOM (needs 3.08 GiB at 25.9/29.5 used)
# What the budget is spent on is a CHOICE: the peak tracks turns x cap, so within
# a fixed token budget more turns are strictly better for the task (the agent gets
# more interaction steps, which is what ALFWorld needs) at the cost of shorter
# generations. Measured averages are ~290 generated tokens/turn, so cap=384 is
# ~1.3x the mean and the brevity instruction keeps most turns inside it.
#   turns=5 / cap=640  -> response max 2775  (fits)
#   turns=6 / cap=768  -> response max 3073  (OOM)
#   turns=5 / cap=768  -> OOM on the GRPO arm (24.4/29.5 GB allocated)
#   turns=8 / cap=384  -> fits, but 38% of turns are TRUNCATED mid-<action>
#                         (measured: 96 truncated vs 2 not_admissible), i.e. the
#                         model needs ~1024 tokens to finish thinking, so spending
#                         the budget on more-but-shorter turns destroys the action.
# The binding constraint is that the peak tracks turns x cap AND cap must stay
# large enough for the format to complete. That leaves turns ~= 5 on this card size.
#
# The HPI arms have ~3.4 GB LESS headroom than GRPO: the frozen privileged teacher
# (a second 1.7B in bf16, `methods.opsd_loss._get_teacher`) is resident during the
# loss, so its activations stack on top of the actor's. Measured: GRPO survived
# response max 2775; HPI OOMed at 2769. Both arms share this cap so the comparison
# stays matched -- the ceiling is set by the more expensive arm.
# Lifting this needs a TP-aware LoRA provider (TP=2 currently dies with
# "4096 and 2048 cannot broadcast"), not a different batch size.
MAX_TURNS="${MAX_TURNS:-5}"
# Per-turn generation cap. 1024 was the first guess and the Gate A/B smoke showed
# it is too small: 44% of turns hit it while still inside <think>, so the episode
# burned its turn budget without ever emitting an action. Completed turns run
# p50=487 / p90=899, so 2048 gives real headroom while still bounding the layout.
# Per-turn generation cap. 1024 was too small BEFORE the brevity instruction
# landed (44% of turns were cut off inside <think>); with that instruction in the
# prompt the truncation rate fell to 6%, so 1024 is affordable again -- and it has
# to be, because the loss materialises full-vocab fp32 logits over the whole packed
# sequence (slime `get_responses`: `logits = logits.float()`), and 2048 x 8 turns
# OOMs a 32 GB card.
MAX_RESPONSE="${MAX_RESPONSE:-512}"          # per-turn generation cap
NUM_ROLLOUT="${NUM_ROLLOUT:-150}"
EVAL_ARGS=()
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"         # -> 25/50/75/100/150
OPSD_SEED="${OPSD_SEED:-42}"
ALFWORLD_ACTION_HISTORY="${ALFWORLD_ACTION_HISTORY:-8}"

# ---- HPI knobs (only meaningful for the HPI arms) -----------------------------
# Scale is *required* for the HPI arms and must come from a calibration run on
# this model/task; the default here is a placeholder that the launcher refuses to
# accept silently (see the ALFWORLD_HPI_SCALE check below).
ALFWORLD_HPI_ALPHA="${ALFWORLD_HPI_ALPHA:-0.5}"
ALFWORLD_HPI_FUTURE_GAP="${ALFWORLD_HPI_FUTURE_GAP:-2}"
ALFWORLD_HPI_SCALE="${ALFWORLD_HPI_SCALE:-}"

# ---- method switch ------------------------------------------------------------
case "$METHOD" in
  grpo)
    # Stock GRPO: slime's own advantage estimator and policy loss, w == 1.
    ALFWORLD_HPI_MODE="none"
    LOSS_ARGS=( --loss-type policy_loss )
    ;;
  hpi_outcome_only)
    ALFWORLD_HPI_MODE="outcome_only"
    LOSS_ARGS=(
      --loss-type custom_loss
      --custom-loss-function-path tasks.alfworld.v1.loss.alfworld_hpi_loss
      --advantage-estimator grpo
      --eps-clip 0.2
    )
    ;;
  hpi_future)
    ALFWORLD_HPI_MODE="future_outcome"
    LOSS_ARGS=(
      --loss-type custom_loss
      --custom-loss-function-path tasks.alfworld.v1.loss.alfworld_hpi_loss
      --advantage-estimator grpo
      --eps-clip 0.2
    )
    ;;
  base_eval)
    # Eval-only: same rollout, no training step. --num-rollout 0 makes slime run
    # the eval loop and exit, exactly like train_scripts/run_dev_eval.sh.
    ALFWORLD_HPI_MODE="none"
    LOSS_ARGS=( --loss-type policy_loss )
    NUM_ROLLOUT=0
    EVAL_ARGS=(
      --eval-interval 1
      --eval-config "${ALFWORLD_EVAL_CONFIG:-$PROJ/tasks/alfworld/v1/configs/eval_alfworld.json}"
      --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
      --eval-max-response-len "$MAX_RESPONSE"
    )
    ;;
  *)
    echo "[alfworld] FATAL: unknown METHOD=$METHOD" >&2; exit 2 ;;
esac

if [ "$METHOD" != "grpo" ] && [ "$METHOD" != "base_eval" ] && [ -z "$ALFWORLD_HPI_SCALE" ]; then
  echo "[alfworld] FATAL: METHOD=$METHOD needs ALFWORLD_HPI_SCALE (calibrate it first: see README)." >&2
  echo "[alfworld]        The weight is w = 1 + alpha*tanh(|e|/scale); an uncalibrated scale silently sets the strength." >&2
  exit 2
fi

export ALFWORLD_HPI_MODE ALFWORLD_HPI_ALPHA ALFWORLD_HPI_FUTURE_GAP ALFWORLD_ACTION_HISTORY
export ALFWORLD_MAX_TURNS="$MAX_TURNS"
[ -n "$ALFWORLD_HPI_SCALE" ] && export ALFWORLD_HPI_SCALE
# ---- GPU split ----------------------------------------------------------------
# Eval-only has no real training step, so give the engines as many cards as
# possible: 7 rollout + 1 actor is the split the math dev-eval launcher uses.
# Training keeps the 4+4 split from the proven scripts/run_grpo.sh.
if [ "$METHOD" = "base_eval" ]; then
  NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_GPUS:-7}"
  NUM_ACTOR_GPUS="${NUM_ACTOR_GPUS:-1}"
else
  NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_GPUS:-4}"
  NUM_ACTOR_GPUS="${NUM_ACTOR_GPUS:-4}"
fi

# ---- tensor parallelism --------------------------------------------------------
# TP=1 is what every proven math/GRPO launcher here uses, but ALFWorld needs TP>1
# for MEMORY, not for speed: the per-turn layout reaches ~3.4k tokens per episode
# and the loss materialises full-vocab fp32 tensors over all of it
# (`get_responses` does `.float()` and `.div()`; Megatron's vocab-parallel
# cross-entropy adds more). Measured: 21 GB free at step start, step needs >25 GB,
# so it OOMed at TP=1 even with full activation recompute. TP shards the vocab
# dimension and would halve those tensors -- but TP=2 is INCOMPATIBLE with this
# repo's LoRA path: `methods.lora_megatron.custom_model_provider` is not
# TP-aware and dies with `4096 and 2048 cannot broadcast` (the hidden dim is
# sharded to 1024/rank while the provider still assumes 2048). So TP stays 1 and
# the memory budget is met by bounding the episode length instead.
TP="${TP:-1}"

export ALFWORLD_OVERLAY_SITE="${ALFWORLD_OVERLAY_SITE:-$BASE/envs/alfworld_overlay/lib/python3.10/site-packages}"

# ---- eval-only LR-schedule fix -------------------------------------------------
# slime computes `train_iters = num_rollout * rollout_bs * G // GBS`, then falls
# back to `lr_decay_iters = train_iters` when --lr-decay-iters is absent. With
# --num-rollout 0 (eval-only) that is 0, and Megatron aborts on
# `assert self.lr_decay_steps > 0` during actor init -- before any rollout runs.
# The proven eval launcher (train_scripts/run_dev_eval.sh:216) passes 1 for this
# reason; with --lr-decay-style constant the value cannot affect the schedule, so
# this only unblocks the assert and leaves the training arms untouched.
LR_DECAY_ARGS=()
if [ "${NUM_ROLLOUT:-1}" -eq 0 ]; then
  LR_DECAY_ARGS=( --lr-decay-iters 1 )
fi

SAVE_DIR="${SAVE_DIR:-$PROJ/runs/$RUN_NAME}"
LOG="${LOG:-$PROJ/logs/${RUN_NAME}.log}"
LOCK="/tmp/opsd_${RUN_NAME}.lock"

mkdir -p "$SAVE_DIR" "$(dirname "$LOG")" "$PROJ/logs"

# ---- single-instance guard ----------------------------------------------------
exec 8>"$LOCK"
if ! flock -n 8; then echo "[alfworld] another $RUN_NAME run holds $LOCK -- exiting" >> "$LOG"; exit 1; fi

# ---- resume / refuse-to-clobber ----------------------------------------------
RESUME_FROM="${RESUME_FROM:-}"
RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  [ -s "$RESUME_FROM/latest_checkpointed_iteration.txt" ] || {
    echo "[alfworld] FATAL: RESUME_FROM=$RESUME_FROM has no latest_checkpointed_iteration.txt" >> "$LOG"; exit 1; }
  RESUME_ARGS=(--no-load-optim --no-load-rng)
  echo "[alfworld] RESUME from $RESUME_FROM iter $(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")" >> "$LOG"
elif [ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "[alfworld] FATAL: $SAVE_DIR already holds a run -- refusing to overwrite (set RESUME_FROM to continue)" >> "$LOG"
  exit 1
fi

# ---- environment --------------------------------------------------------------
source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NODE_IFNAME="${NODE_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME=$NODE_IFNAME TP_SOCKET_IFNAME=$NODE_IFNAME HCCL_SOCKET_IFNAME=$NODE_IFNAME
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONBUFFERED=16
# G=8 per task plus 15-turn episodes means many live sequences; the anti-fragmentation
# allocator is the arm most exposed to dynamic-memory fragmentation. Must be exported
# BEFORE `ray start` so Ray actors inherit it (same reasoning as scripts/run_grpo.sh).
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/Megatron-LM:$PROJ/frameworks/Megatron-Bridge/src:$PROJ/frameworks/mbridge:$PROJ/frameworks/MindSpeed:$PROJ/frameworks/sglang/python:$PROJ/frameworks/slime-ascend:$PROJ:$PYTHONPATH"
export WANDB_MODE=disabled WANDB_DISABLED=true
export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"

# LoRA: mirrors scripts/run_grpo.sh so the arms are comparable with the math runs.
export OPSD_LORA="${OPSD_LORA:-0}"
export OPSD_LORA_R="${OPSD_LORA_R:-64}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"
LORA_ARGS=(--custom-model-provider-path methods.lora_megatron.custom_model_provider)
if [ "$OPSD_LORA" = "1" ]; then
  LORA_ARGS+=(--only-train-params-name-list '.*\.lora_.*')
fi
GRAPH_ARGS=()
if [ "$SGLANG_ENABLE_NPU_GRAPH" = "0" ]; then
  GRAPH_ARGS=(--sglang-disable-cuda-graph)
else
  GRAPH_ARGS=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-8}")
fi

MODEL_ARGS=(
   --swiglu --num-layers 28 --hidden-size 2048 --ffn-hidden-size 6144
   --num-attention-heads 16 --group-query-attention --num-query-groups 8
   --use-rotary-position-embeddings --disable-bias-linear --normalization "RMSNorm"
   --norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 151936
   --padded-vocab-size 151936 --kv-channels 128 --qk-layernorm
)

# ---- sanity: the harness must be importable before we burn a Ray startup ------
if ! python -c "import tasks.alfworld.v1.rollout" 2>/dev/null; then
  echo "[alfworld] FATAL: 'import tasks.alfworld.v1.rollout' failed under this interpreter/PYTHONPATH." >> "$LOG"
  echo "[alfworld]        Run tasks/alfworld/v1/scripts/smoke_alfworld_cpu.sh first." >> "$LOG"
  exit 3
fi
if [ ! -s "$ALFWORLD_DATA_DIR/train.jsonl" ]; then
  echo "[alfworld] FATAL: $ALFWORLD_DATA_DIR/train.jsonl missing; run tasks/alfworld/v1/scripts/setup_alfworld.sh" >> "$LOG"
  exit 3
fi

echo "[alfworld] method=$METHOD mode=$ALFWORLD_HPI_MODE G=$G rollout_bs=$ROLLOUT_BS GBS=$GLOB_BS steps=$NUM_ROLLOUT max_turns=$MAX_TURNS seed=$OPSD_SEED" | tee -a "$LOG"

# ---- Ray ----------------------------------------------------------------------
if ! ray status 2>/dev/null | grep -q "1 node"; then
  ray stop --force 2>/dev/null || true
  ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats \
      --dashboard-host 0.0.0.0 --dashboard-port 8265 2>&1 | tail -2
fi
cleanup_ray() { ray stop --force 2>/dev/null || true; }
trap cleanup_ray EXIT

cd "$PROJ"

# Activation recomputation (flags in TRAIN_ARGS): the ALFWorld per-turn layout is
# long -- ~6 turns x (prompt + generation) reaches ~3.8k tokens even at a
# 1024-token cap -- and the forward + loss materialise full-vocab fp32 logits over
# all of it. Without recompute, Gate C OOMed at 25.6/29.5 GB allocated while the
# 1.7B weights are only ~4 GB, i.e. activations were the peak, not weights.
# full/block/1 is the strongest setting (one layer at a time); ~30% step time.
#
# `--recompute-loss-function` additionally recomputes the loss in backward. The
# loss is where the peak is: it holds several full-vocab fp32 tensors of shape
# [tokens, 151936] (`get_responses` does .float() then .div(), and Megatron's
# vocab-parallel cross-entropy adds more). Layer recompute alone did not fix the OOM.
python "$PROJ/frameworks/slime-ascend/train.py" \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "$MODEL" \
   --load "${RESUME_FROM:-$MODEL}" \
   --save "$SAVE_DIR" \
   --save-interval "$SAVE_INTERVAL" \
   "${RESUME_ARGS[@]}" \
   --prompt-data "$ALFWORLD_DATA_DIR/train.jsonl" \
   --input-key prompt \
   --metadata-key metadata \
   --rollout-function-path tasks.alfworld.v1.rollout.generate_rollout \
   --custom-convert-samples-to-train-data-path tasks.alfworld.v1.train_data.convert_samples_to_train_data \
   --rollout-batch-size "$ROLLOUT_BS" \
   --over-sampling-batch-size "$ROLLOUT_BS" \
   --n-samples-per-prompt "$G" \
   --rollout-max-response-len "$MAX_RESPONSE" \
   --rollout-temperature 0.6 \
   --rollout-top-p 0.95 \
   --rollout-top-k 20 \
   --rollout-shuffle \
   --rollout-num-gpus "$NUM_ROLLOUT_GPUS" \
   --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
   --sglang-attention-backend ascend \
   --sglang-mem-fraction-static 0.82 \
   --sglang-context-length 40960 \
   --sglang-chunked-prefill-size 8192 \
   "${GRAPH_ARGS[@]}" \
   --sglang-disable-radix-cache \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "$NUM_ACTOR_GPUS" \
   --num-gpus-per-node 8 \
   --num-rollout "$NUM_ROLLOUT" \
   --transformer-impl local \
   --qkv-format bshd \
   --tensor-model-parallel-size "$TP" \
   --pipeline-model-parallel-size 1 \
   --context-parallel-size 1 \
   --expert-model-parallel-size 1 \
   --global-batch-size "$GLOB_BS" \
   --micro-batch-size 1 \
   --calculate-per-token-loss \
   "${LOSS_ARGS[@]}" \
   --optimizer adam \
   --lr 5e-6 \
   --lr-decay-style constant \
   "${LR_DECAY_ARGS[@]}" \
   --weight-decay 0.0 \
   --adam-beta1 0.9 \
   --adam-beta2 0.999 \
   --clip-grad 0.1 \
   --recompute-loss-function \
   --recompute-granularity full \
   --recompute-method block \
   --recompute-num-layers 1 \
   --attention-dropout 0.0 \
   --hidden-dropout 0.0 \
   --attention-softmax-in-fp32 \
   --attention-backend flash \
   --use-flash-attn \
   --no-gradient-accumulation-fusion \
   --log-interval 1 \
   --seed "$OPSD_SEED" \
   "${EVAL_ARGS[@]}" \
   --save-debug-rollout-data "$SAVE_DIR/rollout_{rollout_id}.pt" \
   "${LORA_ARGS[@]}" \
   >> "$LOG" 2>&1
echo "[alfworld] $METHOD exited rc=$? $(date '+%F %H:%M:%S')" >> "$LOG"
