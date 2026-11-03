#!/bin/bash
# ============================================================
# opsd-slime: math OPSD / Reward-CV training on 8x Ascend 910B (slime harness).
#
# ONE launcher for every model size. The size-specific values live in
# configs/models/<slug>.sh and are selected with MODEL=<slug>:
#   MODEL=qwen3-1p7b  (default)      MODEL=qwen3-4b      MODEL=qwen3-8b
# Convention: train_scripts/run_math_opsd_slime_npu_<size>.sh is a 3-line shim
# that sets MODEL and execs this file.
#
# Official OPSD config this harness reproduces (opsd-npu oracle):
#   GBS 32, G=1, T=1.1, top_p 0.95, top_k 20, max_completion 1024,
#   beta=0, lambda=1, jsd_token_clip=0.05, full-vocab, fixed teacher
#   (= initial policy), lr 5e-6 constant, max_grad_norm 0.1.
#
# Deviations are listed in reports/STATUS.md, and the per-size ones in the model
# card that introduces them.
#
# Usage:
#   MODEL=qwen3-4b bash train_scripts/run_math_opsd_slime_npu.sh        # full 4+4
#   SMOKE=1 MODEL=qwen3-8b bash ...                                     # 1 train + 1 rollout, 2 prompts, 128 tokens
#   SMOKE10=1 MODEL=qwen3-8b bash ...                                   # 4+4, 10 steps
# ============================================================
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/model_env.sh
source "$PROJ/train_scripts/lib/model_env.sh" || exit 1

BASE="$CUA_BASE"
MODEL="$MODEL_DIR"
DATA=$BASE/data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet
# SAVE_DIR / LOG are allocated below, once the topology has fixed the variant.

source "$PROJ/train_scripts/lib/npu_env.sh"
opsd_enter_env
# $MODEL_ALLOC_CONF comes from the model card (empty on 1.7B, expandable_segments
# on the cards near the ceiling); must be exported before `ray start`.
opsd_export_npu_env "$PROJ" "$MODEL_ALLOC_CONF"

# ---- method switch: resolved from opsd_core/registry.py ----
# The method vocabulary used to live here as a ~200-line if-chain, maintained
# independently alongside tasks.alfworld/scripts/_common_v2.sh (as `ARM`) and
# scripts/run_names.py (as METHOD_SLUG). The three could drift. The registry is
# now the single source; this block consumes the shell fragment it emits.
#
# A retired or unknown method makes the registry exit 2, which fails the launch
# LOUDLY instead of silently falling through to the `opsd` default.
export OPSD_METHOD="${OPSD_METHOD:-opsd}"
REGISTRY_FRAG=$(python -m opsd_core.registry --task math --method "$OPSD_METHOD" --shell) || {
  echo "run_math_opsd_slime_npu: cannot resolve method '$OPSD_METHOD'" >&2
  exit 2
}
eval "$REGISTRY_FRAG"

# Rows emitted per prompt by the HSI rollout (methods/hsi_rollout.py): one row
# per chunk. Every other arm emits one row per completion.
HSI_CHUNKS=1
if [ "$OPSD_METHOD" = "hsi" ]; then
  HSI_CHUNKS="$OPSD_HSI_CHUNKS"
  # GBS is the ROW count of one optimizer step and the framework runs
  # `rows_returned // GBS` steps per rollout, so GBS is pinned to
  # HSI_STEP_PROMPTS x OPSD_HSI_CHUNKS to get exactly one step per rollout
  # (rollout cadence == step cadence, so save-interval-in-rollouts is also
  # save-interval-in-steps). ROLLOUT_BS sits above HSI_STEP_PROMPTS so a prompt
  # that yields fewer than K rows still leaves a full step.
  GBS=$(( HSI_STEP_PROMPTS * OPSD_HSI_CHUNKS ))
  # slime's httpx rollout client runs with Timeout(None) and its retry loop only
  # fires on an error, so a /generate that never returns hangs the rollout
  # forever -- the silent 2h47m/2h54m wedge. Bound each attempt and retry.
  export OPSD_ROLLOUT_TIMEOUT_S="${OPSD_ROLLOUT_TIMEOUT_S:-900}"
  export OPSD_ROLLOUT_ATTEMPTS="${OPSD_ROLLOUT_ATTEMPTS:-3}"
fi

# Method-intrinsic extra CLI flags, emitted by the registry as a bash array
# (e.g. --use-opsd --use-rollout-logprobs for the HSI arms).
HSI_ARGS=("${EXTRA_ARGS[@]}")

# Full-softmax arms must NOT pass --rollout-top-p/-k: slime's own defaults are
# already 1.0/-1, which is exactly what the estimator needs, and passing
# `--rollout-top-k -1` (a negative number as a separate token) corrupts slime's
# arg parser and kills the run at import (root-caused 2026-09-23).
VNEXT_OMIT_SAMPLER_FLAGS="${OPSD_OMIT_SAMPLER_FLAGS}"

# official-config LoRA (r64/alpha128/dropout0, 7 HF target modules). On for the
# arms the registry marks LoRA; `opsd` itself runs full-parameter.
if [ "$OPSD_LORA" = "1" ]; then
  export OPSD_LORA_R="${OPSD_LORA_R:-64}"
  export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
  export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"
  # --no-save-optim: the DistributedOptimizer's param objects are rebound to its
  # fp32 main-param buffers, so its sharded_state_dict cannot be matched against
  # the model tree and the step-25 save crashes. Model weights (incl. adapters)
  # still save; optimizer state does not -> mid-run resume loses Adam moments
  # (documented deviation; the official baseline runs plain accelerate-Adam).
  # The HSI arm overrides MODEL_PROVIDER to publish the theta_old handle the
  # scorer needs; it wraps the stock LoRA provider, so the adapter tree is
  # unchanged.
  LORA_ARGS=(--custom-model-provider-path "${MODEL_PROVIDER:-methods.lora_megatron.custom_model_provider}"
             --only-train-params-name-list '.*\.lora_.*'
             --no-save-optim
             --no-save-rng)
else
  LORA_ARGS=
fi

# Activation recomputation. OFF by default on every size.
#
# ============================ DO NOT RE-ENABLE ============================
# D2 as originally set (full/block/36 = ALL layers) SILENTLY ZEROES EVERY
# GRADIENT on a LoRA arm. Proven 2026-09-25 on 4B: the run it "fixed" logged
# train/grad_norm == 0.0 for all 100 steps, and its cp24 vs cp99 model shards
# were byte-identical (md5 8b418666bb12 / d6da4cdb4945 / efddb47c3065) -- 75
# optimizer steps that changed not one weight. Cause: with
# --only-train-params-name-list '.*\.lora_.*' the base weights are frozen, so
# the checkpointed block's inputs carry no grad and Megatron's reentrant
# CheckpointFunction (frameworks/Megatron-LM/.../tensor_parallel/random.py,
# backward) detaches the recomputed graph from the LoRA parameters.
# It looked like a memory fix because a detached graph needs no backward
# activations -- the OOM went away for the wrong reason.
# Attempt 2 (same config, NO recompute) logged grad_norm 0.0055-0.0068: healthy.
# =========================================================================
#
# Memory is instead handled at the source of the spike: the redundant
# single-element torch.cat in teacher_forward_completion_logits (~594 MB, the
# exact allocation that OOMed) plus the allocator settings below.
if [ "${OPSD_RECOMPUTE:-0}" = "1" ]; then
  RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method block
                  --recompute-num-layers "${OPSD_RECOMPUTE_LAYERS:-36}")
else
  RECOMPUTE_ARGS=
fi

# Sampler flags. Omitted for the vNext arms so slime's own defaults
# (top_p=1.0 / top_k=-1) apply; set explicitly for every other arm.
if [ "${VNEXT_OMIT_SAMPLER_FLAGS:-0}" = "1" ]; then
  SAMPLER_ARGS=
else
  SAMPLER_ARGS=(--rollout-top-p 0.95 --rollout-top-k 20)
fi

# ---- official OPSD constants (loss) ----
export OPSD_BETA=0
export OPSD_JSD_TOKEN_CLIP=0.05
export OPSD_LOSS_TEMPERATURE=1.1
export OPSD_TEACHER_CHUNK="${OPSD_TEACHER_CHUNK:-2}"

# ---- topology ----
if [ "${SMOKE:-0}" = "1" ]; then
  # The actor must still be able to build TP x PP ranks, so the 1-actor smoke is
  # only valid for a card that fits on one card (1.7B/4B). The 8B card is PP2 and
  # its smoke therefore drives 2 actor NPUs.
  NUM_TRAIN_GPUS=$(( MODEL_TP * MODEL_PP ))
  NUM_ROLLOUT_GPUS=1; ROLLOUT_BS=2; OVERSAMPLE_BS=2
  MAX_RESPONSE=128; GBS=$(( ROLLOUT_BS * HSI_CHUNKS )); MBS=1; NUM_ROLLOUT=1; SAVE_INTERVAL=1000000
elif [ "${SMOKE10:-0}" = "1" ]; then
  NUM_TRAIN_GPUS=4; NUM_ROLLOUT_GPUS=4; ROLLOUT_BS=32; OVERSAMPLE_BS=32
  MAX_RESPONSE=256; GBS=$(( ROLLOUT_BS * HSI_CHUNKS )); MBS="${MBS:-$MODEL_MBS_DEFAULT}"; NUM_ROLLOUT=10; SAVE_INTERVAL=1000000
else
  NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-4}"; NUM_ROLLOUT_GPUS="${NUM_ROLLOUT_GPUS:-4}"
  ROLLOUT_BS="${ROLLOUT_BS:-32}"; OVERSAMPLE_BS="${OVERSAMPLE_BS:-32}"
  MAX_RESPONSE="${MAX_RESPONSE:-1024}"; GBS="${GBS:-32}"; MBS="${MBS:-$MODEL_MBS_DEFAULT}"
  NUM_ROLLOUT="${NUM_ROLLOUT:-25}"; SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
fi

# ---- run identity ------------------------------------------------------------
# <task>-<model>-<method>[-<variant>]-<stamp>, from the one rule implementation.
# The variant records only what this run does DIFFERENTLY from its model card's
# own default recipe -- the card already fixes the geometry that belongs to the
# model, and repeating it in every name would bury the one knob that matters.
VARIANT=""
_add_variant() { VARIANT="${VARIANT:+$VARIANT-}$1"; }
[ "${SMOKE:-0}" = "1" ] && _add_variant "smoke1"
[ "${SMOKE10:-0}" = "1" ] && _add_variant "smoke10"
[ "$MBS" != "$MODEL_MBS_DEFAULT" ] && _add_variant "mbs$MBS"
[ "$MAX_RESPONSE" != "1024" ] && _add_variant "c$MAX_RESPONSE"
[ "$NUM_ROLLOUT" != "100" ] && _add_variant "r$NUM_ROLLOUT"
VARIANT="${RUN_VARIANT:-$VARIANT}"
NAME_METHOD="$(name_rule_method "$OPSD_METHOD")" || exit 1

RESUME_FROM="${RESUME_FROM:-}"
if [ -n "$RESUME_FROM" ]; then
  # A resumed segment continues in the directory it resumed FROM -- that one
  # already conforms, and allocating a fresh stamped dir here would scatter one
  # experiment's checkpoints across two directories with no way back. Only the
  # log gets a segment marker, so a resumed run cannot overwrite its own history.
  RUN_NAME="${RUN_NAME:-$(basename "$RESUME_FROM")}"
  SAVE_DIR="${SAVE_DIR:-$RESUME_FROM}"
  LOG="${LOG:-$PROJ/logs/${RUN_NAME}.resume.$(date +%Y%m%d_%H%M%S).log}"
else
  RUN_NAME="${RUN_NAME:-$(allocate_run_name "$NAME_METHOD" "$VARIANT")}" || exit 1
  SAVE_DIR="${SAVE_DIR:-$PROJ/runs/$RUN_NAME}"
  LOG="${LOG:-$PROJ/logs/$RUN_NAME.log}"
fi
# LOG_TAG is the pre-rule way of naming a run's log. It is honoured as a LOG
# override so an existing queue keeps writing where it expects to -- it no longer
# has any say over the RUN DIRECTORY, which always comes from the rule.
[ -n "${LOG_TAG:-}" ] && : "${LOG:=$PROJ/logs/$LOG_TAG.log}"

# ---- parallelism budget ------------------------------------------------------
# world_size = TP x PP x DP, so DP is whatever is left of the actor NPUs. The
# Megatron invariant GBS % (MBS x DP) == 0 is checked here rather than being
# rediscovered as a cryptic megatron assert 10 minutes into the run (a gate run
# once died on GBS=4 with exactly this).
DP=$(( NUM_TRAIN_GPUS / ( MODEL_TP * MODEL_PP ) ))
if [ "$DP" -lt 1 ]; then
  echo "[opsd-slime] FATAL: --actor-num-gpus-per-node $NUM_TRAIN_GPUS < TP x PP = $(( MODEL_TP * MODEL_PP )) for $MODEL_NAME"
  exit 1
fi
if [ "$(( GBS % ( MBS * DP ) ))" -ne 0 ]; then
  echo "[opsd-slime] FATAL: GBS=$GBS is not a multiple of MBS x DP = $MBS x $DP = $(( MBS * DP ))"
  exit 1
fi

mkdir -p "$SAVE_DIR" "$PROJ/logs"

# ---- model args: from the model card (configs/models/$MODEL_SLUG.sh) ----

# --load points at the HF checkpoint: slime detects a non-megatron dir and loads
# HF weights into the megatron model via mbridge (avoids torch-2.1 dcp load,
# which megatron-core 0.16's planner kwargs don't support).
# Resume support (2026-09-18): pointing --load at a SAVE_DIR that holds
# latest_checkpointed_iteration.txt makes slime CONTINUE from that iteration
# (slime/utils/arguments.py only zeroes start_rollout_id for fresh/HF loads).
# RESUME MUST USE THE SAME TP/PP AS THE INTERRUPTED RUN: the checkpoint is
# sharded on that geometry. Both come from the model card here, so resuming an
# 8B run means MODEL=qwen3-8b, not a flag you have to remember.
# The LoRA checkpoints are written with --no-save-optim (megatron's own optimizer
# save crashes on this LoRA tree), so megatron's state file carries no optimizer
# moments and the load side is pinned to --no-load-optim. When
# OPSD_OPTIM_SIDECAR=1 the Adam moments are restored from the per-rank sidecar
# instead (methods/optim_sidecar.py), which is what makes a resumed arm the
# same experiment as a continuous one. Arms without the gate still resume on a
# reset Adam -- prefer a single continuous run when the schedule allows.
# (RESUME_FROM was read and consumed up in the run-identity block)
RESUME_ARGS=
if [ -n "$RESUME_FROM" ]; then
  [ -s "$RESUME_FROM/latest_checkpointed_iteration.txt" ] || {
    echo "[opsd] FATAL: RESUME_FROM=$RESUME_FROM has no latest_checkpointed_iteration.txt"; exit 1; }
  RESUME_ARGS=(--no-load-optim --no-load-rng)
  if [ "${OPSD_OPTIM_SIDECAR:-0}" = "1" ]; then
    echo "[opsd] RESUMING from $RESUME_FROM (iter $(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")) -- Adam state restored from the per-rank sidecar"
  else
    echo "[opsd] RESUMING from $RESUME_FROM (iter $(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")) -- optimizer state absent, Adam restarts"
  fi
fi

# slime derives `train_iters = num_rollout * rollout_bs * G // GBS` and falls back
# to `lr_decay_iters = train_iters` when --lr-decay-iters is absent. A fresh HSI
# smoke (1 rollout, small GBS) makes that product 0, and Megatron aborts on
# `assert self.lr_decay_steps > 0` during actor init -- before any rollout runs.
# With --lr-decay-style constant the value cannot affect the schedule, so pin it to
# 1 only when the derived value would be non-positive (harmless for every other arm).
TRAIN_ITERS=$(( NUM_ROLLOUT * ROLLOUT_BS / GBS ))
LR_DECAY_ARGS=
if [ "$TRAIN_ITERS" -le 0 ]; then
  LR_DECAY_ARGS=( --lr-decay-iters 1 )
fi

CKPT_ARGS=(
   --hf-checkpoint "$MODEL"
   --load "${RESUME_FROM:-$MODEL}"
   --save "$SAVE_DIR"
   --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
   --prompt-data "$DATA"
   --input-key problem
   --label-key solution
   --rollout-function-path "${ROLLOUT_FN:-tasks.math.math_rollout.generate_rollout}"
   --rollout-batch-size "$ROLLOUT_BS"
   --over-sampling-batch-size "$OVERSAMPLE_BS"
   --n-samples-per-prompt 1
   --rollout-max-response-len "$MAX_RESPONSE"
   --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.1}"
   # Reward-CV vNext samples from the SAME full softmax the actor loss uses
   # (spec §3): top_p=1.0 / top_k=-1, taken from slime's defaults -- see the
   # "DO NOT pass --rollout-top-p/--rollout-top-k" note above. Every other arm
   # keeps the inherited official OPSD sampler (0.95 / 20).
   "${SAMPLER_ARGS[@]}"
   --rollout-shuffle
   --rollout-num-gpus "$NUM_ROLLOUT_GPUS"
   --rollout-num-gpus-per-engine 1
   --sglang-attention-backend ascend
   --sglang-mem-fraction-static "$MODEL_SGLANG_MEM_FRACTION"
   --sglang-context-length 32768
   --sglang-chunked-prefill-size "$MODEL_SGLANG_CHUNKED_PREFILL"
   --sglang-disable-radix-cache
)

TRAIN_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "$NUM_TRAIN_GPUS"
   # node NPU count visible to ray (lower it to coexist with other NPU jobs)
   --num-gpus-per-node "${NUM_NODE_GPUS:-8}"
   --num-rollout "$NUM_ROLLOUT"

   --transformer-impl local
   # bshd is the ONLY numerically validated qkv format on this stack: the
   # MindSpeed npu_fusion_attention TND/packed path is broken (Gate-1 null
   # control: identical-context HF-vs-Megatron KL 6.9 nats with thd vs 0.0006
   # nats with bshd). Override with OPSD_QKV_FORMAT=thd only for diagnostics.
   --qkv-format "${OPSD_QKV_FORMAT:-bshd}"
   --tensor-model-parallel-size "$MODEL_TP"
   --pipeline-model-parallel-size "$MODEL_PP"
   --context-parallel-size 1
   --expert-model-parallel-size 1

   --global-batch-size "$GBS"
   --micro-batch-size "$MBS"
   --calculate-per-token-loss

   --loss-type custom_loss
   --custom-loss-function-path "${CUSTOM_LOSS_FN:-methods.opsd_loss.opsd_loss_function}"
   --custom-convert-samples-to-train-data-path "${CONVERT_FN:-tasks.math.math_data.convert_samples_to_train_data}"
   # The postprocess hook is a METHOD dimension (the registry models it as such),
   # not a stray extra arg. Emitted here so the launcher honours it; previously it
   # rode inside HSI_ARGS, and dropping it silently zeroes the HSI advantages.
   ${POSTPROCESS_FN:+--rollout-data-postprocess-path "$POSTPROCESS_FN"} \
   "${HSI_ARGS[@]+"${HSI_ARGS[@]}"}"

   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   "${LR_DECAY_ARGS[@]+"${LR_DECAY_ARGS[@]}"}"
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.999
   --clip-grad "${OPSD_CLIP_GRAD:-0.1}"

   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --no-gradient-accumulation-fusion
   "${RECOMPUTE_ARGS[@]}"

   --log-interval 1
   # align the data order with the official baseline (config_resolved.yaml: seed 42)
   --seed "${OPSD_SEED:-42}"
   "${LORA_ARGS[@]}"
)
if [ -n "${OPSD_SAVE_ROLLOUT_TEMPLATE:-}" ]; then
  TRAIN_ARGS+=(--save-debug-rollout-data "$OPSD_SAVE_ROLLOUT_TEMPLATE")
fi

# Effective-config banner: the point of it is that a run can be audited from its
# own log line (grep '[opsd-slime] effective') without trusting the caller's
# intent -- the 4B campaign's invalid eval batch came from exactly that gap.
echo "[opsd-slime] run=$RUN_NAME (task=$MODEL_TASK model=$MODEL_SLUG method=$NAME_METHOD variant=${VARIANT:-none})"
echo "[opsd-slime] model=$MODEL_NAME slug=$MODEL_SLUG dir=$MODEL"
echo "[opsd-slime] parallel: tp=$MODEL_TP pp=$MODEL_PP dp=$DP (actor_npus=$NUM_TRAIN_GPUS)"
echo "[opsd-slime] train=$NUM_TRAIN_GPUS rollout=$NUM_ROLLOUT_GPUS GBS=$GBS MBS=$MBS rollout_bs=$ROLLOUT_BS max_response=$MAX_RESPONSE num_rollout=$NUM_ROLLOUT method=$OPSD_METHOD lora=${OPSD_LORA:-0}"
echo "[opsd-slime] save_dir=$SAVE_DIR log=$LOG"

# Ray must be started WITHOUT --num-gpus so ray's Ascend NPU accelerator plugin
# auto-detects the 8 NPUs, registers the "NPU" resource that slime's placement
# group bundles request, and manages per-worker ASCEND_RT_VISIBLE_DEVICES.
if ! ray status 2>/dev/null | grep -q "1 node"; then
  ray stop --force 2>/dev/null || true
  ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats \
      --dashboard-host 0.0.0.0 --dashboard-port 8265 2>&1 | tail -2
fi

cleanup_ray() { ray stop --force 2>/dev/null || true; }
trap cleanup_ray EXIT

cd "$PROJ"
# Log directly to the file instead of `| tee "$LOG"`: on 2026-09-18 a full disk
# made tee exit, the python stdout pipe lost its reader, and the training log
# froze for 2.5h while training itself kept running. Writing straight to the file
# removes the pipe, so neither a caller redirect nor a tee failure can decouple
# the log from the process. `tail -f "$LOG"` still works.
python "$PROJ/frameworks/slime-ascend/train.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${RESUME_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" \
   >> "$LOG" 2>&1
