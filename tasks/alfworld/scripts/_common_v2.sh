#!/bin/bash
# ============================================================
# v2 shared launcher -- the two nodes run this with ARM=grpo|gspo|hsi (|hsi_v3).
#
# Shared task protocol. ARM selects the objective; backbone wrappers may set
# estimator-specific optimizer and clipping parameters.
#   grpo : ALFWORLD_V2_HPI=0, no postprocess hook, plain trajectory-mean GRPO loss
#   gspo : identical to grpo except --advantage-estimator gspo, i.e. the SAME
#          trajectory-mean reduction over a SEQUENCE-level surrogate
#          (mean-KL ratio per row instead of per token). No HPI, no scorer.
#   hsi  : ALFWORLD_V2_HPI=1, the postprocess JSD estimator, the HPI-weighted loss.
#          `hsi_v3` is the historic launcher spelling for the same objective (the
#          registry aliases it to `hsi`). The retired ALFWorld credit arms
#          (hpi_future / hsi_v2 / hsi_candidate) now fail closed in the registry.
# Model, schedule, sampling, task limits and LoRA defaults are shared here.
#
# `--advantage-estimator` is passed on EVERY arm (default grpo), because leaving
# it off is what let the gspo arm run GRPO numbers under a GSPO label: the flag
# only reaches the surrogate through the custom loss, so it has to be spelled out.
#
# Usage: ARM=<grpo|gspo|hsi|hsi_v3|eval> RUN_NAME=<name> bash .../run_alfworld_v2_<arm>.sh
# ============================================================
set -uo pipefail

# gy-rl-4 adaptation: overridable so an experiment can run this tree's
# code (CUA_PROJ) while still reading the shared checkpoints (CUA_BASE).
BASE="${CUA_BASE:-/workspace/cua}"
PROJ="${CUA_PROJ:-$BASE/repos/opsd-slime}"
source "$PROJ/tasks/alfworld/scripts/_runtime_defaults.sh"
# The one size-independent ALFWorld recipe (protocol, placement, budget, LoRA
# ranks, serving and logging defaults). Per-size backbones override only model
# geometry and the parallelism/memory values the hardware forces.
source "$PROJ/tasks/alfworld/scripts/_v2_defaults.sh"
# Backbone geometry. The defaults are the validated Qwen3-1.7B numbers, so the
# 1.7B arms are byte-for-byte unchanged. The 4B port is a pure override of these
# five values + MODEL (see TRANSFER_4B_PROMPT.md §6) -- nothing about the METHOD
# is allowed to differ between the two backbones.
MODEL="${MODEL:-$BASE/checkpoints/Qwen3-1.7B}"
NUM_LAYERS="${NUM_LAYERS:-28}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-6144}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-16}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-8}"

ARM="${ARM:?ARM must be set by the calling run_alfworld_v2_*.sh}"
RUN_NAME="${RUN_NAME:-alfworld_v2_${ARM}}"

# Default rollout efficiency protocol, shared by every backbone and arm.
export SGLANG_PREFIX_CACHE="${SGLANG_PREFIX_CACHE:-1}"
export ALFWORLD_THINKING_STYLE="${ALFWORLD_THINKING_STYLE:-brief}"
export KL_COEF="${KL_COEF:-0.001}"

# ---- matched knobs ------------------------------------------------------------
# Protocol defaults live in _v2_defaults.sh; this block only derives launch
# values and keeps the multi-node controls shared.
GLOB_BS=$(( ROLLOUT_BS * G ))        # nominal trajectory budget; rows accumulate once
# Multi-node actors. ACTOR_NODES>1 places the trainer on that many nodes (one
# Ray worker each, all 8 cards per node); the driver itself still runs on the
# node where the launcher was started. Everything downstream (trace padding in
# train_data.convert_samples_to_train_data, the SRLVR reductions) already derives
# DP from actor_num_nodes x actor_num_gpus_per_node, so this is the only switch.
ACTOR_NODES="${ACTOR_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TP="${TP:-1}"                        # single-node only; see the v1 notes on TP+LoRA
# GSPO wrappers supply a narrow sequence-ratio clip instead of inheriting
# GRPO's token-ratio window. The registry selects the surrogate below.
ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"
# Extra train.py flags an arm needs (empty for every legacy arm).
EXTRA_ARGS=
# Which model provider builds the actor. Only the HSI arm overrides it (to publish
# the theta_old handle the scorer needs); every other arm keeps the stock provider,
# so the built model is identical across arms.
#
# NOTE the two variables. `ARM_PROVIDER_PATH` is what the ARM switch selects and
# `MODEL_PROVIDER_PATH` is the operator override; the effective value is resolved
# once, at TRAIN_ARGS time. Setting MODEL_PROVIDER_PATH to its own default up here
# and then letting the arm re-default it with `${MODEL_PROVIDER_PATH:-...}` silently
# keeps the DEFAULT -- measured: the HSI arm ran for a preflight with the stock
# provider, which is exactly the hook the scorer needs to be absent.
ARM_PROVIDER_PATH=""
MODEL_PROVIDER_PATH_OVERRIDE="${MODEL_PROVIDER_PATH:-}"

# The trajectory-balanced reducer. Without it every turn-row is weighted equally,
# which is the bug that collapsed the previous runs; see
# tasks/alfworld/train_data.py::trajectory_weighting. Set it here rather than
# relying on anyone remembering, and keep `legacy` reachable for reproducing
# historical numbers.
SCHEDULE="${SCHEDULE:-$PROJ/tasks/alfworld/configs/train_schedule_seed42.jsonl}"

EVAL_ARGS=

# Both sides are explicit overrides; leaving them unset keeps slime defaults.
EPS_CLIP_ARGS=()
if [ -n "${EPS_CLIP:-}" ]; then
  EPS_CLIP_ARGS=(--eps-clip "$EPS_CLIP")
fi
EPS_CLIP_HIGH_ARGS=()
if [ -n "${EPS_CLIP_HIGH:-}" ]; then
  EPS_CLIP_HIGH_ARGS=(--eps-clip-high "$EPS_CLIP_HIGH")
fi

# ---- arm switch ---------------------------------------------------------------
# ---- arm switch: resolved from opsd_core/registry.py ----
# The arm vocabulary used to be a ~150-line `case` here, maintained separately
# from the math launcher's if-chain and run_names.py's METHOD_SLUG. The registry
# is now the single source; this block consumes the shell fragment it emits.
#
# Two names need care:
#   ARM=hsi_v3   is our method (exact normalized JSD). The registry aliases it to
#                the canonical `hsi`, so `--method hsi` means the same thing here
#                as it does for math and code.
#   ARM=hsi      IS the canonical `hsi` (the registry resolves it directly). It no
#                longer means the retired top-128 candidate arm -- that arm is
#                retired and the registry now refuses `hsi_v2` with rc=2.
#   ARM=eval     is a MODE, not a method; handled below, not by the registry.
#
# Keep this stdlib-only registry subprocess isolated from the training
# PYTHONPATH: editable compatibility hooks may print startup diagnostics, which
# would corrupt the shell fragment captured below.
case "$ARM" in
  eval) _REGISTRY_METHOD=grpo ;;
  *)    _REGISTRY_METHOD="$ARM" ;;
esac
REGISTRY_FRAG=$(PYTHONPATH="$PROJ" python3 \
  -m opsd_core.registry --task alfworld --method "$_REGISTRY_METHOD" --shell) || {
  echo "[v2] FATAL: cannot resolve ARM=$ARM (registry method=$_REGISTRY_METHOD); see" >&2
  exit 2
}
eval "$REGISTRY_FRAG"
# The registry force-exports ALFWORLD_V2_GIGPO=1 for the GiGPO arm only. Clear
# any value inherited from a previous run in the same supervisor shell so a GRPO
# cell can never silently run GiGPO's advantage estimator.
if [ "$_REGISTRY_METHOD" != "gigpo" ]; then
  export ALFWORLD_V2_GIGPO=0
else
  # GiGPO is a standalone estimator. Clear every channel another arm could have
  # leaked into this shell; the rollout also fails closed, but this keeps the
  # launch banner and effective_config truthful.
  export ALFWORLD_V2_HSI=0 ALFWORLD_V2_HSI_V2=0 ALFWORLD_V2_HSI_CANDIDATE=0
  export ALFWORLD_V2_ADVANTAGE_MODE=group
  export OPSD_SRLVR=0
  unset SRLVR_VARIANT
fi

EXTRA_ARGS+=(--disable-rollout-trim-samples)
export ALFWORLD_TASK_SAMPLING="${ALFWORLD_TASK_SAMPLING:-frontier}"
export ALFWORLD_CURRICULUM_METHOD="$_REGISTRY_METHOD"
export ALFWORLD_CURRICULUM_DIR="${ALFWORLD_CURRICULUM_DIR:-${RESUME_FROM:-${SAVE_DIR:-$PROJ/runs/$RUN_NAME}}}"
export ALFWORLD_V2_PROTOCOL=1
# Baselines need sampled-token evidence, but no top-k candidate table.
if [ "${ALFWORLD_V2_HPI:-0}" != "1" ]; then
  export ALFWORLD_V2_HSI_CANDIDATE_K="${ALFWORLD_V2_HSI_CANDIDATE_K:-0}"
fi

LOSS_FN="${CUSTOM_LOSS_FN:?registry provided no loss function for ARM=$ARM}"
# The postprocess hook is a METHOD dimension, not a stray extra arg.
POSTPROCESS_ARGS=
[ -n "${POSTPROCESS_FN:-}" ] && POSTPROCESS_ARGS=( --rollout-data-postprocess-path "$POSTPROCESS_FN" )
# The registry's effective provider (it is never empty: the stock LoRA provider is
# the arm default, matching what this launcher resolved for the empty case).
ARM_PROVIDER_PATH="${MODEL_PROVIDER:-}"
# EXTRA_ARGS comes from the registry as a real bash array.

if [ "$ARM" = "eval" ]; then
  # Eval-only: --num-rollout 0 runs the eval loop and exits. Uses the SAME rollout
  # so the prompts and the per-turn layout are identical to training.
  NUM_ROLLOUT=0
  EVAL_ARGS=(
    --eval-interval 1
    --eval-config "${ALFWORLD_V2_EVAL_CONFIG:-$PROJ/tasks/alfworld/v1/configs/eval_alfworld.json}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-response-len "$MAX_RESPONSE"
  )
  if [ "${EVAL_USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    EVAL_ARGS+=( --use-checkpoint-opt-param-scheduler )
  fi
fi

# Everything that depends on G must be derived HERE, after the arm switch -- the
# switch is what sets G=1 for HSI. Computing it earlier silently used G=8, which
# disabled the LR guard below and then aborted actor init.
GLOB_BS="${GBS_OVERRIDE:-$(( ROLLOUT_BS * G ))}"

# slime derives `train_iters = num_rollout * rollout_bs * G // GBS` and falls back
# to `lr_decay_iters = train_iters` when --lr-decay-iters is absent. With HSI's G=1
# that product can be 0 (1 rollout x 4 tasks x 1 // 32 == 0), and Megatron aborts on
# `assert self.lr_decay_steps > 0` during actor init -- before any rollout runs.
# With --lr-decay-style constant the value cannot affect the schedule.
# ---- eval-only ----------------------------------------------------------------
# Mirrors v1's `base_eval` arm: --num-rollout 0 makes slime run the eval datasets
# and exit without a training step. Eval goes through the SAME rollout function,
# model, LoRA provider and sampling knobs as the training arms, so a checkpoint is
# scored by exactly the code that produced it.
#
# This block must stay ABOVE the LR-schedule guard: it is what sets NUM_ROLLOUT=0,
# and that guard is the only thing standing between eval-only and the
# `assert self.lr_decay_steps > 0` abort in Megatron actor init.
EVAL_ONLY="${EVAL_ONLY:-0}"
# Two entry points mean the same thing (run_alfworld_v2_eval.sh sets ARM=eval,
# run_alfworld_v2_eval_4b.sh sets EVAL_ONLY=1). Funnel them here so both get the
# same GPU split and the same LR-guard placement -- a split that differed per
# entry point would silently make two eval numbers incomparable.
[ "$ARM" = "eval" ] && EVAL_ONLY=1
if [ "$EVAL_ONLY" = "1" ]; then
  NUM_ROLLOUT=0
  # Eval produces no training metrics. Leaving the shared default at 1 makes
  # `experiment_metrics.finish` read a `common_metrics.jsonl` that only training
  # writes, and the eval exits rc=1 after producing valid splits.
  export OPSD_EXPERIMENT_METRICS=0
  # Eval actors only load and publish weights. Share the node with the rollout
  # engines so the default 8-card node generates on all 8 TP1 engines. Explicit
  # COLOCATE=0 retains the separate actor/rollout layout below.
  export COLOCATE="${COLOCATE:-1}"
  if [ "${COLOCATE:-0}" = "1" ]; then
    ACTOR_GPUS="${ACTOR_GPUS:-$NUM_GPUS_PER_NODE}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-$NUM_GPUS_PER_NODE}"
  else
    ACTOR_GPUS="${ACTOR_GPUS:-$TP}"
    ROLLOUT_GPUS="${ROLLOUT_GPUS:-$(( NUM_GPUS_PER_NODE - ACTOR_GPUS ))}"
  fi
  if [ "$ACTOR_GPUS" -le 0 ] || [ "$ROLLOUT_GPUS" -le 0 ]; then
    echo "[v2] FATAL: eval split actor=$ACTOR_GPUS rollout=$ROLLOUT_GPUS on $NUM_GPUS_PER_NODE GPUs" >&2
    exit 2
  fi
  if [ $(( ACTOR_GPUS % TP )) -ne 0 ]; then
    echo "[v2] FATAL: eval actor GPUs $ACTOR_GPUS not divisible by TP $TP" >&2
    exit 2
  fi
  if [ "${COLOCATE:-0}" != "1" ] && [ $(( ACTOR_GPUS + ROLLOUT_GPUS )) -gt "$NUM_GPUS_PER_NODE" ]; then
    echo "[v2] FATAL: eval split actor=$ACTOR_GPUS + rollout=$ROLLOUT_GPUS exceeds the $NUM_GPUS_PER_NODE GPUs on this node" >&2
    exit 2
  fi
  EVAL_ARGS=(
    --eval-interval 1
    --eval-config "${EVAL_CONFIG:-$PROJ/tasks/alfworld/v1/configs/eval_alfworld.json}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-response-len "$MAX_RESPONSE"
  )
  # Eval-only builds train_iters=0, while a full-model checkpoint carries the
  # training scheduler (total_iters=3200 for 100 rollouts); without this the
  # OptimizerParamScheduler load asserts 32 != 3200. No optimizer step runs in
  # eval, so taking the checkpoint's scheduler values is safe.
  if [ "${EVAL_USE_CHECKPOINT_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
    EVAL_ARGS+=( --use-checkpoint-opt-param-scheduler )
  fi
else
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
  ACTOR_GPUS="${ACTOR_GPUS:-2}"
fi

TRAIN_ITERS=$(( NUM_ROLLOUT * ROLLOUT_BS * G / GLOB_BS ))
LR_DECAY_ARGS=
if [ "$TRAIN_ITERS" -le 0 ]; then
  LR_DECAY_ARGS=( --lr-decay-iters 1 )
fi

# ---- HSI credit: the frozen common baseline for this campaign ----------------
# `full`: credit over the WHOLE generated turn (think + action). The historical
#   `action` span was a workaround for the turn-level form, where a think-span
#   score was dominated by a condition-independent magnitude shift; the
#   trajectory-wide centring in the full-token form removes that.
# `token`: per-token d_j with trajectory-wide weighted centre/scale, instead of
#   one scalar per turn broadcast across its tokens.
# Both nodes must agree on these; only Node1's credit CONSTRUCTION differs.
export ALFWORLD_V2_HPI_SPAN="${ALFWORLD_V2_HPI_SPAN:-full}"
export ALFWORLD_V2_HSI_GRANULARITY="${ALFWORLD_V2_HSI_GRANULARITY:-token}"

export ALFWORLD_V2_MAX_TURNS="$MAX_TURNS"
export ALFWORLD_V2_MAX_RESPONSE="$MAX_RESPONSE"
export ALFWORLD_V2_CONTINUAL="${ALFWORLD_V2_CONTINUAL:-1}"
export ALFWORLD_V2_CONTINUAL_MAX_INFLIGHT="${ALFWORLD_V2_CONTINUAL_MAX_INFLIGHT:-$ROLLOUT_BS}"
export ALFWORLD_V2_POOL_CAPACITY="${ALFWORLD_V2_POOL_CAPACITY:-$((2 * ROLLOUT_BS))}"
export ALFWORLD_V2_HISTORY="$HISTORY_LENGTH"
export ALFWORLD_V2_FUTURE_GAP="$FUTURE_GAP"
export ALFWORLD_V2_SCHEDULE="$SCHEDULE"
export ALFWORLD_ACTION_HISTORY="$HISTORY_LENGTH"
export ALFWORLD_MAX_TURNS="$MAX_TURNS"

# ---- HSI engine/sampler conventions ------------------------------------------
# The engine must report the SAMPLER's candidates and log-probs (no re-forward):
#   * top_logprobs_num is set per request in episode._generate (= CANDIDATE_K).
#   * SGLANG_RETURN_ORIGINAL_LOGPROB is deliberately LEFT AT ITS DEFAULT (False).
#     That default makes the engine return `log_softmax(logits / rollout_temperature)`
#     (sglang/srt/layers/sampler.py::_forward_ascend_backend), which is exactly the
#     convention the TRAINER uses -- `megatron_utils/loss.py::get_responses` divides
#     the logits by `args.rollout_temperature` before its log-softmax (loss.py:102
#     and loss.py:128). With both sides in the same convention the PPO ratio is ~1
#     on-policy, which is what makes `--use-rollout-logprobs` legitimate.
#
#     Setting SGLANG_RETURN_ORIGINAL_LOGPROB=1 breaks that: the engine then returns
#     `log_softmax(logits)` and the ratio becomes exp(log pi_{T}(y) - log pi_{T=1}(y)),
#     a SYSTEMATIC per-token bias rather than noise. Measured consequence of exactly
#     that mistake on run ...153227: response length 800 -> 25 tokens, reward
#     0.58 -> 0.0, action_valid 97% -> 44%, and the policy drifting onto
#     `<boolean>false</boolean>`. Because it is a convention mismatch, the
#     temperature compensation has to be removed WITH it: dividing pre-temperature
#     values by T would double-apply it.
#
#     For the same reason ALFWORLD_V2_HSI_SAMPLER_TEMP stays unset (0 = "the engine
#     already reports the sampling distribution"), so `sampling_probabilities`
#     softmaxes the returned values directly and recovers the true p_t.
export ALFWORLD_OVERLAY_SITE="${ALFWORLD_OVERLAY_SITE:-$BASE/envs/alfworld_overlay/lib/python3.10/site-packages}"

# ---- sampler flags ------------------------------------------------------------
# DO NOT pass `--rollout-top-k` / `--rollout-top-p` / `--rollout-temperature`
# explicitly for the V2 protocol. A NEGATIVE value as its own token (`--rollout-top-k
# -1`) leaves an EMPTY-STRING key in slime's args namespace; MindSpeed's
# `transformer_config_init_subclass` copies every unknown arg key into
# `TransformerConfig.__annotations__`, dataclasses then renders a field literally
# named "" and the generated `__init__` blows up with
# `SyntaxError: invalid syntax` -- in `parse_args`, before a single NPU is touched.
# Measured twice on this stack (2026-09-23 and again 2026-10-04); see the campaign
# notes. slime's own defaults are ALREADY the V2 values
# (`slime/utils/arguments.py`: temperature 1.0, top_p 1.0, top_k -1), so the correct
# way to ask for full-softmax sampling is to ask for NOTHING.
#
# The retired arms keep the 2026-10-03 sampler explicitly, so their historical
# numbers stay reproducible from this file.
SAMPLER_ARGS=
if [ "${ALFWORLD_V2_PROTOCOL:-0}" = "1" ]; then
  # V2: rely on slime's defaults (T=1.0 / top_p=1.0 / top_k=-1). An operator may
  # still override deliberately, but only with a POSITIVE/normal token.
  [ -n "${ROLLOUT_TEMP:-}" ] && SAMPLER_ARGS+=( --rollout-temperature "$ROLLOUT_TEMP" )
  [ -n "${ROLLOUT_TOP_P:-}" ] && SAMPLER_ARGS+=( --rollout-top-p "$ROLLOUT_TOP_P" )
  [ -n "${ROLLOUT_TOP_K:-}" ] && SAMPLER_ARGS+=( --rollout-top-k "$ROLLOUT_TOP_K" )
else
  SAMPLER_ARGS+=(
    --rollout-temperature "${ROLLOUT_TEMP:-0.6}"
    --rollout-top-p "${ROLLOUT_TOP_P:-0.95}"
    --rollout-top-k "${ROLLOUT_TOP_K:-20}"
  )
fi

SAVE_DIR="${SAVE_DIR:-$PROJ/runs/$RUN_NAME}"
LOG="${LOG:-$PROJ/logs/${RUN_NAME}.log}"
LOCK="/tmp/opsd_${RUN_NAME}.lock"
mkdir -p "$SAVE_DIR" "$(dirname "$LOG")"

exec 8>"$LOCK"
if ! flock -n 8; then echo "[v2] another $RUN_NAME holds $LOCK" >> "$LOG"; exit 1; fi

RESUME_FROM="${RESUME_FROM:-}"
RESUME_ARGS=
if [ -n "$RESUME_FROM" ]; then
  [ -s "$RESUME_FROM/latest_checkpointed_iteration.txt" ] || { echo "[v2] FATAL: no ckpt in $RESUME_FROM" >> "$LOG"; exit 1; }
  if [ "${SAVE_TRAIN_STATE:-0}" != "1" ] || [ "${RESUME_WEIGHTS_ONLY:-0}" = "1" ]; then
    RESUME_ARGS=(--no-load-optim --no-load-rng)
  fi
elif [ -e "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
  echo "[v2] FATAL: $SAVE_DIR already holds a run" >> "$LOG"; exit 1
fi

export SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
source "$PROJ/train_scripts/lib/npu_env.sh"
opsd_enter_env

# Ascend torch_memory_saver (sgl-kernel-npu fork) cannot pause allocations made
# under the expandable-segments allocator: offload silently frees nothing and the
# colocated engines then OOM. Colocate therefore runs with a plain allocator and
# the NPU hook mode (torch allocator + MemPool), which also needs the CANN
# coredump handler off. Non-colocate arms keep the validated expandable setting.
ALLOC_CONF_DEFAULT="expandable_segments:True"
if [ "${COLOCATE:-0}" = "1" ]; then
  ALLOC_CONF_DEFAULT=""
  unset PYTORCH_NPU_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
  export TMS_HOOK_MODE="${TMS_HOOK_MODE:-torch}"
  export ASCEND_COREDUMP_SIGNAL="${ASCEND_COREDUMP_SIGNAL:-none}"
  # Trainer and SGLang engines are separate processes on the same cards; with the
  # default base port the second HCCL communicator cannot bind 16666 and dies with
  # HCCL_E_UNAVAIL (=7). The upstream colocate example uses these ranges.
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
  # SRLVR defaults to one TP1 engine per card (the wrapper supplies that
  # default). TP>1 engines are allowed now: the strict sampler only warns, so
  # full fine-tuning can opt into e.g. ROLLOUT_GPUS_PER_ENGINE=4 explicitly.
  # (The sampler assumes all-gathered logits, which SGLang provides for TP
  # sampling; the warning exists so this can be verified.)
  :
fi
opsd_export_npu_env "$PROJ" "$ALLOC_CONF_DEFAULT"
# Ray workers import the Ascend SGLang fork in fresh interpreters. Resolve the
# repo's torch2.1/CANN compatibility shim there as well; it fails loudly if an
# unavailable fused kernel is actually called, while dense Qwen3 uses the
# existing torch-native fallback.
if [ "${OPSD_QWEN35_GDN:-0}" = "1" ]; then
  python "$PROJ/scripts/check_gdn_kernels.py" || exit 1
else
  export PYTHONPATH="$PROJ/compat/sgl_kernel_npu_shim:$PYTHONPATH"
fi
export WANDB_MODE=disabled WANDB_DISABLED=true

case "$OPSD_LORA" in
  0|1) ;;
  *) echo "OPSD_LORA must be 0 (full RL) or 1 (LoRA)" >&2; exit 1 ;;
esac
TRAINABLE_ARGS=()
if [ "$OPSD_LORA" = "1" ]; then
  TRAINABLE_ARGS=(--only-train-params-name-list '.*\.lora_.*')
fi

# Tied vs untied embeddings is a per-backbone fact, not a knob: Qwen3-1.7B and
# -4B tie (config.json tie_word_embeddings=true), Qwen3-8B does not. Getting it
# wrong does not error -- it builds a different module tree than the checkpoint,
# so the flag lives on the backbone card (_8b_backbone.sh sets it) rather than
# being remembered here.
UNTIE_ARGS=
[ "${UNTIE_EMBEDDINGS:-0}" = "1" ] && UNTIE_ARGS=( --untie-embeddings-and-output-weights )

MODEL_ARGS=(
   --swiglu --num-layers "$NUM_LAYERS" --hidden-size "$HIDDEN_SIZE" --ffn-hidden-size "$FFN_HIDDEN_SIZE"
   --num-attention-heads "$NUM_ATTENTION_HEADS" --group-query-attention --num-query-groups "$NUM_QUERY_GROUPS"
   --use-rotary-position-embeddings --disable-bias-linear --normalization "RMSNorm"
   --norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 151936
   --padded-vocab-size 151936 --kv-channels 128 --qk-layernorm
   "${UNTIE_ARGS[@]+"${UNTIE_ARGS[@]}"}"
)

[ -s "$SCHEDULE" ] || { echo "[v2] FATAL: schedule missing at $SCHEDULE" >> "$LOG"; exit 3; }

# Sequence parallelism is incompatible with the fused-LoRA wrapper: a
# ColumnParallelLinear with sequence_parallel all-gathers the sequence shard for
# its base product, while the adapter delta is computed from the rank-local
# shard, so the residual add dies with
#   "The size of tensor a (S) must match the size of tensor b (S/TP) at dim 0".
# It is therefore opt-in and never enabled for LoRA arms.
SEQ_PARALLEL_ARGS=()
if [ "${OOM_SEQUENCE_PARALLEL:-0}" = "1" ] && [ "${OPSD_LORA:-0}" != "1" ]; then
  SEQ_PARALLEL_ARGS=(--sequence-parallel)
fi

echo "[v2] arm=$ARM hpi=$ALFWORLD_V2_HPI estimator=$ADVANTAGE_ESTIMATOR G=$G bs=$ROLLOUT_BS GBS=$GLOB_BS turns=$MAX_TURNS cap=$MAX_RESPONSE hist=$HISTORY_LENGTH gap=$FUTURE_GAP continual=$ALFWORLD_V2_CONTINUAL inflight=$ALFWORLD_V2_CONTINUAL_MAX_INFLIGHT pool_capacity=$ALFWORLD_V2_POOL_CAPACITY steps=$NUM_ROLLOUT" | tee -a "$LOG"
echo "[v2-efficiency] actor=$ACTOR_GPUS train_tp=$TP lora=$OPSD_LORA rollout=$ROLLOUT_GPUS inference_tp=${ROLLOUT_GPUS_PER_ENGINE:-1} prefix_cache=${SGLANG_PREFIX_CACHE:-1} thinking=${ALFWORLD_THINKING_STYLE:-legacy} history=complete kl=$KL_COEF candidate_k=${ALFWORLD_V2_HSI_CANDIDATE_K:-auto}" | tee -a "$LOG"
if [ "${OPSD_SRLVR:-0}" = "1" ] && [ "${SRLVR_VARIANT:-}" = "v2" ]; then
  # v2/v3 hindsight compaction is read by rollout.py from SRLVR_COMPACT_FUTURE
  # (default 0). Print the value that will actually be used: a hardcoded
  # compact=0 here once contradicted a run that really ran compact=1.
  echo "[srlvr-hsi-config] variant=v2 gamma=${SRLVR_GAMMA:-0.99} future=full compact=${SRLVR_COMPACT_FUTURE:-0} advantage_mode=${SRLVR_ADVANTAGE_MODE:-batch} thinking=${ALFWORLD_THINKING_STYLE:-brief} thinking_tokens=${ALFWORLD_THINKING_TOKENS:-unset} max_response=$MAX_RESPONSE context_train=$SEQ_LENGTH context_inference=$SEQ_LENGTH continual=$ALFWORLD_V2_CONTINUAL inflight=$ALFWORLD_V2_CONTINUAL_MAX_INFLIGHT reference_kl=low_var_kl:${KL_COEF:-0.001} ref=$MODEL save_interval=$SAVE_INTERVAL rollouts=$NUM_ROLLOUT" | tee -a "$LOG"
fi
if [ "${OPSD_SRLVR:-0}" = "1" ] && [ "${SRLVR_VARIANT:-}" = "v4" ]; then
  echo "[srlvr-v4-config] beta=${SRLVR_V4_BETA:-0.0} token_clip=${SRLVR_V4_TOKEN_CLIP:-0.05} teacher=${MODEL} teacher_attn=${SRLVR_V4_TEACHER_ATTN:-sdpa} compact=${SRLVR_V4_COMPACT_FUTURE:-1} thinking=${ALFWORLD_THINKING_STYLE:-brief} thinking_tokens=${ALFWORLD_THINKING_TOKENS:-unset} max_response=$MAX_RESPONSE context_train=$SEQ_LENGTH context_inference=$SEQ_LENGTH reference_kl=low_var_kl:${KL_COEF:-0.001} ref=$MODEL save_interval=$SAVE_INTERVAL rollouts=$NUM_ROLLOUT" | tee -a "$LOG"
fi
if [ "$_REGISTRY_METHOD" = "gigpo" ]; then
  # Anchor grouping is exact (task_id, observation); remind every reader of the
  # arm's three knobs so a log is self-describing.
  echo "[gigpo-config] step_w=${ALFWORLD_V2_GIGPO_STEP_W:-1.0} gamma=${ALFWORLD_V2_GIGPO_GAMMA:-0.95} mode=${ALFWORLD_V2_GIGPO_MODE:-mean_std_norm} grouping=exact(task_id,anchor_state) G=$G think=${ALFWORLD_THINKING_STYLE:-brief} max_response=$MAX_RESPONSE rollouts=$NUM_ROLLOUT" | tee -a "$LOG"
fi

PREFIX_CACHE_ARGS=()
if [ "${SGLANG_PREFIX_CACHE:-1}" != "1" ]; then
  PREFIX_CACHE_ARGS=(--sglang-disable-radix-cache)
fi

# Optimizer CPU offload only pays when the FULL model is trained. The ALFWorld
# arms train r128 LoRA adapters; offloading them forces torch.optim's fused
# AdamW onto the CPU backend, which this torch 2.1 build does not implement
# (`aten::_fused_adamw_` NotImplementedError, measured 2026-10-27 on the v4
# smoke). Keep the historical default ON for existing arms; v4 opts out.
OPTIMIZER_ARGS=(--optimizer adam)
if [ "${OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --optimizer-offload-fraction 1
    --use-precision-aware-optimizer
    --overlap-cpu-optimizer-d2h-h2d
  )
fi

COLOCATE_ARGS=()
if [ "${COLOCATE:-0}" = "1" ]; then
  COLOCATE_ARGS=(--colocate)
fi

# The full train.py argument vector, as an array so it can be (a) executed and
# (b) replayed through slime's own parser without touching an NPU -- see the
# DRY_RUN_PARSE block below and tasks/alfworld/scripts/argv_preflight.py.
# The Full HSI teacher is independent of the initial-model reference policy.
# Preserve the existing PPO + initial-model KL configuration for every arm.
REFERENCE_ARGS=(--use-kl-loss --kl-loss-coef "${KL_COEF:-0.001}" --kl-loss-type low_var_kl --ref-load "$MODEL")

# Default ON for every arm/backbone (_v2_defaults.sh owns the default); the
# only reason to set 0 is a debug session that needs graph captures disabled.
GRAPH_ARGS=(--sglang-disable-cuda-graph)
case "${SGLANG_ENABLE_NPU_GRAPH:-1}" in
  0) ;;
  1) GRAPH_ARGS=(--sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS:-8}") ;;
  *) echo "SGLANG_ENABLE_NPU_GRAPH must be 0 or 1" >&2; exit 1 ;;
esac

TRAIN_ARGS=(
   "${MODEL_ARGS[@]}"
   --hf-checkpoint "$MODEL"
   --load "${EVAL_LOAD:-${EVAL_LOAD_FROM:-${RESUME_FROM:-$MODEL}}}"
   --save "$SAVE_DIR"
   --save-interval "$SAVE_INTERVAL"
   "${RESUME_ARGS[@]}"
   --prompt-data "$PROJ/tasks/alfworld/v1/configs/data/train.jsonl"
   --input-key prompt
   --metadata-key metadata
   --rollout-function-path tasks.alfworld.rollout.generate_rollout
   --custom-convert-samples-to-train-data-path "$CONVERT_FN"
   "${POSTPROCESS_ARGS[@]}"
   --rollout-batch-size "$ROLLOUT_BS"
   --over-sampling-batch-size "$ROLLOUT_BS"
   --n-samples-per-prompt "$G"
   --rollout-max-response-len "$MAX_RESPONSE"
   ${SAMPLER_ARGS[@]+"${SAMPLER_ARGS[@]}"}
   --rollout-shuffle
   --rollout-num-gpus "$ROLLOUT_GPUS"
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE:-1}"
   --sglang-attention-backend ascend
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.42}"
   --sglang-context-length "$SEQ_LENGTH"
   --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-8}"
   --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS:-65536}"
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-8}"
   "${GRAPH_ARGS[@]}"
   "${PREFIX_CACHE_ARGS[@]+${PREFIX_CACHE_ARGS[@]}}"
   "${COLOCATE_ARGS[@]+${COLOCATE_ARGS[@]}}"
   --actor-num-nodes "${ACTOR_NODES:-1}"
   --actor-num-gpus-per-node "$ACTOR_GPUS"
   --num-gpus-per-node "$NUM_GPUS_PER_NODE"
   --num-rollout "$NUM_ROLLOUT"
   --transformer-impl local
   --qkv-format bshd
   --tensor-model-parallel-size "$TP"
   --pipeline-model-parallel-size "${PP:-1}"
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --global-batch-size "$GLOB_BS"
   --micro-batch-size 1
   --seq-length "$SEQ_LENGTH"
   --calculate-per-token-loss
   --loss-type custom_loss
   --custom-loss-function-path "$LOSS_FN"
   --advantage-estimator "$ADVANTAGE_ESTIMATOR"
   --use-opsd
   ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
   "${OPTIMIZER_ARGS[@]+"${OPTIMIZER_ARGS[@]}"}"
   --lr "${LR:-1e-6}"
   "${REFERENCE_ARGS[@]+${REFERENCE_ARGS[@]}}"
   --lr-decay-style constant
   "${LR_DECAY_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   --weight-decay 0.0
   --adam-beta1 0.9
   --adam-beta2 0.999
   --clip-grad "${CLIP_GRAD:-1.0}"
   "${EPS_CLIP_ARGS[@]+${EPS_CLIP_ARGS[@]}}"
   "${EPS_CLIP_HIGH_ARGS[@]+${EPS_CLIP_HIGH_ARGS[@]}}"
   --recompute-granularity full
   --recompute-method "${RECOMPUTE_METHOD:-uniform}"
   --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   ${SEQ_PARALLEL_ARGS[@]+"${SEQ_PARALLEL_ARGS[@]}"}
   --no-gradient-accumulation-fusion
   --log-interval 1
   --seed "$SEED"
   "${EVAL_ARGS[@]}"
   --save-debug-rollout-data "${DEBUG_ROLLOUT_DIR:-$SAVE_DIR}/rollout_{rollout_id}.pt"
   --custom-model-provider-path "${MODEL_PROVIDER_PATH_OVERRIDE:-${ARM_PROVIDER_PATH:-methods.lora_megatron.custom_model_provider}}" \
   "${TRAINABLE_ARGS[@]+${TRAINABLE_ARGS[@]}}"
)

if [ "${OPSD_SRLVR:-0}" = "1" ] && [ "${SRLVR_VARIANT:-}" != "v4" ]; then
  # Sum over actor tokens; framework row averaging is compensated in actor_loss.
  # SRLVR-v4 KEEPS --calculate-per-token-loss: its objective is the global
  # actor-token mean (the same reduction OPSD uses), so the framework's token
  # normaliser is exactly the one the loss wants.
  _SRLVR_ARGS=()
  for _arg in "${TRAIN_ARGS[@]}"; do
    [ "$_arg" = "--calculate-per-token-loss" ] || _SRLVR_ARGS+=("$_arg")
  done
  TRAIN_ARGS=("${_SRLVR_ARGS[@]}")
fi

if [ -n "${SGLANG_DTYPE:-}" ]; then
  TRAIN_ARGS+=(--sglang-dtype "$SGLANG_DTYPE")
fi

if [ "${SAVE_TRAIN_STATE:-1}" != "1" ]; then
  TRAIN_ARGS+=(--no-save-optim --no-save-rng)
fi

# Execution-only memory controls and replay of a saved failing batch. Defaults
# preserve the 8B protocol; the 32 GB 4B wrappers opt in explicitly.
if [ -n "${LOG_PROBS_CHUNK_SIZE:-}" ]; then
  TRAIN_ARGS+=(--log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE")
fi
if [ "${RECOMPUTE_LOSS:-0}" = "1" ]; then
  TRAIN_ARGS+=(--recompute-loss-function)
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  TRAIN_ARGS+=(--load-debug-rollout-data "$LOAD_DEBUG_ROLLOUT_DATA")
fi

TRAIN_ENTRYPOINT=train.py
if [ "${ASYNC_TRAIN:-0}" = "1" ]; then
  if [ "${COLOCATE:-0}" = "1" ]; then
    echo "[v2] FATAL: ASYNC_TRAIN=1 requires separated actor and rollout GPUs" >> "$LOG"
    exit 2
  fi
  if [ "${OPSD_SRLVR:-0}" = "1" ]; then
    if [ "${SRLVR_STALE_POLICY_CORRECTION:-1}" != "1" ]; then
      echo "[v2] FATAL: ASYNC_TRAIN=1 with SRLVR needs SRLVR_STALE_POLICY_CORRECTION=1" >> "$LOG"
      exit 2
    fi
    if [ "${SRLVR_VARIANT:-}" != "v2" ] && [ "${SRLVR_VARIANT:-}" != "v3" ]; then
      echo "[v2] FATAL: async SRLVR implements stale-policy stamping for v2/v3 only (got ${SRLVR_VARIANT:-unset})" >> "$LOG"
      exit 2
    fi
    # The weight-update interval is pinned to 1 by validate_runtime, so an
    # overlapped rollout is exactly one weight version behind the actor that
    # trains it. The rows are stamped with that generation version and the
    # actor loss applies the clip on exp(logp_now - logp_rollout).
    if [ -n "${RESUME_FROM:-}" ]; then
      echo "[v2] FATAL: async SRLVR resume is not supported yet (the rollout manager cannot see start_rollout_id); start a fresh run" >> "$LOG"
      exit 2
    fi
    export SRLVR_STALE_LAG="${SRLVR_STALE_LAG:-1}"
    export SRLVR_MAX_STALE="${SRLVR_MAX_STALE:-1}"
    echo "[v2-stale] SRLVR async: stale_lag=$SRLVR_STALE_LAG max_stale=$SRLVR_MAX_STALE correction=truncated_is" >> "$LOG"
  else
    export SRLVR_STALE_LAG="${SRLVR_STALE_LAG:-0}"
  fi
  TRAIN_ENTRYPOINT=train_async.py
  TRAIN_ARGS+=(--use-rollout-logprobs)
fi
echo "[v2-driver] entrypoint=$TRAIN_ENTRYPOINT async_train=${ASYNC_TRAIN:-0} rollout_logprobs=$([ "${ASYNC_TRAIN:-0}" = "1" ] && echo 1 || echo 0) colocate=${COLOCATE:-0}" | tee -a "$LOG"

# Replay the argv through slime's parser and exit. This is the cheap gate for the
# one failure mode that has cost this project real time twice: a flag that parses
# into an EMPTY-STRING arg key (a bare negative value is the usual way in) crashes
# dataclass codegen inside `parse_args` with a SyntaxError, several minutes into
# a launch, with a traceback that points at Megatron-Bridge rather than at the
# offending flag. Costs seconds and no NPU.
if [ "${DRY_RUN_PARSE:-0}" = "1" ]; then
  python "$PROJ/tasks/alfworld/scripts/argv_preflight.py" "${TRAIN_ARGS[@]}"
  exit $?
fi

if [ "${ACTOR_NODES:-1}" -gt 1 ] || [ "${OPSD_RAY_EXTERNAL:-0}" = "1" ]; then
  # Multi-node (or externally managed) Ray: the cluster is started by the
  # campaign harness on every participating node, and the driver only connects.
  # NEVER `ray stop` here -- that would tear down the peer's raylet mid-run.
  : "${RAY_ADDRESS:?multi-node/external Ray needs RAY_ADDRESS=<head-ip>:<gcs-port>}"
  if ! ray status --address "$RAY_ADDRESS" 2>/dev/null | grep -q "node_"; then
    echo "[v2] FATAL: no Ray cluster at $RAY_ADDRESS; start it with the campaign harness" >> "$LOG"
    exit 4
  fi
  cleanup_ray() { :; }
else
  if ! ray status 2>/dev/null | grep -q "1 node"; then
    ray stop --force 2>/dev/null || true
    # Bound the prestarted worker pool. On this 175-CPU node the default pool
    # (175 prestarted workers) leaves the raylet reporting workers that never
    # register, and placement-group creation then never completes. The proven
    # two-node harness used 16 CPUs / 8 GiB object store; match it here.
    ray start --head --node-ip-address 127.0.0.1 \
      --num-cpus "${RAY_NUM_CPUS:-16}" \
      --object-store-memory "${RAY_OBJECT_STORE:-8000000000}" \
      --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265 2>&1 | tail -2
  fi
  cleanup_ray() { ray stop --force 2>/dev/null || true; }
fi
trap cleanup_ray EXIT

cd "$PROJ"

# `--use-opsd` makes slime hand the loss the rollout's own per-token advantages
# verbatim, bypassing its positional group normalisation -- which is what keeps
# the semantics "GRPO over episodes" now that the rows are turns.
export OPSD_ENABLED=1

python "$PROJ/frameworks/slime-ascend/$TRAIN_ENTRYPOINT" "${TRAIN_ARGS[@]}" >> "$LOG" 2>&1
train_rc=$?
echo "[v2] $ARM exited rc=$train_rc $(date '+%F %H:%M:%S')" >> "$LOG"
exit "$train_rc"
