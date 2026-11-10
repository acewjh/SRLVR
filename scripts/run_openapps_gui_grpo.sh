#!/bin/bash
# GUI GRPO on OpenApps with Qwen2.5-VL-3B (slime + Megatron + SGLang, NPU).
#
# Task = OpenApps (274 browsergym tasks) via scripts/openapps_gateway.py.
# Rollout = slime.osworld_rollout (screenshot -> VLM tool call -> pyautogui
# action -> env step), already part of the vendored slime fork.
#
# Env knobs:
#   NUM_ROLLOUT=1        rollouts to run           (smoke default; 100 for a real run)
#   ROLLOUT_BS=1         prompts per rollout
#   G=2                  samples per prompt        (GRPO group size)
#   GBS=2                rows per optimizer step
#   MAX_RESPONSE=8192    total response tokens per episode (observations included)
#   OSWORLD_MAX_TURNS=16 model turns per episode (OpenApps uses 25 env steps)
#   MODEL_SIZE=3B|7B     defaults actor TP to 2|4 from the checkpoint config
#   COLOCATE=1           optional colocated actor/rollout layout; smoke-test first
#   ACTOR_GPUS=4 ROLLOUT_GPUS=4
#
# Run: `nohup bash scripts/run_openapps_gui_grpo.sh > logs/gui_grpo_launcher.log 2>&1 &`
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="/workspace/cua"
MODEL_SIZE="${MODEL_SIZE:-}"
if [ -z "${MODEL_PATH:-}" ]; then
  MODEL_SIZE="${MODEL_SIZE:-3B}"
  case "$MODEL_SIZE" in
    3B) MODEL_PATH="$BASE/checkpoints/Qwen2.5-VL-3B-Instruct" ;;
    7B) MODEL_PATH="$BASE/checkpoints/Qwen2.5-VL-7B-Instruct" ;;
    *) echo "[gui-grpo] FATAL: MODEL_SIZE must be 3B or 7B (got $MODEL_SIZE)" >&2; exit 1 ;;
  esac
fi
if [ -z "$MODEL_SIZE" ]; then
  case "${MODEL_PATH##*/}" in
    *7[Bb]*) MODEL_SIZE=7B ;;
    *) MODEL_SIZE=3B ;;
  esac
fi
DATA_PATH="${DATA_PATH:-$PROJ/data/openapps_manifest.parquet}"
# DEBUG_ROLLOUT_ONLY uses HF weights and skips Megatron checkpoint loading.
# Trained-checkpoint evaluation must load/publish the actor weights first.
LOAD_PATH="${LOAD_PATH:-$MODEL_PATH}"
DEBUG_ROLLOUT_ONLY="${DEBUG_ROLLOUT_ONLY:-0}"
ROLLOUT_SHUFFLE="${ROLLOUT_SHUFFLE:-1}"
DEBUG_ARGS=()
[ "$DEBUG_ROLLOUT_ONLY" = "1" ] && DEBUG_ARGS+=(--debug-rollout-only)
EVAL_ONLY="${EVAL_ONLY:-0}"
EVAL_ARGS=()
if [ "$EVAL_ONLY" = "1" ]; then
  [ "$DEBUG_ROLLOUT_ONLY" = "0" ] || { echo "eval must load and publish actor weights" >&2; exit 1; }
  EVAL_ARGS=(--eval-interval 1 --eval-prompt-data openapps "$DATA_PATH"
    --eval-function-path "${OPENAPPS_EVAL_FUNCTION:-methods.openapps_eval.generate_rollout}"
    --lr-decay-iters 1 --start-rollout-id 0)
  export OPSD_OPTIM_SIDECAR=0 OPSD_EXPERIMENT_METRICS=0
fi
if [ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]; then
  DEBUG_ARGS+=(--debug-train-only --load-debug-rollout-data "$LOAD_DEBUG_ROLLOUT_DATA")
fi
SHUFFLE_ARGS=()
[ "$ROLLOUT_SHUFFLE" = "1" ] && SHUFFLE_ARGS+=(--rollout-shuffle)

# Read the Qwen2.5-VL language geometry from the checkpoint rather than
# silently building the 3B shape when MODEL_PATH points at the 7B model.
MODEL_CONFIG="$MODEL_PATH/config.json"
[ -f "$MODEL_CONFIG" ] || { echo "[gui-grpo] FATAL: model config missing: $MODEL_CONFIG" >&2; exit 1; }
read -r MODEL_TYPE NUM_LAYERS HIDDEN_SIZE FFN_HIDDEN_SIZE NUM_ATTENTION_HEADS NUM_QUERY_GROUPS KV_CHANNELS VOCAB_SIZE ROTARY_BASE < <(
  python3 - "$MODEL_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
text = cfg.get("text_config") or cfg
model_type = cfg.get("model_type", "")
hidden = int(text["hidden_size"])
heads = int(text["num_attention_heads"])
kv_heads = int(text["num_key_value_heads"])
head_dim = int(text.get("head_dim", hidden // heads))
if hidden % heads or head_dim != hidden // heads:
    raise SystemExit(f"unsupported Qwen2.5-VL head geometry: hidden={hidden}, heads={heads}, head_dim={head_dim}")
print(model_type, int(text["num_hidden_layers"]), hidden,
      int(text["intermediate_size"]), heads, kv_heads, head_dim,
      int(text.get("vocab_size", cfg.get("vocab_size", 151936))),
      int(text.get("rope_theta", cfg.get("rope_theta", 1000000))))
PY
)
[ "$MODEL_TYPE" = "qwen2_5_vl" ] || {
  echo "[gui-grpo] FATAL: expected a Qwen2.5-VL checkpoint, got model_type=$MODEL_TYPE" >&2
  exit 1
}

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
[ "$EVAL_ONLY" = "1" ] && NUM_ROLLOUT=0
ROLLOUT_BS="${ROLLOUT_BS:-1}"
G="${G:-2}"
GBS="${GBS:-2}"
# The response budget charges observation tokens too: a 1280x720 screenshot is
# ~1200 tokens. 24576 and 16384 both OOMed in the colocate backward pass
# (3.5 GiB request with 4 GiB free, then 2.6 GiB with 3.3 GiB free): the
# allocator cannot get a contiguous block. 8192 response + 12288 sequence is
# the stable point on 32 GB cards; longer episodes truncate cleanly.
OSWORLD_VCS="${OSWORLD_VCS:-1}"
if [ "$OSWORLD_VCS" = "1" ]; then
  MAX_RESPONSE="${MAX_RESPONSE:-24576}"
else
  MAX_RESPONSE="${MAX_RESPONSE:-4096}"
fi
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-16384}"
# A 8192-token prefill chunk asked for 5.46 GiB of activation and OOMed
# the 4-engine non-colocate layout (4.43 GiB free); 4096 halves it.
SGLANG_CHUNKED_PREFILL="${SGLANG_CHUNKED_PREFILL:-4096}"
# OpenApps' own screenshot preset runs max_steps=25 (default 30) environment
# steps; the old 8 model turns was far below that. 16 turns x ~1200-token
# screenshot still fits the 24576 training context, so start there.
OSWORLD_MAX_TURNS="${OSWORLD_MAX_TURNS:-16}"
OSWORLD_MAX_TOKENS_PER_TURN="${OSWORLD_MAX_TOKENS_PER_TURN:-1024}"
TP_DEFAULT=2
[ "$HIDDEN_SIZE" -gt 2048 ] && TP_DEFAULT=4
TP="${TP:-$TP_DEFAULT}"
COLOCATE="${COLOCATE:-0}"
if [ "$COLOCATE" = "1" ]; then
  # Framework-supported, but not yet validated for Qwen2.5-VL on this task.
  # Eight actor ranks share all eight NPUs with rollout engines; TMS/offload
  # handles the hand-off between training and inference allocations.
  ACTOR_GPUS="${ACTOR_GPUS:-8}"
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
  [ "$ACTOR_GPUS" -eq "$ROLLOUT_GPUS" ] || {
    echo "[gui-grpo] FATAL: colocate requires equal actor/rollout GPU counts" >&2
    exit 1
  }
  ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-$TP}"
  # Leave headroom for the actor allocations retained after an optimizer step.
  # At 0.30, restoring the KV cache failed on rank 0 after the first update.
  SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.15}"
  NUM_GPUS_PER_NODE="$ACTOR_GPUS"
  COLOCATE_ARGS=(--colocate)
else
  ACTOR_GPUS="${ACTOR_GPUS:-4}"
  ROLLOUT_GPUS="${ROLLOUT_GPUS:-4}"
  if [ "$TP" -ge 4 ]; then
    ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-4}"
  else
    # Two TP2 engines shard both language and vision attention on four NPUs.
    ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
  fi
  # This fraction budgets weights + KV, not the engine's peak total usage.
  # TP1 at 0.5 OOMed in vision SDPA (5.46 GiB requested / 4.43 GiB free).
  # Keep space for vision activations; validate TP2 at this conservative value.
  SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.4}"
  NUM_GPUS_PER_NODE="$((ACTOR_GPUS + ROLLOUT_GPUS))"
  COLOCATE_ARGS=()
fi
[ $((ACTOR_GPUS % TP)) -eq 0 ] || {
  echo "[gui-grpo] FATAL: actor GPUs $ACTOR_GPUS must be divisible by TP=$TP" >&2
  exit 1
}
[ $((ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE)) -eq 0 ] || {
  echo "[gui-grpo] FATAL: rollout GPUs $ROLLOUT_GPUS must be divisible by rollout TP=$ROLLOUT_GPUS_PER_ENGINE" >&2
  exit 1
}
[ "$NUM_QUERY_GROUPS" -ge "$TP" ] && [ $((NUM_QUERY_GROUPS % TP)) -eq 0 ] || {
  echo "[gui-grpo] FATAL: TP=$TP must divide num_query_groups=$NUM_QUERY_GROUPS" >&2
  exit 1
}
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
KL_ARGS=()
if [ "$KL_LOSS_COEF" != "0" ] && [ "$KL_LOSS_COEF" != "0.0" ]; then
  KL_ARGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF" --kl-loss-type low_var_kl)
fi
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-0}"
ROLLOUT_LOGPROB_ARGS=()
VCS_ARGS=()
if [ "$OSWORLD_VCS" = "1" ]; then
  VCS_ARGS=(
    --disable-rollout-trim-samples
    --custom-convert-samples-to-train-data-path methods.openapps_vcs_grpo.convert_samples_to_train_data
    --loss-type custom_loss
    --custom-loss-function-path methods.openapps_vcs_grpo.openapps_vcs_grpo_loss
  )
fi
# This skips the actor's old-policy scoring pass, but only use it after the
# train/rollout parity metric is small; recent OpenApps smokes are not there yet.
if [ "$USE_ROLLOUT_LOGPROBS" = "1" ]; then
  ROLLOUT_LOGPROB_ARGS=(--use-rollout-logprobs)
fi
[ -d "$MODEL_PATH" ] || { echo "[gui-grpo] FATAL: model missing: $MODEL_PATH" >&2; exit 1; }
[ -f "$DATA_PATH" ]  || { echo "[gui-grpo] FATAL: manifest missing: $DATA_PATH" >&2; exit 1; }

if [ "${PRINT_CONFIG_ONLY:-0}" = "1" ]; then
  printf '%s\n' \
    "model_size=$MODEL_SIZE" \
    "model_path=$MODEL_PATH" \
    "geometry=${NUM_LAYERS}x${HIDDEN_SIZE} ffn=$FFN_HIDDEN_SIZE heads=$NUM_ATTENTION_HEADS kv_heads=$NUM_QUERY_GROUPS" \
    "train_tp=$TP actor_gpus=$ACTOR_GPUS rollout_gpus=$ROLLOUT_GPUS rollout_tp=$ROLLOUT_GPUS_PER_ENGINE colocate=$COLOCATE" \
    "max_turns=$OSWORLD_MAX_TURNS max_tokens_per_turn=$OSWORLD_MAX_TOKENS_PER_TURN response_cap=$MAX_RESPONSE train_context=$SEQ_LENGTH sglang_context=$SGLANG_CONTEXT_LENGTH" \
    "vcs=$OSWORLD_VCS images_to_keep=${OSWORLD_IMAGES_TO_KEEP:-3} removal_threshold=${OSWORLD_MIN_REMOVAL_THRESHOLD:-1}" \
    "kl_loss_coef=$KL_LOSS_COEF use_rollout_logprobs=$USE_ROLLOUT_LOGPROBS"
  exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-gui-qwen25vl${MODEL_SIZE,,}-grpo-$STAMP}"
RUNS_ROOT="${OPENAPPS_RUNS_ROOT:-$PROJ/runs}"
SAVE_DIR="$RUNS_ROOT/$RUN_NAME"
LOG="$PROJ/logs/$RUN_NAME.log"

mkdir -p "$PROJ/logs" "$PROJ/runs" "$SAVE_DIR" || exit 1
# Preserve dashboard discovery when large run artifacts use host-local storage.
if [ "$RUNS_ROOT" != "$PROJ/runs" ] && [ ! -e "$PROJ/runs/$RUN_NAME" ] && [ ! -L "$PROJ/runs/$RUN_NAME" ]; then
  ln -s "$SAVE_DIR" "$PROJ/runs/$RUN_NAME" || exit 1
fi
# Keep the launcher log in the run's canonical dashboard-readable location as
# well. The external logs/ copy remains the stable operator-facing path, while
# this symlink lets the dashboard parse rollout/train metrics before telemetry.
ln -sfn "$LOG" "$SAVE_DIR/train.log"

# ---- services (app server + browsergym gateway) ---------------------------
if [ "${START_OPENAPPS_SERVICES:-1}" = "1" ]; then
  bash "$PROJ/scripts/start_openapps_services.sh" || exit 1
fi

# ---- conda + CANN + PYTHONPATH -------------------------------------------
# shellcheck source=../train_scripts/lib/npu_env.sh
source "$PROJ/train_scripts/lib/npu_env.sh" || exit 1
OPSD_CONDA_ENV="${OPSD_CONDA_ENV:-PyTorch-2.9.0}"
opsd_enter_env
if [ "$COLOCATE" = "1" ]; then
  # Ascend's memory saver cannot release allocations made under expandable
  # segments. Use the same torch/TMS allocator contract as the validated
  # colocated ALFWorld launchers.
  opsd_export_npu_env "$PROJ" ""
  unset PYTORCH_NPU_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
  export TMS_HOOK_MODE="${TMS_HOOK_MODE:-torch}"
  export ASCEND_COREDUMP_SIGNAL="${ASCEND_COREDUMP_SIGNAL:-none}"
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
  # The SGLang scheduler imports this when memory saver is enabled. Fail before
  # starting Ray if the selected conda environment lacks the dependency.
  python -c 'import torch_memory_saver' || exit 1
else
  opsd_export_npu_env "$PROJ" "expandable_segments:True"
fi

# Local gateway/app traffic must bypass the corporate proxy.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"

# OpenApps GUI metrics are part of the run contract.  Keep local telemetry on
# unless an operator explicitly disables it, and make the repository root
# available to compat/sitecustomize in every actor process.
export OPSD_DASHBOARD="${OPSD_DASHBOARD:-1}"
export OPSD_PROJECT_ROOT="$PROJ"

# ---- rollout environment contract ----------------------------------------
export OSWORLD_GATEWAY_URL="${OSWORLD_GATEWAY_URL:-http://127.0.0.1:5111}"
export OSWORLD_GATEWAY_ENV_TYPE="${OSWORLD_GATEWAY_ENV_TYPE:-openapps}"
export OSWORLD_MAX_TURNS
export OSWORLD_MAX_TOKENS_PER_TURN
# Truncation used to cost -0.2 while a natural stop cost evaluate() (>=0), so
# stopping early was strictly safer than continuing to work. With the budget
# raised, make "ran out of turns" neutral instead of punished.
export OSWORLD_INCOMPLETE_PUNISHMENT="${OSWORLD_INCOMPLETE_PUNISHMENT:-0.0}"
export OSWORLD_PARSE_RETRIES="${OSWORLD_PARSE_RETRIES:-2}"
# Qwen2.5-VL grounds in absolute pixels of the 1280x720 input image, not in the
# fictional 1000x1000 grid the Qwen3.6-VL prompt assumes. Verified with a clean
# prompt: the OpenMaps dock icon comes back as [573,682,604,713] (pixels).
export OSWORLD_COORD_MODE="${OSWORLD_COORD_MODE:-image_pixels}"
export OSWORLD_TRAIN_VISION="${OSWORLD_TRAIN_VISION:-1}"
# Project only loss-bearing positions in bounded full-vocabulary chunks. This
# avoids the full [sequence,vocab] fp32 output allocation on long visual rows.
export OPSD_SRLVR_SELECTIVE_LOGITS="${OPENAPPS_SELECTIVE_LOGITS:-1}"

# ---- checkpoint space policy (ALFWorld-style) ----------------------------
# The weights-only checkpoint (--no-save-optim --no-save-rng) is ~7 GiB and is
# enough to benchmark (--load iter_N --debug-rollout-only). To also resume
# exactly, the optimizer is dumped as a per-rank sidecar next to each save
# (methods/optim_sidecar.py). scripts/gui_ckpt_rotate.py then keeps ONLY the
# newest sidecar and trims the older ones, so old checkpoints stay
# benchmark-able but not resume-exact.
export OPSD_OPTIM_SIDECAR="${OPSD_OPTIM_SIDECAR:-1}"
export OSWORLD_REQUEST_TIMEOUT="${OSWORLD_REQUEST_TIMEOUT:-180}"
export OSWORLD_VCS
export OSWORLD_IMAGES_TO_KEEP="${OSWORLD_IMAGES_TO_KEEP:-3}"
export OSWORLD_MIN_REMOVAL_THRESHOLD="${OSWORLD_MIN_REMOVAL_THRESHOLD:-1}"
export OPSD_MAX_CONCURRENT_ENVS="${OPSD_MAX_CONCURRENT_ENVS:-16}"

# ---- ray, then wait until a remote task actually round-trips -------------
# Same wedge as the math launcher: a driver started on a cold raylet hangs in
# placement_group() with no error. Wall clock is not a signal; a probe is.
ray stop --force >/dev/null 2>&1
sleep 3
ray start --head --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265 >/dev/null 2>&1

probe() {
  timeout 90 python - <<'PY' 2>/dev/null
import ray
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
@ray.remote
def ping():
    return 1
assert ray.get(ping.remote(), timeout=60) == 1
print("RAY_READY")
PY
}
ready=0
for attempt in $(seq 1 30); do
  probe | grep -q RAY_READY && { ready=1; break; }
  echo "[gui-grpo] ray not responsive yet (probe $attempt)" >&2
  pgrep -f "raylet --raylet_socket_name" >/dev/null 2>&1 || {
    ray stop --force >/dev/null 2>&1
    ray start --head --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265 >/dev/null 2>&1
  }
  sleep 20
done
[ "$ready" = 1 ] || { echo "[gui-grpo] FATAL: ray never became responsive" >&2; exit 1; }

echo "[gui-grpo] run=$RUN_NAME model=$MODEL_PATH geometry=${NUM_LAYERS}x${HIDDEN_SIZE} TP=$TP actor_gpus=$ACTOR_GPUS rollout_gpus=$ROLLOUT_GPUS rollout_tp=$ROLLOUT_GPUS_PER_ENGINE colocate=$COLOCATE rollouts=$NUM_ROLLOUT bs=$ROLLOUT_BS G=$G gbs=$GBS turns=$OSWORLD_MAX_TURNS response_cap=$MAX_RESPONSE train_context=$SEQ_LENGTH kl_coef=$KL_LOSS_COEF" | tee -a "$LOG"

cd "$PROJ/frameworks/slime-ascend" || exit 1
TRAIN_ENTRYPOINT=train_async.py
[ "$COLOCATE" = "1" ] && TRAIN_ENTRYPOINT=train.py
[ "$EVAL_ONLY" = "1" ] && TRAIN_ENTRYPOINT=train.py
echo "[gui-grpo] entrypoint=$TRAIN_ENTRYPOINT" | tee -a "$LOG"
python "$TRAIN_ENTRYPOINT" \
  --custom-model-provider-path methods.openapps_vision.model_provider \
  --rollout-function-path slime.osworld_rollout.generate_rollout \
  --train-backend megatron \
  --balance-data \
  --distributed-timeout-minutes "${OPENAPPS_DISTRIBUTED_TIMEOUT_MINUTES:-60}" \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "$ACTOR_GPUS" \
  --rollout-num-gpus "$ROLLOUT_GPUS" \
  --num-gpus-per-node "$NUM_GPUS_PER_NODE" \
  --hf-checkpoint "$MODEL_PATH" \
  --load "$LOAD_PATH" \
  --ref-load "$MODEL_PATH" \
  --no-load-optim \
  --no-load-rng \
  --rotary-base "$ROTARY_BASE" \
  --swiglu \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --group-query-attention \
  --num-query-groups "$NUM_QUERY_GROUPS" \
  --use-rotary-position-embeddings \
  --disable-bias-linear \
  --normalization RMSNorm \
  --norm-epsilon 1e-6 \
  --vocab-size "$VOCAB_SIZE" \
  --kv-channels "$KV_CHANNELS" \
  --seq-length "$SEQ_LENGTH" \
  --micro-batch-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --log-probs-chunk-size 256 \
  --update-weight-buffer-size 134217728 \
  --tensor-model-parallel-size "$TP" \
  --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE" \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --megatron-to-hf-mode bridge \
  --prompt-data "$DATA_PATH" \
  --input-key instruction \
  --multimodal-keys '{"image": "images"}' \
  --apply-chat-template \
  ${SHUFFLE_ARGS[@]+"${SHUFFLE_ARGS[@]}"} \
  ${DEBUG_ARGS[@]+"${DEBUG_ARGS[@]}"} \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
  --rm-type none \
  --num-rollout "$NUM_ROLLOUT" \
  --rollout-batch-size "$ROLLOUT_BS" \
  --over-sampling-batch-size "$ROLLOUT_BS" \
  --n-samples-per-prompt "$G" \
  --global-batch-size "$GBS" \
  --rollout-max-response-len "$MAX_RESPONSE" \
  --rollout-temperature 1.0 \
  --advantage-estimator grpo \
  --normalize-advantages \
  ${VCS_ARGS[@]+"${VCS_ARGS[@]}"} \
  --save-debug-rollout-data "$SAVE_DIR/rollout_{rollout_id}.pt" \
  "${ROLLOUT_LOGPROB_ARGS[@]}" \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --kl-coef 0.0 \
  "${KL_ARGS[@]}" \
  --optimizer adam \
  --optimizer-cpu-offload \
  --overlap-cpu-optimizer-d2h-h2d \
  --use-precision-aware-optimizer \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --save "$SAVE_DIR" \
  --save-interval "$SAVE_INTERVAL" \
  --no-save-optim \
  --no-save-rng \
  --sglang-attention-backend ascend \
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION" \
  --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
  --sglang-chunked-prefill-size "$SGLANG_CHUNKED_PREFILL" \
  --sglang-disable-radix-cache \
  "${COLOCATE_ARGS[@]}" \
  --attention-backend flash \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --no-gradient-accumulation-fusion \
  --log-interval 1 \
  --seed 42 \
  >> "$LOG" 2>&1
train_rc=$?
echo "[gui-grpo] exited rc=$train_rc $(date '+%F %H:%M:%S')" | tee -a "$LOG"
exit "$train_rc"
