#!/bin/bash
# Fixed dev eval (SERVER-A DEV ONLY / NON-PAPER): eval-only restore + sync + generate.
#
# ONE launcher for every model size: MODEL=<slug> selects the card in
# configs/models/. Convention: train_scripts/run_dev_eval_<size>.sh is a shim.
#
# Reuses the VALIDATED training pipeline end to end:
#   Megatron checkpoint -> slime actor restore (--load) -> update_weights
#   (HCCL/ ray, 310/310 verified) -> existing SGLang rollout engines -> math_verify RM.
# No optimizer update, no second weight mapping, no HF conversion.
#
# Usage:
#   MODEL=qwen3-8b bash train_scripts/run_dev_eval.sh <LOAD_PATH> <TAG>
#     <LOAD_PATH>: HF dir (Base) or a checkpoint staging dir containing
#                  latest_checkpointed_iteration.txt + iter_XXXXXXX (see
#                  scripts/make_eval_staging.sh)
#     <TAG>:       row label for reports/dev_eval_table.md and dump filename.
#                  It must be a conforming run name (scripts/run_names.py), with
#                  the checkpoint appended when the cell is not the final one:
#                  `math-qwen3-8b-rewardcv-vnext-20260930_144822` or
#                  `math-qwen3-8b-rewardcv-vnext-20260930_144822-cp24`.
#                  dev_eval/ is only the store; the TAG is the name that carries
#                  the rule, because that is what a result table cites.
# Env:
#   EVAL_K (samples per problem, default OPSD_EVAL_K=12), EVAL_LORA (default:
#   derived from the load path), NPU budget overrides.

set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib/model_env.sh
source "$PROJ/train_scripts/lib/model_env.sh" || exit 1
# Eval-protocol constants (OPSD_EVAL_*) — single source, R12.
# shellcheck source=../scripts/eval_protocol.sh
source "$PROJ/scripts/eval_protocol.sh" || exit 1

BASE="$CUA_BASE"
MODEL="$MODEL_DIR"
DATA=$BASE/data/Openthoughts_math_30k_opsd/data/train-00000-of-00002.parquet
LOAD_PATH="${1:?usage: run_dev_eval.sh <LOAD_PATH> <TAG>}"
TAG="${2:?usage: run_dev_eval.sh <LOAD_PATH> <TAG>}"
EVAL_K="${EVAL_K:-$OPSD_EVAL_K}"   # formal protocol (K=12); see scripts/eval_protocol.sh

DUMP_DIR=$PROJ/runs/dev_eval
DUMP_TEMPLATE=$DUMP_DIR/${TAG}_{rollout_id}.pt

if [ -s "$DUMP_DIR/${TAG}_eval_0.pt" ]; then
  echo "[dev-eval] dump for tag '$TAG' already exists: $DUMP_DIR/${TAG}_eval_0.pt"
  echo "[dev-eval] delete it to re-run."
  exit 0
fi

source "$PROJ/train_scripts/lib/npu_env.sh"
opsd_enter_env
# PYTORCH_NPU_ALLOC_CONF is set just below, keeping its rationale comment.
opsd_export_npu_env "$PROJ" ""
# OPSD harness patch (2026-09-17): torch_npu expandable allocator — long-horizon
# cp100 generation repeatedly triggered fatal EL0004 (Memory_Allocation_Failure
# inside attention ops / the ascend sampler) after 19-53/360 despite
# mem/chunk/concurrency reductions; evidence points to dynamic serving memory
# fragmentation. expandable_segments makes the caching allocator grow segments
# instead of relying on fixed-size blocks. Exported BEFORE ray start so ray
# actors inherit it; the in-worker effective value is logged by the
# OPSD_ALLOC_STATS_JSONL telemetry (OPSD harness patch in scheduler.py).
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

# --- all-NPU eval default (user directive 2026-09-20) -------------------------
# Single source of truth: scripts/eval_gpu_defaults.sh sets NUM_TRAIN_GPUS /
# NUM_ROLLOUT_GPUS / NUM_NODE_GPUS to ALL visible NPUs and aborts if the budget
# is short. An under-provisioned eval silently wastes hours, so this is a hard
# error rather than a warning. Escape hatch: EVAL_ALLOW_PARTIAL_GPUS=1.
#
# The actor has to rebuild the checkpoint's parallelism, so the minimum driver
# size is TP x PP -- 1 for 1.7B/4B, 2 for the PP2 8B card.
export EVAL_ACTOR_NPUS="$(( MODEL_TP * MODEL_PP ))"
eval_gpu_defaults="$PROJ/scripts/eval_gpu_defaults.sh"
if [ -f "$eval_gpu_defaults" ]; then
  # shellcheck source=../scripts/eval_gpu_defaults.sh
  source "$eval_gpu_defaults"
else
  # Self-contained fallback so a copied/partial checkout still uses every card.
  if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    TOTAL_NPUS=$(printf '%s' "$ASCEND_RT_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
  else
    TOTAL_NPUS=$(npu-smi info -l 2>/dev/null | awk '/Total Count/{print $4; exit}')
  fi
  TOTAL_NPUS="${TOTAL_NPUS:-8}"
  : "${NUM_TRAIN_GPUS:=$EVAL_ACTOR_NPUS}"
  : "${NUM_ROLLOUT_GPUS:=$(( TOTAL_NPUS - NUM_TRAIN_GPUS ))}"
  : "${NUM_NODE_GPUS:=$TOTAL_NPUS}"
  if [ "$(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS ))" -lt "$TOTAL_NPUS" ] \
     && [ "${EVAL_ALLOW_PARTIAL_GPUS:-0}" != "1" ]; then
    echo "[dev-eval] FATAL: eval would use $(( NUM_TRAIN_GPUS + NUM_ROLLOUT_GPUS )) of $TOTAL_NPUS NPUs."
    echo "[dev-eval]        Raise NUM_ROLLOUT_GPUS, or set EVAL_ALLOW_PARTIAL_GPUS=1 on purpose."
    exit 1
  fi
  export NUM_TRAIN_GPUS NUM_ROLLOUT_GPUS NUM_NODE_GPUS
fi
echo "[dev-eval] NPU budget: train=$NUM_TRAIN_GPUS rollout=$NUM_ROLLOUT_GPUS total=${TOTAL_NPUS:-?}"
mkdir -p "$DUMP_DIR" "$PROJ/logs"
LOG=$PROJ/logs/dev_eval_${TAG}.log

# LoRA wiring: checkpoints saved by a LoRA run contain adapter params; the eval
# model must build the SAME LoRA modules or the adapters are silently dropped
# (dist_ckpt_strictness=assume_ok_unexpected) and the eval scores a model that
# is neither base nor the trained policy.
#
# DERIVED from the load path rather than defaulted, because getting it backwards
# fails silently: EVAL_LORA=0 on a LoRA checkpoint invalidated an entire 4B
# checkpoint batch (reports/QWEN3_4B_MANIFEST.md, "B3 BUG"), in which every
# trained cell landed within noise of Base. The discriminator is unambiguous --
# an HF directory has config.json, a megatron dist checkpoint has
# latest_checkpointed_iteration.txt. An explicit EVAL_LORA still wins.
if [ -z "${EVAL_LORA:-}" ]; then
  if [ -f "$LOAD_PATH/latest_checkpointed_iteration.txt" ]; then
    EVAL_LORA=1
  elif [ -f "$LOAD_PATH/config.json" ]; then
    EVAL_LORA=0
  else
    echo "[dev-eval] FATAL: $LOAD_PATH is neither an HF dir (config.json) nor a megatron checkpoint"
    echo "[dev-eval]        (latest_checkpointed_iteration.txt) -- set EVAL_LORA=0|1 to override."
    exit 1
  fi
  echo "[dev-eval] lora=$EVAL_LORA (auto-detected from the load path)"
fi
if [ "$EVAL_LORA" = "1" ]; then
  export OPSD_LORA=1
  export OPSD_LORA_R="${OPSD_LORA_R:-64}"
  export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
  export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"
  EVAL_LORA_ARGS=(--custom-model-provider-path methods.lora_megatron.custom_model_provider)
else
  EVAL_LORA_ARGS=
fi

# --load: HF dir for Base, Megatron dist-ckpt staging dir for checkpoints
CKPT_ARGS=(
   --hf-checkpoint "$MODEL"
   --load "$LOAD_PATH"
   --save "$PROJ/runs/_eval_scratch"
   --save-interval 1000000
)

ROLLOUT_ARGS=(
   --prompt-data "$DATA"
   --input-key problem
   --label-key solution
   --rollout-function-path tasks.math.math_rollout.generate_rollout
   --eval-function-path slime.rollout.sglang_rollout.generate_rollout
   --rollout-batch-size 32
   --n-samples-per-prompt 1
   --rollout-num-gpus "$NUM_ROLLOUT_GPUS"
   --rollout-num-gpus-per-engine 1
   --sglang-attention-backend ascend
   # Ascend fused sampling path (canonical formal-eval default; see
   # reports/formal_eval_throughput_diagnosis.md P2). NOTE: slime auto-registers
   # sglang ServerArgs with the --sglang- prefix — the UN-prefixed
   # --sampling-backend is accepted by slime's parser but never forwarded to the
   # engines (verified: engines booted with sampling_backend='pytorch'). Startup
   # log must show sampling_backend=ascend - silent fallback is a config error.
   --sglang-sampling-backend ascend
   # mem 0.82 restored: the [chunk,V] logprobs spike (the 0.72-era OOM source) is
   # eliminated at source by return_logprob=False. Fallback ladder on OOM: 0.82 ->
   # 0.80 -> 0.78 (set EVAL_MEM_FRACTION explicitly); do NOT go back to 0.72.
   --sglang-mem-fraction-static "${EVAL_MEM_FRACTION:-$MODEL_SGLANG_MEM_FRACTION}"
   # Qwen3 native max_position_embeddings = 40960 (matches the formal eval budget)
   --sglang-context-length "$OPSD_EVAL_CONTEXT_LENGTH"
   # 32GB: the full-vocab logits spike scales with the prefill chunk
   # (8192 x 151936 x 4B ~ 5GB/chunk OOMed engines). 2048 keeps the spike ~1.2GB.
   --sglang-chunked-prefill-size "${EVAL_CHUNKED_PREFILL:-$OPSD_EVAL_CHUNKED_PREFILL}"
   --sglang-disable-radix-cache
   # eval fires 180 x 8192-token thinking requests at once; unthrottled this
   # exhausts the engines' KV pool (503 on /generate). 12 in-flight per engine.
   #
   # DEFAULT: KEEP AT 8.  conc=16 was tried on 2026-10-04 and REVERTED on the
   # evidence below. Do not raise it again without re-measuring per-cell
   # CUMULATIVE s/sample (not the early-window rate).
   #
   # Measured, cp25-aime24 K=12 / Qwen3-4B, same 8 NPUs, same script:
   #     conc=2  -> 30.8 s/sample cumulative   (live control: GRPO-4B cell)
   #     conc=16 ->  9.1 s/sample at 132/360   (warm window only)
   #                 16.3 s/sample at 275/360  (cumulative, degrading)
   #                100.0 s/it     at 292/360  (instantaneous)
   #   => projected cell ~3h35m vs the conc=2 control's ~3h05m: NET SLOWER.
   # The early ~3x is real but does not survive the tail, so quoting it as the
   # headline (as an earlier revision of this comment did) was wrong.
   #
   # MECHANISM: K=12 response lengths are heavy-tailed, so the END of a cell is
   # dominated by LONG requests. Early short responses leave KV slack and conc=16
   # wins; the tail then saturates the pool -- engine logs **~97 x "KV cache pool
   # is full. Retract requests."** over one 360-sample cell (token-usage p50 0.83
   # / p90 0.97 / max 1.00; only 2-6 in flight at conc=4). NOTE: a grep for
   # '503'/'KV pool' MISSES that string, so a "0 events" reading taken that way is
   # meaningless -- grep 'Retract' instead.
   # The retractions are sglang preemption, not corruption: the sample is still
   # drawn, so pass@k/avg@k stay unbiased either way. This is a THROUGHPUT
   # finding, not a correctness one.
   #
   # NEXT: sweep conc 4/8/16 on ONE cell, compare CUMULATIVE s/sample, and move
   # the default only to the knee. 8 is the current (untested) candidate.
   --sglang-server-concurrency "${EVAL_CONCURRENCY:-8}"   # KV budget for 38912-token requests
)

# Fixed per-sample-j seeds without the engine-side deterministic flag (the
# ascend attention backend rejects it). Eval-only env gate in sglang_rollout.
export OPSD_EVAL_FIXED_SEEDS=1

TRAIN_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node "$NUM_TRAIN_GPUS"
   --num-gpus-per-node "${NUM_NODE_GPUS:-8}"

   --num-rollout 0
   --eval-interval 1

   # eval restores MODEL weights only: Megatron builds the live optimizer's
   # sharded state dict during load and dies on `assert len(steps)==1`
   # (fresh optimizer, empty state). Eval needs neither optimiser nor RNG state.
   --no-load-optim
   --no-load-rng

   --transformer-impl local
   --tensor-model-parallel-size "$MODEL_TP"
   --pipeline-model-parallel-size "$MODEL_PP"
   --context-parallel-size 1
   --expert-model-parallel-size 1

   --micro-batch-size 2
   --global-batch-size 32
   --calculate-per-token-loss

   --loss-type custom_loss
   --custom-loss-function-path methods.opsd_loss.opsd_loss_function
   --custom-convert-samples-to-train-data-path tasks.math.math_data.convert_samples_to_train_data
   --custom-rm-path tasks.math.rm_math_verify.rm_math_verify

   --eval-config "${EVAL_CONFIG:-$PROJ/configs/eval_aime24_dev.json}"
   --apply-chat-template
   --apply-chat-template-kwargs '{"enable_thinking": true}'
   --n-samples-per-eval-prompt "$EVAL_K"
   # formal budget: NEVER lower for formal eval (formal_eval.sh); 8192 truncates
   # 85% of thinking traces (see reports/param_alignment_vs_baseline.md §3)
   --eval-max-response-len "${EVAL_MAX_NEW:-$OPSD_EVAL_MAX_NEW}"
   --log-passrate
   --rollout-seed 0
   --save-debug-rollout-data "$DUMP_TEMPLATE"

   --optimizer adam
   --lr 5e-6
   --lr-decay-style constant
   # eval-only: num_rollout=0 would drive slime's lr_decay_steps to 0 and trip
   # the actor's `assert self.lr_decay_steps > 0`; constant LR makes this inert.
   --lr-decay-iters 1
   --clip-grad 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --no-gradient-accumulation-fusion
   --log-interval 1
)

echo "[dev-eval] model=$MODEL_NAME slug=$MODEL_SLUG dir=$MODEL tp=$MODEL_TP pp=$MODEL_PP"
echo "[dev-eval] tag=$TAG load=$LOAD_PATH k=$EVAL_K dump=$DUMP_TEMPLATE"
echo "[dev-eval] log: $LOG"
export OPSD_ALLOC_STATS_JSONL="${OPSD_ALLOC_STATS_JSONL:-$PROJ/logs/alloc_stats_${TAG}.jsonl}"
echo "FORMAL_EVAL_EFFECTIVE_CONFIG: return_logprob=$( [ "${OPSD_EVAL_RETURN_LOGPROB:-0}" = "1" ] && echo True || echo False ) sampling_backend=ascend mem_fraction_static=${EVAL_MEM_FRACTION:-$MODEL_SGLANG_MEM_FRACTION} chunked_prefill_size=${EVAL_CHUNKED_PREFILL:-$OPSD_EVAL_CHUNKED_PREFILL} dispatch=client-round-robin context=$OPSD_EVAL_CONTEXT_LENGTH K=$EVAL_K max_new_tokens=${EVAL_MAX_NEW:-$OPSD_EVAL_MAX_NEW} lora=${EVAL_LORA:-0} tp=${MODEL_TP} pp=${MODEL_PP} actor_npus=$NUM_TRAIN_GPUS"
echo "FORMAL_EVAL_ALLOCATOR_CONFIG: PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-unset} alloc_stats_jsonl=${OPSD_ALLOC_STATS_JSONL}"

# env sanity: a stale ray head started under another conda env poisons every
# actor (ModuleNotFoundError inside the engine); fail fast instead.
if ! python -c "import sglang_router" 2>/dev/null; then
  echo "[dev-eval] FATAL: active python is not the slime-npu env (sglang_router import failed)"
  which python
  exit 1
fi

# always start a FRESH head: reusing a stale one inherits its env
ray stop --force > /dev/null 2>&1 || true
pkill -f "sglang.launch_serve[r]" 2>/dev/null || true
sleep 5
ray start --head --node-ip-address 127.0.0.1 --disable-usage-stats \
    --dashboard-host 0.0.0.0 --dashboard-port 8265 2>&1 | tail -2

cleanup_ray() {
  ray stop --force 2>/dev/null || true
  pkill -f "sglang.launch_serve[r]" 2>/dev/null || true
  # SIGKILLed SGLangEngine actors orphan their `sglang::scheduler` /
  # `sglang::detokenizer` children (~27GB/card held; invisible to both the
  # launch_serve pattern and npu-smi "python" checks) — sweep them too.
  # (OPSD harness patch, 2026-09-17)
  pkill -9 -f 'sglang[:]:sched' 2>/dev/null || true
  pkill -9 -f 'sglang[:]:detok' 2>/dev/null || true
}
trap cleanup_ray EXIT

cd "$PROJ"
python "$PROJ/frameworks/slime-ascend/train.py" \
   "${MODEL_ARGS[@]}" \
   "${EVAL_LORA_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" \
   2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}

if [ "$RC" = "0" ] && [ -s "$DUMP_DIR/${TAG}_eval_0.pt" ]; then
  python "$PROJ/tasks/math/aggregate_dev_eval.py" --dump "$DUMP_DIR/${TAG}_eval_0.pt" --tag "$TAG" \
    2>&1 | tee "$PROJ/logs/dev_eval_${TAG}_aggregate.log"
else
  echo "[dev-eval] FAILED rc=$RC (dump present: $(ls "$DUMP_DIR" 2>/dev/null | grep "$TAG" | tr '\n' ' '))"
fi
exit "$RC"
