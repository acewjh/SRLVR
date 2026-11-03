# shellcheck shell=bash
# ============================================================================
# Size-independent model resolution for the opsd-slime math harness.
#
#   MODEL=<slug>        (env; default qwen3-1p7b)   # see configs/models/
#
# MUST BE SOURCED, not executed:
#     source "$PROJ/train_scripts/lib/model_env.sh" || exit 1
#
# Every math launcher (train / dev-eval / GRPO) sources this and then uses the
# MODEL_* variables below. That is what makes "run 1.7B, 4B or 8B" a matter of
# picking a config file rather than picking a script: the launchers carry no
# size-specific value of their own.
#
# On success, the caller has:
#   MODEL_SLUG / MODEL_NAME        identity of the selected card
#   MODEL_DIR                      HF checkpoint directory (--hf-checkpoint)
#   MODEL_ARGS[]                   vendored Megatron model arguments
#   MODEL_TP / MODEL_PP            actor parallelism (DP is implied: the launcher
#                                  derives it from --actor-num-gpus-per-node)
#   MODEL_MBS_DEFAULT              default micro-batch size (env MBS overrides)
#   MODEL_ALLOC_CONF                PYTORCH_NPU_ALLOC_CONF default ("" = leave stock)
#   MODEL_SGLANG_*                 rollout-engine sizing defaults
#   MODEL_TASK                     'math' | 'alfworld', the task component of the
#                                  run name (the card describes the model; the run
#                                  directory is named by scripts/run_names.py)
#   CUA_BASE / CUA_PROJ / CUA_CHECKPOINTS
#
# On failure it prints a diagnostic and `return 1`s, so the caller's
# `|| exit 1` turns a bad MODEL slug into a fail-fast instead of a run that
# silently trains the wrong geometry.
# ============================================================================

CUA_BASE="${CUA_BASE:-/workspace/cua}"
CUA_PROJ="${CUA_PROJ:-$CUA_BASE/repos/opsd-slime}"
CUA_CHECKPOINTS="${CUA_CHECKPOINTS:-$CUA_BASE/checkpoints}"
CUA_MODELS_DIR="${CUA_MODELS_DIR:-$CUA_PROJ/configs/models}"

MODEL_SLUG="${MODEL:-qwen3-1p7b}"

_model_cfg="$CUA_MODELS_DIR/$MODEL_SLUG.sh"
if [ ! -f "$_model_cfg" ]; then
  echo "[model-env] FATAL: MODEL=$MODEL_SLUG has no card (looked for $_model_cfg)." >&2
  echo "[model-env] available: $(ls -1 "$CUA_MODELS_DIR" 2>/dev/null | sed 's/\.sh$//' | tr '\n' ' ')" >&2
  unset _model_cfg
  return 1
fi

# shellcheck source=/dev/null
source "$_model_cfg"
unset _model_cfg

# ---- validation: a half-filled card must fail here, not 20 minutes into a run --
if [ -z "${MODEL_NAME:-}" ] || [ -z "${MODEL_DIR:-}" ]; then
  echo "[model-env] FATAL: $MODEL_SLUG.sh must set MODEL_NAME and MODEL_DIR." >&2
  return 1
fi
if [ -z "${MODEL_ARGS[*]:-}" ]; then
  echo "[model-env] FATAL: $MODEL_SLUG.sh must set MODEL_ARGS." >&2
  return 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[model-env] FATAL: MODEL_DIR=$MODEL_DIR does not look like an HF checkpoint (no config.json)." >&2
  return 1
fi

: "${MODEL_TP:=1}"
: "${MODEL_PP:=1}"
: "${MODEL_MBS_DEFAULT:=1}"
: "${MODEL_ALLOC_CONF:=}"
: "${MODEL_SGLANG_MEM_FRACTION:=0.82}"
: "${MODEL_SGLANG_CHUNKED_PREFILL:=8192}"
: "${MODEL_TASK:=math}"

for _n in "$MODEL_TP" "$MODEL_PP" "$MODEL_MBS_DEFAULT"; do
  case "$_n" in
    ''|*[!0-9]*|0)
      echo "[model-env] FATAL: $MODEL_SLUG.sh has a non-positive integer knob ($_n)." >&2
      unset _n
      return 1
      ;;
  esac
done
unset _n

export MODEL_SLUG MODEL_NAME MODEL_DIR MODEL_TP MODEL_PP MODEL_MBS_DEFAULT
export MODEL_ALLOC_CONF MODEL_TASK
export CUA_BASE CUA_PROJ CUA_CHECKPOINTS

# ---------------------------------------------------------------------------
# Run identity: <task>-<model>-<method>[-<variant>]-<YYYYMMDD_HHMMSS>
#
# Allocated here, through the one implementation (scripts/run_names.py), so a
# launcher cannot invent its own naming scheme -- which is how this tree ended up
# with `math_opsd_slime`, `qwen3-4b-math-opsd-v2` and `reward_cv_vnext_clean_bshd`
# side by side.
# ---------------------------------------------------------------------------
RUN_NAMES_PY="$CUA_PROJ/scripts/run_names.py"

# name_rule_method <OPSD_METHOD> -> the method slug the rule uses
#
# PYTHONPATH is cleared for these two helpers on purpose. run_names.py is pure
# stdlib, but the launcher's own PYTHONPATH puts `compat/sitecustomize.py` on the
# import path, and that shim prints diagnostics to STDOUT -- which the `$(...)`
# capture below would fold into the name ("[compat] ...\nhsi"), turning a valid
# method into an unknown one. A naming tool must not depend on the training
# PYTHONPATH, so it is run without it.
name_rule_method() {
  PYTHONPATH= python3 "$RUN_NAMES_PY" --method-for "$1"
}

# allocate_run_name <method-slug> [variant] -> a fresh conforming name
allocate_run_name() {
  local method="$1" variant="${2:-}"
  if [ -n "$variant" ]; then
    PYTHONPATH= python3 "$RUN_NAMES_PY" --allocate --task "$MODEL_TASK" --model "$MODEL_SLUG" \
        --method "$method" --variant "$variant"
  else
    PYTHONPATH= python3 "$RUN_NAMES_PY" --allocate --task "$MODEL_TASK" --model "$MODEL_SLUG" \
        --method "$method"
  fi
}
