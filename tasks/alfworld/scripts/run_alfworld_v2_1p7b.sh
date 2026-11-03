#!/usr/bin/env bash
# Common model wrapper; algorithms use the existing v2 training entry.
set -eo pipefail
export ARM="${ARM:-grpo}"
case "$ARM" in grpo|gigpo|gspo|srlvr|srlvr_v2|srlvr_v3|srlvr_v4) ;; *) echo "unsupported ARM=$ARM" >&2; exit 2;; esac
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
source "$PROJ/tasks/alfworld/scripts/_runtime_defaults.sh"
# The shared, size-independent ALFWorld recipe (protocol, placement 8+8
# colocated, rollout budget, LoRA ranks, serving/logging defaults).
source "$PROJ/tasks/alfworld/scripts/_v2_defaults.sh"
# 1.7B-specific model-parallel geometry. Everything else is shared.
export TP="${TP:-2}" PP=1
# Match 32 real environment trajectories per rollout: GRPO/GSPO 4x8, SRLVR 32x1.
if [[ "$ARM" == srlvr* ]]; then
  export ROLLOUT_BS="${ROLLOUT_BS:-32}" SRLVR_CALIBRATION="${SRLVR_CALIBRATION:-4}"
fi
# Baselines need sampled-token evidence, but no top-k candidate table.
export ALFWORLD_V2_HSI_CANDIDATE_K=0
# GiGPO knobs (only read when ARM=gigpo). Defaults are the official
# implementation's: return-to-go discount 0.95, step weight 1.0, mean+std
# normalisation inside each anchor state group.
export ALFWORLD_V2_GIGPO_STEP_W="${ALFWORLD_V2_GIGPO_STEP_W:-1.0}"
export ALFWORLD_V2_GIGPO_GAMMA="${ALFWORLD_V2_GIGPO_GAMMA:-0.95}"
export ALFWORLD_V2_GIGPO_MODE="${ALFWORLD_V2_GIGPO_MODE:-mean_std_norm}"
export SRLVR_RATIO_CLIP_LOW="${SRLVR_RATIO_CLIP_LOW:-0.8}" SRLVR_RATIO_CLIP_HIGH="${SRLVR_RATIO_CLIP_HIGH:-1.2}"
if [ "$ARM" = gspo ]; then
  export EPS_CLIP="${EPS_CLIP:-0.0003}" EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.0004}" LR="${LR:-1e-6}"
fi
if [[ "$ARM" == srlvr* ]]; then
  export OPSD_SRLVR=1 SRLVR_MODE="${SRLVR_MODE:-empirical}"
  case "$ARM" in
    srlvr_v2) export SRLVR_VARIANT=v2 ;;
    srlvr_v3) export SRLVR_VARIANT=v3 ;;
    srlvr_v4) export SRLVR_VARIANT=v4 ;;
  esac
  if [[ "$ARM" =~ ^srlvr_v[23]$ ]]; then
    # Keep the original output-head path by default: the chunked response-only
    # implementation showed slower training. Explicit =1 remains available.
    export OPSD_SRLVR_SELECTIVE_LOGITS="${OPSD_SRLVR_SELECTIVE_LOGITS:-0}"
    export OPSD_SRLVR_SELECTIVE_CHUNK="${OPSD_SRLVR_SELECTIVE_CHUNK:-128}"
  fi
  # Settings validation only. Prefer cua-rl where it exists; on clusters where
  # that env is absent, use the launcher's python (or SRLVR_PYTHON explicitly).
  # This does not change any training knob -- it only selects the interpreter.
  SRLVR_PY="${SRLVR_PYTHON:-}"
  if [ -z "$SRLVR_PY" ]; then
    if [ -x /opt/conda/envs/cua-rl/bin/python ]; then
      SRLVR_PY=/opt/conda/envs/cua-rl/bin/python
    else
      SRLVR_PY="${PYTHON:-python3}"
    fi
  fi
  if [ "$ARM" = srlvr ]; then
    PYTHONPATH="$PROJ" "$SRLVR_PY" -c 'from methods.srlvr import Settings; Settings.from_env()'
  elif [ "$ARM" = srlvr_v4 ]; then
    PYTHONPATH="$PROJ" "$SRLVR_PY" -c 'from methods.srlvr_v4 import Settings; Settings.from_env()'
  else
    PYTHONPATH="$PROJ" "$SRLVR_PY" -c 'from methods.srlvr_hsi import Settings; Settings.from_env()'
  fi
fi
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_VARIANT=""
[ "$OPSD_LORA" = "0" ] && TRAIN_VARIANT="-full"
export RUN_NAME="${RUN_NAME:-alfworld-qwen3-1p7b-$ARM${TRAIN_VARIANT}-s$SEED-r$NUM_ROLLOUT-n$ROLLOUT_BS-$STAMP}"
ROOT="${RUN_ROOT:-$PROJ/runs/$RUN_NAME}"
export SAVE_DIR="${SAVE_DIR:-$ROOT/checkpoints}" DEBUG_ROLLOUT_DIR="${DEBUG_ROLLOUT_DIR:-$ROOT/rollouts}" LOG="${LOG:-$ROOT/train.log}"
mkdir -p "$(dirname "$ROOT")"
if [ -z "${RESUME_FROM:-}" ]; then
  if [ "${ALLOW_EXISTING_RUN_ROOT:-0}" = "1" ]; then
    # Supervisor restart before the first checkpoint: keep the same run root so
    # the run name (and its logs) stay one identity.
    mkdir -p "$ROOT"
  else
    mkdir "$ROOT" || { echo "experiment output already exists: $ROOT" >&2; exit 2; }
  fi
fi
mkdir -p "$ROOT" "$DEBUG_ROLLOUT_DIR"
/usr/bin/python3 - "$ROOT/effective_config.json" <<'PY'
import json, os, sys
prefixes = ("ARM", "MODEL", "SRLVR_", "ROLLOUT_", "NUM_ROLLOUT", "ACTOR_", "SAVE_", "ALFWORLD_", "SGLANG_", "TP", "PP", "LR", "EPS_", "SEED", "MAX_", "HISTORY_", "FUTURE_", "SEQ_", "RUN_", "OPSD_SRLVR", "OPSD_LORA", "CLIP_", "KL_")
with open(sys.argv[1], "w") as f:
    json.dump({k:v for k,v in os.environ.items() if k.startswith(prefixes)}, f, indent=2)
PY
printf '[experiment] root=%s host=%s algorithm=%s rounds=%s tasks_per_round=%s\n' "$ROOT" "$(hostname)" "$ARM" "$NUM_ROLLOUT" "$ROLLOUT_BS"
if [[ "$ARM" =~ ^srlvr_v[23]$ ]] && [ "${SRLVR_AUTO_BENCHMARK:-0}" = 1 ]; then
  bash "$PROJ/tasks/alfworld/scripts/_common_v2.sh"
  [ "${DRY_RUN_PARSE:-0}" != 1 ] || exit 0
  # Benchmark the final successfully completed checkpoint using the existing workflow.
  /usr/bin/python3 - "$ROOT" "$SAVE_DIR" "$NUM_ROLLOUT" <<'PYVERIFY'
import json, pathlib, sys
root, checkpoint = map(pathlib.Path, sys.argv[1:3])
rounds = int(sys.argv[3])
final = rounds - 1
records = [json.loads(line) for line in (checkpoint.parent / "srlvr_metrics.jsonl").read_text().splitlines()]
assert [r["rollout"] for r in records] == list(range(rounds)), "incomplete rollout sequence"
assert sum(r["optimizer_steps"] for r in records) == rounds, "expected one successful actor update per rollout"
assert records[-1]["cumulative_optimizer_steps"] == rounds
assert (checkpoint / f"iter_{final:07d}").is_dir(), "missing final checkpoint"
assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == str(final)
PYVERIFY
  TRAIN_CHECKPOINT="$SAVE_DIR"
  TRAIN_RUN_NAME="$RUN_NAME"
  # Inherit model, LoRA, prompt, sampling and evaluation settings from this
  # wrapper; use the existing benchmark and checkpoint-load verification.
  export RUN_NAME="${TRAIN_RUN_NAME}-benchmark-cp${NUM_ROLLOUT}"
  export SAVE_DIR="$ROOT/benchmark/checkpoints" LOG="$ROOT/benchmark.log"
  export DEBUG_ROLLOUT_DIR="$ROOT/benchmark/rollouts"
  export METHOD_SLUG="$ARM" MODEL_SLUG=qwen3-1p7b BACKBONE=1p7b
  export OPSD_SRLVR=0 SRLVR_VARIANT="" OPSD_EXPERIMENT_METRICS=0
  export RESUME_WEIGHTS_ONLY=1 SAVE_TRAIN_STATE=0
  export ACTOR_GPUS=8 ROLLOUT_GPUS=8 TP=2
  unset OPSD_SRLVR_SELECTIVE_LOGITS
  exec bash "$PROJ/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$TRAIN_CHECKPOINT" "$((NUM_ROLLOUT - 1))"
fi
exec bash "$PROJ/tasks/alfworld/scripts/_common_v2.sh"
