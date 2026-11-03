#!/usr/bin/env bash
# ============================================================
# Post-training evaluation for the Candidate-Centered HSI 1.7B run.
#
#   Stage A: the frozen 64-game dev subset at cp10 / cp25 / cp50.
#   Stage B: full ALFWorld (valid_seen 140 + valid_unseen 134) at cp50, and at the
#            dev-best checkpoint when that is not cp50.
#
# Why the evals run AFTER training rather than between rollouts: eval needs the NPUs
# (memory: 评测必须吃满全部显卡), and `eval_alfworld_ckpt.sh` refuses to start while a
# trainer holds them. The checkpoints persist, so scoring them once training stops
# produces exactly the same numbers without stealing cards from the run.
#
# Every evaluation goes through eval_alfworld_ckpt.sh, which pins the iteration with a
# mirror dir and refuses to report numbers unless the log carries
# "successfully loaded checkpoint ... at iteration N" -- the guard against silently
# scoring Base instead of the checkpoint (memory: alfworld-eval-silent-base).
#
# Never deletes; never touches another run's directory.
#
# Usage: bash eval_candidate_campaign.sh <run-name> [model-slug] [backbone] [iters...]
# ============================================================
set -uo pipefail

R=/workspace/cua/repos/opsd-slime
RUN="${1:?usage: eval_candidate_campaign.sh <run-name> [model-slug] [backbone] [iters...]}"
MODEL_SLUG="${2:-qwen3-1p7b}"
BACKBONE="${3:-1p7b}"
shift 3 2>/dev/null || shift $#
# The positional numbers are UPDATE CYCLES (cp10 / cp25 / cp50), not directory
# indices. slime saves on `(rollout_id + 1) % save_interval == 0`
# (frameworks/slime-ascend/slime/utils/misc.py::should_run_periodic_action), so with
# SAVE_INTERVAL=5 the directories are iter_0000004, iter_0000009, … iter_0000049 --
# i.e. cycle N lands in `iter_{N-1}` and `iter_0000010/25/50` are NEVER written.
# Getting this wrong is silent: every Stage-A eval just reports "no checkpoint" and
# the run finishes with an empty eval table.
CYCLES=("$@")
[ "${#CYCLES[@]}" -gt 0 ] || CYCLES=(10 25 50)
ITERS=()
for c in "${CYCLES[@]}"; do ITERS+=("$(( c - 1 ))"); done
LAST="${ITERS[${#ITERS[@]}-1]}"
log_cycles="${CYCLES[*]}"
log_iters="${ITERS[*]}"

SRC="$R/runs/$RUN"
OUT="$SRC/eval"
DEV_CFG="$R/tasks/alfworld/v1/configs/eval_alfworld_dev64.json"
FULL_CFG="$R/tasks/alfworld/v1/configs/eval_alfworld.json"
mkdir -p "$OUT"
PIPE_LOG="$OUT/pipeline.log"
SUMMARY="$OUT/EVAL_SUMMARY.md"
DONE_MARK="$OUT/EVAL_OK"

# Execution knobs held identical to the training arm's protocol.
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export G=1 ROLLOUT_BS=4
export BACKBONE MODEL_SLUG METHOD_SLUG=hsi
export OPSD_V2_LOSS_NORM=traj
# Eval runs the arm's own code path, so the candidate switch stays ON: the module
# tree that scores a checkpoint must be the one that produced it.
export ALFWORLD_V2_HPI=1 ALFWORLD_V2_HSI=1 ALFWORLD_V2_HSI_CANDIDATE=1
export ALFWORLD_V2_HPI_SPAN="${ALFWORLD_V2_HPI_SPAN:-full}"
export ALFWORLD_V2_HSI_GRANULARITY="${ALFWORLD_V2_HSI_GRANULARITY:-token}"

MAX_WAIT_H="${MAX_WAIT_H:-40}"

log() { echo "[pipe] $(date '+%F %T') $*" | tee -a "$PIPE_LOG"; }

if [ -f "$DONE_MARK" ]; then log "EVAL_OK present -- nothing to do"; exit 0; fi

iter_dir() { printf '%s/iter_%07d' "$SRC" "$1"; }

wait_for_training() {
  local waited=0 cap=$(( MAX_WAIT_H * 60 ))
  while :; do
    if pgrep -f '[s]lime-ascend/train.py' > /dev/null; then
      sleep 300; waited=$(( waited + 5 )); continue
    fi
    # No trainer: either it finished, or it is between supervisor restarts. Only
    # proceed once the LAST requested checkpoint exists, and give the supervisor a
    # grace window to bring the run back before declaring it over.
    if [ -d "$(iter_dir "$LAST")" ]; then
      sleep 600
      if pgrep -f '[s]lime-ascend/train.py' > /dev/null; then continue; fi
      log "training idle and cp$LAST present -> evaluating"
      return 0
    fi
    if [ "$waited" -ge "$cap" ]; then
      log "WARN: waited ${MAX_WAIT_H}h without cp$LAST; evaluating what exists"
      return 1
    fi
    sleep 300; waited=$(( waited + 5 ))
    [ $(( waited % 60 )) -eq 0 ] && log "waiting for training (${waited}min elapsed)"
  done
}

eval_one() { # $1=iter $2=cfg $3=tag
  local iter="$1" cfg="$2" tag="$3" name
  name="alfworld-${MODEL_SLUG}-hsi-candidate-s42-cp${iter}-${tag}-$(date +%Y%m%d_%H%M%S)"
  log "eval start iter=$iter tag=$tag name=$name"
  EVAL_CONFIG="$cfg" RUN_NAME="$name" MODEL_SLUG="$MODEL_SLUG" BACKBONE="$BACKBONE" \
    bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$SRC" "$iter" >> "$PIPE_LOG" 2>&1
  local rc=$?
  # The index records the CYCLE (iter + 1), so the summary table reads cp10/cp25/cp50
  # -- the names the task book uses -- rather than the directory's 0-based index.
  echo "$(( iter + 1 )) $tag $name $rc" >> "$OUT/eval_index.txt"
  log "eval done  iter=$iter tag=$tag rc=$rc"
  return $rc
}

success_rate() {
  python3 - "$R/logs/$1.log" <<'PY'
import re, sys
try:
    txt = open(sys.argv[1], errors="replace").read()
except OSError:
    print(""); raise SystemExit
rates = [float(m) for m in re.findall(r"\[v2-eval\][^\n]*success_rate=([0-9.]+)", txt)]
print(sum(rates) / len(rates) if rates else "")
PY
}

log "pipeline start for $RUN (model=$MODEL_SLUG backbone=$BACKBONE cycles=$log_cycles -> iters=$log_iters)"
wait_for_training || true

: > "$OUT/eval_index.txt"

# ---- Stage 0: the cp0 yardstick ---------------------------------------------
# cp0 is the base Qwen3-1.7B: LoRA is zero-initialised, so the untrained adapter is
# the identity and the base model IS cp0. Scored on the SAME dev subset as the
# checkpoints, which is what makes the three numbers comparable; a full-set base
# number from another campaign would not be.
BASE_MODEL="${BASE_MODEL:-/workspace/cua/checkpoints/Qwen3-1.7B}"
if [ -d "$BASE_MODEL" ]; then
  base_name="alfworld-${MODEL_SLUG}-hsi-candidate-s42-cp0-base-$(date +%Y%m%d_%H%M%S)"
  log "eval start iter=0 tag=dev64 (cp0/base) name=$base_name"
  EVAL_CONFIG="$DEV_CFG" RUN_NAME="$base_name" MODEL_SLUG="$MODEL_SLUG" BACKBONE="$BACKBONE" \
    bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" --base "$BASE_MODEL" >> "$PIPE_LOG" 2>&1
  echo "0 dev64 $base_name $?" >> "$OUT/eval_index.txt"
  log "eval done  iter=0 tag=dev64 (cp0/base) rc=$?"
else
  log "skip cp0/base (no HF dir at $BASE_MODEL)"
fi

# ---- Stage A: frozen dev subset ---------------------------------------------
for iter in "${ITERS[@]}"; do
  if [ ! -d "$(iter_dir "$iter")" ]; then log "skip iter=$iter (no checkpoint)"; continue; fi
  eval_one "$iter" "$DEV_CFG" dev64 || log "WARN dev64 eval failed at iter=$iter"
done

# ---- dev-best ----------------------------------------------------------------
BEST_ITER=""; BEST_RATE=""
while read -r iter tag name rc; do
  [ "$tag" = "dev64" ] || continue
  [ "$rc" = "0" ] || continue
  rate=$(success_rate "$name")
  [ -n "$rate" ] || continue
  if [ -z "$BEST_RATE" ] || python3 -c "import sys; sys.exit(0 if float('$rate')>float('$BEST_RATE') else 1)"; then
    BEST_ITER="$iter"; BEST_RATE="$rate"
  fi
done < "$OUT/eval_index.txt"
log "dev-best: iter=${BEST_ITER:-none} rate=${BEST_RATE:-n/a}"

# ---- Stage B: full ALFWorld --------------------------------------------------
eval_one "$LAST" "$FULL_CFG" full || log "WARN full eval failed at iter=$LAST"
if [ -n "$BEST_ITER" ] && [ "$BEST_ITER" != "$LAST" ]; then
  eval_one "$BEST_ITER" "$FULL_CFG" full || log "WARN full eval failed at dev-best iter=$BEST_ITER"
fi

# ---- summary -----------------------------------------------------------------
{
  echo "# Candidate-Centered HSI -- eval summary"
  echo
  echo "Run: \`$RUN\` (Qwen3-1.7B, LoRA r64/a128, G=1, cap 4096, turns 50)."
  echo
  echo "Frozen dev subset: \`tasks/alfworld/v1/configs/data/dev64.jsonl\` "
  echo "(md5 \`$(md5sum "$R/tasks/alfworld/v1/configs/data/dev64.jsonl" | cut -d' ' -f1)\`)."
  echo "Full set: valid_seen 140 + valid_unseen 134."
  echo
  echo "| iter | tag | success_rate | eval run |"
  echo "|------|-----|--------------|----------|"
  while read -r iter tag name rc; do
    rate=$(success_rate "$name")
    echo "| cp$iter | $tag | ${rate:-FAILED(rc=$rc)} | \`$name\` |"
  done < "$OUT/eval_index.txt"
  echo
  echo "dev-best: cp${BEST_ITER:-none} (dev64 rate ${BEST_RATE:-n/a})"
} > "$SUMMARY"

# The efficiency ledger is part of the run's deliverables, so it is produced HERE
# rather than left to a human to remember. `--dumps` reads the per-cycle teacher
# tokens straight off the frozen rollout dumps.
python "$R/scripts/accounting_hsi_candidate.py" "$RUN" --dumps > "$OUT/accounting.out" 2>&1 \
  && log "ledger written -> runs/$RUN/ACCOUNTING.md" \
  || log "WARN ledger generation failed (see $OUT/accounting.out)"

touch "$DONE_MARK"
log "pipeline complete -> $SUMMARY"
