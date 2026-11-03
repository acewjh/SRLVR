#!/usr/bin/env bash
# ============================================================
# Post-training evaluation for alfworld-qwen3-4b-hsi-repeat-stop-s42.
#
# Stage A: the frozen 64-game dev subset at cp25 / cp50 / cp75 / cp100.
# Stage B: full ALFWorld (valid_seen 140 + valid_unseen 134) at cp100 and, when it
#          differs, at the dev-best checkpoint.
#
# Every evaluation goes through eval_alfworld_ckpt.sh, which pins the iteration
# with a mirror dir and refuses to report numbers unless the log carries
# "successfully loaded checkpoint ... at iteration N". Eval is eval-only
# (--num-rollout 0); the training loop never scores anything.
#
# Never deletes, never touches another run's directory.
# ============================================================
set -uo pipefail

R=/workspace/cua/repos/opsd-slime
RUN="${RUN_NAME:-alfworld-qwen3-4b-hsi-repeat-stop-s42-20261003_022250}"
SRC="$R/runs/$RUN"
OUT="$SRC/eval"
DEV_CFG="$R/tasks/alfworld/v1/configs/eval_alfworld_dev64.json"
FULL_CFG="$R/tasks/alfworld/v1/configs/eval_alfworld.json"

mkdir -p "$OUT"
PIPE_LOG="$OUT/pipeline.log"
SUMMARY="$OUT/EVAL_SUMMARY.md"

# Execution knobs held identical to the training arm's protocol.
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export G=1 ROLLOUT_BS=4
export BACKBONE=4b
export METHOD_SLUG=hsi
export OPSD_V2_LOSS_NORM=traj
export ALFWORLD_V2_HPI_SPAN=full
export ALFWORLD_V2_HSI_GRANULARITY=token
# Eval is the HSI arm's own code path; the repeat-stop flag stays ON so the
# loaded run is scored by the same module tree it was trained with. It has no
# effect on generation (it only shapes the loss), but keeping it set avoids
# scoring with a different import surface than the one that produced the weights.
export ALFWORLD_V2_REPEAT_STOP=1

log() { echo "[pipe] $(date '+%F %T') $*" | tee -a "$PIPE_LOG"; }

eval_one() {
  # $1 = iteration, $2 = config path, $3 = tag (dev64|full)
  local iter="$1" cfg="$2" tag="$3" name
  name=$(python "$R/scripts/run_names.py" --allocate --task alfworld --model qwen3-4b \
           --method hsi --variant "repeat-stop-s42-cp${iter}-${tag}") || return 1
  log "eval start iter=$iter tag=$tag name=$name"
  EVAL_CONFIG="$cfg" RUN_NAME="$name" \
    bash "$R/tasks/alfworld/scripts/eval_alfworld_ckpt.sh" "$SRC" "$iter" >> "$PIPE_LOG" 2>&1
  local rc=$?
  echo "$iter $tag $name $rc" >> "$OUT/eval_index.txt"
  log "eval done  iter=$iter tag=$tag rc=$rc"
  return $rc
}

success_rate() {
  # Average the per-dataset success rates printed by the v2 eval rollout.
  local name="$1"
  python3 - "$R/logs/$name.log" <<'PY'
import re, sys
try:
    txt = open(sys.argv[1], errors="replace").read()
except OSError:
    print(""); raise SystemExit
rates = [float(m) for m in re.findall(r"\[v2-eval\][^\n]*success_rate=([0-9.]+)", txt)]
print(sum(rates) / len(rates) if rates else "")
PY
}

iter_of() { grep -E "^$1 $2 " "$OUT/eval_index.txt" 2>/dev/null | tail -1 | awk '{print $3}'; }

log "pipeline start for $RUN"

# ---- Stage A: dev subset -----------------------------------------------------
: > "$OUT/eval_index.txt"
for iter in 25 50 75 100; do
  [ -d "$SRC/$(printf 'iter_%07d' "$iter")" ] || { log "skip iter=$iter (no checkpoint)"; continue; }
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
    BEST_ITER="$iter"; BEST_RATE="$rate"; BEST_NAME="$name"
  fi
done < "$OUT/eval_index.txt"
log "dev-best: iter=${BEST_ITER:-none} rate=${BEST_RATE:-n/a}"

# ---- Stage B: full ALFWorld --------------------------------------------------
eval_one 100 "$FULL_CFG" full || log "WARN full eval failed at iter=100"
if [ -n "$BEST_ITER" ] && [ "$BEST_ITER" != "100" ]; then
  eval_one "$BEST_ITER" "$FULL_CFG" full || log "WARN full eval failed at dev-best iter=$BEST_ITER"
fi

# ---- summary -----------------------------------------------------------------
{
  echo "# Repeat-stop eval summary -- $RUN"
  echo
  echo "Frozen dev subset: \`tasks/alfworld/v1/configs/data/dev64.jsonl\` "
  echo "(md5 \`$(md5sum "$R/tasks/alfworld/v1/configs/data/dev64.jsonl" | cut -d' ' -f1)\`, 32 valid_seen + 32 valid_unseen,"
  echo "first 32 by sorted game_id per split -- fixed before any result was seen)."
  echo
  echo "| iter | tag | success_rate | eval run |"
  echo "|------|-----|--------------|----------|"
  while read -r iter tag name rc; do
    rate=$(success_rate "$name")
    echo "| cp$iter | $tag | ${rate:-FAILED(rc=$rc)} | \`$name\` |"
  done < "$OUT/eval_index.txt"
  echo
  echo "dev-best: cp${BEST_ITER:-none} (rate ${BEST_RATE:-n/a})"
} > "$SUMMARY"

log "pipeline complete -> $SUMMARY"
