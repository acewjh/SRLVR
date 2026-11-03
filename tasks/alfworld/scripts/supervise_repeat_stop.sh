#!/bin/bash
# ============================================================
# Unattended supervisor for alfworld-qwen3-4b-hsi-repeat-stop-s42.
#
# Responsibilities, in priority order:
#   1. keep exactly one instance alive (flock), so a cron heartbeat can restart
#      it without racing;
#   2. if the training process is gone and training is not finished, resume from
#      the newest complete checkpoint -- automatically, no human in the loop;
#   3. on an OOM death, step the sparse-CE chunk down the FIXED ladder
#      256 -> 128 -> 64. Only the execution chunk changes; the algorithm, the
#      samples and the effective batch are untouched;
#   4. when training finishes, run the eval pipeline and leave EVAL_DONE;
#   5. never touch another run's directory, and never delete anything.
#
# Idempotency comes from marker files under $STATE, so re-running this script is
# always safe. Log freshness guards against declaring a live run dead.
# ============================================================
set -uo pipefail

R=/workspace/cua/repos/opsd-slime
RUN="${RUN_NAME:-alfworld-qwen3-4b-hsi-repeat-stop-s42-20261003_022250}"
SAVE="$R/runs/$RUN"
LOG="$R/logs/$RUN.log"
STATE="/tmp/afw_repeat_stop/$RUN"
# New lock generation. The previous file could be held by an orphaned child that
# had inherited the fd; a fresh name makes that stale holder irrelevant.
LOCK="/tmp/afw_repeat_stop.v2.lock"
#: Execution chunks, as "pos:view:score". Only these move; the algorithm, the
#: samples and the effective batch are fixed. The FIRST lever is the teacher's
#: forced-scoring position chunk, because that transient [tlen, vocab] fp32
#: tensor is what actually overruns a 32G card.
#: "pos:view:score:recompute_layers". The last field is the activation-checkpointing
#: headroom: the shipped `--recompute-num-layers 1` checkpoints ONE layer of 36, so
#: it buys almost nothing on a memory-bound step. It is the LAST rung because the
#: math campaign found that checkpointing under `--only-train-params-name-list`
#: can detach the LoRA graph (silent zero gradient), so it is only reached after
#: every cheaper lever, and the no-op guard below watches for exactly that.
#: NOTE: a `...:36` rung (full activation checkpointing) was tried first and
#: REMOVED -- measured 2026-10-03 05:49, step 0 reported train/grad_norm exactly
#: 0.0, i.e. checkpointing under --only-train-params-name-list detaches the LoRA
#: graph. A no-op arm is worse than an OOM, so it is not a lever on this stack.
CHUNK_ORDER=("256:4:256:1" "128:4:256:1" "64:4:256:1" "64:2:128:1" "64:1:64:1")
POLL_SECONDS="${POLL_SECONDS:-300}"
#: A live run writes to its log far more often than this; only used as a
#: secondary guard *after* the process check has already failed.
STALE_SECONDS="${STALE_SECONDS:-1800}"

mkdir -p "$STATE" "$(dirname "$LOG")" "$SAVE"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[sup] another supervisor holds $LOCK" >> "$STATE/supervisor.log"
  exit 0
fi

log() { echo "[sup] $(date '+%F %T') $*" >> "$STATE/supervisor.log"; }

latest_iter() {
  [ -s "$SAVE/latest_checkpointed_iteration.txt" ] || { echo ""; return; }
  tr -dc '0-9' < "$SAVE/latest_checkpointed_iteration.txt"
}

train_alive() {
  # pgrep -f cannot self-match here: this supervisor's own argv is
  # `bash .../supervise_repeat_stop.sh`, which contains neither "train.py" nor
  # the run name. The pattern is anchored on train.py first for the same reason.
  pgrep -f "train\.py.*$RUN" > /dev/null 2>&1
}

# Resume from the newest checkpoint whose adapter and trainer state are readable.
resume_point() {
  local best=""
  for d in "$SAVE"/iter_*; do
    [ -d "$d" ] || continue
    [ -s "$d/latest_checkpointed_iteration.txt" ] || [ -s "$SAVE/latest_checkpointed_iteration.txt" ] || true
    [ -f "$d/trainer_state.json" ] || [ -f "$d/common.pt" ] || continue
    best="$d"
  done
  echo "$best"
}

launch() {
  local chunk="$1" resume_from="$2"
  local pos view score rec
  IFS=: read -r pos view score rec <<< "$chunk"
  local args=(env RUN_NAME="$RUN" NUM_ROLLOUT=100 SAVE_INTERVAL=25 \
              OPSD_HPI_POS_CHUNK="$pos" OPSD_HPI_CHUNK="$view" ALFWORLD_V2_SCORE_CHUNK="$score" \
              RECOMPUTE_NUM_LAYERS="${rec:-1}")
  [ -n "$resume_from" ] && args+=(RESUME_FROM="$resume_from")
  log "launch chunk=$chunk resume=${resume_from:-none}"
  echo "$chunk" > "$STATE/chunk"
  # Byte offset of the shared log at launch time: OOM evidence is only ever read
  # from AFTER this point, so a crash from an earlier attempt cannot be
  # misdiagnosed as an OOM of the current one.
  stat -c %s "$LOG" > "$STATE/log_offset" 2>/dev/null || echo 0 > "$STATE/log_offset"
  # The subshell closes the inherited lock fd before detaching, so descendants
  # never hold the orchestrator's flock.
  ( exec 9>&-; setsid nohup "${args[@]}" bash "$R/tasks/alfworld/scripts/run_alfworld_v2_repeat_stop_4b.sh" \
      >> "$STATE/launcher.log" 2>&1 < /dev/null & )
}

chunk_index() { cat "$STATE/chunk" 2>/dev/null || echo "${CHUNK_ORDER[0]}"; }

next_chunk_or_fail() {
  local cur="$1" i
  for i in "${!CHUNK_ORDER[@]}"; do
    if [ "${CHUNK_ORDER[$i]}" = "$cur" ]; then
      if [ $((i + 1)) -lt ${#CHUNK_ORDER[@]} ]; then
        echo "${CHUNK_ORDER[$((i + 1))]}"; return 0
      fi
      return 1
    fi
  done
  echo "${CHUNK_ORDER[0]}"
}

died_by_oom() {
  # Only text written SINCE THE LAST LAUNCH counts. Reading the tail alone made
  # every later death look like the first OOM, so the ladder walked itself down
  # on unrelated crashes.
  local off
  off=$(cat "$STATE/log_offset" 2>/dev/null || echo 0)
  tail -c +$(( off + 1 )) "$LOG" 2>/dev/null \
    | grep -qiE "out of memory|NPU OOM|HBM out of|torch\.OutOfMemory|OutOfMemoryError"
}

record_peak() {
  npu-smi info 2>/dev/null | awk '/[0-9]+ *\/ *32768/ {for(i=1;i<=NF;i++) if($i ~ /^[0-9]+\/[0-9]+$/) print $i}' \
    | tr -d ' ' | awk -F/ '{if($1+0>m) m=$1+0} END{if(m) print m}' > "$STATE/peak_hbm_mb" 2>/dev/null || true
}

# A fresh supervisor must not inherit the previous instance's scan window: the
# log already contains every past crash, so a stale offset made the first tick
# count an old OOM as this run's. Start the window at the current end.
stat -c %s "$LOG" > "$STATE/log_offset" 2>/dev/null || echo 0 > "$STATE/log_offset"

log "supervisor up (run=$RUN)"

while true; do
  record_peak

  if [ -f "$STATE/TRAIN_DONE" ]; then
    if [ ! -f "$STATE/EVAL_DONE" ]; then
      log "training done -> running eval pipeline"
      if bash "$R/tasks/alfworld/scripts/eval_repeat_stop_pipeline.sh" >> "$STATE/eval.log" 2>&1; then
        touch "$STATE/EVAL_DONE"; log "eval pipeline complete"
      else
        log "eval pipeline FAILED rc=$? (will retry next tick)"
      fi
    fi
    log "all done -- standing down"
    exit 0
  fi

  if [ -f "$STATE/TRAIN_DONE" ]; then :; fi
  if train_alive; then
    # Silent-no-op guard: `grad_norm` exactly 0 means no weight moved (the math
    # campaign lost a whole arm to this).
    #
    # LAUNCH-SCOPED, like died_by_oom: an unscoped grep matches the PREVIOUS
    # attempt's line and fires on a healthy run. That false positive happened on
    # 2026-10-03 07:18 -- it killed a run whose own step 0 reported grad_norm
    # 1.52, and the orphaned trainer then died of an unrelated OOM.
    if tail -c +$(( $(cat "$STATE/log_offset" 2>/dev/null || echo 0) + 1 )) "$LOG" 2>/dev/null \
         | grep -aq "grad_norm': 0.0\|grad_norm': 0,"; then
      log "FATAL: grad_norm is exactly 0 -- silent no-op (recompute+LoRA detach?). BLOCKED."
      touch "$STATE/BLOCKED"
      exit 1
    fi
    # A live run: confirm it is not wedged on a silent log.
    age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$STALE_SECONDS" ]; then
      log "WARNING train.py alive but log idle ${age}s (secondary guard only, not killing)"
    fi
    sleep "$POLL_SECONDS"
    continue
  fi

  # No training process. Decide: finished, or died?
  iter="$(latest_iter)"
  cp_dirs=$(ls -d "$SAVE"/iter_* 2>/dev/null | wc -l)
  if [ "${iter:-0}" -ge 99 ] 2>/dev/null; then
    log "no process and latest iteration=$iter -> training complete"
    touch "$STATE/TRAIN_DONE"
    continue
  fi
  if grep -q "exited rc=0" "$LOG" 2>/dev/null && [ "$cp_dirs" -eq 0 ]; then
    log "FATAL: rc=0 with zero checkpoints and iter=$iter -- nothing to resume from"
    touch "$STATE/BLOCKED"; exit 1
  fi

  cur="$(chunk_index)"
  if died_by_oom; then
    if nxt="$(next_chunk_or_fail "$cur")"; then
      log "OOM detected at chunk=$cur -> ladder down to $nxt"
      echo "$nxt" > "$STATE/chunk"
      cur="$nxt"
    else
      log "FATAL: OOM at chunk=$cur and the ladder is exhausted -> 4B/32G BLOCKED"
      npu-smi info > "$STATE/oom_evidence_npu.txt" 2>&1 || true
      tail -n 200 "$LOG" > "$STATE/oom_evidence_log.txt" 2>/dev/null || true
      touch "$STATE/BLOCKED"; exit 1
    fi
  fi

  rp="$(resume_point)"
  log "no train.py (iter=$iter, ckpts=$cp_dirs) -> resuming on chunk=$cur from ${rp:-base}"
  launch "$cur" "$rp"
  sleep 120
done
