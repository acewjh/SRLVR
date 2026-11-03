#!/bin/bash
# Shared helpers for every eval queue on this node.
#
# These three were copy-pasted across thirteen `eval_queue_*.sh` /
# `eval_*.sh` scripts. `count_samples` was byte-identical in all six files that
# had it; `cleanup_orphans` existed in two spellings that differ only in line
# wrapping; `resolve_tag` was duplicated verbatim between the v2 and v3 queues.
#
# Source it the same way as eval_gpu_defaults.sh:
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
#
# `run_eval_cell` below is the CANONICAL body of the newest lineage
# (2026-10-04/06). Three queues already shared it verbatim apart from their log
# tag; the remaining variants are older lineages (a 5-arg form with EVAL_K=, a
# 3-arg fixed-grader form, and the hf-dir form) and need per-queue adaptation
# before they can call it — see

# Eval-protocol constants (OPSD_EVAL_MAX_NEW / _K / _CONTEXT_LENGTH /
# _MEM_FRACTION / _CHUNKED_PREFILL) — single source, R12. Sourced here so every
# queue that already sources eval_common.sh gets them without a second line.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_protocol.sh"

# One eval cell: stage the checkpoint, run the eval, verify the sample count.
#
#   run_eval_cell <cp> <dataset> <config> <k> <prefix> <conc> [<expected_problems=30>]
#
# `cp` carries the literal "cp" (e.g. "cp25"), so the checkpoint iteration is
# `cp` minus the prefix, not a re-interpolation of it. `expected_problems`
# defaults to 30 (AIME24/25, HMMT25); pass the dataset's own count otherwise —
# the assertion exists so a silently short dump is caught at the cell, not at
# summary time. Set EVAL_CELL_LOG_TAG to control the log prefix.
run_eval_cell() {
  local cp=$1 ds=$2 cname=$3 k=$4 prefix=$5 conc=$6 problems=${7:-30}
  local log="${EVAL_CELL_LOG_TAG:-eval-cell}"
  local iter=$(( ${cp#cp} - 1 ))
  local tag="${prefix}_${cp}_${ds}"
  local staging=$P/runs/eval_staging/${prefix}_${cp}_${iter}
  local dump=$P/runs/dev_eval/${tag}_eval_0.pt
  if [ -s "$dump" ]; then echo "[$log] $tag already has a dump -- skipping" >> "$LOG"; return 0; fi
  echo "[$log] === $tag K=$k ($(date '+%F %H:%M:%S')) ===" >> "$LOG"
  bash $P/scripts/make_eval_staging.sh "$RUN_DIR" "$iter" "$staging" >> "$LOG" 2>&1
  EVAL_CONCURRENCY="$conc" EVAL_CONFIG="$P/configs/${cname}" \
    bash $P/train_scripts/run_dev_eval.sh "$staging" "$tag" >> "$LOG" 2>&1
  local rc=$?
  echo "[$log] rc=$rc for $tag" >> "$LOG"
  if [ "$rc" = "0" ] && [ -s "$dump" ]; then
    local n; n=$(count_samples "$dump")
    if [ "$n" != "$(( problems * k ))" ]; then
      echo "[$log] FATAL: $tag produced $n samples, expected $(( problems * k ))" >> "$LOG"
      cleanup_orphans; exit 1
    fi
    echo "[$log] $tag verified: $n samples" >> "$LOG"
  fi
  cleanup_orphans
}

# Kill leftover SGLang shards/schedulers and the ray cluster between cells.
# Without this an orphaned engine from the previous cell keeps holding NPUs and
# the next cell silently runs on a partial card set.
cleanup_orphans() {
  pkill -9 -f 'sglang[:]:sched' 2>/dev/null
  pkill -9 -f 'sglang[:]:detok' 2>/dev/null
  pkill -9 -f 'sglang[.]launch_serve' 2>/dev/null
  sleep 5; ray stop --force >/dev/null 2>&1; sleep 3
}

# Number of samples in an eval dump.
count_samples() {
  python - "$1" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
s = d["samples"] if isinstance(d, dict) else d
print(len(s))
PY
}

# resolve_tag <cp> <dataset> <prefix>... -> echoes the FIRST tag whose dump exists.
#
# Aliases in priority order. "hpisel" is listed for the formal phase because the
# selector cells turned out to run the FULL formal protocol (K=12 / 38912;
# --n-samples-per-eval-prompt does not override the eval-config's
# n_samples_per_eval_prompt), so a selector dump IS a formal AIME24 result and
# must never be recomputed.
resolve_tag() {
  local cp=$1 ds=$2; shift 2
  local pfx cand
  for pfx in "$@"; do
    for cand in "${pfx}_${cp}_${ds}" "${pfx}_cp${cp}_${ds}"; do
      if [ -s "$P/runs/dev_eval/${cand}_eval_0.pt" ]; then echo "$cand"; return; fi
    done
  done
  echo "$1_${cp}_${ds}"
}
