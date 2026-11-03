#!/bin/bash
# FULL-GRID K=12 formal eval for the unified-HSI Math Qwen3-4B arm.
#
# Why this exists: eval_queue_hsi_4b.sh mirrors eval_queue_grpo.sh's TWO-STAGE
# design (K=4 selector on AIME24 for cp25/50/75/100 -> K=12 on the top-2 only).
# But the reference rows already on reports/dev_eval_table.md (4b_grpo_cp{24,49,
# 74,99}_{aime24,aime25,hmmt25} and the 4b_oursv2 set) are a FULL 4-cp x 3-dataset
# K=12 grid. Comparing HSI's 2 cps against their 4 leaves holes. So: run the full
# grid for HSI too.
#
# Idempotent: run_eval_cell() skips any cell whose dump already exists, so if the
# two-stage queue already produced the top-2 K=12 cells, this only fills the gaps.
# Uses the SAME lock as the two-stage queue, so the two can never interleave.
#
# Usage: nohup bash scripts/eval_hsi_4b_k12_full.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
EVAL_CELL_LOG_TAG=hsi4b-k12

P=/workspace/cua/repos/opsd-slime
# The run this grid evaluates. Overridable so the same script serves every
# HSI run; the 2026-10-04 default below is the voided candidate-centred run and
# MUST be overridden for the v2 (JSD) run (see the campaign handoff).
RUN_DIR=${HSI4B_RUN_DIR:-$P/runs/math-qwen3-4b-hsi-s42-20261004_040413}
LOG=$P/logs/eval_hsi_4b_k12_full.log
LOCK=/tmp/opsd_eval_hsi4b.lock
DONE=$P/logs/hsi_4b_eval.K12_FULL_DONE

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu
export MODEL="${MODEL:-qwen3-4b}"
export EVAL_LORA=1
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.82}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-2048}"

# ---- client HTTP timeout (2026-10-06, cell cp25_aime25 endgame livelock) ------
# slime's http_utils defaults to 900 s (SLIME_HTTP_TIMEOUT_S unset) and retries
# the SAME engine URL on failure. At the END of a cell the remaining samples are
# the long-tail ones and they converge onto a single engine (measured: 17 running
# reqs on port 15006, 0 on the other six, ~200 tok/s all on that one engine).
# Per-request wall time is then L*N/200 tok/s ~ 38912*18/200 ~ 3500 s >> 900 s,
# so every attempt is aborted at 900 s, retried on the same URL, and the work
# restarts from zero. That is a stable livelock: generation happens, no sample
# can EVER complete (measured 2026-10-06: tqdm frozen at 342/360 for 2h08m,
# zero `POST /generate 200 OK` after 12:57:44, exact abort rate 18/900 s =
# 1.2/min ~= the observed 1.1/min). 342/360 samples were discarded and the cell
# restarted; see logs/hsi_4b_math_code/INCIDENT_20261006_cp25_aime25_endgame_livelock.md.
# 7200 s = 2x the no-thrash worst case above, and still bounded, so a genuine
# silent wedge is converted into a retry within 2 h on a cell that runs 3-10 h.
# Execution-level only: K, temperature, top_p, top_k, max_response_len and the
# grader are untouched (task book §8).
export SLIME_HTTP_TIMEOUT_S="${SLIME_HTTP_TIMEOUT_S:-7200}"

# ---- engine control-plane HTTP timeout (2026-10-06, cell cp25_hmmt25 rc=1) ---
# sglang_engine.py's _http_post/_http_get default to SLIME_SGLANG_HTTP_TIMEOUT=600 s
# (frameworks/slime-ascend/.../sglang_engine.py:36) and this script never set it.
# hsi4bv2_cp25_hmmt25 (started 22:25:06) died at 22:41 with, on ALL SEVEN engines,
#   urllib3.exceptions.ReadTimeoutError: HTTPConnectionPool(host=..., port=1500X):
#   Read timed out. (read timeout=600.0)
#   ray::SGLangEngine.update_weights_from_distributed() -> TimeoutError: timed out
# i.e. the LoRA weight-sync POST, not generation. Same class of bug as the 900 s
# rollout timeout fixed above: a client-side bound firing on an engine that is
# merely slow (here: all 7 simultaneously, at cell start). 3600 s is 6x the
# healthy value (measured healthy weight update ~1 s) and still bounded, so a
# truly dead engine still fails the cell instead of hanging the grid forever.
# Execution-level only: K / temperature / top_p / top_k / max_response_len and the
# grader are untouched (task book s8).
export SLIME_SGLANG_HTTP_TIMEOUT="${SLIME_SGLANG_HTTP_TIMEOUT:-3600}"

exec 9>"$LOCK"
if ! flock -n 9; then echo "[hsi4b-k12] another eval queue holds $LOCK -- exiting" >> "$LOG"; exit 1; fi

# Paused by the user directive of 2026-10-05 (training throughput > exhaustive
# eval; see logs/hsi_4b_math_code/scripts/pause_math_then_code_v2.sh). Choke
# point: whatever re-launches this grid -- the keeper's repair path, the old
# eval->code chain, a manual re-run -- must not take the 8 NPUs back from the
# Code arm. The marker is written BEFORE the grid is killed, so this can never
# race a grid that is already starting.
if [ -e "/workspace/logs/hsi_4b_math_code/MATH_EVAL_PAUSED" ]; then
  echo "[hsi4b-k12] MATH_EVAL_PAUSED present -- grid paused, exiting (no cell started)" >> "$LOG"; exit 0
fi

echo "[hsi4b-k12] full-grid start $(date '+%F %H:%M:%S')" >> "$LOG"
[ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ] || { echo "[hsi4b-k12] FATAL: no checkpoints" >> "$LOG"; exit 1; }

# Concurrency: per-dataset historical values (aime24 2, aime25/hmmt25 3) unless
# HSI_K12_CONC overrides ALL of them -- that is how the 503/throughput probe runs.
# HSI_K12_CELLS=probe runs ONLY cp25/aime24 (the probe cell); default runs the grid.
CONC="${HSI_K12_CONC:-}"
probe_conc() { [ -n "$CONC" ] && printf '%s' "$CONC" || printf '%s' "$1"; }

# AIME24 first for all 4 cps (the headline comparable column), then the other two.
if [ "${HSI_K12_CELLS:-grid}" = "probe" ]; then
  echo "[hsi4b-k12] PROBE mode: cp25/aime24 only at conc=$(probe_conc 2)" >> "$LOG"
  run_eval_cell cp25 aime24 eval_aime24_dev.json 12 "${HSI4B_PREFIX:-hsi4b}" "$(probe_conc 2)"
else
  for cp in 25 50 75 100; do run_eval_cell "cp${cp}" aime24 eval_aime24_dev.json 12 "${HSI4B_PREFIX:-hsi4b}" "$(probe_conc 8)"; done
  for cp in 25 50 75 100; do run_eval_cell "cp${cp}" aime25 eval_aime25_dev.json 12 "${HSI4B_PREFIX:-hsi4b}" "$(probe_conc 8)"; done
  for cp in 25 50 75 100; do run_eval_cell "cp${cp}" hmmt25 eval_hmmt25_dev.json 12 "${HSI4B_PREFIX:-hsi4b}" "$(probe_conc 8)"; done
fi

if [ "${HSI_K12_CELLS:-grid}" = "probe" ]; then
  echo "[hsi4b-k12] PROBE DONE $(date '+%F %H:%M:%S')" >> "$LOG"
  touch "$P/logs/hsi_4b_eval.K12_PROBE_DONE"
else
  echo "[hsi4b-k12] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
  touch "$DONE"
fi
