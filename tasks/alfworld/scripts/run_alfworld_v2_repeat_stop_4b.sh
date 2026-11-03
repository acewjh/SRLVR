#!/bin/bash
# ============================================================
# Qwen3-4B / 8x32G -- HSI + repeat-stop, 100 update cycles, seed 42.
#
# This arm is the COMMON HSI arm (ARM=hpi_future => ALFWORLD_V2_HPI=1, G=1,
# ALFWORLD_V2_HPI_SPAN=full, ALFWORLD_V2_HSI_GRANULARITY=token, trajectory-
# balanced reducer) with exactly ONE method addition:
#
#   ALFWORLD_V2_REPEAT_STOP=1
#
# which replaces the PPO surrogate at gated positions with the conditional stop
# loss (tasks/alfworld/repeat_stop.py). Nothing else about the common version is
# touched -- with the flag unset the loss is the shipped baseline by identity.
#
# The backbone geometry comes from _4b_backbone.sh, the same file the other 4B
# arms source, so "matched by construction" keeps holding.
#
# Usage: bash tasks/alfworld/scripts/run_alfworld_v2_repeat_stop_4b.sh
# ============================================================
set -uo pipefail

# The orchestrator holds its flock on fd 9. Children inherit that fd, and an
# orphaned child then keeps holding the lock after the orchestrator dies (the
# repo has been bitten by exactly this). Close it at the top of the long-running
# branch so this process tree never owns the lock.
exec 9>&-

. "$(dirname "$0")/_4b_backbone.sh"

# ---- frozen experiment configuration (launch spec sections 1 and 8) ----------
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"

export G="${G:-1}"                   # HSI is single-rollout; the arm switch pins it anyway
export ROLLOUT_BS="${ROLLOUT_BS:-4}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export SEED="${SEED:-42}"
export LR="${LR:-5e-6}"
export TP="${TP:-1}"
# This frozen arm carries a teacher and per-position chunking; its measured
# layout is 2 actor + 6 rollout, non-colocated. Keep it explicit.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"

# The trajectory-balanced reducer (already the harness default; spelled out so a
# launch log can prove it ran).
export OPSD_V2_LOSS_NORM="${OPSD_V2_LOSS_NORM:-traj}"

# ---- execution chunks (never the algorithm) ----------------------------------
# Teacher forced-scoring positions per forward. The default 0 scores the whole
# target span at once, materialising a [tlen, vocab] fp32 tensor (~2.5 GiB at
# tlen=4096, doubled by log_softmax) -- measured OOM on this 32G card once the
# student, the frozen teacher and the batch are all resident. log_softmax is
# per-position, so chunking is exact: identical log-probs, bounded peak.
export OPSD_HPI_POS_CHUNK="${OPSD_HPI_POS_CHUNK:-256}"
# Views per teacher forward (upstream knob).
export OPSD_HPI_CHUNK="${OPSD_HPI_CHUNK:-4}"

# ---- the one method addition -------------------------------------------------
export ALFWORLD_V2_REPEAT_STOP="${ALFWORLD_V2_REPEAT_STOP:-1}"
# Sparse-CE row-gather chunk. 256 by default; the supervisor steps it down the
# fixed 256 -> 128 -> 64 ladder on OOM without touching the algorithm, the
# samples, or the effective batch.
export ALFWORLD_V2_SCORE_CHUNK="${ALFWORLD_V2_SCORE_CHUNK:-256}"

# ---- run identity ------------------------------------------------------------
# Repo naming rule: <task>-<model>-<method>[-<variant>]-<stamp>. The variant
# records the deviation from the model card (the repeat-stop surrogate).
export RUN_NAME="${RUN_NAME:-alfworld-qwen3-4b-hsi-repeat-stop-s42}"

echo "[repeat-stop] cap=$MAX_RESPONSE turns=$MAX_TURNS hist=$HISTORY_LENGTH seq=$SEQ_LENGTH \
G=$G bs=$ROLLOUT_BS steps=$NUM_ROLLOUT save=$SAVE_INTERVAL seed=$SEED \
pos_chunk=$OPSD_HPI_POS_CHUNK view_chunk=$OPSD_HPI_CHUNK score_chunk=$ALFWORLD_V2_SCORE_CHUNK"

export ARM=hpi_future
exec bash "$(dirname "$0")/_common_v2.sh"
