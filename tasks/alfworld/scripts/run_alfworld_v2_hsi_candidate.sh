#!/bin/bash
# Candidate-Centered HSI on Qwen3-1.7B -- the 2026-10-03 `hsi_candidate_center_1p7b_s42`
# run, meant to be compared against a matched GRPO-1.7B on the other 32G node.
#
# This file is the 1.7B twin of run_alfworld_v2_hsi_candidate_4b.sh and differs from
# it in exactly one thing: it does NOT source a backbone override, because the
# defaults in _common_v2.sh ARE the validated Qwen3-1.7B geometry (model path,
# 28 layers, 2048 hidden, 16 heads / 8 query groups). Sourcing nothing is what makes
# the 1.7B arms byte-for-byte the validated configuration rather than a re-listing
# of it.
#
# One rollout + one future-conditioned frozen-teacher forward + one policy train
# step. No matched/control teacher, no old-actor replay, no full-vocab credit; see
# tasks/alfworld/candidate_center.py for the estimator and _common_v2.sh's
# `hsi_candidate` arm for the engine/sampler conventions (top_logprobs_num per
# request, SGLANG_RETURN_ORIGINAL_LOGPROB for the PPO old-policy convention).
#
# Usage: RUN_NAME=... NUM_ROLLOUT=... bash run_alfworld_v2_hsi_candidate.sh
set -uo pipefail
PROJ=/workspace/cua/repos/opsd-slime

# The task book's fixed protocol for this arm. Exported here rather than left to
# _common_v2.sh's defaults because several of them differ from the previous
# campaign's (clip-grad 0.1 -> 1.0, cap 512 -> 4096).
export CLIP_GRAD="${CLIP_GRAD:-1.0}"
export LR="${LR:-5e-6}"
export MAX_TURNS="${MAX_TURNS:-50}"
export MAX_RESPONSE="${MAX_RESPONSE:-4096}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-8}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export SEED="${SEED:-42}"
export OPSD_LORA_R="${OPSD_LORA_R:-64}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"

# ---- source provenance -------------------------------------------------------
# This repository is on a shared NFS volume that other nodes also write to, so the
# sources can move under a launch. Snapshot exactly what this run will import and
# record any drift from the preflight fingerprint -- the running process is immune
# once Python has imported, but the record has to exist either way.
SNAP_DIR="$PROJ/runs/${RUN_NAME:-unknown}/SOURCE_SNAPSHOT"
FILES=(candidate_center.py rollout.py episode.py train_data.py config.py loss.py privilege.py)
mkdir -p "$SNAP_DIR"
for f in "${FILES[@]}"; do
  cp -f "$PROJ/tasks/alfworld/$f" "$SNAP_DIR/$f" 2>/dev/null || true
done
cp -f "$PROJ/tasks/alfworld/scripts/_common_v2.sh" "$SNAP_DIR/_common_v2.sh" 2>/dev/null || true
cp -f "$PROJ/tasks/alfworld/scripts/run_alfworld_v2_hsi_candidate.sh" "$SNAP_DIR/run_alfworld_v2_hsi_candidate.sh" 2>/dev/null || true
if [ -s /tmp/hsi_cc_fingerprint.txt ]; then
  cp -f /tmp/hsi_cc_fingerprint.txt "$SNAP_DIR/CODE_FILES.sha256"
  # The manifest holds repo-relative paths, so verify from the repo root.
  if ! ( cd "$PROJ" && sha256sum -c --quiet "$SNAP_DIR/CODE_FILES.sha256" ) >/dev/null 2>&1; then
    {
      echo "SOURCE DRIFT vs the preflight fingerprint, recorded $(date '+%F %T')"
      ( cd "$PROJ" && sha256sum -c "$SNAP_DIR/CODE_FILES.sha256" 2>&1 | grep -v ': OK$' ) || true
    } >> "$PROJ/runs/${RUN_NAME:-unknown}/SOURCE_DRIFT.txt"
    echo "[hsi-candidate] WARNING: sources differ from the preflight fingerprint (see SOURCE_DRIFT.txt)" >&2
  fi
fi

export ARM=hsi_candidate RUN_NAME="${RUN_NAME:-alfworld_v2_hsi_candidate_1p7b}"
exec bash "$(dirname "$0")/_common_v2.sh"
