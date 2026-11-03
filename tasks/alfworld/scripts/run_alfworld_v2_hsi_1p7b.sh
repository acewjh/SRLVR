#!/bin/bash
# ALFWorld HSI on Qwen3-1.7B -- Node0 of the 1.7B matrix (campaign_1p7b).
#
# ADDITIVE file (created 2026-10-05). It does NOT source a backbone file: the
# defaults inside _common_v2.sh are already the validated Qwen3-1.7B geometry
# (MODEL=checkpoints/Qwen3-1.7B, 28 layers / 2048 hidden / 6144 ffn / 16 heads /
# 8 kv groups) -- see _common_v2.sh:28-35. Sourcing _4b_backbone.sh here would
# silently run the 4B model.
#
# The METHOD is untouched: it selects one branch of _common_v2.sh, which is the
# only place the estimator, sampler and postprocess plug points are allowed to
# live, so Node0 and Node1 cannot drift apart. G=1 and the short-batch top-up
# both come from that branch (ALFWORLD_V2_HSI=1 ->
# tasks/alfworld/rollout.py::_hsi_mode).
#
# ARM=hsi_v3 is our method: the exact-normalized-JSD estimator
# (OPSD_HSI_ESTIMATOR=jsd, postprocess tasks.alfworld.hsi2_jsd). The registry
# aliases `hsi_v3` to the canonical `hsi`, so both spellings select it. The
# launcher's historic bare `hsi` used to mean the retired top-128 candidate arm;
# that arm is gone (2026-10-15) and `hsi` now resolves to the same canonical
# objective. `hsi_v3` is kept as the default so the frozen command strings of any
# in-flight lane supervisor keep working.
#
# Task protocol fixed by BG_TASK_PROMPT_V4.md / CAMPAIGN_QUEUE_1p7b.md:
#   MAX_TURNS=50  HISTORY_LENGTH=0  MAX_RESPONSE=128  SEQ_LENGTH=32768
# NUM_ROLLOUT=100 with save_interval=25 -> cp25/50/75/100.
#
# Usage: RUN_NAME=... bash run_alfworld_v2_hsi_1p7b.sh
export ARM="${ARM:-hsi_v3}"
# Full-softmax sampler (T=1.0 / top_p=1.0 / top_k=-1). Without this the shared
# launcher falls through to the 2026-10-03 LEGACY sampler (T=0.6 / top_p 0.95 /
# top_k 20) -- see _common_v2.sh:362-390. The 4B arms inherit it from
# launch_c64.sh; this wrapper calls _common_v2.sh directly, so it must set it.
export ALFWORLD_V2_PROTOCOL=1
export RUN_NAME="${RUN_NAME:-alfworld_v2_hsi_1p7b}"
export MAX_TURNS="${MAX_TURNS:-50}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-0}"
export MAX_RESPONSE="${MAX_RESPONSE:-128}"
export SEQ_LENGTH="${SEQ_LENGTH:-32768}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"

# --- resume a partially-completed cell in place (additive, 2026-10-06) --------
# A mid-cell death used to cost the WHOLE cell: lane_supervisor.sh always starts a
# fresh RUN_NAME, so the re-armed supervisor began a new run from the base model.
# The 2026-10-05 alfworld cell died at rollout 55/100 (IndexError in
# train_data.py::trajectory_weighting -- fixed in the same change), which would
# have thrown away 2 checkpoints and ~9 h of the frozen task schedule.
# Default to CONTINUING that run in place instead:
#   --save stays on the existing dir, so hsi_cell_guard.sh's ">=4 ckpts" completion
#   rule and the lane's PENDING_EVAL entry keep pointing at the right directory,
#   and `hsi2_baseline.json` (which resolves from --save, see hsi2_jsd.py:496)
#   keeps the historical-baseline ledger instead of restarting it at 0;
#   --load reads latest_checkpointed_iteration.txt plus the dataset cursor
#   (rollout/global_dataset_state_dict_49.pt), so the task schedule resumes too.
# The default DISARMS ITSELF once the dir holds >=4 ckpts (the cell is done), so a
# later fresh cell is unaffected. Escape hatches: ALFWORLD_HSI_FRESH=1 forces a
# clean run; ALFWORLD_HSI_RESUME_DIR=<dir> resumes a different directory.
if [ "${ALFWORLD_HSI_FRESH:-0}" != 1 ]; then
  _hsi_resume_dir="${ALFWORLD_HSI_RESUME_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/runs/alfworld-qwen3-1p7b-hsi-s42-20261005_221026}"
  _hsi_resume_ck=$(ls -d "$_hsi_resume_dir"/iter_* 2>/dev/null | wc -l)
  if [ -s "$_hsi_resume_dir/latest_checkpointed_iteration.txt" ] && [ "${_hsi_resume_ck:-0}" -lt 4 ]; then
    export RESUME_FROM="${RESUME_FROM:-$_hsi_resume_dir}"
    export SAVE_DIR="${SAVE_DIR:-$_hsi_resume_dir}"
    export RUN_NAME="$(basename "$_hsi_resume_dir")"
  fi
fi
exec bash "$(dirname "$0")/_common_v2.sh"
