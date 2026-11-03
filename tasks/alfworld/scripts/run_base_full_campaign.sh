#!/usr/bin/env bash
# Standalone base inference, using the same task modules and strict sampler as training.
# Does not start/stop Ray or instantiate a trainer.
set -eo pipefail
PROJ=$(cd "$(dirname "$0")/../../.." && pwd)
REPORT="${1:?usage: run_base_full_campaign.sh <new-report-directory> <codex-thread-id>}"
THREAD_ID="${2:?completion notification needs the owning thread ID}"
mkdir -p "$REPORT"
source "$PROJ/train_scripts/lib/npu_env.sh"
opsd_enter_env
opsd_export_npu_env "$PROJ" 'expandable_segments:True'
export PYTHONPATH="$PROJ/compat/sgl_kernel_npu_shim:$PYTHONPATH"
export ALFWORLD_OVERLAY_SITE="$PROJ/../../envs/alfworld_overlay/lib/python3.10/site-packages"
export ALFWORLD_V2_MAX_TURNS=50 ALFWORLD_V2_MAX_RESPONSE=4096
export ALFWORLD_ACTION_HARNESS=command-normalization-v2-form
export ALFWORLD_SHOW_ADMISSIBLE=1 ALFWORLD_THINKING_STYLE=brief
unset ALFWORLD_THINKING_TOKENS SGLANG_RETURN_ORIGINAL_LOGPROB
export ALFWORLD_ENV_ACTORS=1 ALFWORLD_ENV_WORKERS=8
# In inference this selects the strict CDF sampler and actual-token logprobs only;
# base_full_eval directly calls run_episode, so no SRLVR loss or teacher is involved.
export OPSD_SRLVR=1 SRLVR_VARIANT=v2 SRLVR_SAMPLING=cdf
export ALFWORLD_V2_HPI=0 ALFWORLD_V2_HSI_CANDIDATE_K=0
export OPSD_DASHBOARD=0
export ALFWORLD_BASE_SOURCE_SNAPSHOT="$REPORT/source_snapshot"
export PYTHONPATH="$ALFWORLD_BASE_SOURCE_SNAPSHOT:$PYTHONPATH"
cd "$PROJ"
exec python "$PROJ/tasks/alfworld/scripts/base_full_campaign.py" \
  --project "$PROJ" --report "$REPORT" --thread "$THREAD_ID"
