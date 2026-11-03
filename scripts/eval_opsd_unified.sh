#!/bin/bash
# A3: put GT-OPSD cp100 and Base on the SAME evaluator as every other arm.
#
# The existing GT-OPSD numbers (cua/runs/opsd_official_qwen3_1p7b_seed42/eval/
# cp{25,50,75_cheap,100}/) and the Base numbers (cua/runs/eval_base_qwen3_1p7b/)
# were produced by the baseline node with vllm-ascend + the official
# evaluate_math.py. Every other arm on this server is scored by the SGLang
# harness with the repaired official grader. Absolute values are NOT comparable
# across those two backends, so this re-runs cp100 and Base here to get one
# common yardstick. Only deltas may be read off the combined table.
#
# Base is the untouched Qwen3-1.7B; GT-OPSD cp100 is a LoRA adapter dir that is
# merged first, so both run through the identical EVAL_LORA=0 HF path.
#
# Usage: nohup bash scripts/eval_opsd_unified.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"

P=/workspace/cua/repos/opsd-slime
ROOT=/workspace/cua
BASE_HF=$ROOT/checkpoints/Qwen3-1.7B
ADAPTER=$ROOT/runs/opsd_official_qwen3_1p7b_seed42/opsd_official_qwen3_1p7b_seed42_seed42/checkpoint-100
MERGED=$ROOT/runs/gt_opsd_cp100_merged
MERGER=$ROOT/repos/opsd-baseline-node/scripts/merge_adapter.py
LOG=$P/logs/eval_opsd_unified.log
LOCK=/tmp/opsd_eval_unified.lock

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu

export EVAL_LORA=0                       # both sources are merged/plain HF dirs
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.82}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-2048}"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[unified] another queue holds $LOCK -- exiting" >> "$LOG"
  exit 1
fi

# How many samples in this dump actually received a nonzero reward. Added
# 2026-09-25 after `opsd100_aime24` produced 360 samples of token salad scoring
# 0/360 and still passed the sample-count check.
nonzero_rewards() {
  python - "$1" <<'PY'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
s = d["samples"] if isinstance(d, dict) else d
print(sum(1 for x in s if float(x.get("reward", 0.0)) > 0.0))
PY
}

run_cell() { # <hf_dir> <tag> <dataset> <conc>
  local hf=$1 tag=$2 ds=$3 conc=$4
  local dump=$P/runs/dev_eval/${tag}_eval_0.pt
  if [ -s "$dump" ]; then
    echo "[unified] $tag already has a dump -- skipping" >> "$LOG"; return 0
  fi
  echo "[unified] === $tag K=12 conc=$conc ($(date '+%F %H:%M:%S')) ===" >> "$LOG"
  EVAL_K=12 EVAL_CONCURRENCY="$conc" EVAL_CONFIG="$P/configs/eval_${ds}_dev.json" \
    bash $P/train_scripts/run_dev_eval.sh "$hf" "$tag" >> "$LOG" 2>&1
  local rc=$?
  echo "[unified] rc=$rc for $tag" >> "$LOG"
  if [ "$rc" = "0" ] && [ -s "$dump" ]; then
    local n; n=$(count_samples "$dump")
    if [ "$n" != "360" ]; then
      echo "[unified] FATAL: $tag produced $n samples, expected 360 (30 x K=12)" >> "$LOG"
      cleanup_orphans; exit 1
    fi
    # CONTENT guard, added 2026-09-25 after `opsd100_aime24` produced 360
    # samples of token salad with 0/360 nonzero reward and still sailed through
    # the count check above. Sample count proves the CELL ran; it says nothing
    # about the output being real. A cell that scores nothing at all is
    # overwhelmingly more likely to be a broken model/prompt than a genuine
    # 0% -- treat it as fatal so it cannot be reported as a result.
    local nz; nz=$(nonzero_rewards "$dump")
    if [ "$nz" = "0" ]; then
      echo "[unified] FATAL: $tag has 0/$n nonzero rewards -- output is not being graded (broken model/prompt?), refusing to accept this cell" >> "$LOG"
      # the (garbage) dump is deliberately KEPT as evidence -- never deleted
      cleanup_orphans; exit 1
    fi
    echo "[unified] $tag verified: $n samples == 30 x K=12, $nz/$n nonzero rewards" >> "$LOG"
  else
    echo "[unified] WARNING: $tag produced no usable dump (rc=$rc)" >> "$LOG"
  fi
  cleanup_orphans
}

echo "[unified] start $(date '+%F %H:%M:%S')" >> "$LOG"

# --- merge GT-OPSD cp100 once (idempotent) ---------------------------------
if [ ! -f "$MERGED/config.json" ]; then
  if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    echo "[unified] FATAL: no adapter at $ADAPTER" >> "$LOG"; exit 1
  fi
  echo "[unified] merging cp100 adapter -> $MERGED ($(date '+%F %H:%M:%S'))" >> "$LOG"
  python "$MERGER" "$BASE_HF" "$ADAPTER" "$MERGED" >> "$LOG" 2>&1
  rc=$?
  [ "$rc" = "0" ] && [ -f "$MERGED/config.json" ] || {
    echo "[unified] FATAL: merge failed rc=$rc" >> "$LOG"; exit 1; }
  echo "[unified] merge OK" >> "$LOG"
fi

[ -f "$BASE_HF/config.json" ] || { echo "[unified] FATAL: no base HF at $BASE_HF" >> "$LOG"; exit 1; }

# --- Base and GT-OPSD cp100, same protocol as every other arm ---------------
for ds in aime24 aime25 hmmt25; do
  conc=3; [ "$ds" = "aime24" ] && conc=2
  run_cell "$BASE_HF" "baseu_${ds}" "$ds" "$conc"
  # opsd100_* SKIPPED 2026-09-25. `opsd100_aime24` produced token salad with
  # 0/360 nonzero reward (its dump is KEPT as evidence under runs/dev_eval/).
  # Root cause NOT established: the merged weights are finite and ~= base
  # (mean|d| 2e-6), the tokenizer/config/prompt are identical to base's, and the
  # cell log shows model_path=.../Qwen3-1.7B -- that is the one unexplained lead.
  # Until it is resolved the whole OPSD half is untrustworthy and the remaining
  # two cells would cost ~5 h for nothing. To re-enable, restore the
  # `run_cell "$MERGED" "opsd100_${ds}"` line.
  echo "[unified] opsd100_${ds} SKIPPED (broken merged model -- see comment)" >> "$LOG"
done

echo "[unified] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
