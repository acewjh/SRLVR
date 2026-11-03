#!/bin/bash
# Eval for the standard-GRPO arm: K=4 selector on cp25/50/75/100, then formal K=12
# on the top-2 (AIME24 + AIME25 + HMMT25). Mirrors the HPI-v2 queue exactly so the
# two arms are measured identically.
#
# Usage: nohup bash scripts/eval_queue_grpo.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
EVAL_CELL_LOG_TAG=grpo-eval

P=/workspace/cua/repos/opsd-slime
RUN_DIR=$P/runs/math-qwen3-1p7b-grpo-20260921_224239
LOG=$P/logs/eval_queue_grpo.log
LOCK=/tmp/opsd_eval_grpo.lock

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu
export EVAL_LORA=1
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.82}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-2048}"

exec 9>"$LOCK"
if ! flock -n 9; then echo "[grpo-eval] another queue holds $LOCK -- exiting" >> "$LOG"; exit 1; fi

echo "[grpo-eval] start $(date '+%F %H:%M:%S')" >> "$LOG"
[ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ] || { echo "[grpo-eval] FATAL: no checkpoints" >> "$LOG"; exit 1; }

for cp in 25 50 75 100; do run_eval_cell "cp${cp}" aime24 eval_aime24_dev_k4.json 4 grposel 4; done

SEL=$P/reports/grpo_selector.json
python - "$SEL" <<'PY' >> "$LOG" 2>&1
import glob, json, os, subprocess, sys
P = "/workspace/cua/repos/opsd-slime"
scores = {}
for cp in (25, 50, 75, 100):
    cands = sorted(glob.glob(f"{P}/runs/dev_eval/grposel*cp{cp}_aime24_eval_0.pt"))
    if not cands: print(f"[grpo-eval] no dump for cp{cp}"); continue
    dump = cands[0]; tag = os.path.basename(dump).replace("_eval_0.pt", "")
    r = subprocess.run(["python", f"{P}/tasks/math/aggregate_dev_eval.py", "--dump", dump,
                        "--tag", tag, "--out", f"{P}/reports/grpo_eval_table.md"],
                       capture_output=True, text=True)
    try:
        obj, _ = json.JSONDecoder().raw_decode(r.stdout.lstrip()); scores[cp] = obj["average@k"]
    except Exception as e: print(f"[grpo-eval] parse cp{cp}: {e}")
top = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
json.dump({"selector_k4_aime24": scores, "top2": [c for c, _ in top]}, open(sys.argv[1], "w"), indent=2)
print(f"[grpo-eval] selector {scores} -> top2 {[c for c, _ in top]}")
PY
TOP2=$(python -c "import json;print(' '.join(str(c) for c in json.load(open('$SEL'))['top2']))" 2>/dev/null)
[ -z "$TOP2" ] && { echo "[grpo-eval] FATAL: no top-2" >> "$LOG"; exit 1; }
echo "[grpo-eval] formal on cp: $TOP2" >> "$LOG"
for cp in $TOP2; do
  run_eval_cell "cp${cp}" aime24 eval_aime24_dev.json 12 grpo 2
  run_eval_cell "cp${cp}" aime25 eval_aime25_dev.json 12 grpo 3
  run_eval_cell "cp${cp}" hmmt25 eval_hmmt25_dev.json 12 grpo 3
done
echo "[grpo-eval] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
