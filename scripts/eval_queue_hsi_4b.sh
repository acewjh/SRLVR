#!/bin/bash
# Eval for the unified-HSI Math Qwen3-4B arm (campaign math-qwen3-4b-hsi-s42).
#
# Same protocol as eval_queue_grpo.sh so the numbers slot straight into the
# existing 4B table (Base / GRPO / Ours-vNext are already on it at K=12):
#   * K=4 selector on AIME24 for cp25/50/75/100  -> top-2 by average@k
#   * formal K=12 on AIME24 + AIME25 + HMMT25 for the top-2
#
# The 4 official cps (iter_24/49/74/99) are hardlink-guarded in
# logs/hsi_4b_math_code/ckpt_guard/ so a shared-FS reaper cannot take them while
# this runs. make_eval_staging.sh only symlinks; run dirs are never modified.
#
# Usage: nohup bash scripts/eval_queue_hsi_4b.sh > /dev/null 2>&1 &
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_gpu_defaults.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_common.sh"
EVAL_CELL_LOG_TAG=hsi4b-eval

P=/workspace/cua/repos/opsd-slime
RUN_DIR=$P/runs/math-qwen3-4b-hsi-s42-20261004_040413
LOG=$P/logs/eval_queue_hsi_4b.log
LOCK=/tmp/opsd_eval_hsi4b.lock
SEL=$P/reports/hsi_4b_selector.json
DONE=$P/logs/hsi_4b_eval.EVAL_DONE

source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl 2>/dev/null || conda activate slime-npu
# MUST come before run_dev_eval.sh: MODEL selects configs/models/<slug>.sh and its
# default is qwen3-1p7b. Without this the queue silently evaluates the 4B checkpoint
# against the 1.7B card (caught 2026-10-04 by reading the `[dev-eval] model=` line).
export MODEL="${MODEL:-qwen3-4b}"
export EVAL_LORA=1                        # the arm is LoRA r64
export EVAL_MEM_FRACTION="${EVAL_MEM_FRACTION:-0.82}"
export EVAL_CHUNKED_PREFILL="${EVAL_CHUNKED_PREFILL:-2048}"

exec 9>"$LOCK"
if ! flock -n 9; then echo "[hsi4b-eval] another queue holds $LOCK -- exiting" >> "$LOG"; exit 1; fi

echo "[hsi4b-eval] start $(date '+%F %H:%M:%S')" >> "$LOG"
[ -s "$RUN_DIR/latest_checkpointed_iteration.txt" ] || { echo "[hsi4b-eval] FATAL: no checkpoints" >> "$LOG"; exit 1; }

for cp in 25 50 75 100; do run_eval_cell "cp${cp}" aime24 eval_aime24_dev_k4.json 4 hsi4bsel 4; done

python - "$SEL" <<'PY' >> "$LOG" 2>&1
import glob, json, os, subprocess, sys
P = "/workspace/cua/repos/opsd-slime"
scores = {}
for cp in (25, 50, 75, 100):
    cands = sorted(glob.glob(f"{P}/runs/dev_eval/hsi4bsel*cp{cp}_aime24_eval_0.pt"))
    if not cands: print(f"[hsi4b-eval] no dump for cp{cp}"); continue
    dump = cands[0]; tag = os.path.basename(dump).replace("_eval_0.pt", "")
    r = subprocess.run(["python", f"{P}/math_opsd/aggregate_dev_eval.py", "--dump", dump,
                        "--tag", tag, "--out", f"{P}/reports/hsi_4b_eval_table.md"],
                       capture_output=True, text=True)
    try:
        obj, _ = json.JSONDecoder().raw_decode(r.stdout.lstrip()); scores[cp] = obj["average@k"]
    except Exception as e: print(f"[hsi4b-eval] parse cp{cp}: {e}")
top = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
json.dump({"selector_k4_aime24": scores, "top2": [c for c, _ in top]}, open(sys.argv[1], "w"), indent=2)
print(f"[hsi4b-eval] selector {scores} -> top2 {[c for c, _ in top]}")
PY
TOP2=$(python -c "import json;print(' '.join(str(c) for c in json.load(open('$SEL'))['top2']))" 2>/dev/null)
[ -z "$TOP2" ] && { echo "[hsi4b-eval] FATAL: no top-2" >> "$LOG"; exit 1; }
echo "[hsi4b-eval] formal on cp: $TOP2" >> "$LOG"
for cp in $TOP2; do
  run_eval_cell "cp${cp}" aime24 eval_aime24_dev.json 12 hsi4b 2
  run_eval_cell "cp${cp}" aime25 eval_aime25_dev.json 12 hsi4b 3
  run_eval_cell "cp${cp}" hmmt25 eval_hmmt25_dev.json 12 hsi4b 3
done
echo "[hsi4b-eval] DONE $(date '+%F %H:%M:%S')" >> "$LOG"
touch "$DONE"
