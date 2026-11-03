#!/bin/bash
# Formal-eval throughput sweep (runs ONLY when the box is free; see directive).
# Phase 1 (mem sweep): 1 engine TP1, conc 16, mem 0.72/0.76/0.80/0.82
# Phase 2 (load sweep): best mem, conc 4/8/16/32
# Each config: 8-min measurement window; metrics parsed from the engine log.
# All knobs: return_logprob=False (client), chunk 2048, ctx 40960, formal sampling.

set -uo pipefail
P=/workspace/cua/repos/opsd-slime
BASE=/workspace/cua
LOG=$P/logs/eval_sweep.log
SWEEP_DIR=$P/runs/eval_sweep
PORT=37995
WINDOW="${SWEEP_WINDOW:-480}"

mkdir -p "$SWEEP_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate cua-rl
source /usr/local/Ascend/cann-8.5.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES="${SWEEP_NPU:-0}"
export PYTHONPATH="$P/compat:$P/frameworks/sglang/python:$P"

launch_engine() { # <mem>
  ray stop --force > /dev/null 2>&1 || true
  pkill -9 -f "sglang" > /dev/null 2>&1 || true
  sleep 8
  python -m sglang.launch_server \
    --model-path "$BASE/checkpoints/Qwen3-1.7B" \
    --tp 1 --port $PORT --host 127.0.0.1 \
    --device npu --attention-backend ascend \
    --mem-fraction-static "$1" --context-length 40960 \
    --chunked-prefill-size 2048 --disable-radix-cache \
    > "$SWEEP_DIR/engine_mem$1.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 5
    c=$(curl -s --noproxy '*' --max-time 4 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$c" = "200" ] && { echo "[sweep] engine mem=$1 healthy"; return 0; }
  done
  echo "[sweep] engine mem=$1 FAILED to become healthy"; return 1
}

measure() { # <tag> <conc>
  local tag="$1" conc="$2"
  local t0=$(date +%s)
  python "$P/tests/sweep_client.py" --ports $PORT --concurrency "$conc" --window "$WINDOW" \
    > "$SWEEP_DIR/client_${tag}.log" 2>&1
  # metrics from the engine log within the window
  local line=$(grep -a "Decode batch" "$SWEEP_DIR/engine_mem"*.log 2>/dev/null | tail -n $((WINDOW / 2)) | tail -1)
  python - "$SWEEP_DIR" "$tag" "$WINDOW" <<'PYEOF'
import glob, json, re, sys, time
swdir, tag, window = sys.argv[1], sys.argv[2], int(sys.argv[3])
logs = glob.glob(f"{swdir}/engine_mem*.log")
logs.sort(key=lambda f: __import__("os").path.getmtime(f))
log = logs[-1]
rows = []
for line in open(log, errors="ignore"):
    if "Decode batch" not in line:
        continue
    m = re.search(r"\[(\d{4}-\d\d-\d\d [0-9:]+)\].*#running-req: (\d+), #token: (\d+)", line)
    if m:
        t = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        rows.append((t, int(m.group(2)), int(m.group(3))))
t_end = time.time()
rows = [r for r in rows if r[0] > t_end - window - 60]
agg = []
for i in range(1, len(rows)):
    dt = rows[i][0] - rows[i-1][0]
    dtok = rows[i][2] - rows[i-1][2]
    if dt > 0 and dtok > 0:
        agg.append((dtok / dt, rows[i][1]))
if agg:
    import statistics
    toks = [a for a, _ in agg]
    b = [b for _, b in agg]
    res = {
        "tag": tag,
        "agg_tok_s_mean": round(statistics.mean(toks), 1),
        "agg_tok_s_max": round(max(toks), 1),
        "running_batch_mean": round(statistics.mean(b), 1),
        "running_batch_max": max(b),
        "n_intervals": len(agg),
    }
    print("[sweep-metrics] " + json.dumps(res), flush=True)
else:
    print("[sweep-metrics] " + json.dumps({"tag": tag, "error": "no decode rows in window"}), flush=True)
PYEOF
  local hbm=$(npu-smi info 2>/dev/null | grep -E "^\| 0 " | grep -oE "[0-9]+ / 32768" | head -1)
  echo "[sweep] $tag done; NPU0 HBM now: $hbm" >> "$LOG"
}

echo "[sweep] START $(date)" >> "$LOG"
: > "$SWEEP_DIR/.phase"

echo "[sweep] === PHASE 1: mem sweep at conc 16 ===" >> "$LOG"
BEST_MEM=0.72
BEST_TOK=0
for mem in 0.72 0.76 0.80 0.82; do
  echo "[sweep] --- mem=$mem ---" >> "$LOG"
  if launch_engine "$mem"; then
    # 3-min warmup/request ramp, then measure
    python "$P/tests/sweep_client.py" --ports $PORT --concurrency 16 --window 180 \
      > "$SWEEP_DIR/warmup_mem$mem.log" 2>&1
    measure "mem$mem" 16
  else
    echo "[sweep] mem=$mem unstable -> STOP ascending" >> "$LOG"
    break
  fi
done

echo "[sweep] === PHASE 2: load sweep at best mem ===" >> "$LOG"
BEST_MEM=$(grep -aoE '"tag": "mem[0-9.]+", "agg_tok_s_mean": [0-9.]+' "$LOG" | awk -F'[:,]' '{print $2, $4}' | sort -k2 -rn | head -1 | awk '{print $1}' | tr -d 'mem')
[ -z "$BEST_MEM" ] && BEST_MEM=0.72
echo "[sweep] best mem = $BEST_MEM" >> "$LOG"
if launch_engine "$BEST_MEM"; then
  for conc in 4 8 16 32; do
    echo "[sweep] --- conc=$conc ---" >> "$LOG"
    measure "conc$conc" "$conc"
  done
fi

echo "[sweep] DONE $(date)" >> "$LOG"
pkill -9 -f "sglang" 2>/dev/null || true
