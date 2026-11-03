#!/bin/bash
# Phase 3: standalone SGLang rollout smoke for the math OPSD harness.
# 1 TP1 text-only engine on one NPU (default 7), Qwen3-1.7B, 8 math prompts,
# official rollout sampling params (T=1.1, top_p=0.95, top_k=20, max_tokens=128).
# Writes reports/sglang_smoke.md and logs/sglang_smoke_server.log.

set -euo pipefail

BASE=/workspace/cua
PROJ=$BASE/repos/opsd-slime
MODEL=$BASE/checkpoints/Qwen3-1.7B
NPU_ID="${NPU_ID:-7}"
PORT="${PORT:-37881}"
MAX_TOKENS="${MAX_TOKENS:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

source /opt/conda/etc/profile.d/conda.sh
# cua-rl = this machine's slime env since the 09-14 container rebuild;
# slime-npu = pre-rebuild naming kept as fallback
conda activate cua-rl 2>/dev/null || conda activate slime-npu
source /usr/local/Ascend/cann-8.5.1/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH="$PROJ/compat:$PROJ/frameworks/sglang/python:$PROJ:$PYTHONPATH"

mkdir -p "$PROJ/logs" "$PROJ/reports"
LOG=$PROJ/logs/sglang_smoke_server.log
REPORT=$PROJ/reports/sglang_smoke.md

cleanup() {
  pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
}
trap cleanup EXIT
cleanup || true
sleep 2

echo "[smoke] launching TP1 engine on NPU $NPU_ID, port $PORT"
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp 1 --port "$PORT" --host 127.0.0.1 \
  --device npu --attention-backend ascend \
  --mem-fraction-static "$MEM_FRACTION" --context-length 32768 \
  --chunked-prefill-size 8192 \
  --disable-radix-cache \
  > "$LOG" 2>&1 &
ENGINE_PID=$!

healthy=0
for i in $(seq 1 90); do
  sleep 5
  code=$(curl -s --noproxy '*' --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" || echo 000)
  if [ "$code" = "200" ]; then echo "[smoke] engine healthy after $((i*5))s"; healthy=1; break; fi
  kill -0 "$ENGINE_PID" 2>/dev/null || { echo "[smoke] FATAL: engine died"; tail -40 "$LOG"; exit 1; }
done
[ "$healthy" = "1" ] || { echo "[smoke] FATAL: engine never healthy"; tail -40 "$LOG"; exit 1; }

echo "[smoke] running generation smoke"
MAX_TOKENS="$MAX_TOKENS" NUM_PROMPTS="$NUM_PROMPTS" PORT="$PORT" \
python "$PROJ/scripts/_sglang_smoke_client.py" 2>&1 | tee "$PROJ/logs/sglang_smoke_client.log"
CLIENT_RC=${PIPESTATUS[0]}

npu-smi info | grep -A2 "NPU $NPU_ID " | tee -a "$PROJ/logs/sglang_smoke_client.log" || true
exit "$CLIENT_RC"
