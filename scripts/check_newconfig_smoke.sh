#!/bin/bash
# Sanity checker for the new formal-eval default configuration smoke
# (run after ~12 min of the smoke_newconfig eval). Checks:
#   1. dispatch: all 4 engines received requests (per-pid Decode lines, running>0)
#   2. sampling_backend=ascend in the ServerArgs dump (no silent fallback)
#   3. eval payload return_logprob=False (log line) + no [chunk,V] spike
#   4. mem_fraction_static=0.82 / chunked_prefill_size=2048 in effect
#   5. no OOM / no RayTaskError
P=/workspace/cua/repos/opsd-slime
LOG=$P/logs/dev_eval_smoke_newconfig.log

echo "=== 1. Dispatch (per-engine decode activity) ==="
for pid in $(grep -a "Decode batch" "$LOG" | grep -aoE "SGLangEngine pid=[0-9]+" | sort -u | grep -oE "[0-9]+"); do
  echo -n "  engine pid $pid: max running="
  grep -a "SGLangEngine pid=$pid" "$LOG" | grep -aoE "#running-req: [0-9]+" | awk -F': ' '{if($2>m)m=$2} END {print m+0}'
done
N_ACTIVE=$(for pid in $(grep -a "Decode batch" "$LOG" | grep -aoE "SGLangEngine pid=[0-9]+" | sort -u); do
  grep -a "SGLangEngine pid=$pid" "$LOG" | grep -aoE "#running-req: [0-9]+" | awk -F': ' '{if($2>m)m=$2}' | grep -vq "^0$" && echo active
done | wc -l)
echo "  engines actively decoding: $N_ACTIVE / 4"

echo "=== 2. sampling_backend ==="
grep -a -m1 -oE "sampling_backend[ =:']+[a-z]+" "$LOG" | head -1

echo "=== 3. return_logprob ==="
grep -a -m1 -oE "FORMAL_EVAL_EFFECTIVE_CONFIG: [^ ]*return_logprob=[A-Za-z]+" "$LOG" | head -1

echo "=== 4. memory ==="
grep -a -m1 -oE "mem_fraction_static[ =:]+[0-9.]+" "$LOG" | head -1
grep -a -m1 -oE "chunked_prefill_size[ =:]+[0-9]+" "$LOG" | head -1

echo "=== 5. errors ==="
echo -n "  OOM count: "; grep -a -cE "out of memory" "$LOG"
echo -n "  RayTaskError count: "; grep -a -cE "RayTaskError" "$LOG"
