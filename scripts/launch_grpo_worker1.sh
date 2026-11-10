#!/bin/bash
# Launch the standard math GRPO run on worker-1 against a warmed-up Ray cluster.
#
# Why this wrapper exists (measured on 2026-11-06, worker-1):
#   `ray start` prelaunches a pool of python workers (one per CPU, 175 here).
#   Each of them imports the full stack (torch_npu + sglang + Megatron). If the
#   driver reaches `placement_group()` while that pool is still importing, the
#   driver's core-worker registration never completes and the run wedges right
#   after "Connected to Ray cluster" (reproduced at 03:02 and 03:58; the same
#   driver on a 10+ minute old cluster passes in seconds). So: start Ray, wait
#   for the registration-timeout storm in raylet.err to go quiet, then launch.
#
# The HIS gateway proxy is unset for the whole tree: the sglang router's
# health checks to http://<pod-ip>:1500x otherwise traverse the proxy and come
# back 429/504, forcing retry rounds (the engines still come up, but slowly).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
P=$(pwd)

# shellcheck source=../train_scripts/lib/npu_env.sh
source train_scripts/lib/npu_env.sh || exit 1
OPSD_CONDA_ENV="${OPSD_CONDA_ENV:-PyTorch-2.9.0}"
opsd_enter_env
opsd_export_npu_env "$P" "expandable_segments:True"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# The raylet is started HERE, so the actor processes inherit THIS environment,
# not the driver's. run_grpo.sh exports OPSD_LORA=1 only after `ray start` in its
# own flow; if the cluster is pre-started without it, the LoRA provider falls back
# to "full-model, no adapters" while the CLI still passes
# --only-train-params-name-list '.*\.lora_.*' -- an optimizer with zero matching
# params, i.e. grad_norm 0.0 and byte-identical checkpoints (measured 04:45 run).
export OPSD_LORA="${OPSD_LORA:-1}"
export OPSD_LORA_R="${OPSD_LORA_R:-64}"
export OPSD_LORA_ALPHA="${OPSD_LORA_ALPHA:-128}"
export OPSD_LORA_DROPOUT="${OPSD_LORA_DROPOUT:-0.0}"

start_ray() {
  ray stop --force >/dev/null 2>&1
  sleep 5
  ray start --head --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265 >/dev/null 2>&1 || true
  sleep 5
  if ! ray status >/dev/null 2>&1; then
    # Observed once: the raylet dies at startup on a metrics-agent port-file
    # timeout when the machine is still busy with a previous run's processes.
    echo "[launch] ray start crashed; cleaning and retrying" >&2
    ray stop --force >/dev/null 2>&1
    pkill -9 -f "/ray/core/src/ray" 2>/dev/null || true
    sleep 10
    ray start --head --disable-usage-stats --dashboard-host 0.0.0.0 --dashboard-port 8265 >/dev/null 2>&1 || true
    sleep 5
  fi
}

echo "[launch] restarting ray" >&2
start_ray

# Readiness = a remote task actually round-trips. Wall-clock alone is not a
# signal (90 s of quiet raylet.err still wedged the driver at 04:37), and the
# raylet does not answer a driver's registration until its worker pool has
# finished importing the stack. The probe worker pays the same import cost, so
# a successful round-trip means the driver will get through too.
probe() {
  timeout 90 python - <<'PY'
import ray

ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")


@ray.remote
def ping():
    return 1


assert ray.get(ping.remote(), timeout=60) == 1
print("RAY_READY")
PY
}

ready=0
for attempt in $(seq 1 30); do
  if probe 2>/dev/null | grep -q RAY_READY; then
    echo "[launch] ray responsive after ${attempt} probe(s)" >&2
    ready=1
    break
  fi
  echo "[launch] ray not responsive yet (probe ${attempt})" >&2
  if ! pgrep -f "raylet --raylet_socket_name" >/dev/null 2>&1; then
    echo "[launch] raylet is gone -- restarting ray" >&2
    start_ray
  fi
  sleep 20
done
if [ "$ready" != 1 ]; then
  echo "[launch] FATAL: ray never became responsive" >&2
  exit 1
fi

exec env OPSD_CONDA_ENV="$OPSD_CONDA_ENV" MODEL="${MODEL:-qwen3-1p7b}" \
  NUM_ROLLOUT="${NUM_ROLLOUT:-100}" SAVE_INTERVAL="${SAVE_INTERVAL:-25}" \
  bash scripts/run_grpo.sh
