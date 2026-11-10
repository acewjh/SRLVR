#!/bin/bash
# Start (idempotently) the OpenApps app servers and the browsergym gateway.
#
#   app servers: FastAPI/uvicorn serving the OpenApps apps  (default 5058..)
#   gateway     : scripts/openapps_gateway.py, speaks slime's /v1 env protocol
#                 on top of browsergym                          (default 5111)
#
# Episodes mutate the app databases, so a single server shared by concurrent
# rollouts leaks state between them. Start OPENAPPS_POOL_SIZE servers (default
# 8 = one per concurrent rollout sample), each seeded from its own logs_dir,
# and the gateway resets a server before every episode it hosts.
#
# All live on 127.0.0.1 of whichever pod runs the training (the rollout calls
# them from Ray actors on the same node). CPU-only; no NPU is touched.
set -uo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PORT="${OPENAPPS_APP_PORT:-5058}"
GW_PORT="${OPENAPPS_GW_PORT:-5111}"
POOL_SIZE="${OPENAPPS_POOL_SIZE:-8}"
# Chromium refuses some ports outright (net::ERR_UNSAFE_PORT); 5060/5061 (SIP)
# sit right after the legacy single-server port and killed every rollout that
# landed on them. Start the pool at 5070 and skip the restricted ones.
POOL_PORT_BASE="${OPENAPPS_POOL_PORT_BASE:-5070}"
is_unsafe_port() {
  case "$1" in
    2049|3659|4045|5060|5061|6000|6566|6665|6666|6667|6668|6669|6697|10080) return 0 ;;
    *) return 1 ;;
  esac
}

source /opt/conda/etc/profile.d/conda.sh
conda activate "${OPSD_CONDA_ENV:-PyTorch-2.9.0}" || exit 1

_unset_proxies() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
  export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
}

up() {  # up <url>
  curl -s --noproxy "*" -o /dev/null --max-time 2 "$1" 2>/dev/null
}

_unset_proxies
mkdir -p "$PROJ/logs"

POOL_URLS=""
POOL_PORTS=()
for ((i = 0; i < POOL_SIZE; i++)); do
  port=$((POOL_PORT_BASE + i))
  is_unsafe_port "$port" && port=$((port + 100))
  POOL_PORTS+=("$port")
  url="http://127.0.0.1:${port}"
  POOL_URLS="${POOL_URLS:+${POOL_URLS},}${url}"
done

# Launch every missing pool server at once; each seeds its own databases dir.
miss=0
for ((i = 0; i < POOL_SIZE; i++)); do
  port="${POOL_PORTS[$i]}"
  if ! up "http://127.0.0.1:${port}/"; then
    [ "$miss" -eq 0 ] && echo "[services] starting OpenApps app servers on ${POOL_PORTS[*]}" >&2
    miss=$((miss + 1))
    srv_logs="$PROJ/logs/openapps_pool/srv${i}"
    mkdir -p "$srv_logs"
    (
      cd "$PROJ/openapps" || exit 1
      OPENAPPS_WEB_HOST=127.0.0.1 \
      OPENAPPS_RESET_SKIP="${OPENAPPS_RESET_SKIP:-onlineshop}" \
      nohup setsid python launch.py \
        use_wandb=False headless=True ++web_app_port="${port}" \
        logs_dir="$srv_logs" \
        > "$srv_logs/server.log" 2>&1 < /dev/null &
    )
  fi
done

for ((i = 0; i < POOL_SIZE; i++)); do
  port="${POOL_PORTS[$i]}"
  ok=0
  for _ in $(seq 1 150); do
    if up "http://127.0.0.1:${port}/"; then ok=1; break; fi
    sleep 2
  done
  [ "$ok" -eq 1 ] || { echo "[services] FATAL: app server ${port} did not come up (log: $PROJ/logs/openapps_pool/srv${i}/server.log)" >&2; exit 1; }
done

if ! up "http://127.0.0.1:${GW_PORT}/health"; then
  echo "[services] starting OpenApps gateway on ${GW_PORT}" >&2
  OPENAPPS_BASE_URL="${POOL_URLS%%,*}" \
  OPENAPPS_POOL_BASE_URLS="${POOL_URLS}" nohup setsid \
    python "$PROJ/scripts/openapps_gateway.py" --port "${GW_PORT}" \
    > "$PROJ/logs/openapps_gateway.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    sleep 1
    up "http://127.0.0.1:${GW_PORT}/health" && break
  done
fi
up "http://127.0.0.1:${GW_PORT}/health" || { echo "[services] FATAL: gateway did not come up" >&2; exit 1; }

echo "[services] apps=${POOL_URLS} gateway=${GW_PORT} OK" >&2
curl -s --noproxy "*" "http://127.0.0.1:${GW_PORT}/health"; echo
