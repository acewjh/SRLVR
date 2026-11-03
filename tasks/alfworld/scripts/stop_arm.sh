#!/usr/bin/env bash
# Stop a running ALFWorld arm on THIS node, by process group.
#
# Used over ssh from the other node. Two deliberate choices:
#   * kill by PGID, never by `pkill -f <name>` -- a -f pattern matches the command
#     line of the killer itself (this repo has been bitten by that before), and an
#     inlined pattern also matches any monitoring shell that echoes it.
#   * the pattern below is written with a bracket class for the same reason.
set -uo pipefail

PATS=('[c]ommon_v2.sh' '[t]rain.py' '[l]aunch_c64.sh')
hits=0
for pat in "${PATS[@]}"; do
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$g" ] && { kill -TERM -"$g" 2>/dev/null; hits=$((hits+1)); }
  done
done
sleep 15
for pat in "${PATS[@]}" '[s]glang' '[S]GLangEngine'; do
  for p in $(pgrep -f "$pat" 2>/dev/null); do
    g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$g" ] && kill -KILL -"$g" 2>/dev/null
  done
done
sleep 5
left=$(pgrep -cf '[t]rain.py|[c]ommon_v2.sh|[s]glang' 2>/dev/null || echo 0)
echo "[stop_arm] signalled=$hits remaining=$left on $(hostname)"
