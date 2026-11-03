#!/usr/bin/env bash
# Legacy entry name; one optimizer update is now the common default.
export ARM=grpo
exec bash "$(dirname "$0")/run_alfworld_v2_1p7b.sh"
