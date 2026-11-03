#!/bin/bash
# Qwen3-4B dev-eval entry point.
#
# Merged 2026-09-28 from the Server-B replica `cua/repos/opsd-slime-cua-rl`. All
# logic lives in run_dev_eval.sh; the size-specific values live in
# configs/models/qwen3-4b.sh. This file only picks the card.
set -uo pipefail
export MODEL="${MODEL:-qwen3-4b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dev_eval.sh" "$@"
