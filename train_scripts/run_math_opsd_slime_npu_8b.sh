#!/bin/bash
# Qwen3-8B entry point.
#
# Everything shared lives in run_math_opsd_slime_npu.sh; the size-specific values
# live in configs/models/qwen3-8b.sh (read its header for why 8B is PP2 x DP2
# rather than the 1.7B/4B TP1 x PP1 x DP4 -- short version: 32 GB cards, and the
# harness needs a Megatron student and an HF teacher copy resident at once).
set -uo pipefail
export MODEL="${MODEL:-qwen3-8b}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_math_opsd_slime_npu.sh" "$@"
