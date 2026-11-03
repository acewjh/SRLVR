#!/bin/bash
# Single source of truth for the eval-protocol constants.
#
# These numbers were copy-pasted across the eval queues and
# train_scripts/run_dev_eval.sh. They live here now and are consumed from here:
#
#   OPSD_EVAL_MAX_NEW        38912   formal max_new_tokens / max_response_len
#   OPSD_EVAL_K              8      n_samples_per_eval_prompt (AIME24/25, HMMT25)
#   OPSD_EVAL_CONTEXT_LENGTH 40960   Qwen3 native max_position_embeddings
#   OPSD_EVAL_MEM_FRACTION   0.82    --sglang-mem-fraction-static (this 8-NPU node)
#   OPSD_EVAL_CHUNKED_PREFILL 2048   keeps the prefill logits spike ~1.2 GB/engine
#
# Source it (it only sets/exports variables -- no side effects; a preset value
# wins, so a caller can still override):
#     source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/eval_protocol.sh"
#
# DELIBERATELY NOT defined here -- the single source for these is elsewhere:
#   temperature=1.0, top_p=0.95, top_k=-1, and the *effective* (final)
#   n_samples_per_eval_prompt / max_response_len live ONLY in
#   configs/eval_*.json. The eval engine resolves a dataset's
#   n_samples_per_eval_prompt / max_response_len / temperature / top_p / top_k
#   from the config FIRST and falls back to the CLI flags
#   (--n-samples-per-eval-prompt / --eval-max-response-len / ...) only when the
#   config omits them -- see frameworks/slime-ascend/slime/utils/eval_config.py
#   (_apply_dataset_field_overrides / _first_not_missing). Every
#   configs/eval_*.json sets them, so EVAL_K / EVAL_MAX_NEW below are FALLBACKS,
#   not the effective values. Adding a second copy of the sampling params here
#   would silently drift from the configs -- so do not.

: "${OPSD_EVAL_MAX_NEW:=38912}"
: "${OPSD_EVAL_K:=8}"
: "${OPSD_EVAL_CONTEXT_LENGTH:=40960}"
: "${OPSD_EVAL_MEM_FRACTION:=0.82}"
: "${OPSD_EVAL_CHUNKED_PREFILL:=2048}"

export OPSD_EVAL_MAX_NEW OPSD_EVAL_K OPSD_EVAL_CONTEXT_LENGTH \
       OPSD_EVAL_MEM_FRACTION OPSD_EVAL_CHUNKED_PREFILL
