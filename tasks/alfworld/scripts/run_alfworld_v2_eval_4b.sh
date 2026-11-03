#!/bin/bash
# Eval a 4B arm -- or Base -- on valid_seen + valid_unseen (ALFWorld).
#
#   EVAL_LOAD_FROM=$PWD/runs/alfworld_v2_hpi_4b RUN_NAME=alfworld_v2_hpi_4b_eval \
#       bash tasks/alfworld/scripts/run_alfworld_v2_eval_4b.sh
#
# EVAL_LOAD_FROM defaults to the HF dir, i.e. Base. Pointing it at a training run
# dir evaluates that checkpoint through the LoRA model provider that _common_v2.sh
# always installs -- which is mandatory: loading a LoRA checkpoint without the
# provider gives a model indistinguishable from Base.
#
# ARM only selects the loss/postprocess wiring, and no training step runs in an
# eval-only pass, so both trained arms are scored by this one path. It is pinned
# to grpo so the HSI teacher scorer is not instantiated during eval.
. "$(dirname "$0")/_4b_backbone.sh"
export EVAL_ONLY=1
# Inherit placement and graph defaults from _v2_defaults.sh through
# _common_v2.sh: 8 actors and 8 rollout engines share the eight cards.
# _4b_backbone.sh owns TP=2 and the 4B serving memory limits.
export ARM=grpo
export RUN_NAME="${RUN_NAME:-alfworld_v2_eval_4b}"
exec bash "$(dirname "$0")/_common_v2.sh"
