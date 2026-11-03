#!/bin/bash
# HSI on Qwen3-4B -- Node0 of the ALFWorld campaign.
#
# One rollout with theta_old, one SAME-policy future-conditioned scoring pass
# issued before the optimizer step, one PPO update with the detached centred
# hindsight advantage, and one policy train step. No frozen teacher, no matched
# or control teacher, no old-actor replay, no full-vocab credit tensor.
#
# The estimator is the shared math_opsd/hsi2_jsd.py (aliased at
# tasks/alfworld/hsi2_jsd.py); the engine/sampler conventions are driven by
# _common_v2.sh, which is the only place they are allowed to live so the two nodes
# cannot drift apart.
#
# Usage: RUN_NAME=... NUM_ROLLOUT=... bash run_alfworld_v2_hsi_4b.sh
. "$(dirname "$0")/_4b_backbone.sh"
# HSI 4B carries a teacher; keep its measured 2 actor + 6 rollout non-colocated
# layout. The shared 8+8 colocated default targets full-RL GRPO/GSPO.
export COLOCATE="${COLOCATE:-0}" ACTOR_GPUS="${ACTOR_GPUS:-2}" ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"
# ARM defaults to `hsi_v3` (our method, the registry's canonical `hsi`) but must
# yield to a caller that exported a different arm: launch_c64.sh's `hsi_v3` case
# routes through this same script, and a hard assignment would be wrong for a
# caller deliberately selecting another live arm. `hsi` now names the same
# canonical objective via the registry, so either spelling is equivalent.
export ARM="${ARM:-hsi_v3}" RUN_NAME="${RUN_NAME:-alfworld_v2_hsi_4b}"
exec bash "$(dirname "$0")/_common_v2.sh"
