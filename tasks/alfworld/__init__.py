"""ALFWorld Future-HPI v2 -- per-turn training rows, Full-vs-Control credit.

v2 exists because v1 made three structural choices that turned out to be wrong,
all of them diagnosable and all of them fixed here:

1. **One Megatron row per episode.** The row grew with the turn budget, and the
   LoRA lm_head (full-vocab fp32 over the whole row) set the horizon, so only ~5
   turns fitted. v2 emits one row PER TURN: horizon and row length are decoupled,
   and ``MAX_TURNS`` goes to 20.
2. **Teacher scoring inside the custom loss.** ``--recompute-loss-function``
   re-executes the loss during the backward, so every view was scored twice and a
   second 1.7B had to be shuttled between NPU and CPU between microbatches. ~30-70
   min/step. v2 scores once, batched, in ``--rollout-data-postprocess-path`` --
   before the step, outside the loss, with the teacher resident.
3. **Hand-shaped weights** (``1 + alpha*tanh(|e|/scale)``, a calibrated scale, a
   clamp). v2 uses the unshaped, self-normalised density ratio ``r_t = exp(e_t)``
   with per-trajectory token-count normalisation, so ``Full == Control`` reduces to
   vanilla GRPO exactly and there is no free hyper-parameter.

Reused from v1 UNCHANGED, by import only: ``paths``, ``env_adapter``,
``action_parser``, ``trajectory``, ``reward``, ``eval``. No v1 file is modified --
the v1 GRPO run on worker-1 keeps running against the bytes it imported.
"""

import os as _os

# Env-actor worker processes import ONLY ``tasks.alfworld.v1.env_adapter``. The
# eager imports below pull in torch/ray/sglang through rollout/loss, which would
# make every TextWorld worker a 500 MB torch process and defeat the point of a
# CPU-only env actor. The worker sets this flag before importing the adapter.
if _os.environ.get("ALFWORLD_ENV_ACTOR_PROCESS") != "1":  # pragma: no branch
    from . import (  # noqa: F401
        advantages,
        config,
        episode,
        hsi_shared,
        loss,
        privilege,
        prompts,
        rollout,
        schedule,
    )

__all__ = [
    "advantages",
    "config",
    "episode",
    "hsi_shared",
    "loss",
    "privilege",
    "prompts",
    "rollout",
    "schedule",
]
