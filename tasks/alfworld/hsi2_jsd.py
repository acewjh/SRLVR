"""ALFWorld entry point for the HSI-v3 estimator (running-baseline SIGN x exact JSD).

The estimator itself is ``math_opsd/hsi2_jsd.py`` -- task-agnostic by construction
(it re-implements its own model-access helpers rather than importing them, and its
only imports are stdlib).  It already reads the ALFWorld rollout layout
(``turn_ids`` / ``hpi_views`` / ``hpi_prompt_lengths`` / ``hpi_target_lengths``)
and imports ``tasks.alfworld.config.TURN_ID_STRIDE``, so running it on the ALFWorld
harness needs no adaptation -- only a name the ALFWorld launcher can point
``--rollout-data-postprocess-path`` at.

This module is that name.  It is a **thin alias**: nothing is copied, so the two
campaigns cannot drift apart.  See ``TASK_MATH_HSI_V2_SPEC.md`` for the authority
on the estimator and ``BG_TASK_PROMPT_V3.md`` for the ALFWorld wiring.

Why an alias rather than pointing the launcher straight at ``math_opsd``: the arm
switch in ``scripts/_common_v2.sh`` names ALFWorld-side modules, and keeping the
ALFWorld/ dependency explicit gives the next reader one obvious place to look.
"""

from __future__ import annotations

from methods.hsi2_jsd import (  # noqa: F401  (re-exported on purpose)
    JSD_POSITION_CHUNK,
    baseline_state,
    estimator,
    jsd_token_weights,
    paired_jsd_weights,
    postprocess,
    render_outcome_descriptor,
    reset_counters,
    stats,
)

__all__ = [
    "JSD_POSITION_CHUNK",
    "estimator",
    "render_outcome_descriptor",
    "jsd_token_weights",
    "paired_jsd_weights",
    "baseline_state",
    "postprocess",
    "reset_counters",
    "stats",
]
