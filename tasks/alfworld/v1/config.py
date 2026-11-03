"""Configuration for the ALFWorld harness.

Two sources, one precedence order: an explicit ``args`` attribute wins when it
exists, otherwise the ``ALFWORLD_*`` environment variable, otherwise the default.

Environment variables are the primary interface because that is this repo's
established convention for custom-loss/rollout knobs (``OPSD_HPI_*``,
``OPSD_CV_*``): the vendored slime argument parser does not know about them, and
extending it would mean editing the shared framework instead of plugging into it.
The ``args`` branch is kept so that a future slime that *does* register these
flags works without a code change.

Naming mirrors the repo's HPI knobs so a reader can map one onto the other:

    OPSD_HPI_ALPHA / OPSD_HPI_SCALE      ->  ALFWORLD_HPI_ALPHA / ALFWORLD_HPI_SCALE
"""

import os
from argparse import Namespace

__all__ = [
    "action_history_limit",
    "env_flag",
    "future_gap",
    "hpi_alpha",
    "hpi_scale",
    "max_turns",
    "privilege_mode",
    "resolve",
]

#: Cap on agent turns per episode.
DEFAULT_MAX_TURNS = 15

#: How many recent actions are rendered into the transcript. Keeps prompt growth
#: (and therefore the O(T^2) layout) bounded on long episodes.
DEFAULT_ACTION_HISTORY = 8

DEFAULT_HPI_ALPHA = 0.5
DEFAULT_HPI_SCALE = 1.0


def resolve(args: Namespace | None, attr: str, env: str, default):
    """``args.attr`` if present, else ``$env``, else ``default``."""
    if args is not None:
        value = getattr(args, attr, None)
        if value is not None:
            return value
    raw = os.environ.get(env)
    if raw is None or raw == "":
        return default
    return raw


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def max_turns(args: Namespace | None = None) -> int:
    value = int(resolve(args, "alfworld_max_turns", "ALFWORLD_MAX_TURNS", DEFAULT_MAX_TURNS))
    if value < 1:
        raise ValueError(f"max turns must be >= 1, got {value}")
    return value


def action_history_limit(args: Namespace | None = None) -> int:
    value = int(
        resolve(args, "alfworld_action_history", "ALFWORLD_ACTION_HISTORY", DEFAULT_ACTION_HISTORY)
    )
    if value < 0:
        raise ValueError(f"action history limit must be >= 0, got {value}")
    return value


def privilege_mode(args: Namespace | None = None) -> str:
    from .privilege import MODES, MODE_NONE

    mode = str(resolve(args, "alfworld_hpi_mode", "ALFWORLD_HPI_MODE", MODE_NONE))
    if mode not in MODES:
        raise ValueError(f"ALFWORLD_HPI_MODE must be one of {MODES}, got {mode!r}")
    return mode


def future_gap(args: Namespace | None = None) -> int:
    from .privilege import DEFAULT_FUTURE_GAP

    value = int(
        resolve(args, "alfworld_hpi_future_gap", "ALFWORLD_HPI_FUTURE_GAP", DEFAULT_FUTURE_GAP)
    )
    if value < 0:
        raise ValueError(f"future gap must be >= 0, got {value}")
    return value


def hpi_alpha(args: Namespace | None = None) -> float:
    return float(resolve(args, "alfworld_hpi_alpha", "ALFWORLD_HPI_ALPHA", DEFAULT_HPI_ALPHA))


def hpi_scale(args: Namespace | None = None) -> float:
    from .loss import SCALE_FLOOR

    return max(float(resolve(args, "alfworld_hpi_scale", "ALFWORLD_HPI_SCALE", DEFAULT_HPI_SCALE)), SCALE_FLOOR)
