"""Filesystem + dependency-isolation paths for the ALFWorld harness.

Dependency isolation contract
-----------------------------
The cua-rl base environment is shared with running experiments and MUST NOT be
modified. ALFWorld's Python dependencies therefore live in a separate overlay
venv created with ``--system-site-packages``::

    cua/envs/alfworld_overlay

That overlay is activated for standalone scripts, but the *training* process runs
under cua-rl. To let the rollout import ``textworld``/``alfworld`` from there we
append the overlay's ``site-packages`` to ``sys.path``.

Appending is the load-bearing detail: ``sys.path`` entries that come later can
only *supply* modules the interpreter is missing, never *replace* ones it already
found. Verified overlap between the overlay and cua-rl is limited to
``pip``/``setuptools``/``pkg_resources``/``_distutils_hack``, none of which are
imported by the training path.
"""

import os
import sys

__all__ = [
    "ALFWORLD_ROOT",
    "DEFAULT_ALFWORLD_DATA",
    "DEFAULT_OVERLAY_SITE",
    "SPLITS",
    "ensure_alfworld_importable",
    "game_data_root",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This package lives at ``$CUA/repos/opsd-slime/tasks/alfworld/v1``, so the cua
# root is five levels up from this file
# (v1 -> alfworld -> tasks -> opsd-slime -> repos -> cua).
_CUA_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
    )
)

#: Root of the ALFWorld assets (game files + traj_data.json).
ALFWORLD_ROOT = os.environ.get("ALFWORLD_ROOT", os.path.join(_CUA_ROOT, "data", "alfworld"))

# Capture the data-root environment ONCE, at import time.
#
# This is load-bearing: importing the `alfworld` package sets
# ``os.environ["ALFWORLD_DATA"]`` to ``~/.cache/alfworld`` when the variable is
# unset, so a late lookup would silently redirect every game path to an empty
# cache directory. Reading it here (before `ensure_alfworld_importable` runs)
# and pinning it avoids that. `ALFWORLD_HPI_DATA` is our own override and takes
# precedence because nothing else writes it.
_ENV_DATA_AT_IMPORT = os.environ.get("ALFWORLD_DATA")
_ENV_HPI_DATA = os.environ.get("ALFWORLD_HPI_DATA")

#: Directory that holds ``train`` / ``valid_seen`` / ``valid_unseen``.
DEFAULT_ALFWORLD_DATA = os.path.join(ALFWORLD_ROOT, "json_2.1.1")

#: Overlay venv site-packages holding textworld / alfworld / jericho.
DEFAULT_OVERLAY_SITE = os.path.join(
    _CUA_ROOT, "envs", "alfworld_overlay", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"
)

#: ALFWorld split names, in the order used by the eval report.
SPLITS = ("train", "valid_seen", "valid_unseen")


def _candidate_overlay_sites() -> list[str]:
    """Overlay site-packages candidates, most specific first.

    ``ALFWORLD_OVERLAY_SITE`` wins so a run can be repointed without code edits;
    otherwise the literal venv path is derived from the active interpreter's
    version rather than a hardcoded ``python3.10``.
    """
    candidates = []
    env_site = os.environ.get("ALFWORLD_OVERLAY_SITE")
    if env_site:
        candidates.append(env_site)
    candidates.append(DEFAULT_OVERLAY_SITE)
    return candidates


def ensure_alfworld_importable() -> str | None:
    """Make ``textworld`` / ``alfworld`` importable, returning the path used.

    Idempotent. Returns ``None`` when ALFWorld is already importable without the
    overlay (e.g. a standalone script running under the overlay interpreter), in
    which case nothing is appended.
    """
    if _already_importable():
        return None

    for site in _candidate_overlay_sites():
        if not os.path.isdir(site):
            continue
        if site not in sys.path:
            sys.path.append(site)

    if not _already_importable():
        raise ImportError(
            "ALFWorld is not importable. Expected the overlay site-packages at "
            f"{DEFAULT_OVERLAY_SITE!r} (override with ALFWORLD_OVERLAY_SITE). "
            "Run tasks/alfworld/v1/scripts/setup_alfworld.sh to build it."
        )
    return sys.path[-1]


def _already_importable() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(mod) is not None for mod in ("textworld", "alfworld"))


def game_data_root() -> str:
    """Root directory containing the ALFWorld split folders.

    Resolution order: ``ALFWORLD_HPI_DATA``, then the ``ALFWORLD_DATA`` value
    captured at import time, then the in-repo default. See the note on
    ``_ENV_DATA_AT_IMPORT`` for why this is not a live ``os.environ`` lookup.
    """
    root = _ENV_HPI_DATA or _ENV_DATA_AT_IMPORT or DEFAULT_ALFWORLD_DATA
    # Keep the alfworld/textworld packages pointed at the same tree.
    os.environ["ALFWORLD_DATA"] = root
    return root
