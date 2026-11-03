"""The live tree has no dangling import after the 2026-10-15 layout refactor. Two invariants:

1. **Every absolute import in the live tree resolves.** The refactor moved
   ``math_opsd`` / ``alfworld_hpi`` / ``alfworld_hpi_v2`` / ``code_grpo`` into
   ``methods/`` + ``tasks/``. The old-name alias shims were retired on
   2026-10-15, so a lingering ``import math_opsd.*`` now shows up here as an
   unresolvable spec.

2. **Every module path the registry hands to slime resolves.** ``registry`` is
   the single source for what the launcher passes to ``train.py``; a typo there
   kills a run at import time on the training ranks, hours in.

``TRACKED_ROOTS`` keeps the four retired names deliberately: if one ever
reappears in live code the first invariant turns red. Deliberately *not*
asserted: that the live tree contains no old package names in prose -- the frozen
``scripts/serverA_*`` pipeline and ``runs/*/SOURCE_SNAPSHOT`` archives keep
naming them (both are outside the walk).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib

import pytest

from opsd_core import registry

PROJ = pathlib.Path(__file__).resolve().parent.parent

TRACKED_ROOTS = {
    "opsd_core",
    "methods",
    "tasks",
    # retired 2026-10-15: kept so a stray import turns red
    "math_opsd",
    "alfworld_hpi",
    "alfworld_hpi_v2",
    "code_grpo",
}

# Trees that are records, not live code. `runs/*/SOURCE_SNAPSHOT` is a verbatim
# archive of an old working tree; `_retired/` is the retirement archive; the rest
# are captured logs and generated reports.
SKIP_DIRS = {"__pycache__", ".git", "_retired", "runs", "logs", "reports", "frameworks", "docs"}


def _live_py_files():
    for dp, dn, fn in os.walk(PROJ):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            if f.endswith(".py"):
                yield pathlib.Path(dp) / f


def _absolute_imports(path: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolved by its own package
                continue
            yield (node.module or ""), node.lineno


def _resolves(dotted: str) -> bool:
    try:
        return importlib.util.find_spec(dotted) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


@pytest.mark.unit
def test_every_live_import_resolves():
    dangling: list[str] = []
    for path in _live_py_files():
        rel = path.relative_to(PROJ)
        for dotted, lineno in _absolute_imports(path):
            root = dotted.split(".")[0]
            if root not in TRACKED_ROOTS:
                continue
            if not _resolves(dotted):
                dangling.append(f"{rel}:{lineno} -> {dotted}")
    assert not dangling, "unresolvable imports in the live tree:\n" + "\n".join(sorted(dangling))


@pytest.mark.unit
def test_registry_module_paths_resolve():
    paths: set[str] = set()
    for task_spec in registry.TASKS.values():
        for value in (task_spec.rm,):
            if value:
                paths.add(value)
    for table in registry.METHODS.values():
        for spec in table.values():
            for value in (spec.loss, spec.rollout, spec.convert, spec.postprocess, spec.provider, spec.rm):
                if value:
                    paths.add(value)

    # Each registry entry is ``<dotted.module>.<attribute>``; slime resolves it
    # with importlib at launch. Assert the MODULE half resolves (the attribute
    # half is only checkable by importing, which would drag torch in).
    unresolved = sorted(p for p in paths if not _resolves(p.rsplit(".", 1)[0]))
    assert not unresolved, f"registry names modules that do not exist: {unresolved}"
