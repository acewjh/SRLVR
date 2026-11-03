"""Alias shims for packages that moved in the 2026-10-15 layout refactor.

The refactor split the flat top-level module bags into

    opsd_core/   L0 task- and method-agnostic primitives
    methods/     L1 algorithm implementations (one module per objective)
    tasks/       L2 task-specific modules (tasks/math, tasks/code, tasks/alfworld)

Every historical import path must keep resolving, because three kinds of caller
cannot be edited to follow the move:

  * ``frameworks/slime-ascend`` (vendored, read-only) imports
    ``math_opsd.lora_megatron`` / ``math_opsd.optim_sidecar``;
  * the running Server-A math pipeline and its launchers
    (``scripts/serverA_*``, ``run_grpo*.sh``) name module paths on the CLI;
  * historical ``runs/*/SOURCE_SNAPSHOT`` trees are archived verbatim.

A shim package therefore lives at each old name (``math_opsd/``,
``alfworld_hpi/``, ``alfworld_hpi_v2/``, ``code_grpo/``). Its ``__init__``
installs a meta-path finder that maps ``<old>.<sub>`` onto the real dotted
module.

Two properties matter and both are tested (``tests/test_shim_packages.py``):

1. **Single module object.** ``math_opsd.opsd_loss`` is *the same* module as
   ``methods.opsd_loss`` — not a second execution. Modules with global state
   (e.g. the HSI counters in ``hsi2_jsd``) would otherwise silently fork.
2. **Lazy.** Importing a shim costs nothing until a submodule is asked for, so
   ``import math_opsd`` stays cheap and cannot drag torch/slime into a caller
   that only wanted ``math_opsd.optim_sidecar``.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import sys
from typing import Mapping

__all__ = ["install_alias_finder"]


class _AliasLoader(importlib.abc.Loader):
    """Load ``<old>.<sub>`` as the *same object* as the real module."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec):  # type: ignore[no-untyped-def]
        # Import the real module and hand it straight back: the import machinery
        # then stores it under ``spec.name`` too, so both names share one object.
        return importlib.import_module(self._target)

    def exec_module(self, module) -> None:  # type: ignore[no-untyped-def]
        # Already executed as its real name; nothing left to run.
        return None


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``<package>.<sub>`` names listed in ``mapping``."""

    def __init__(self, package: str, mapping: Mapping[str, str]) -> None:
        self._prefix = package + "."
        self._mapping = dict(mapping)

    def find_spec(self, fullname, path=None, target=None):  # type: ignore[no-untyped-def]
        if not fullname.startswith(self._prefix):
            return None
        sub = fullname[len(self._prefix) :]
        # Only direct children: ``<pkg>.a.b`` is resolved by the real package.
        if "." in sub:
            return None
        real = self._mapping.get(sub)
        if real is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(real))


def install_alias_finder(package: str, mapping: Mapping[str, str]) -> None:
    """Install a lazy alias finder for ``package`` -> ``mapping``.

    ``mapping`` maps the historical submodule name (``"opsd_loss"``) to the real
    dotted path (``"methods.opsd_loss"``). The finder is appended to
    ``sys.meta_path`` so the package's own real files always win.
    """
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder) and finder._prefix == package + ".":
            finder._mapping.update(mapping)
            return
    sys.meta_path.append(_AliasFinder(package, mapping))
