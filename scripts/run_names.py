#!/usr/bin/env python3
"""The run-name rule for opsd-slime, in one place: allocate, rename, verify.

Rule
----
Every run directory under `runs/` is named

    <task>-<model>-<method>[-<variant>]-<YYYYMMDD_HHMMSS>

    task     math | alfworld | code        (the TASKS tuple; additive)
             what the policy is trained (or evaluated) on
    model    qwen3-1p7b | qwen3-4b | qwen3-8b
             the policy checkpoint's short name, as used by configs/models/
    method   the objective, one of METHOD_SLUG's keys. The live vocabulary
             (opsd, grpo, gspo, hsi) is held to opsd_core/registry.py by
             tests/test_run_names.py; METHOD_SLUG also carries the retired/legacy
             slugs (rewardcv*, rewardonly-vnext, outcome-posterior-cv, hpi_future,
             hsi_v2, hsi_candidate, ...) so run directories written before a method
             was retired or renamed still parse
    variant  OPTIONAL and `-`-joined: every special configuration that separates
             this run from the default recipe of the same task+model+method --
             mbs1, pp4, c1024, think, clean, smoke10, gate1024, wedged, ...
             Omit it when the run IS the default recipe.
    stamp    local launch time, YYYYMMDD_HHMMSS: unique, and sorts chronologically

Examples
    math-qwen3-1p7b-grpo-20260921_224239
    math-qwen3-8b-rewardcv-vnext-20260930_144822
    alfworld-qwen3-4b-hsi-think-c1024-20260928_120948
    code-qwen3-1p7b-grpo-gbs64-c4096-r1-20261001_142408

`<model>` is the configs/models/ slug, deliberately spelled `qwen3-1p7b` and not
`qwen3-1.7b`: there is exactly ONE model vocabulary in this repo, and it is the
one every launcher, card and checkpoint lookup already uses.

Why a rule at all: before 2026-09-30 this tree held `math_opsd_slime`,
`qwen3-4b-math-opsd-v2`, `afw64b_hsi`, `reward_cv_vnext_clean_bshd` and
`alfworld_v2_hpi_think_4b_c1024` side by side -- four different orderings of the
same four facts, so no directory could be read without its reports, and no two
could be sorted together.

New runs never need this tool: the launchers allocate through `allocate()` (shell
side: `python scripts/run_names.py --allocate ...`), so a run is born conforming.
What is left here is the maintenance path for what predates the rule:

    python scripts/run_names.py --check          # audit runs/ against the rule
    python scripts/run_names.py --plan           # dry run of the legacy renames
    python scripts/run_names.py --apply          # rename, then repoint references
    python scripts/run_names.py --allocate --task math --model qwen3-4b --method grpo

Reserved names under runs/ that are NOT runs, and are therefore exempt (see
RESERVED): `dev_eval/`, `eval_staging/`, and anything starting with `_`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RUNS = PROJ / "runs"
LEGACY = Path(__file__).resolve().parent / "run_names_legacy.json"

# ---------------------------------------------------------------------------
# The method vocabulary
#
# Three tables, merged into METHOD_SLUG below. The split is load-bearing:
#   LIVE   -- one entry per live objective; held one-for-one to
#             opsd_core/registry.py (the single source -- REFACTOR_PLAN.md §4/R13)
#             by tests/test_run_names.py, so a method added to or removed from the
#             registry without a matching edit here turns the suite red instead of
#             silently minting (or rejecting) a run name.
#   EXTRA  -- run-name tokens in current use that the registry does not model.
#   LEGACY -- retired slugs and historic spellings; PARSE-ONLY, frozen forever, so
#             the run directories already on disk keep resolving.
#
# The LIVE table is NOT imported from the registry on purpose: every launcher runs
# this file as a bare `python3` with PYTHONPATH cleared (train_scripts/lib/
# model_env.sh), and the naming tool must not fail because the L2 registry happens
# to be mid-edit on the shared filesystem. The tests are what keep them in step.
# ---------------------------------------------------------------------------

# Live objectives: registry method name -> the token that names its runs.
LIVE_METHOD_SLUG = {
    # Official OPSD objective.
    "opsd": "opsd",
    "grpo": "grpo",
    # GiGPO: GRPO episode advantage + anchor-state step advantage
    # (arXiv:2505.10978). A distinct objective, so it must not sort under `grpo`.
    "gigpo": "gigpo",
    # GSPO: the GRPO recipe with a sequence-level surrogate
    # (alfworld_hpi_v2/loss.py:compute_ppo_kl). A distinct method, not a grpo
    # variant -- the objective differs, so the run must not sort under `grpo`.
    "gspo": "gspo",
    "srlvr": "srlvr",
    "srlvr_v2": "srlvr_v2",
    "srlvr_v3": "srlvr_v3",
    "srlvr_v4": "srlvr_v4",
    # Our method (user, 2026-09-30): the hindsight same-policy-credit objective,
    # canonical on every task. One spelling: the run directory says `hsi`, never
    # the historic internal names (`hpi_future`, `hsi_v3`). Those are parse-only
    # legacy tokens now (see LEGACY_METHOD_SLUG).
    "hsi": "hsi",
}

# LIVE entries whose run token deliberately differs from the registry's own
# `slug`. Enumerated so the registry cross-check allows exactly these deviations
# and flags every other one. Empty since the ALFWorld arms retired 2026-10-15:
# every live method's run token now equals its registry slug.
REGISTRY_SLUG_REMAPS: frozenset[str] = frozenset()

# Run-name tokens in current use that the registry does not model: the eval-only
# pseudo-methods and the 2026-10-03 ALFWorld campaign arms, which name their runs
# after the campaign rather than after a registry method.
EXTRA_METHOD_SLUG = {
    # Candidate-Centered HSI (2026-10-03): the single-rollout, ONE-frozen-teacher,
    # candidate-centered hindsight-credit objective (alfworld_hpi_v2/candidate_center.py).
    # A distinct method from `hsi` (the two-teacher Full/Control scorer) -- the
    # objective differs, so it must not sort under it. (The registry models the ARM
    # as `hsi_candidate`; this is the run token the campaign used.)
    "hsi_candidate_center": "hsi_candidate_center",
    # The matched GRPO baseline for that campaign: identical harness, seed, task
    # schedule and sampler, G=8. Kept distinct from the generic `grpo` slug so the
    # pair is unambiguous in runs/.
    "grpo_matched": "grpo_matched",
    "base": "base",
    "null-control": "null_control",
}

# Retired objectives (REFACTOR_PLAN.md §7) and the spelling the launcher used for
# each. Kept so the run directories written before retirement still parse; a live
# run must never be allocated one of these.
LEGACY_METHOD_SLUG = {
    "rewardcv": "rewardcv",
    "rewardcv-cal": "rewardcv-cal",
    "rewardcv-hpi": "rewardcv-hpi",
    "rewardcv-hpi-v2": "rewardcv-hpi-v2",
    "rewardcv-hpi-v15": "rewardcv-hpi-v15",
    "rewardcv-vnext": "rewardcv-vnext",
    "rewardonly-vnext": "rewardonly-vnext",
    "outcome-posterior-cv": "outcome-posterior-cv",
    # accepted on the launcher side as well, so --method-for resolves them
    "reward_cv": "rewardcv",
    "reward_cv_calibrated": "rewardcv-cal",
    "reward_cv_hpi": "rewardcv-hpi",
    "reward_cv_hpi_v2": "rewardcv-hpi-v2",
    "reward_cv_hpi_v15": "rewardcv-hpi-v15",
    "reward_cv_vnext": "rewardcv-vnext",
    "reward_only_vnext": "rewardonly-vnext",
    "outcome_posterior_cv": "outcome-posterior-cv",
    # ALFWorld credit arms retired 2026-10-15 (_retired/README.md Batch 3). The
    # objective key is kept so run directories written under these names still
    # parse; a live run is never allocated one.
    "hpi_future": "hsi",
    "hsi_v2": "hsi_v2",
    "hsi_candidate": "hsi_candidate",
}

#: method token -> the objective key it stands for. LIVE wins over the others so a
#: live objective can never be shadowed by a legacy spelling.
METHOD_SLUG = {**LEGACY_METHOD_SLUG, **EXTRA_METHOD_SLUG, **LIVE_METHOD_SLUG}

# Directories under runs/ that are stores or tooling, not runs. They are exempt
# from the rule; everything else must match it.
RESERVED = {"dev_eval", "eval_staging"}

# Tasks that own a run namespace. 'code' was added 2026-10-01 for the Qwen3-1.7B
# code-GRPO baseline. Purely additive: the math/alfworld grammar and every
# existing run name are unaffected.
TASKS = ("math", "alfworld", "code")

STAMP_RE = r"\d{8}_\d{6}"

# The method is matched against the KNOWN vocabulary, longest alternative first,
# not against a loose `[a-z0-9-]+`. Both halves of the token are hyphen-joined, so
# a loose class is ambiguous: `-rewardcv-vnext-mbs1-pp4-<stamp>` would happily
# parse as method `rewardcv-vnext-mbs1-pp4` with no variant, and nothing about the
# result would look wrong. The model is lazy for the same reason -- it must yield
# to the method at the first split that works.
_METHOD_ALT = "|".join(re.escape(m) for m in sorted(METHOD_SLUG, key=len, reverse=True))
NAME_RE = re.compile(
    rf"^(?P<task>{'|'.join(TASKS)})-(?P<model>[a-z0-9.\-]+?)-(?P<method>{_METHOD_ALT})"
    rf"(?:-(?P<variant>[a-z0-9\-]+))?-(?P<stamp>{STAMP_RE})$"
)


def allocate(
    task: str,
    model: str,
    method: str,
    variant: str = "",
    stamp: str | None = None,
) -> str:
    """Return the conforming name for a run. Raises on anything off-rule.

    `stamp` is injectable so tests (and renames) are deterministic; production
    callers leave it None and get the current local time.
    """
    if task not in TASKS:
        raise ValueError(f"task {task!r} must be one of {TASKS}")
    if method not in METHOD_SLUG:
        raise ValueError(
            f"method {method!r} is not a known objective; one of: "
            + ", ".join(sorted(METHOD_SLUG))
        )
    if not re.fullmatch(r"[a-z0-9.\-]+", model):
        raise ValueError(f"model {model!r} must be lowercase [a-z0-9.-]")
    variant = (variant or "").strip("-")
    if variant and not re.fullmatch(r"[a-z0-9\-]+", variant):
        raise ValueError(f"variant {variant!r} must be lowercase [a-z0-9-]")
    if stamp is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(STAMP_RE, stamp):
        raise ValueError(f"stamp {stamp!r} must be YYYYMMDD_HHMMSS")

    name = f"{task}-{model}-{method}"
    if variant:
        name += f"-{variant}"
    return f"{name}-{stamp}"


def parse(name: str) -> dict[str, str] | None:
    """Split a conforming name, or None if it does not conform."""
    m = NAME_RE.match(name)
    if not m:
        return None
    out = m.groupdict()
    out["variant"] = out.get("variant") or ""
    return out


def is_reserved(name: str) -> bool:
    return name in RESERVED or name.startswith("_") or name.startswith(".")


#: Audit categories. `offrule` is the only failing one; everything else is either
#: conforming, exempt by convention, or known and explained.
CATEGORIES = ("conforming", "reserved", "pending", "deferred", "offrule")


def audit() -> dict[str, list[str]]:
    """Partition every directory under runs/ into exactly one CATEGORIES bucket.

    A complete partition is the point: a directory that is neither conforming nor
    explicitly accounted for is how a naming standard quietly rots. `pending` is
    "off-rule, rename planned, not yet applied"; `deferred` is "off-rule, and
    deliberately left alone" (blocked on a live job, or not a run at all) -- those
    must NOT be reported as failures, or the audit stays red for reasons everyone
    already knows and stops being read.
    """
    buckets: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    if not RUNS.is_dir():
        return buckets
    planned = _legacy_map()
    deferred = _deferred()
    for entry in sorted(RUNS.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if is_reserved(name):
            buckets["reserved"].append(name)
        elif parse(name):
            buckets["conforming"].append(name)
        elif name in planned:
            buckets["pending"].append(name)
        elif name in deferred:
            buckets["deferred"].append(name)
        else:
            buckets["offrule"].append(name)
    return buckets


def _legacy_data() -> dict:
    if not LEGACY.is_file():
        return {"renames": {}, "deferred": {}}
    return json.loads(LEGACY.read_text())


def _legacy_map() -> dict[str, str]:
    return {old: rec["new"] for old, rec in _legacy_data().get("renames", {}).items()}


def _deferred() -> dict[str, str]:
    """name -> blocking group, excluding the `_groups` legend entry."""
    raw = _legacy_data().get("deferred", {})
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _deferred_groups() -> dict[str, str]:
    return _legacy_data().get("deferred", {}).get("_groups", {})


# ---------------------------------------------------------------------------
# references: a rename is only done when every live reference moves with it
# ---------------------------------------------------------------------------

REFERENCE_GLOBS = (
    "scripts/*.sh",
    "train_scripts/*.sh",
    "train_scripts/lib/*.sh",
    "configs/models/*.sh",
    "math_opsd/*.py",
    "tests/*.py",
    "eval_node/*.sh",
    "alfworld_hpi/scripts/*.sh",
    "alfworld_hpi_v2/scripts/*.sh",
)


def reference_files() -> list[Path]:
    files: list[Path] = []
    for glob in REFERENCE_GLOBS:
        files.extend(sorted(PROJ.glob(glob)))
    return files


def repoint(old: str, new: str, *, apply: bool) -> list[str]:
    """Rewrite `runs/<old>` references to `runs/<new>`. Returns the hit list.

    Only the run-directory token is rewritten -- bare mentions of the same words
    elsewhere (a log tag, a variable name) are deliberately left alone, because a
    rename must not silently retarget something that is not this directory.
    """
    hits: list[str] = []
    pattern = re.compile(rf"(runs/){re.escape(old)}(?![\w.\-])")
    for path in reference_files():
        text = path.read_text()
        if not pattern.search(text):
            continue
        hits.append(f"{path.relative_to(PROJ)}")
        if apply:
            path.write_text(pattern.sub(rf"\g<1>{new}", text))
    return hits


def symlinks_into(planned: dict[str, str]) -> list[tuple[Path, str, str]]:
    """Symlinks under runs/ whose target is a renamed run directory.

    These are the easy way to break a rename without touching a single script:
    `runs/eval_staging/<cell>/iter_N` is a symlink into the run it stages, and
    renaming the run silently turns every staged cell into a dangling link. The
    2026-09-30 rename did exactly that to 38 links before this function existed.

    Returns ``[(link, old_name, target_relpath)]``.
    """
    root = str(RUNS) + os.sep
    found: list[tuple[Path, str, str]] = []
    for link in RUNS.rglob("*"):
        if not link.is_symlink():
            continue
        target = link.readlink()
        resolved = target if target.is_absolute() else link.parent / target
        resolved = Path(os.path.normpath(resolved))
        # Only THIS tree's runs/. The filesystem is shared, and a sibling repo
        # (`opsd-slime-cua-rl`) keeps its own runs/ whose names this map does not
        # govern -- matching on the bare "runs/" segment flagged those as ours.
        if not str(resolved).startswith(root):
            continue
        old, _, rest = str(resolved.relative_to(RUNS)).partition(os.sep)
        if old in planned:
            found.append((link, old, rest))
    return found


def plan() -> list[tuple[str, str, str, list[str]]]:
    """[(old, new, stamp_source, [referencing files])], checked for collisions."""
    data = json.loads(LEGACY.read_text())
    out = []
    seen = set()
    for old, rec in sorted(data["renames"].items()):
        new = rec["new"]
        # Targets in the `_` namespace are infrastructure/residue, exempt by
        # design; only real runs have to conform.
        if not is_reserved(new) and not parse(new):
            raise SystemExit(f"legacy map: {new!r} does not conform to the rule")
        if new in seen:
            raise SystemExit(f"legacy map: {new!r} is produced twice")
        seen.add(new)
        if old not in RESERVED and not old.startswith("_"):
            src = RUNS / old
            if src.is_dir() and (RUNS / new).exists():
                raise SystemExit(f"legacy map: target {new!r} already exists")
        out.append((old, new, rec.get("stamp_source", "?"), repoint(old, new, apply=False)))
    return out


def dangling_into_renamed() -> list[tuple[Path, str]]:
    """Dangling symlinks under runs/ that point at a name this map renames.

    This is the failure mode a directory rename has and a reference rewrite does
    not catch, so it is a hard failure of the audit rather than a note.
    """
    out: list[tuple[Path, str]] = []
    for link, old, _rest in symlinks_into(_legacy_map()):
        if not link.exists():
            out.append((link, old))
    return out


def cmd_check() -> int:
    buckets = audit()
    for category in CATEGORIES:
        print(f"{category:>10}: {len(buckets[category])}")
    for name in buckets["offrule"]:
        print(f"  BAD  {name}")

    dangling = dangling_into_renamed()
    if dangling:
        print(f"\n{len(dangling)} dangling symlink(s) into a RENAMED directory:")
        for link, old in dangling:
            print(f"  BROKEN {link} -> runs/{old}/...")
        print("  run --plan/--apply to relink, or these staged cells are unusable")
    flagged = {link for link, _ in dangling}
    other_dangling = [
        link for link in RUNS.rglob("*")
        if link.is_symlink() and not link.exists() and link not in flagged
    ]
    if other_dangling:
        print(
            f"\n({len(other_dangling)} other dangling symlink(s), NOT caused by the rule: "
            "pre-existing, or pointing into a sibling tree such as opsd-slime-cua-rl)"
        )

    if buckets["pending"]:
        print("\npending (rename planned; run --apply):")
        print(f"  {', '.join(buckets['pending'])}")

    groups = _deferred_groups()
    by_group: dict[str, list[str]] = {}
    for name in buckets["deferred"]:
        by_group.setdefault(_deferred()[name], []).append(name)
    for group in sorted(by_group):
        print(f"\ndeferred -- {group} ({len(by_group[group])}):")
        print(f"  {groups.get(group, 'no reason recorded')}")
        print(f"  {', '.join(sorted(by_group[group]))}")
    return 1 if (buckets["offrule"] or dangling) else 0


def cmd_plan() -> int:
    rows = plan()
    print(f"{len(rows)} rename(s) planned\n")
    for old, new, source, refs in rows:
        mark = " " if (RUNS / old).is_dir() else "?"
        print(f"{mark} runs/{old}\n    -> runs/{new}   [stamp: {source}]")
        if refs:
            print(f"       refs: {', '.join(refs)}")
    return 0


def cmd_apply() -> int:
    rows = plan()
    applied, skipped = [], []
    for old, new, _source, _refs in rows:
        src, dst = RUNS / old, RUNS / new
        if not src.is_dir():
            skipped.append((old, new, "source missing"))
            continue
        if dst.exists():
            skipped.append((old, new, "target exists"))
            continue
        src.rename(dst)
        applied.append((old, new))
    for old, new in applied:
        repoint(old, new, apply=True)
    # A rename is not done until the symlinks that point INTO the renamed
    # directory move with it -- staged eval cells and `_mirror_*` dirs are links,
    # not copies, so a script-only rewrite leaves them dangling.
    applied_map = dict(applied)
    relinked = 0
    for link, old, rest in symlinks_into(applied_map):
        link.unlink()
        new_target = RUNS / applied_map[old] / rest if rest else RUNS / applied_map[old]
        link.symlink_to(new_target)
        relinked += 1
    if applied:
        pointer = RUNS / "LAYOUT.md"
        lines = [
            "# runs/ name map",
            "",
            "Every directory here follows `<task>-<model>-<method>[-<variant>]-<stamp>`"
            " (`scripts/run_names.py`). Renamed on 2026-09-30; forward-only, nothing",
            "was deleted.",
            "",
            "| was | now |",
            "|---|---|",
        ]
        lines += [f"| `runs/{o}` | `runs/{n}` |" for o, n in applied]
        pointer.write_text("\n".join(lines) + "\n")
    print(f"applied {len(applied)}, skipped {len(skipped)}, relinked {relinked}")
    for old, new, why in skipped:
        print(f"  SKIP runs/{old} -> runs/{new}: {why}")
    return 0


def cmd_allocate(args: argparse.Namespace) -> int:
    print(allocate(args.task, args.model, args.method, args.variant or ""))
    return 0


def cmd_method_for(launcher_method: str) -> int:
    """Launcher's OPSD_METHOD -> the name-rule method slug.

    Single source for the mapping: the shell launchers ask, so `reward_cv_vnext`
    and `rewardcv-vnext` cannot drift apart in two places.
    """
    if launcher_method not in METHOD_SLUG:
        raise SystemExit(
            f"OPSD_METHOD={launcher_method!r} has no name-rule slug; add it to "
            "METHOD_SLUG in scripts/run_names.py"
        )
    print(METHOD_SLUG[launcher_method])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="audit runs/ against the rule")
    ap.add_argument("--plan", action="store_true", help="dry run of the legacy renames")
    ap.add_argument("--apply", action="store_true", help="rename and repoint references")
    ap.add_argument("--allocate", action="store_true", help="print a new conforming name")
    ap.add_argument("--method-for", metavar="OPSD_METHOD", help="launcher method -> name-rule slug")
    ap.add_argument("--task", default="math")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--method", default="opsd")
    ap.add_argument("--variant", default="")
    args = ap.parse_args(argv)

    if args.method_for:
        return cmd_method_for(args.method_for)
    if args.allocate:
        return cmd_allocate(args)
    if args.apply:
        return cmd_apply()
    if args.plan:
        return cmd_plan()
    return cmd_check()


if __name__ == "__main__":
    sys.exit(main())
