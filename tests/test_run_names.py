"""The run-name rule: allocation, parsing, audit, and the launcher contract.

User directive (2026-09-30, and not for the first time): every run directory is
`<task>-<model>-<method>[-<variant>]-<YYYYMMDD_HHMMSS>`, and that is the standing
standard for everything launched from now on -- not a one-off cleanup.

These tests hold both halves of that:
  * the rule itself (allocate/parse/round-trip, and that it REJECTS junk rather
    than emitting a name that merely looks close);
  * that the launchers actually go through it, because a rule nobody calls is how
    this tree acquired four different naming schemes.

Run:  python -m pytest tests/test_run_names.py -q
      (cua-rl env: `conda activate cua-rl` first -- scripts/run_names.py imports
       nothing outside the stdlib, but the launcher tests shell out to bash.)
"""

import pathlib
import re
import subprocess
import sys

import pytest

PROJ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from scripts import run_names  # noqa: E402


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------


def test_allocate_joins_the_five_components_in_order() -> None:
    name = run_names.allocate("math", "qwen3-8b", "rewardcv-vnext", "mbs1-pp4", "20260930_144822")
    assert name == "math-qwen3-8b-rewardcv-vnext-mbs1-pp4-20260930_144822"


def test_variant_is_optional_and_absent_names_stay_short() -> None:
    name = run_names.allocate("math", "qwen3-1p7b", "grpo", "", "20260921_224239")
    assert name == "math-qwen3-1p7b-grpo-20260921_224239"


def test_allocate_defaults_the_stamp_to_now() -> None:
    name = run_names.allocate("math", "qwen3-4b", "opsd")
    assert re.fullmatch(r"math-qwen3-4b-opsd-\d{8}_\d{6}", name)


@pytest.mark.parametrize(
    "name",
    [
        "math-qwen3-8b-rewardcv-vnext-mbs1-pp4-20260930_144822",
        "math-qwen3-1p7b-grpo-20260921_224239",
        "alfworld-qwen3-4b-hsi-think-c1024-20260928_120948",
    ],
)
def test_parse_round_trips_every_example_in_the_docstring(name: str) -> None:
    parsed = run_names.parse(name)
    assert parsed is not None
    rebuilt = run_names.allocate(
        parsed["task"],
        parsed["model"],
        parsed["method"] if parsed["method"] in run_names.METHOD_SLUG else "grpo",
        parsed["variant"],
        parsed["stamp"],
    )
    assert rebuilt == name


@pytest.mark.parametrize(
    "bad",
    [
        "math_opsd_slime",  # the old shape
        "qwen3-4b-math-opsd-v2",  # model first, no stamp
        "afw64b_hsi",
        "reward_cv_vnext_clean_bshd",
        "math-qwen3-8b-rewardcv-vnext",  # no stamp
        "math-qwen3-8b-rewardcv-vnext-2026-09-30",  # stamp not YYYYMMDD_HHMMSS
    ],
)
def test_parse_rejects_the_legacy_shapes(bad: str) -> None:
    assert run_names.parse(bad) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"task": "chem"},
        {"method": "not-a-method"},
        {"model": "Qwen3-8B"},  # uppercase
        {"variant": "has space"},
        {"stamp": "yesterday"},
    ],
)
def test_allocate_refuses_off_rule_input(kwargs) -> None:
    base = {"task": "math", "model": "qwen3-8b", "method": "grpo"}
    with pytest.raises(ValueError):
        run_names.allocate(**{**base, **kwargs})


def test_reserved_names_are_exempt() -> None:
    """Stores and tooling are not runs; the rule must not demand a stamp of them."""
    for name in ("dev_eval", "eval_staging", "_mirror_x", "_residue-whatever", ".lock"):
        assert run_names.is_reserved(name)
    for name in ("math-qwen3-1p7b-opsd-20260924_015038", "afw64b_hsi"):
        assert not run_names.is_reserved(name)


def test_method_for_maps_every_launcher_method_to_a_slug() -> None:
    """The shell asks for the slug, so the two vocabularies cannot drift apart."""
    for launcher_method in (
        "opsd",
        "reward_cv",
        "reward_cv_calibrated",
        "reward_cv_hpi",
        "reward_cv_hpi_v2",
        "reward_cv_hpi_v15",
        "reward_cv_vnext",
        "reward_only_vnext",
        "grpo",
    ):
        slug = run_names.METHOD_SLUG[launcher_method]
        assert slug in run_names.METHOD_SLUG, f"{slug} is not itself a name-rule method"


def test_method_for_rejects_an_unmapped_method() -> None:
    with pytest.raises(SystemExit):
        run_names.cmd_method_for("totally_new_method")


# ---------------------------------------------------------------------------
# the live vocabulary must mirror the registry (REFACTOR_PLAN.md §4 / R13)
#
# run_names.py keeps its own vocabulary instead of importing the L2 registry (it
# is invoked as a bare `python3` with PYTHONPATH cleared, and must not depend on
# the registry being importable). That freedom is exactly what lets the two
# drift, so these tests are the pin.
# ---------------------------------------------------------------------------


def test_live_vocabulary_mirrors_the_registry() -> None:
    """A method added to (or removed from) the registry must be mirrored here."""
    from opsd_core import registry

    registered = {m for table in registry.METHODS.values() for m in table}
    assert set(run_names.LIVE_METHOD_SLUG) == registered


def test_every_live_registry_slug_is_a_recognised_run_token() -> None:
    """A run the registry can name must survive the rule's allocate/parse."""
    from opsd_core import registry

    for slug in sorted(set(registry.canonical_slugs().values())):
        assert slug in run_names.METHOD_SLUG, f"run_names does not know the slug {slug!r}"
        name = run_names.allocate("math", "qwen3-8b", slug, "", "20260101_000000")
        parsed = run_names.parse(name)
        assert parsed is not None, name
        assert parsed["method"] == slug, f"{name} parsed as {parsed['method']!r}"


def test_live_slug_values_agree_with_the_registry() -> None:
    """Only the documented remaps may name a run differently from the registry."""
    from opsd_core import registry

    registry_slug = {
        m: spec.slug for table in registry.METHODS.values() for m, spec in table.items()
    }
    for method, slug in run_names.LIVE_METHOD_SLUG.items():
        if method in run_names.REGISTRY_SLUG_REMAPS:
            continue
        assert slug == registry_slug[method], (
            f"{method}: run_names says {slug!r}, registry says {registry_slug[method]!r}"
        )
    # every documented remap must be a real live method -- no stale exceptions
    assert run_names.REGISTRY_SLUG_REMAPS <= set(registry_slug)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------


def test_audit_partitions_every_entry() -> None:
    """Every directory is conforming, reserved, planned, deferred, or a failure.

    Notably it must not report a DEFERRED directory as a failure: an audit that
    stays red for a known, explained reason stops being read, and then it stops
    catching the real thing.
    """
    buckets = run_names.audit()
    assert set(buckets) == set(run_names.CATEGORIES)
    seen = {name for names in buckets.values() for name in names}
    on_disk = {p.name for p in run_names.RUNS.iterdir() if p.is_dir()}
    assert seen == on_disk, f"unaccounted: {on_disk - seen}, unknown: {seen - on_disk}"
    assert not buckets["offrule"], f"off-rule and unexplained: {buckets['offrule']}"


def test_every_planned_rename_conforms_and_is_unique() -> None:
    planned = run_names._legacy_map()
    assert planned, "the legacy map should not be empty"
    assert len(set(planned.values())) == len(planned), "two dirs map to the same name"
    for old, new in planned.items():
        # Residue/infrastructure moves to the `_` namespace, which is exempt from
        # the rule by design -- only real runs have to conform.
        if run_names.is_reserved(old) or run_names.is_reserved(new):
            continue
        assert run_names.parse(new), f"{old} -> {new} does not conform"


def test_deferred_entries_all_carry_a_group_with_a_reason() -> None:
    groups = run_names._deferred_groups()
    for name, group in run_names._deferred().items():
        assert group in groups, f"{name} is deferred to an undocumented group {group!r}"
        assert len(groups[group]) > 40, f"group {group} has no real explanation"


# ---------------------------------------------------------------------------
# symlinks: the rename hazard the script rewrite does not cover
# ---------------------------------------------------------------------------


def test_symlinks_into_finds_links_pointing_at_a_renamed_dir(monkeypatch, tmp_path) -> None:
    """`runs/eval_staging/<cell>/iter_N` is a LINK into the run it stages.

    Renaming the run leaves every staged cell dangling, and no amount of
    grep-and-replace over .sh/.py files notices -- which is how the 2026-09-30
    rename broke 38 staged cells before relinking was added.
    """
    monkeypatch.setattr(run_names, "RUNS", tmp_path)
    (tmp_path / "eval_staging" / "cell").mkdir(parents=True)
    planned = {"old_name": "new-name-20260101_000000"}

    # a link into a planned rename, one into an untouched dir, and one into a
    # SIBLING tree (shared FS: opsd-slime-cua-rl has its own runs/)
    (tmp_path / "eval_staging" / "cell" / "iter_0000024").symlink_to(run_names.RUNS / "old_name" / "iter_0000024")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "l").symlink_to(tmp_path / "untouched" / "x")
    (tmp_path / "other" / "s").symlink_to("/tmp/sibling-repo/runs/old_name/iter_1")

    hits = run_names.symlinks_into(planned)
    assert [old for _link, old, _rest in hits] == ["old_name"]
    assert hits[0][2] == "iter_0000024", "the path below the run dir must be preserved"


def test_symlinks_into_ignores_relative_and_resolves_them(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(run_names, "RUNS", tmp_path)
    (tmp_path / "cell").mkdir()
    (tmp_path / "cell" / "iter_1").symlink_to("../old_name/iter_1")
    hits = run_names.symlinks_into({"old_name": "new-name-20260101_000000"})
    assert len(hits) == 1 and hits[0][1] == "old_name"


def test_dangling_into_renamed_is_a_hard_audit_failure(monkeypatch, tmp_path) -> None:
    """A dangling link into a renamed dir means staged cells are unusable."""
    monkeypatch.setattr(run_names, "RUNS", tmp_path)
    monkeypatch.setattr(
        run_names, "_legacy_data", lambda: {"renames": {"gone": {"new": "math-qwen3-1p7b-opsd-20260101_000000"}}, "deferred": {}}
    )
    (tmp_path / "cell").mkdir()
    (tmp_path / "cell" / "iter_1").symlink_to(tmp_path / "gone" / "iter_1")
    dangling = run_names.dangling_into_renamed()
    assert len(dangling) == 1
    assert dangling[0][1] == "gone"


# ---------------------------------------------------------------------------
# the launchers must go through the rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "launcher",
    [
        "train_scripts/run_math_opsd_slime_npu.sh",
        "train_scripts/run_dev_eval.sh",
        "scripts/run_grpo.sh",
    ],
)
def test_launcher_sources_the_model_env_which_owns_the_rule(launcher: str) -> None:
    text = (PROJ / launcher).read_text()
    assert "model_env.sh" in text


@pytest.mark.parametrize(
    "launcher", ["train_scripts/run_math_opsd_slime_npu.sh", "scripts/run_grpo.sh"]
)
def test_launcher_allocates_its_run_name(launcher: str) -> None:
    text = (PROJ / launcher).read_text()
    assert "allocate_run_name" in text, (
        f"{launcher} does not allocate a run name; a run launched from it would be "
        "born off-rule"
    )
    assert "run_names.py" in (PROJ / "train_scripts/lib/model_env.sh").read_text()


@pytest.mark.parametrize(
    "card", sorted(p.stem for p in (PROJ / "configs/models").glob("*.sh"))
)
def test_model_card_carries_no_run_identity(card: str) -> None:
    """A card describes the model. Run identity is the rule's job.

    `MODEL_SAVE_DIR_DEFAULT=".../math_opsd_slime"` living in the 1.7B card is
    precisely how the directory names diverged from the rule.
    """
    text = (PROJ / "configs/models" / f"{card}.sh").read_text()
    for banned in ("MODEL_SAVE_DIR_DEFAULT", "MODEL_GRPO_SAVE_DIR_DEFAULT", "MODEL_LOG_TAG"):
        assert banned not in text, f"{card}.sh still sets {banned}"
    assert 'MODEL_TASK="' in text, f"{card}.sh must declare MODEL_TASK"


def test_allocate_run_name_shell_helper_round_trips() -> None:
    """The shell -> python seam, exercised the way a launcher uses it."""
    script = (
        f'set -uo pipefail; export CUA_PROJ={PROJ}; export MODEL=qwen3-8b;'
        f'source {PROJ}/train_scripts/lib/model_env.sh;'
        'method="$(name_rule_method reward_cv_vnext)" || exit 9;'
        'allocate_run_name "$method" "mbs1-pp4"'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    name = proc.stdout.strip()
    assert re.fullmatch(r"math-qwen3-8b-rewardcv-vnext-mbs1-pp4-\d{8}_\d{6}", name), name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
