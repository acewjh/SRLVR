"""ALFWorld environment, dataset discovery and eval aggregation tests.

The environment tests are marked ``alfworld`` and skip cleanly when the overlay
venv or the game data is absent, so this file still contributes a signal (dataset
plumbing, eval maths) on a bare machine.
"""

import os

import pytest

from tasks.alfworld.v1.eval import CATEGORY_ORDER, EvalSample, aggregate, format_report
from tasks.alfworld.v1.paths import game_data_root

OVERLAY = os.environ.get("ALFWORLD_OVERLAY_SITE") or None


def _alfworld_ready() -> bool:
    try:
        from tasks.alfworld.v1.paths import ensure_alfworld_importable

        ensure_alfworld_importable()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not ready", not "test error"
        return False


def _data_ready(split: str) -> bool:
    return os.path.isdir(os.path.join(game_data_root(), split))


requires_alfworld = pytest.mark.skipif(
    not _alfworld_ready(), reason="ALFWorld overlay packages are not importable"
)
requires_train = pytest.mark.skipif(not _data_ready("train"), reason="ALFWorld train split missing")
requires_seen = pytest.mark.skipif(
    not _data_ready("valid_seen"), reason="ALFWorld valid_seen split missing"
)
requires_unseen = pytest.mark.skipif(
    not _data_ready("valid_unseen"), reason="ALFWorld valid_unseen split missing"
)


# ---------------------------------------------------------------------------
# 1. import isolation
# ---------------------------------------------------------------------------


@requires_alfworld
def test_alfworld_and_textworld_import():
    import alfworld  # noqa: F401
    import textworld

    assert textworld.__version__


def _site_packages_overlap() -> tuple[set[str], set[str]]:
    """Top-level package names present in BOTH the overlay and the base env."""
    from tasks.alfworld.v1.paths import DEFAULT_OVERLAY_SITE

    overlay = os.environ.get("ALFWORLD_OVERLAY_SITE", DEFAULT_OVERLAY_SITE)
    base = os.environ.get("CUA_RL_SITE_PACKAGES")
    if not base or not os.path.isdir(base) or not os.path.isdir(overlay):
        pytest.skip("both site-packages directories are needed for this check")

    def names(path):
        return {
            entry
            for entry in os.listdir(path)
            if not entry.endswith((".dist-info", ".egg-info", ".pth", ".so"))
            and entry != "__pycache__"
            and os.path.isdir(os.path.join(path, entry))
        }

    return names(base), names(overlay)


@requires_alfworld
def test_overlay_shares_no_module_names_with_the_base_environment():
    """The isolation argument in one assertion.

    The overlay is APPENDED to ``sys.path`` on the training interpreter, so it can
    only supply modules cua-rl is missing -- *provided* it does not contain a
    package cua-rl already has. If this set ever grows beyond pip/setuptools, the
    append is no longer sufficient and the shadowing risk becomes real.
    """
    base, overlay = _site_packages_overlap()
    shared = base & overlay
    allowed = {"pip", "setuptools", "pkg_resources", "_distutils_hack", "wheel"}
    assert shared <= allowed, (
        f"overlay package names collide with cua-rl: {sorted(shared - allowed)}; "
        "the sys.path append can no longer guarantee cua-rl's copies win"
    )


@requires_alfworld
def test_overlay_is_appended_not_prepended_under_the_training_interpreter():
    """Under cua-rl the overlay must sit AFTER the base env on ``sys.path``.

    Skipped when this suite runs under the overlay interpreter itself, where the
    overlay IS the primary site-packages and the question does not apply.
    """
    import sys

    from tasks.alfworld.v1.paths import DEFAULT_OVERLAY_SITE

    overlay = os.environ.get("ALFWORLD_OVERLAY_SITE", DEFAULT_OVERLAY_SITE)
    # site-packages -> pythonX.Y -> lib -> <venv root>
    venv_root = os.path.dirname(os.path.dirname(os.path.dirname(overlay)))
    if os.path.realpath(sys.prefix) == os.path.realpath(venv_root):
        pytest.skip("running under the overlay interpreter; append ordering is not applicable")

    base = os.environ.get("CUA_RL_SITE_PACKAGES")
    if not base or not os.path.isdir(base):
        pytest.skip("CUA_RL_SITE_PACKAGES not set")
    overlay_index = next((i for i, p in enumerate(sys.path) if p == overlay), None)
    base_index = next((i for i, p in enumerate(sys.path) if p == base), None)
    if overlay_index is None:
        pytest.skip("overlay not on sys.path for this interpreter")
    assert base_index is not None and overlay_index > base_index, (
        f"overlay at {overlay_index} must come after base at {base_index}"
    )


# ---------------------------------------------------------------------------
# 2. dataset / game file discovery
# ---------------------------------------------------------------------------


@requires_alfworld
@requires_train
def test_discovers_train_games():
    from tasks.alfworld.v1.env_adapter import discover_games

    games = discover_games("train", limit=5)
    assert len(games) == 5
    for game in games:
        assert game.path.endswith("game.tw-pddl")
        assert os.path.exists(game.path)
        assert game.split == "train"
        assert game.task_type in {
            "pick_and_place_simple",
            "look_at_obj_in_light",
            "pick_clean_then_place_in_recep",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_two_obj_and_place",
        }


@requires_alfworld
@requires_train
def test_discovery_is_deterministic():
    from tasks.alfworld.v1.env_adapter import discover_games

    first = [g.game_id for g in discover_games("train", limit=10)]
    second = [g.game_id for g in discover_games("train", limit=10)]
    assert first == second


@requires_alfworld
def test_discovery_rejects_an_unknown_split():
    from tasks.alfworld.v1.env_adapter import discover_games

    with pytest.raises(FileNotFoundError, match="not found"):
        discover_games("no_such_split")


@requires_alfworld
def test_task_type_parsing_prefers_the_longest_prefix():
    from tasks.alfworld.v1.env_adapter import task_type_from_game_id

    assert (
        task_type_from_game_id("pick_clean_then_place_in_recep-Mug-None-CounterTop-1")
        == "pick_clean_then_place_in_recep"
    )
    assert task_type_from_game_id("pick_and_place_simple-Watch-None-Table-2") == "pick_and_place_simple"


@requires_alfworld
def test_task_description_extraction():
    from tasks.alfworld.v1.env_adapter import extract_task_description

    observation = (
        "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room.\n\n"
        "Your task is to: put a cool tomato in microwave."
    )
    assert extract_task_description(observation) == "put a cool tomato in microwave."
    assert extract_task_description("no objective here") == ""


# ---------------------------------------------------------------------------
# 3 + 4. env.reset / env.step
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def seen_env():
    from tasks.alfworld.v1.env_adapter import AlfWorldEnv, discover_games

    games = discover_games("valid_seen", limit=1)
    if not games:
        pytest.skip("no valid_seen games available")
    env = AlfWorldEnv(games[0], max_steps=8)
    yield env
    env.close()


@requires_alfworld
@requires_seen
def test_reset_returns_observation_actions_and_task(seen_env):
    result = seen_env.reset()
    assert result.observation
    assert result.admissible_actions
    assert result.done is False
    assert seen_env.task_description


@requires_alfworld
@requires_seen
def test_step_with_a_valid_action_advances_the_environment(seen_env):
    start = seen_env.reset()
    action = start.admissible_actions[0]
    result = seen_env.step(action)
    assert result.observation
    assert isinstance(result.done, bool)
    # the won flag must be a real boolean, not a truthy one-element list
    assert result.won in (True, False)


@requires_alfworld
@requires_seen
def test_step_does_not_report_won_on_a_non_terminal_move(seen_env):
    """Regression: textworld reports `won` as a batch-of-one list.

    Reading it without unwrapping makes `bool([False])` truthy and every episode
    look successful -- which would have made the whole eval meaningless.
    """
    start = seen_env.reset()
    result = seen_env.step(start.admissible_actions[0])
    if not result.done:
        assert result.won is False


# ---------------------------------------------------------------------------
# 5 + 6. seen / unseen samples
# ---------------------------------------------------------------------------


@requires_alfworld
@requires_seen
def test_valid_seen_sample_is_loadable():
    from tasks.alfworld.v1.env_adapter import AlfWorldEnv, discover_games

    games = discover_games("valid_seen", limit=1)
    with AlfWorldEnv(games[0]) as env:
        result = env.reset()
    assert result.admissible_actions


@requires_alfworld
@requires_unseen
def test_valid_unseen_sample_is_loadable():
    from tasks.alfworld.v1.env_adapter import AlfWorldEnv, discover_games

    games = discover_games("valid_unseen", limit=1)
    assert games[0].split == "valid_unseen"
    with AlfWorldEnv(games[0]) as env:
        result = env.reset()
    assert result.admissible_actions


# ---------------------------------------------------------------------------
# 18. eval aggregation
# ---------------------------------------------------------------------------


def _sample(success, turns, split, task_type, truncated=False, invalid=0):
    return EvalSample(
        task_id="t",
        task_type=task_type,
        split=split,
        success=success,
        num_turns=turns,
        truncated=truncated,
        invalid_action_count=invalid,
    )


def test_aggregate_computes_overall_and_split_success_rates():
    report = aggregate(
        [
            _sample(True, 5, "valid_seen", "pick_and_place_simple"),
            _sample(False, 7, "valid_seen", "pick_and_place_simple"),
            _sample(True, 9, "valid_unseen", "look_at_obj_in_light"),
        ]
    )
    payload = report.as_dict()
    assert payload["overall"]["n"] == 3
    assert payload["overall"]["success_rate"] == pytest.approx(2 / 3)
    assert payload["seen_success_rate"] == pytest.approx(0.5)
    assert payload["unseen_success_rate"] == pytest.approx(1.0)


def test_aggregate_reports_per_category_success():
    report = aggregate(
        [
            _sample(True, 5, "valid_seen", "pick_clean_then_place_in_recep"),
            _sample(False, 5, "valid_seen", "pick_clean_then_place_in_recep"),
            _sample(True, 5, "valid_seen", "pick_two_obj_and_place"),
        ]
    )
    categories = report.as_dict()["overall"]["by_category"]
    assert categories["Clean"] == {"n": 2, "success_rate": 0.5}
    assert categories["Pick2"] == {"n": 1, "success_rate": 1.0}


def test_aggregate_reports_turns_truncation_and_invalid_actions():
    report = aggregate(
        [
            _sample(False, 10, "train", "pick_and_place_simple", truncated=True, invalid=2),
            _sample(True, 20, "train", "pick_and_place_simple", truncated=False, invalid=0),
        ]
    )
    overall = report.as_dict()["overall"]
    assert overall["mean_turns"] == pytest.approx(15.0)
    assert overall["truncation_rate"] == pytest.approx(0.5)
    # invalid actions are normalised by turns, not by episodes
    assert overall["invalid_action_rate"] == pytest.approx(2 / 30)


def test_empty_report_is_well_defined():
    payload = aggregate([]).as_dict()
    assert payload["overall"]["n"] == 0
    assert payload["overall"]["success_rate"] == 0.0


def test_unknown_task_type_passes_through_as_its_own_category():
    report = aggregate([_sample(True, 3, "train", "brand_new_task")])
    assert "brand_new_task" in report.as_dict()["overall"]["by_category"]


def test_format_report_mentions_every_known_category():
    report = aggregate([_sample(True, 4, "valid_seen", "pick_heat_then_place_in_recep")])
    rendered = format_report(report)
    assert "success_rate" in rendered
    assert "Heat" in rendered
    # only categories with data are listed; the six official labels are the keys
    assert set(CATEGORY_ORDER) >= {"Pick", "Look", "Clean", "Heat", "Cool", "Pick2"}


# ---------------------------------------------------------------------------
# Concurrency: TextWorld is single-threaded, and the rollout builds episodes from
# threads. This is the regression test for the bug that killed the first two NPU
# runs: `textworld.envs.pddl.logic._PARSER` is a module-level tatsu parser whose
# parse state is shared, so concurrent parses raise `FailedToken` at positions
# that move run to run. `env_adapter._TEXTWORLD_LOCK` serialises it.
#
# Removing that lock makes this test fail -- which is the only way it is useful,
# because a single-threaded probe never reproduces the race.
# ---------------------------------------------------------------------------


@requires_alfworld
@requires_seen
def test_concurrent_env_construction_is_safe():
    import threading

    from tasks.alfworld.v1.env_adapter import AlfWorldEnv, discover_games

    games = discover_games("valid_seen", limit=6)
    outcomes: list[str | None] = [None] * len(games)

    def run(game, index):
        try:
            with AlfWorldEnv(game, max_steps=4) as env:
                first = env.reset()
                env.step(first.admissible_actions[0])
            outcomes[index] = "OK"
        except Exception as exc:  # noqa: BLE001 - the point is to surface any failure
            outcomes[index] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=run, args=(g, i)) for i, g in enumerate(games)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failures = [o for o in outcomes if o != "OK"]
    assert not failures, f"concurrent env use corrupted TextWorld state: {failures}"
