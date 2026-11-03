"""Acceptance tests for the opt-in ALFWorld action-repair harness.

The repair sits after parsing and before ``env.step``. It may only:
  * translate a command form to an exact admissible command,
  * navigate to the receptacle named by the model's command,
  * or leave the action invalid.

It must never choose an object or guess a receptacle that the model did not
name, and it must never pull a command out of multi-command/prose output.
"""
import pytest

from tasks.alfworld.action_harness import (
    HARNESS_VERSION,
    parse_action,
    resolve_harness_mode,
)
from tasks.alfworld.action_repair import repair_action


def test_default_mode_is_v2_form_and_v2_is_still_selectable(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    assert resolve_harness_mode() == "v2-form"
    assert resolve_harness_mode("command-normalization-v2") == "v2"
    assert resolve_harness_mode("command-normalization-v1") == "v1"
    assert resolve_harness_mode("v2-form") == "v2-form"
    assert HARNESS_VERSION == "command-normalization-v2-form"
    with pytest.raises(ValueError):
        resolve_harness_mode("not-a-harness")


def test_default_does_not_navigate(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    adm = ["go to fridge 1", "look"]
    assert not parse_action("<action>open fridge 1</action>", adm).parsed.is_valid
    assert parse_action("<action>open fridge 1</action>", adm, repair_mode="v2").parsed.is_valid


@pytest.mark.parametrize("raw", [
    "examine cabinet 2",
    "inspect cabinet 2",
    "look at fridge 1",
    "look in fridge 1",
    "look fridge 1",
    "look around",
])
def test_look_family_maps_to_the_bare_look_action(raw):
    result = repair_action(raw, ["look", "go to cabinet 2"])
    assert result.status == "repaired"
    assert result.action == "look"
    assert result.rule == "look_family"


def test_move_to_put_only_when_exactly_one_form_is_admissible():
    result = repair_action("move cup 1 to coffeemachine 1", ["put cup 1 in coffeemachine 1"])
    assert result.status == "repaired"
    assert result.action == "put cup 1 in coffeemachine 1"
    both = repair_action("move cup 1 to coffeemachine 1",
                         ["put cup 1 in coffeemachine 1", "put cup 1 on coffeemachine 1"])
    assert both.status == "ambiguous"
    assert both.action is None


def test_form_translations():
    assert repair_action("grab apple 1 from shelf 1", ["take apple 1 from shelf 1"]).action == \
        "take apple 1 from shelf 1"
    assert repair_action("go kitchen 1", ["go to kitchen 1"]).action == "go to kitchen 1"
    assert repair_action("wash apple 1 with sinkbasin 1", ["clean apple 1 with sinkbasin 1"]).action == \
        "clean apple 1 with sinkbasin 1"
    assert repair_action("check inventory", ["inventory"]).action == "inventory"
    assert repair_action("open the drawer 1", ["open drawer 1"]).action == "open drawer 1"


def test_precondition_navigation_requires_full_mode():
    adm = ["go to fridge 1", "look"]
    full = repair_action("open fridge 1", adm, allow_precondition=True)
    assert full.status == "repaired"
    assert full.action == "go to fridge 1"
    assert full.rule == "navigate_precondition"
    form_only = repair_action("open fridge 1", adm, allow_precondition=False)
    assert form_only.status == "unrepaired"


def test_precondition_navigation_take_example():
    result = repair_action("take desklamp from desk 1", ["go to desk 1", "look"],
                           allow_precondition=True)
    assert result.status == "repaired"
    assert result.action == "go to desk 1"


def test_never_guesses_object_or_receptacle():
    # The named receptacle is not reachable, and another receptacle holding the
    # same object is not a licence to rewrite the model's command.
    assert repair_action("take mug 1 from cabinet 2",
                         ["take mug 1 from cabinet 1", "look"]).status == "unrepaired"
    assert repair_action("open desklamp 1", ["look"]).status == "unrepaired"
    assert repair_action("go to shelf 2", ["go to shelf 1", "look"]).status == "unrepaired"


def test_multi_command_and_prose_are_never_repaired():
    for raw in ("look; inventory", "look\ninventory", "<action>look</action>",
                "`look`", "I will look around."):
        assert repair_action(raw, ["look", "inventory"]).status == "unrepaired"


def test_empty_admissible_is_never_repaired():
    assert repair_action("examine fridge 1", []).status == "unrepaired"


def test_parse_action_repairs_structured_invalid_by_default(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    result = parse_action("<action>examine fridge 1</action>", ["look"])
    assert result.parsed.is_valid
    assert result.parsed.action == "look"
    assert result.source.startswith("repaired:")


def test_parse_action_v1_keeps_structured_invalid(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    result = parse_action("<action>examine fridge 1</action>", ["look"], repair_mode="v1")
    assert not result.parsed.is_valid
    assert result.parsed.status.value == "not_admissible"


def test_parse_action_v1_has_no_precondition_navigation(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    adm = ["go to fridge 1", "look"]
    assert not parse_action("<action>open fridge 1</action>", adm, repair_mode="v1").parsed.is_valid
    assert parse_action("<action>open fridge 1</action>", adm, repair_mode="v2").parsed.is_valid


def test_parse_action_does_not_repair_truncated_or_multi_command(monkeypatch):
    monkeypatch.delenv("ALFWORLD_ACTION_HARNESS", raising=False)
    assert not parse_action("<action>look at fridge", ["look"], truncated=True).parsed.is_valid
    assert not parse_action("<action>look; inventory</action>", ["look", "inventory"]).parsed.is_valid
