"""Contract tests for the L3 entry point (REFACTOR_PLAN.md §4, §6, R13).

The registry replaced three independently-maintained method vocabularies. These
tests pin the properties that made the old duplication dangerous:

* an unknown or retired method must FAIL CLOSED, never fall through to a default;
* the emitted shell fragment must be valid bash and must let an explicitly
  exported variable win (the launcher's historical override semantics);
* ``run_names.py``'s live vocabulary must agree with the registry's.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from opsd_core import registry

REPO = __file__.rsplit("/tests/", 1)[0]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "opsd_core.registry", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )


# --------------------------------------------------------------------------- #
# fail-closed
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    ("task", "method"),
    [
        ("math", "reward_cv"),  # retired
        ("math", "reward_cv_vnext"),  # retired
        ("alfworld", "hpi_future"),  # retired 2026-10-15
        ("alfworld", "hsi_v2"),  # retired 2026-10-15
        ("alfworld", "hsi_candidate"),  # retired 2026-10-15
        ("math", "no_such_method"),
        ("no_such_task", "opsd"),
        ("alfworld", "opsd"),  # valid method, wrong task
    ],
)
def test_unknown_selection_raises(task, method):
    # Act / Assert
    with pytest.raises(registry.UnknownSelection):
        registry.resolve(task, method)


@pytest.mark.unit
def test_cli_fails_closed_without_leaking_a_config():
    # Act
    result = _run("--task", "math", "--method", "reward_cv")

    # Assert — rc=2 and no KEY=value line that a caller could source
    assert result.returncode == 2
    assert "CUSTOM_LOSS_FN=" not in result.stdout
    assert "retired" in result.stderr


@pytest.mark.unit
def test_shell_emitter_refuses_unsafe_values():
    with pytest.raises(ValueError):
        registry._safe("x; rm -rf /")
    with pytest.raises(ValueError):
        registry._safe("$(whoami)")


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_every_task_has_opsd_or_a_documented_alternative():
    for task in registry.TASKS:
        methods = registry.known_methods(task)
        assert methods, f"{task} has no live methods"
        assert "hsi" in methods


@pytest.mark.unit
def test_hsi_is_shared_across_all_three_tasks():
    # The whole point of the refactor: one objective, three task adapters.
    loss_paths = {registry.resolve(task, "hsi")[1].loss for task in registry.TASKS}
    assert loss_paths == {"tasks.alfworld.loss.alfworld_v2_loss"}


@pytest.mark.unit
def test_full_softmax_arms_omit_the_sampler_flags():
    for task in registry.TASKS:
        for method in registry.known_methods(task):
            spec = registry.resolve(task, method)[1]
            if spec.sampler == registry.SAMPLER_FULL_SOFTMAX:
                assert spec.omit_sampler_flags, f"{task}/{method} wants full softmax but sets flags"


@pytest.mark.unit
@pytest.mark.parametrize("method", ["hpi_future", "hsi_v2", "hsi_candidate"])
def test_the_retired_alfworld_credit_arms_fail_closed(method):
    """The paper reports HP-RLVR (HSI) vs GRPO only; these three earlier credit
    arms were retired 2026-10-15 and must be refused, not silently redirected."""
    assert method in registry.RETIRED_METHODS
    with pytest.raises(registry.UnknownSelection):
        registry.resolve("alfworld", method)


@pytest.mark.unit
def test_alfworld_arm_hsi_is_the_canonical_hsi():
    """`hsi` now means the same objective on every task (it used to resolve to the
    retired top-128 candidate arm). `hsi_v3` -- the launcher's historic name for
    our method -- aliases to it."""
    canonical = registry.resolve("alfworld", "hsi")[1]

    assert canonical.postprocess == "tasks.alfworld.hsi2_jsd.postprocess"
    # ...and `hsi_v3` (the launcher's name for our method) aliases to canonical.
    assert registry.resolve("alfworld", "hsi_v3")[1] is canonical


@pytest.mark.unit
def test_hsi_arms_pin_g_and_step_size():
    """The HSI arms collapse G to 1 and hold the optimizer-step size at 32 rows."""
    for task in ("alfworld", "code"):
        spec = registry.resolve(task, "hsi")[1]
        assert spec.n_samples_per_prompt == 1, task
    assert registry.resolve("alfworld", "hsi")[1].gbs_override == 32
    # code computes GLOB_BS from HSI_STEP_PROMPTS x OPSD_HSI_CHUNKS in the launcher
    assert registry.resolve("code", "hsi")[1].gbs_override is None
    assert dict(registry.resolve("code", "hsi")[1].extra_env)["HSI_STEP_PROMPTS"] == "8"


@pytest.mark.unit
def test_gigpo_is_the_grpo_recipe_plus_the_step_term():
    """GiGPO shares every GRPO knob; only the advantage estimator changes.

    The matched-control claim in the campaign depends on this being true at the
    registry level: same loss, same rollout, same convert, same G, and HPI off.
    """
    gigpo = registry.resolve("alfworld", "gigpo")[1]
    grpo = registry.resolve("alfworld", "grpo")[1]

    assert gigpo.loss == grpo.loss
    assert gigpo.rollout == grpo.rollout
    assert gigpo.convert == grpo.convert
    assert gigpo.postprocess is None
    assert gigpo.n_samples_per_prompt == grpo.n_samples_per_prompt == 8
    env = dict(gigpo.extra_env)
    assert env["ALFWORLD_V2_GIGPO"] == "1"
    assert env["ALFWORLD_V2_HPI"] == "0"


@pytest.mark.unit
def test_alfworld_arms_force_their_env_and_math_does_not():
    """ALFWorld's launcher assigns arm env unconditionally; math defaults it."""
    assert registry.resolve("alfworld", "gspo")[1].force_env is True
    assert registry.resolve("math", "hsi")[1].force_env is False


@pytest.mark.unit
def test_shell_fragment_emits_forced_env_without_a_default_guard():
    # Act
    result = _run("--task", "alfworld", "--method", "gspo", "--shell")

    # Assert — a preset must NOT win over the arm for a forced variable
    assert result.returncode == 0
    assert 'export ADVANTAGE_ESTIMATOR=gspo' in result.stdout
    assert 'G=8' in result.stdout


@pytest.mark.unit
def test_shell_fragment_is_valid_bash_and_respects_overrides(tmp_path):
    # Arrange
    result = _run("--task", "math", "--method", "hsi", "--shell")
    assert result.returncode == 0
    fragment = tmp_path / "frag.sh"
    fragment.write_text(result.stdout)

    # Act
    script = (
        f"set -e\n. {fragment}\n"
        'echo "$CUSTOM_LOSS_FN|$OPSD_HSI_CHUNKS|${EXTRA_ARGS[*]}|$NUM_ROLLOUT"\n'
    )
    default = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    overridden = subprocess.run(
        ["bash", "-c", f"export CUSTOM_LOSS_FN=MY_OVERRIDE OPSD_HSI_CHUNKS=8\n{script}"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Assert
    assert default.returncode == 0, default.stderr
    assert default.stdout.strip() == (
        "tasks.alfworld.loss.alfworld_v2_loss|4|--use-opsd --use-rollout-logprobs|100"
    )
    assert overridden.returncode == 0, overridden.stderr
    assert overridden.stdout.startswith("MY_OVERRIDE|8|")
