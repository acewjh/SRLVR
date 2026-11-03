"""`--advantage-estimator gspo` must reach the surrogate, or say that it did not.

The failure mode this file is written against is the silent one: the launcher
passes a flag that the custom loss never reads, so the run reports GSPO while
computing GRPO. Two separate claims are asserted, and they are not the same claim:

* the branch is REACHED and changes the arithmetic -- whenever the old policy is
  not this forward, the sequence mean of ``old - new`` replaces the per-token
  values, and the loss moves with it;
* when the old policy IS this forward (the v2 launcher's source) the two arms are
  the same step -- and that collapse is detected, warned about and reported, so it
  cannot be mistaken for a working GSPO arm.

CPU-only. The model forward and the context-parallel gather are stubbed out: both
are process-group boundaries, and the gather is the identity at
``--context-parallel-size 1``, the only setting this harness launches with.
"""

import pathlib
from argparse import Namespace

import pytest


@pytest.fixture(autouse=True)
def _no_old_policy_contract_check(monkeypatch):
    """These tests DELIBERATELY pair an old policy with a different current one to
    exercise the surrogate maths, which is exactly the situation
    ``loss.verify_old_policy_contract`` (HSI V2 section 6) asserts against.
    Disable the check here; production runs leave it on."""
    monkeypatch.setenv("ALFWORLD_V2_OLDPOLICY_ASSERT", "0")
import torch

from tasks.alfworld import config, loss as v2_loss

STRIDE = config.TURN_ID_STRIDE

# Per-row log-probs. Every row MIXES token values on purpose: a row that is
# constant would make the sequence mean equal the token values, and this file
# would then pass against the very bug it exists to catch.
NEW_LOG_PROBS = [
    torch.tensor([-1.00, -0.40, -1.60, -0.20], dtype=torch.float32),
    torch.tensor([-0.70, -1.90, -0.30], dtype=torch.float32),
    torch.tensor([-2.20, -0.50, -1.10, -0.90, -0.60], dtype=torch.float32),
]
# The old policy differs from the new one: GSPO and GRPO can only differ if there
# is something to correct for. These are what real rollout log-probs look like --
# the harness's own ``rollout_log_probs`` is a zero placeholder (see below).
OLD_LOG_PROBS = [
    torch.tensor([-0.90, -0.60, -1.30, -0.35], dtype=torch.float32),
    torch.tensor([-0.75, -1.60, -0.55], dtype=torch.float32),
    torch.tensor([-1.80, -0.80, -1.25, -1.05, -0.95], dtype=torch.float32),
]
# Rows 0 and 1 are two turns of one episode; row 2 is a second episode.
TURN_IDS = [[0 * STRIDE + 0], [0 * STRIDE + 1], [1 * STRIDE + 0]]
# Per-token constant within a row, as the v2 rollout writes them.
ADVANTAGES = [[0.8] * 4, [-0.6] * 3, [1.2] * 5]


@pytest.fixture(autouse=True)
def identity_cp_gather(monkeypatch):
    """``all_gather_with_cp`` is the identity at context-parallel size 1."""
    monkeypatch.setattr(v2_loss, "gather_full_sequence", lambda tensor, total_length, response_length: tensor)


def _args(estimator: str, *, use_rollout_logprobs: bool) -> Namespace:
    return Namespace(
        advantage_estimator=estimator,
        use_rollout_logprobs=use_rollout_logprobs,
        eps_clip=0.2,
        eps_clip_high=0.2,
        entropy_coef=0.0,
    )


def _lengths() -> list[int]:
    return [int(row.numel()) for row in NEW_LOG_PROBS]


def _batch(*, real_rollout_logprobs: bool) -> dict:
    """The micro-batch the loss sees, minus what the model forward would produce."""
    lengths = _lengths()
    return {
        "response_lengths": lengths,
        "total_lengths": [length + 8 for length in lengths],
        "unconcat_tokens": [torch.zeros(length + 8, dtype=torch.long) for length in lengths],
        "loss_masks": [torch.ones(length, dtype=torch.long) for length in lengths],
        "turn_ids": TURN_IDS,
        "opsd_advantages": ADVANTAGES,
        # The old policy proper: what slime's pre-train old-actor pass stores
        # (actor.py:476-489). The loss must read THIS, not its own forward --
        # using its own forward makes the PPO ratio identically 1.
        "log_probs": [row.clone() for row in OLD_LOG_PROBS],
        # v2 stores `[0.0] * response_length` here and never sets
        # --use-rollout-logprobs, so this placeholder must never be read.
        "rollout_log_probs": [
            old.clone() if real_rollout_logprobs else torch.zeros(length)
            for old, length in zip(OLD_LOG_PROBS, lengths, strict=True)
        ],
    }


def _stub_model_forward(monkeypatch) -> None:
    """Replace the model forward with the fixed per-row log-probs.

    `slime.backends.megatron_utils` cannot be imported in this process at all
    (its Megatron-Bridge import dies inside a dataclass under pytest), which is
    precisely why `alfworld_v2_loss` reaches it through this seam.
    """
    monkeypatch.setattr(
        v2_loss,
        "log_probs_and_entropy",
        lambda logits, **kwargs: (
            torch.zeros(0),
            {"log_probs": [row.clone().requires_grad_(True) for row in NEW_LOG_PROBS]},
        ),
    )


def _run_loss(monkeypatch, estimator: str, *, real_rollout_logprobs: bool):
    _stub_model_forward(monkeypatch)
    args = _args(estimator, use_rollout_logprobs=real_rollout_logprobs)
    batch = _batch(real_rollout_logprobs=real_rollout_logprobs)
    return v2_loss.alfworld_v2_loss(args, batch, torch.zeros(0), sum_of_sample_mean=None)


def _compute_kl(estimator: str) -> torch.Tensor:
    """The loss's KL stage alone, on one fixed pair of old/new log-probs."""
    lengths = _lengths()
    return v2_loss.compute_ppo_kl(
        _args(estimator, use_rollout_logprobs=True),
        new_log_probs=torch.cat(NEW_LOG_PROBS),
        old_log_probs=torch.cat(OLD_LOG_PROBS),
        new_log_probs_per_sample=[row.clone() for row in NEW_LOG_PROBS],
        old_log_probs_per_sample=[row.clone() for row in OLD_LOG_PROBS],
        loss_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
        total_lengths=[length + 8 for length in lengths],
        response_lengths=lengths,
    )


# --- the branch is reached --------------------------------------------------


def test_gspo_and_grpo_differ_when_the_old_policy_is_not_this_forward(monkeypatch):
    loss_grpo, _ = _run_loss(monkeypatch, "grpo", real_rollout_logprobs=True)
    loss_gspo, metrics_gspo = _run_loss(monkeypatch, "gspo", real_rollout_logprobs=True)

    assert float(metrics_gspo["v2/gspo_sequence_kl"]) == 1.0, "the gspo branch was not taken"
    assert not torch.equal(loss_grpo, loss_gspo), (
        "gspo and grpo returned the same loss: the estimator flag is a no-op"
    )
    assert abs(float(loss_grpo) - float(loss_gspo)) > 1e-6


def test_gspo_kl_is_constant_within_a_row_and_grpo_kl_is_not():
    grpo_kl = _compute_kl("grpo")
    gspo_kl = _compute_kl("gspo")

    offset = 0
    for row, length in enumerate(_lengths()):
        grpo_row = grpo_kl[offset : offset + length]
        gspo_row = gspo_kl[offset : offset + length]
        offset += length

        assert grpo_row.unique().numel() > 1, f"row {row}: the fixture's grpo kl is constant"
        assert gspo_row.unique().numel() == 1, f"row {row}: gspo kl is not one value per sequence"
        # That one value IS the sequence mean, i.e. ratio = exp(-mean kl) is one
        # ratio per sequence rather than exp(-kl_t) per token.
        assert torch.allclose(gspo_row[0], grpo_row.mean(), atol=1e-6)

    assert not torch.allclose(grpo_kl, gspo_kl)


def test_gspo_loss_keeps_a_usable_gradient(monkeypatch):
    """A sequence-level KL that detached itself would train nothing."""
    import slime.utils.ppo_utils as ppo_utils

    _stub_model_forward(monkeypatch)
    captured = {}
    real_compute_policy_loss = ppo_utils.compute_policy_loss

    def _recording_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, *args, **kwargs):
        captured["kl"] = ppo_kl
        return real_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, *args, **kwargs)

    monkeypatch.setattr(ppo_utils, "compute_policy_loss", _recording_compute_policy_loss)
    args = _args("gspo", use_rollout_logprobs=True)
    loss, _ = v2_loss.alfworld_v2_loss(
        args, _batch(real_rollout_logprobs=True), torch.zeros(0), sum_of_sample_mean=None
    )

    assert captured["kl"].requires_grad
    # It is an intermediate, so its gradient only exists if we ask for it.
    captured["kl"].retain_grad()
    loss.backward()
    assert captured["kl"].grad is not None
    assert float(captured["kl"].grad.abs().sum()) > 0.0


# --- the grpo path is untouched ---------------------------------------------


def test_grpo_kl_is_exactly_the_token_level_difference():
    assert torch.equal(_compute_kl("grpo"), torch.cat(OLD_LOG_PROBS) - torch.cat(NEW_LOG_PROBS))


def test_missing_estimator_attribute_means_token_level():
    """Every arm launched before this flag existed has no attribute at all."""
    lengths = _lengths()
    args = Namespace(use_rollout_logprobs=False, eps_clip=0.2, eps_clip_high=0.2, entropy_coef=0.0)

    assert not v2_loss.is_gspo(args)
    assert v2_loss.estimator_name(args) == "grpo"
    kl = v2_loss.compute_ppo_kl(
        args,
        new_log_probs=torch.cat(NEW_LOG_PROBS),
        old_log_probs=torch.cat(OLD_LOG_PROBS),
        new_log_probs_per_sample=[row.clone() for row in NEW_LOG_PROBS],
        old_log_probs_per_sample=[row.clone() for row in OLD_LOG_PROBS],
        loss_masks=[torch.ones(length, dtype=torch.long) for length in lengths],
        total_lengths=[length + 8 for length in lengths],
        response_lengths=lengths,
    )
    assert torch.equal(kl, torch.cat(OLD_LOG_PROBS) - torch.cat(NEW_LOG_PROBS))


def test_grpo_ignores_the_zero_rollout_placeholder():
    """`old` is the stored old-actor pass -- neither the placeholder NOR the
    current forward.

    The second half is the old-policy fix: using this forward as `old` makes the
    ratio identically 1 (this test used to assert exactly that, which is why the
    bug survived a whole campaign).
    """
    batch = _batch(real_rollout_logprobs=False)
    args = _args("grpo", use_rollout_logprobs=False)
    per_sample, concatenated = v2_loss.resolve_old_log_probs(args, batch, NEW_LOG_PROBS)

    assert torch.equal(concatenated, torch.cat(OLD_LOG_PROBS))
    assert not torch.equal(concatenated, torch.cat(NEW_LOG_PROBS)), "old must not be this forward"
    assert not torch.equal(concatenated, torch.cat(batch["rollout_log_probs"]))
    assert all(not row.requires_grad for row in per_sample)


# --- the collapse the launch config causes is reported, not silent ----------


def test_a_gspo_step_no_longer_collapses_onto_grpo(monkeypatch):
    """The old-policy fix invalidates the previous GSPO claims, on purpose.

    This test used to pin the collapse: with `old` taken from the loss's own
    forward, ``ppo_kl`` was zero on both branches and the two arms were the same
    numerical step -- so every GSPO number was a GRPO number wearing a label. Its
    docstring said the test would fail once the arm got a real old policy, and
    that re-deriving the claims was the point. That is what happened.

    What must now hold: with a real old policy the two surrogates genuinely differ,
    and neither reports the degenerate KL.
    """
    monkeypatch.setattr(v2_loss, "_DEGENERATE_SEQUENCE_KL_WARNED", False)
    loss_grpo, metrics_grpo = _run_loss(monkeypatch, "grpo", real_rollout_logprobs=False)
    loss_gspo, metrics_gspo = _run_loss(monkeypatch, "gspo", real_rollout_logprobs=False)

    assert not torch.equal(loss_grpo, loss_gspo), (
        "GRPO and GSPO must differ now that `old` is the stored old-actor pass"
    )
    assert float(metrics_gspo["v2/gspo_degenerate_kl"]) == 0.0
    assert float(metrics_grpo["v2/gspo_degenerate_kl"]) == 0.0


def test_the_collapse_warning_is_raised_once(monkeypatch, caplog):
    monkeypatch.setattr(v2_loss, "_DEGENERATE_SEQUENCE_KL_WARNED", False)
    args = _args("gspo", use_rollout_logprobs=False)
    log_probs = torch.cat(NEW_LOG_PROBS)

    with caplog.at_level("WARNING", logger=v2_loss.logger.name):
        assert v2_loss.detect_degenerate_sequence_kl(args, log_probs, log_probs)
        assert v2_loss.detect_degenerate_sequence_kl(args, log_probs, log_probs)

    warnings = [record for record in caplog.records if "numerically GRPO" in record.getMessage()]
    assert len(warnings) == 1, "the collapse is stated once per process, not per step"


def test_a_real_old_policy_is_not_reported_as_degenerate(monkeypatch):
    monkeypatch.setattr(v2_loss, "_DEGENERATE_SEQUENCE_KL_WARNED", False)
    log_probs = torch.cat(NEW_LOG_PROBS)

    assert not v2_loss.detect_degenerate_sequence_kl(
        _args("gspo", use_rollout_logprobs=True), torch.cat(OLD_LOG_PROBS), log_probs
    )
    # The token-level estimators read the same source by design and claim nothing else.
    assert not v2_loss.detect_degenerate_sequence_kl(
        _args("grpo", use_rollout_logprobs=False), log_probs, log_probs
    )


# --- the launcher actually passes the flag ----------------------------------


def _registry_fragment(root, method):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "opsd_core.registry",
         "--task", "alfworld", "--method", method, "--shell"],
        capture_output=True, text=True, cwd=str(root), check=True,
    ).stdout


def test_launcher_passes_the_estimator_to_slime():
    """A loss-side branch is worthless if the launcher never sets the flag.

    The arm switch moved out of this script and into ``opsd_core/registry.py``
    (REFACTOR_PLAN.md §4), so the `gspo` override is asserted against the registry
    fragment the launcher evals — not against a literal that used to sit here.
    """
    # parents[2] is the repo root: it is both the launcher's home after the
    # 2026-10-15 move (tasks/alfworld/scripts/) and the cwd the registry
    # subprocess below needs in order to import opsd_core.
    root = pathlib.Path(v2_loss.__file__).resolve().parents[2]
    script = (root / "tasks" / "alfworld" / "scripts" / "_common_v2.sh").read_text()

    assert 'ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-grpo}"' in script
    assert '--advantage-estimator "$ADVANTAGE_ESTIMATOR"' in script

    # Exactly one arm overrides the default, and it is `gspo`.
    assert "export ADVANTAGE_ESTIMATOR=gspo" in _registry_fragment(root, "gspo")
    assert "ADVANTAGE_ESTIMATOR=gspo" not in _registry_fragment(root, "grpo")
