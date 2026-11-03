"""PPO's old policy must come from the stored theta_old pass, not this forward.

The bug: `resolve_old_log_probs` returned `log_probs.detach()` -- the loss's own
forward. New and old were then the same tensor, so `ratio == 1` exactly, `ppo_kl`
and `pg_clipfrac` were 0 on every single step of whole runs, and PPO's clipping
branch could never execute. The objective silently became a plain policy gradient
with no trust region.

slime already computes the right thing: `actor.py:476-489` runs a log-prob pass
before any optimizer step (on `old_actor` weights when `--keep-old-actor`, else on
the not-yet-updated actor) and stores it as `rollout_data["log_probs"]`. These
tests pin that this is what the loss consumes, and that the three behaviours the
fix spec asks for actually hold.
"""

import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

PROJ = str(Path(__file__).resolve().parents[3])
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from tasks.alfworld.loss import resolve_old_log_probs  # noqa: E402
from slime.utils.ppo_utils import compute_policy_loss  # noqa: E402


def _args(**over):
    base = {"use_rollout_logprobs": False, "eps_clip": 0.2, "eps_clip_high": 0.2}
    base.update(over)
    return Namespace(**base)


def test_missing_stored_old_log_probs_is_an_error_not_a_ratio_of_one():
    # Arrange: a batch with no `log_probs` key. Silently falling back to this
    # forward is exactly the bug, so it must be loud.
    batch = {}

    # Act / Assert
    with pytest.raises(ValueError, match="old-policy log-probs"):
        resolve_old_log_probs(_args(), batch, [torch.zeros(4)])


def test_old_log_probs_come_from_the_stored_old_actor_pass():
    # Arrange: the stored pass and the current forward deliberately differ, so
    # which one is used is observable.
    stored = [torch.tensor([-1.0, -2.0, -3.0, -4.0])]
    current = [torch.tensor([-9.0, -9.0, -9.0, -9.0])]

    # Act
    per_sample, flat = resolve_old_log_probs(_args(), {"log_probs": stored}, current)

    # Assert
    assert torch.equal(per_sample[0], stored[0])
    assert torch.equal(flat, stored[0])
    assert not torch.equal(flat, current[0])


def test_rollout_logprobs_win_when_the_flag_is_set():
    # Arrange
    batch = {
        "log_probs": [torch.tensor([-1.0, -1.0])],
        "rollout_log_probs": [torch.tensor([-5.0, -5.0])],
    }

    # Act
    _, flat = resolve_old_log_probs(_args(use_rollout_logprobs=True), batch, [torch.zeros(2)])

    # Assert
    assert torch.equal(flat, torch.tensor([-5.0, -5.0]))


def test_old_policy_startup_gate_allows_later_policy_drift(monkeypatch):
    from tasks.alfworld import loss

    monkeypatch.setenv("ALFWORLD_V2_OLDPOLICY_ASSERT", "1")
    monkeypatch.setattr(loss, "_OLD_POLICY_CHECKED", False)
    monkeypatch.setattr(loss, "_OLD_POLICY_FIRST_WINDOW_DONE", False)
    monkeypatch.setattr(loss, "_OLD_POLICY_BIAS_WINDOW", [])
    monkeypatch.setattr(loss, "_OLD_POLICY_WINDOW", 2)
    old = torch.tensor([-2.0, -3.0])
    # A large mismatch before certification still fails.
    with pytest.raises(AssertionError, match="old-policy contract violated"):
        loss.verify_old_policy_contract(_args(), old + 1.0, old)
    loss.verify_old_policy_contract(_args(), old.clone(), old)
    loss.verify_old_policy_contract(_args(), old.clone(), old)
    # After certification this is a PPO ratio, not a sampling convention check.
    assert loss.verify_old_policy_contract(_args(), old + 1.0, old) == pytest.approx(1.0)


def test_identical_new_and_old_gives_ratio_one_and_no_clipping():
    # Arrange: spec check 1 -- when the policy has not moved, ratio == 1 and the
    # clipping branch does not fire. (This is the case the old code forced ALWAYS.)
    new = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    old = new.clone()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

    # Act
    ppo_kl = old - new
    _loss, clipfrac = compute_policy_loss(ppo_kl, adv, 0.2, 0.2)

    # Assert
    assert torch.allclose(torch.exp(-ppo_kl), torch.ones_like(ppo_kl))
    assert float(clipfrac.mean()) == 0.0


def test_a_changed_policy_makes_the_ratio_stop_being_one():
    # Arrange: spec check 2.
    new = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    old = new + torch.tensor([0.05, -0.05, 0.0, 0.1])
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

    # Act
    ratio = torch.exp(resolve_old_log_probs(_args(), {"log_probs": [old]}, [new])[1] - new)

    # Assert
    assert not torch.allclose(ratio, torch.ones_like(ratio))
    _ = adv


def test_the_clipped_branch_executes_when_the_ratio_leaves_the_range():
    # Arrange: spec check 3.
    #
    # Direction matters, and slime's convention is the pessimistic one
    # (`ppo_utils.py:132-136`): ratio = exp(-ppo_kl), the loss keeps
    # `max(-ratio*adv, -clamp(ratio)*adv)`, and clipfrac counts only where the
    # CLAMPED term is the selected one. With adv > 0 that happens when the ratio
    # rises ABOVE 1+eps, i.e. when new > old -- so the log-prob has to go UP, not
    # down. (Getting this backwards is the easy mistake; this test made it first.)
    eps = 0.2
    old = torch.tensor([0.0, 0.0])
    # ppo_kl = old - new = -1.0 -> ratio = e ~ 2.72, far above 1 + eps.
    new = torch.tensor([1.0, 0.0])
    adv = torch.tensor([1.0, 1.0])

    # Act
    ppo_kl = old - new
    _loss, clipfrac = compute_policy_loss(ppo_kl, adv, eps, eps)

    # Assert
    assert float(clipfrac.mean()) > 0.0, "clipping branch never executed for a ratio of e"
    assert float(clipfrac.mean()) == pytest.approx(0.5, abs=1e-6)

    # The mirror image, which is NOT clipping: with adv > 0 the unclipped term is
    # already the pessimistic choice when the ratio falls below 1-eps, so the
    # clipped branch is not selected. Pinning it keeps the direction test above
    # from being satisfied by a symmetric (and therefore wrong) implementation.
    new_down = torch.tensor([-1.0, 0.0])  # ratio = e^-1 ~ 0.37, below 1 - eps
    down_clipfrac = compute_policy_loss(old - new_down, adv, eps, eps)[1]
    assert float(down_clipfrac.mean()) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))
