from types import SimpleNamespace

import pytest
import torch

from tasks.alfworld.episode import _history_pairs
from tasks.alfworld.loss import reference_kl
from tasks.alfworld.prompts import TaskView, build_turn_user_message


@pytest.mark.parametrize('show_admissible', ['0', '1'])
def test_history_includes_first_turn_after_more_than_eight_steps(monkeypatch, show_admissible):
    monkeypatch.setenv('ALFWORLD_SHOW_ADMISSIBLE', show_admissible)
    turns = [SimpleNamespace(observation=f"obs{i}", action_text=f"act{i}") for i in range(30)]
    history = _history_pairs(SimpleNamespace(turns=turns), 29)
    assert len(history) == 29
    text = build_turn_user_message(TaskView("task", "now", ["SECRET_ACTION_LIST"], history))
    assert "obs0" in text and "act28" in text
    assert ("SECRET_ACTION_LIST" in text) == (show_admissible == '1')


def test_reference_kl_has_real_gradient_and_frozen_reference():
    args = SimpleNamespace(use_kl_loss=True, kl_loss_type="low_var_kl")
    lp = torch.tensor([-.7, -.9], requires_grad=True)
    ref = torch.tensor([-1.1, -.3], requires_grad=True)
    kl = reference_kl(args, lp, ref)
    expected = torch.exp(ref.detach() - lp) - 1 - (ref.detach() - lp)
    assert torch.allclose(kl, expected)
    grad, ref_grad = torch.autograd.grad(.001 * kl.sum(), (lp, ref), allow_unused=True)
    assert grad.abs().sum() > 0 and ref_grad is None
    with pytest.raises(ValueError, match="missing"):
        reference_kl(args, lp, None)
