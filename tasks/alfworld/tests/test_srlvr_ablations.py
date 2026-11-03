"""The three v2 ablations preserve causal sampling and calibration semantics."""
from copy import deepcopy

import pytest

from methods import srlvr_hsi as method
from conftest import make_trajectory


def test_uniform_b_uses_effective_tokens_includes_last_turn_and_keeps_episodes_separate():
    def row(p, t, e, mask):
        return dict(meta=dict(position=p, turn=t, e=e), mask=mask)
    rows = [row(0, 2, [.4], [1]), row(1, 0, [-.9], [1]),
            row(0, 0, [.8, -.3, 1.], [1, 1, 0]),
            row(0, 1, [-.2, 1., .5], [1, 0, 1])]
    standard = method.fill_future(deepcopy(rows), .7)
    uniform = method.fill_future(rows, .7, True)
    expected = (2 * .3 * (-.2 + .7*.5 + .7**2*.4) + 2 * .3*.4) / 5
    assert [r['meta']['B'] for r in uniform[0]] == pytest.approx([expected]*3)
    assert standard[0][-1]['meta']['B'] == 0.
    assert uniform[1][0]['meta']['B'] == 0.
    assert [r['meta']['e'] for r in uniform[0]] == [r['meta']['e'] for r in standard[0]]


@pytest.mark.parametrize('success', [True, False])
@pytest.mark.parametrize('compact', ['0', '1'])
def test_outcome_teacher_reuses_no_future_view_and_leaves_student_unchanged(tokenizer, monkeypatch, success, compact):
    from tasks.alfworld.episode import EpisodeResult, TurnSample
    from tasks.alfworld.prompts import TaskView, build_turn_prompt_ids
    from tasks.alfworld.rollout import OutcomeAnchor, build_group_samples, _prompt_ids_for
    from tasks.alfworld.privilege import render_hpi_block, NO_FUTURE_PLACEHOLDER
    from tasks.alfworld.hsi_shared import render_outcome_descriptor
    monkeypatch.setenv('OPSD_SRLVR', '1')
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    monkeypatch.setenv('SRLVR_COMPACT_FUTURE', compact)
    trajectory = make_trajectory(num_turns=4, success=success)
    turns = []
    for source in trajectory.turns:
        view = TaskView(trajectory.task_description, source.observation, source.admissible_actions, [])
        turns.append(TurnSample(0, 0, trajectory.task_id, source.turn_idx,
            build_turn_prompt_ids(tokenizer, view), source.generated_token_ids,
            source.generated_text, view, generated_log_probs=[-.1]*len(source.generated_token_ids)))
    def build():
        return build_group_samples(tokenizer, None, [EpisodeResult(trajectory, turns)],
            OutcomeAnchor([0.], [1.]), 0, with_hpi=False, future_gap=1)
    monkeypatch.setenv('SRLVR_TEACHER_OUTCOME_ONLY', '0')
    standard = build()
    monkeypatch.setenv('SRLVR_TEACHER_OUTCOME_ONLY', '1')
    for row, original, turn in zip(build(), standard, turns, strict=True):
        assert row.tokens == original.tokens == turn.prompt_token_ids + turn.generated_token_ids
        assert row.loss_mask == original.loss_mask
        assert row.rollout_log_probs == original.rollout_log_probs
        block = render_hpi_block(trajectory, turn.turn_idx, with_future=False,
            future_gap=1, outcome_text=render_outcome_descriptor(success))
        assert NO_FUTURE_PLACEHOLDER in block
        assert '[Future transition ' not in block
        teacher, plen, n = row.metadata['hsi_teacher_view']
        assert teacher[:plen] == _prompt_ids_for(tokenizer, turn.view, block)
        assert teacher[plen:] == row.tokens[-n:]


def test_settings_validate_switches_and_resume_identity(monkeypatch):
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    monkeypatch.setenv('SRLVR_ACTOR_LAMBDA_ZERO', '1')
    cfg = method.Settings.from_env()
    assert cfg.actor_lambda_zero
    state = method.prepare_history(None, cfg, 0)
    monkeypatch.setenv('SRLVR_ACTOR_LAMBDA_ZERO', '0')
    with pytest.raises(RuntimeError, match='changed ablation'):
        method.prepare_history(state, method.Settings.from_env(), 0)
    monkeypatch.setenv('SRLVR_TRAJECTORY_UNIFORM_B', 'yes')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        method.Settings.from_env()
