import json
import re

import torch

from tasks.alfworld.privilege import render_compact_future, render_hpi_block
from conftest import make_trajectory


def restore(text):
    result, observations = [], {}
    for part in re.split(r'(?=\[Future transition \d)', text):
        if not part:
            continue
        label, action, body = part.split('\n', 2)
        first, *tail = map(int, re.findall(r'\d+', label))
        last = tail[0] if tail else first
        if first != last:
            repeat, body = body.split('\n', 1)
            assert f'{last-first+1} times.' in repeat
        body = body.rstrip('\n')
        prefix = 'Resulting environment observation: '
        value = body[len(prefix):]
        if value.startswith('line delta from turn '):
            header, payload = value.split('\n', 1)
            ref = int(re.search(r'turn (\d+)', header)[1])
            before = observations[ref].splitlines(keepends=True)
            output, cursor = [], 0
            for edit in json.loads(payload):
                i, j = edit['reference_lines']
                assert before[i:j] == edit['removed']
                output.extend(before[cursor:i])
                output.extend(edit['added'])
                cursor = j
            output.extend(before[cursor:])
            value = ''.join(output)
        for index in range(first, last+1):
            observations[index] = value
            result.append((index, action[len('Executed action: '):], value))
    return result


def test_exact_adjacent_folding_and_short_feedback_fallback():
    items = [(1, 'look', 'Nothing happens.'), (2, 'look', 'Nothing happens.'),
             (3, 'look', 'Nothing happens.'), (4, 'look', 'Nothing happens.'),
             (5, 'open Fridge 2', 'Nothing happens.'),
             (6, 'look', 'Nothing happens.'), (7, 'LOOK', 'Nothing happens.'),
             (9, 'LOOK', 'Nothing happens.')]
    text = render_compact_future(items)
    assert '[Future transition 1-4]' in text
    assert 'consecutively 4 times.' in text
    assert 'line delta' not in text
    assert restore(text) == items


def test_delta_reconstructs_every_change_and_reference_survives_folding():
    base = '\n'.join(f'Location {i}: Object {i}, not open, quantity {i}.' for i in range(80))
    changed = base.replace('Location 4: Object 4, not open, quantity 4.\n', '')
    changed = changed.replace('Location 7: Object 7, not open, quantity 7.', 'ERROR: Object 7 is open; quantity 0.\nAdded at Location 7: bowl 22.')
    items = [(3, 'look', base), (4, 'look', base), (5, 'open bowl 22', changed), (6, 'look', base)]
    text = render_compact_future(items)
    assert 'line delta from turn 4' in text
    assert 'line delta from turn 5' in text
    assert restore(text) == items


def test_hsi_gap_outcome_and_control_are_preserved():
    trajectory = make_trajectory()
    for turn in trajectory.turns:
        turn.next_observation = 'Nothing happens.'
        turn.action_text = 'look'
    full = render_hpi_block(trajectory, 1, with_future=True, future_gap=1, compact_future=True)
    assert '[Future transition 2-3]' in full
    assert '[Future transition 1' not in full
    assert '<think>' not in full
    kwargs = dict(with_future=False, future_gap=1, outcome_text='<outcome>unchanged</outcome>')
    assert render_hpi_block(trajectory, 1, compact_future=True, **kwargs) == render_hpi_block(trajectory, 1, **kwargs)


def test_thinking_teacher_target_and_causal_rows(monkeypatch, tokenizer):
    from tasks.alfworld.episode import EpisodeResult, TurnSample
    from tasks.alfworld.prompts import TaskView, build_turn_prompt_ids, build_turn_messages
    from tasks.alfworld.rollout import OutcomeAnchor, build_group_samples, _prompt_ids_for
    monkeypatch.setenv('OPSD_SRLVR', '1')
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    monkeypatch.setenv('ALFWORLD_THINKING_STYLE', 'brief')
    monkeypatch.setenv('ALFWORLD_THINKING_TOKENS', '64')
    trajectory = make_trajectory(4)
    turns = []
    for i, source in enumerate(trajectory.turns):
        view = TaskView(trajectory.task_description, source.observation, source.admissible_actions, [])
        assert 'at most 64 tokens' in build_turn_messages(view)[0]['content']
        assert 'Admissible actions:' not in build_turn_messages(view)[1]['content']
        turns.append(TurnSample(0, 0, trajectory.task_id, i, build_turn_prompt_ids(tokenizer, view),
            source.generated_token_ids, source.generated_text, view, generated_log_probs=[-.1]*2))
    episode = EpisodeResult(trajectory, turns)
    monkeypatch.setenv('SRLVR_COMPACT_FUTURE', '0')
    original = build_group_samples(tokenizer, None, [episode], OutcomeAnchor([0.], [1.]), 0, with_hpi=False, future_gap=1)
    monkeypatch.setenv('SRLVR_COMPACT_FUTURE', '1')
    compact = build_group_samples(tokenizer, None, [episode], OutcomeAnchor([0.], [1.]), 0, with_hpi=False, future_gap=1)
    for a, b, turn in zip(original, compact, turns):
        for key in ('tokens', 'loss_mask', 'rollout_log_probs', 'reward', 'opsd_advantages', 'turn_ids'):
            assert getattr(a, key) == getattr(b, key)
        ids, plen, n = b.metadata['hsi_teacher_view']
        assert ids[plen:] == turn.generated_token_ids == b.tokens[-n:]
        assert b.metadata['trajectory'] == a.metadata['trajectory']
        assert b.metadata['future_compression']['transitions'] == max(0, 3-turn.turn_idx)


def test_initial_kl_reuses_shared_function_and_probe_excludes_it(monkeypatch):
    from types import SimpleNamespace
    from methods.srlvr_hsi import actor_loss
    from tasks.alfworld import loss as shared
    theta = torch.tensor([-.7, -.4], requires_grad=True)
    ref = torch.tensor([-.6, -.8])
    mask = torch.tensor([1., 0.])
    args = SimpleNamespace(vocab_size=None, srlvr_calibration=False, eps_clip=.2, eps_clip_high=.2,
        srlvr_loss_scale=3., use_kl_loss=True, kl_loss_coef=.001, kl_loss_type='low_var_kl')
    batch = dict(unconcat_tokens=[], total_lengths=[2], response_lengths=[2],
        rollout_log_probs=[theta.detach()], opsd_advantages=[torch.tensor([.2, .2])],
        loss_masks=[mask], ref_log_probs=[ref])
    monkeypatch.setattr(shared, 'log_probs_and_entropy', lambda logits, **kw: (None, {'log_probs': [logits]}))
    actual, _ = actor_loss(args, batch, theta, None)
    expected = (-(theta-theta.detach()).exp().mul(.2).mul(mask).sum() + .001*(shared.reference_kl(args, theta, ref)*mask).sum())*3
    assert torch.allclose(torch.autograd.grad(actual, theta, retain_graph=True)[0], torch.autograd.grad(expected, theta)[0])
    args.srlvr_calibration = True
    batch.pop('ref_log_probs')
    actual, _ = actor_loss(args, batch, theta, None)
    assert torch.allclose(actual, -(theta*.2*mask).sum()*3)
