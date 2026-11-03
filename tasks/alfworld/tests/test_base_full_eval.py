"""Full-eval coverage and denominator must survive failures and interruptions."""
from types import SimpleNamespace

import pytest

from tasks.alfworld.base_full_eval import summarize, validate_games


def test_infrastructure_failure_remains_in_full_denominator():
    records = [dict(status='completed', split='valid_seen', task_type='pick',
                    generated_tokens=10, trajectory=dict(success=True, num_turns=1,
                    invalid_action_count=0, truncated=False, turns=[dict(parse_status='ok')])),
               dict(status='error', split='valid_seen', task_type='pick')]
    result = summarize(records, dict(valid_seen=2))
    assert result['total']['success_rate'] == 0.5
    assert result['total']['errors'] == 1
    assert not result['complete']


def test_partial_run_does_not_publish_a_full_success_rate():
    result = summarize([dict(status='error', split='valid_seen', task_type='pick')],
                       dict(valid_seen=140, valid_unseen=134))
    assert result['total']['success_rate'] is None
    assert result['valid_seen']['success_rate'] is None
    assert not result['complete']


def test_duplicate_or_partial_dataset_is_rejected(tmp_path):
    paths = [tmp_path/'a', tmp_path/'b']
    for path in paths:
        path.touch()
    games = [SimpleNamespace(game_id=str(i), path=str(path), split='valid_seen')
             for i, path in enumerate(paths)]
    with pytest.raises(ValueError, match='duplicate'):
        validate_games([games[0], games[0]], games, 'valid_seen')
    with pytest.raises(ValueError, match='full data split'):
        validate_games(games[:1], games, 'valid_seen')
    validate_games(games, games, 'valid_seen')
