"""The ALFWorld EVAL path must spread episodes over every engine.

Measured 2026-10-07 on the 1.7B HSI alfworld eval
(``alfworld-qwen3-1p7b-hsi-cp24-20261007_133941``): engine 0 (port 15000)
reported ``3549`` requests and 2.246e6 generated tokens while each of the six
sibling engines reported ``num_running_reqs 0`` and *no* request counter at all.
Six of the eval's seven engines -- six of the node's eight NPUs, each holding
~26 GB of weights -- idled for the whole evaluation.

Cause: ``_eval_async`` called ``run_episode`` without ``engine_url``, so
``episode._generate`` fell back to ``OPSD_ENGINE_URL``, which the RolloutManager
pins to ``engines[0]``.  The training path (``_run_group``) has always spread
round-robin over ``_engine_urls(args)``; only the eval path forgot to.

Engine choice cannot change the numbers: ``_sampling_params`` seeds every
episode (``sampling_seed``) and every turn of an episode keeps its own engine,
so the prefix cache is still reused within an episode.

CPU-only: ``GenerateState``, ``load_eval_games`` and ``run_episode`` are
stubbed, so no tokenizer, engine or NPU is involved.
"""

import asyncio
import types

from tasks.alfworld import eval_data as ED
from tasks.alfworld import rollout as R

ENGINES = [f"http://127.0.0.1:{15000 + 2 * i}" for i in range(7)]


class _FakeTrajectory:
    task_description = "get the apple"
    truncated = False
    success = False

    def to_dict(self):
        return {}


class _FakeEpisode:
    trajectory = _FakeTrajectory()
    reward = 0.0


class _FakeGenerateState:
    def __init__(self, args):
        self.tokenizer = None
        self.sampling_params = {}


class _FakeDataset:
    def __init__(self, name):
        self.name = name
        self.path = f"{name}.jsonl"


def _drive(monkeypatch, *, n_games, engines, calls, concurrency="8"):
    """Run ``_eval_async`` with the heavy dependencies stubbed."""

    async def _run_episode(args, tokenizer, game, params, *, episode_id, group_id,
                           engine_url=None):
        calls.append({"episode_id": episode_id, "engine_url": engine_url})
        return _FakeEpisode()

    monkeypatch.setattr(R, "GenerateState", _FakeGenerateState)
    monkeypatch.setattr(R, "run_episode", _run_episode)
    monkeypatch.setattr(R, "_engine_urls", lambda args: list(engines))
    monkeypatch.setattr(ED, "load_eval_games", lambda path: [object()] * n_games)
    monkeypatch.setenv("ALFWORLD_EVAL_CONCURRENCY", concurrency)

    args = types.SimpleNamespace(eval_datasets=[_FakeDataset("valid_seen")])
    return asyncio.run(R._eval_async(args, rollout_id=0))


def test_eval_spreads_episodes_over_every_engine(monkeypatch):
    """14 episodes over 7 engines -> each engine takes exactly 2."""
    calls = []
    _drive(monkeypatch, n_games=14, engines=ENGINES, calls=calls)

    used = [c["engine_url"] for c in calls]
    assert len(used) == 14
    assert set(used) == set(ENGINES), "every engine must serve at least one episode"
    for url in ENGINES:
        assert used.count(url) == 2


def test_eval_never_falls_back_to_the_pinned_engine_url(monkeypatch):
    """``engine_url=None`` makes ``episode._generate`` use engines[0] for all."""
    calls = []
    _drive(monkeypatch, n_games=3, engines=ENGINES, calls=calls)

    assert [c["engine_url"] for c in calls] == ENGINES[:3]
    assert all(c["engine_url"] for c in calls), "an unset engine_url pins the whole eval"


def test_eval_engine_choice_is_stable_across_datasets(monkeypatch):
    """Each dataset re-runs from index 0; that is fine as long as none is pinned."""
    calls = []
    _drive(monkeypatch, n_games=7, engines=ENGINES, calls=calls)

    assert [c["engine_url"] for c in calls] == ENGINES


def test_eval_degrades_to_the_single_exported_engine(monkeypatch):
    """A one-engine deployment must still work (the fallback is not removed)."""
    calls = []
    _drive(monkeypatch, n_games=3, engines=[ENGINES[0]], calls=calls)

    assert [c["engine_url"] for c in calls] == [ENGINES[0]] * 3
