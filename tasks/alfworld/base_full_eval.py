"""Base-only full evaluation through the current training episode implementation.

No actor/trainer or checkpoint conversion is needed: engines load the original
HF model and run_episode supplies the very same prompts, parser and environment.
Each finished episode is saved separately, so interrupted evaluations can resume.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def validate_games(games, discovered, split):
    ids = [g.game_id for g in games]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{split}: duplicate game IDs')
    if {g.path for g in games} != {g.path for g in discovered}:
        raise ValueError(f'{split}: evaluation index differs from the full data split')
    if any(g.split != split or not Path(g.path).is_file() for g in games):
        raise ValueError(f'{split}: wrong split or missing game file')


def summarize(records, expected):
    def stats(rows, n):
        good = [r for r in rows if r['status'] == 'completed']
        trajectories = [r['trajectory'] for r in good]
        turns = sum(t['num_turns'] for t in trajectories)
        successes = sum(t['success'] for t in trajectories)
        invalid = sum(t['invalid_action_count'] for t in trajectories)
        return dict(expected=n, recorded=len(rows), evaluated=len(good),
                    errors=len(rows)-len(good), successes=successes,
                    success_rate=successes/n if len(rows) == n else None,
                    truncated_episodes=sum(t['truncated'] for t in trajectories),
                    response_cap_turns=sum(x['parse_status'] == 'truncated'
                                           for t in trajectories for x in t['turns']),
                    invalid_actions=invalid, turns=turns,
                    action_valid_rate=(turns-invalid)/turns if turns else None,
                    mean_turns=turns/len(good) if good else None,
                    generated_tokens=sum(r['generated_tokens'] for r in good),
                    mean_generated_tokens=sum(r['generated_tokens'] for r in good)/len(good) if good else None)
    result = {split: stats([r for r in records if r['split'] == split], n)
              for split, n in expected.items()}
    result['total'] = stats(records, sum(expected.values()))
    result['task_types'] = {kind: stats([r for r in records if r['task_type'] == kind],
                                       sum(r['task_type'] == kind for r in records))
                            for kind in sorted({r['task_type'] for r in records})}
    result['complete'] = (result['total']['recorded'] == result['total']['expected']
                          and result['total']['errors'] == 0)
    return result


async def evaluate(cli):
    from transformers import AutoTokenizer
    from tasks.alfworld import config
    from tasks.alfworld.action_harness import HARNESS_VERSION
    from tasks.alfworld.episode import run_episode
    from tasks.alfworld.env_actors import shutdown_pool
    from tasks.alfworld.eval_data import load_eval_games
    from tasks.alfworld.rollout import _sampling_params
    from tasks.alfworld.v1.env_adapter import discover_games
    from slime.utils import http_utils

    out = Path(cli.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    models = Path(cli.model).resolve()
    if models.name not in ('Qwen3-1.7B', 'Qwen3-4B', 'Qwen3-8B'):
        raise ValueError('This entry accepts only the three original Qwen3 HF model directories')
    if (models/'adapter_config.json').exists():
        raise ValueError('An adapter is not a base model')
    tokenizer = AutoTokenizer.from_pretrained(str(models), trust_remote_code=True, local_files_only=True)
    args = SimpleNamespace(seq_length=32768, rollout_num_gpus=len(cli.engines),
                           rollout_num_gpus_per_engine=1, sglang_server_concurrency=8,
                           use_distributed_post=False)
    state = SimpleNamespace(sampling_params=dict(
        temperature=1.0, top_p=1.0, top_k=-1, stop_token_ids=None,
        skip_special_tokens=False, no_stop_trim=True, spaces_between_special_tokens=False))
    http_utils.init_http_client(args)
    engine_info = [await http_utils.get(url+'/get_server_info') for url in cli.engines]
    for url, info in zip(cli.engines, engine_info):
        server_args = info.get('server_args', info)
        if Path(server_args['model_path']).resolve() != models:
            raise ValueError(f'{url}: engine loaded a different model')
        if int(server_args['context_length']) != args.seq_length:
            raise ValueError(f'{url}: engine context differs from training')
    groups = {}
    for split in ('valid_seen', 'valid_unseen'):
        games = load_eval_games(str(Path(cli.project)/'tasks/alfworld/v1/configs/data'/f'{split}.jsonl'))
        validate_games(games, discover_games(split), split)
        groups[split] = games[:1] if cli.smoke else games
    expected = {split: len(games) for split, games in groups.items()}
    if not cli.smoke and expected != dict(valid_seen=140, valid_unseen=134):
        raise ValueError(f'Unexpected full benchmark size: {expected}')
    manifest = dict(model=str(models), model_config=json.loads((models/'config.json').read_text()),
                    expected=expected, smoke=cli.smoke, engines=cli.engines,
                    engine_info=engine_info, action_harness=HARNESS_VERSION,
                    max_turns=config.MAX_TURNS, max_response=config.MAX_RESPONSE_TOKENS,
                    context_length=args.seq_length, history='complete observations/actions; excludes past thinking',
                    show_admissible=True, thinking_style='brief', thinking_token_budget=None,
                    sampling=_sampling_params(state, seed=10000, max_response=4096),
                    sampling_seed_rule='10000 + index within each split; unchanged across turns',
                    sampler='strict FP32 full-softmax inverse CDF, same as current SRLVR training',
                    concurrency=cli.concurrency, environment_versions={},
                    source_snapshot=os.environ.get('ALFWORLD_BASE_SOURCE_SNAPSHOT'),
                    start_time=time.time())
    from tasks.alfworld.v1.paths import ensure_alfworld_importable
    ensure_alfworld_importable()
    import importlib.metadata
    for package in ('alfworld', 'textworld', 'torch', 'transformers'):
        manifest['environment_versions'][package] = importlib.metadata.version(package)
    prior = out/'manifest.json'
    if prior.exists():
        old = json.loads(prior.read_text())
        for key in ('model', 'expected', 'sampling', 'action_harness', 'source_snapshot'):
            if old[key] != manifest[key]:
                raise ValueError(f'Resume protocol changed: {key}')
    else:
        write_json(prior, manifest)
    semaphore = asyncio.Semaphore(cli.concurrency)
    records = []

    async def one(split, index, game):
        folder = out/split
        folder.mkdir(exist_ok=True)
        path = folder/f'{index:04d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['task_id'] != game.game_id:
                raise ValueError(f'{path}: result identity differs from the dataset')
        else:
            async with semaphore:
                record = dict(split=split, index=index, task_id=game.game_id,
                              task_type=game.task_type, sampling_seed=10000+index)
                start = time.perf_counter()
                try:
                    episode = await run_episode(args, tokenizer, game,
                        _sampling_params(state, seed=10000+index, max_response=4096),
                        episode_id=index, group_id=index,
                        engine_url=cli.engines[index % len(cli.engines)])
                    record.update(status='completed', trajectory=episode.trajectory.to_dict(),
                        generated_tokens=episode.generated_tokens, wall_seconds=episode.wall_seconds,
                        generation_seconds=episode.generation_seconds,
                        environment_seconds=episode.environment_seconds,
                        harness_sources=[t.harness_source for t in episode.turns])
                except Exception as exc:
                    record.update(status='error', error_type=type(exc).__name__, error=str(exc),
                                  wall_seconds=time.perf_counter()-start)
                write_json(path, record)
        records.append(record)
        write_json(out/'summary.json', summarize(records, expected))
        print(f"[base-eval] {models.name} {split} {index} status={record['status']} "
              f"success={record.get('trajectory', {}).get('success')} recorded={len(records)}/{sum(expected.values())}", flush=True)

    try:
        # Match the current rollout's per-split ordering, seed and engine selection.
        for split, games in groups.items():
            await asyncio.gather(*(one(split, i, g) for i, g in enumerate(games)))
    finally:
        shutdown_pool()
        await http_utils._http_client.aclose()
    result = summarize(records, expected)
    write_json(out/'summary.json', result)
    return 0 if result['complete'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--engines', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    cli = parser.parse_args()
    if cli.concurrency < 1:
        parser.error('concurrency must be positive')
    return asyncio.run(evaluate(cli))


if __name__ == '__main__':
    raise SystemExit(main())
