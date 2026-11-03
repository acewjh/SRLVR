"""Read shared experiment artifacts without changing active training files.

Metric precedence: native telemetry > text log > common/SRLVR ledgers. Derived
EPISODE metrics have their own namespace; they never overwrite training ROW
lengths. One point per metric/step, latest timestamp within a source tier.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading

_LOCK = threading.RLock()
_CACHE = {}
_ARTIFACTS = {'checkpoints', 'checkpoint', 'rollouts', 'rollout', 'dashboard',
              'trajectories', 'episodes', 'SOURCE_SNAPSHOT', '__pycache__'}
_MARKERS = ('dashboard', 'train.log', 'common_metrics.jsonl', 'srlvr_metrics.jsonl',
            'effective_args.json', 'effective_config.json')
_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_LOG = re.compile(r'\b(step|perf|rollout|eval) (\d+):\s*\{(.*)\}')
_KV = re.compile(r"['\"]([\w./@-]+/[^'\"]+)['\"]:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?=\s*[,}])")
_STAMP = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]')


def _exists(path):
    try:
        return path.exists()
    except OSError:
        return False


def canonical_run(path):
    path = Path(path).expanduser().resolve()
    # --save convention: <run>/checkpoints. Old writers already wrote here;
    # read both locations, without moving or interrupting those writers.
    return path.parent if path.name in {'checkpoints', 'checkpoint'} else path


def _parse_log(line):
    line = _ANSI.sub('', line)
    match = _LOG.search(line)
    if not match:
        return None
    kind, step, body = match.groups()
    values = {key: float(value) for key, value in _KV.findall(body + '}')}
    if not values:
        return None
    stamp = _STAMP.search(line)
    ts = datetime.strptime(stamp[1], '%Y-%m-%d %H:%M:%S').timestamp() if stamp else 0
    return {'kind': 'metrics', 'step': int(step), 'time': ts,
            'step_key': 'eval/step' if kind == 'eval' else 'train/step' if kind == 'step' else 'rollout/step',
            'metrics': values, '_line': line[-1800:]}


def _records(path, text_log=False):
    """Incrementally consume COMPLETE lines only, including on network filesystems."""
    path = Path(path)
    with _LOCK:
        try:
            st = path.stat()
            identity = (st.st_dev, st.st_ino)
            key = (str(path), text_log)
            state = _CACHE.get(key)
            if state is None or state['identity'] != identity or st.st_size < state['offset'] or (st.st_size == state['size'] and st.st_mtime_ns != state['mtime']):
                state = {'identity': identity, 'offset': 0, 'rows': [], 'size': -1, 'mtime': None}
                _CACHE[key] = state
            if st.st_size != state['size'] or st.st_mtime_ns != state['mtime']:
                with path.open('rb') as stream:
                    stream.seek(state['offset'])
                    # readline avoids retaining huge non-metric engine log chunks.
                    while True:
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            break
                        state['offset'] = stream.tell()
                        try:
                            decoded = line.decode('utf-8', errors='replace')
                            row = _parse_log(decoded) if text_log else json.loads(decoded)
                            if isinstance(row, dict):
                                state['rows'].append(row)
                        except (ValueError, TypeError):
                            continue
                state['size'], state['mtime'] = st.st_size, st.st_mtime_ns
            return list(state['rows'])
        except OSError:
            return []


def _json(path):
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _events(run):
    rows = []
    for folder in (run / 'dashboard', run / 'checkpoints/dashboard', run / 'checkpoint/dashboard'):
        try:
            for path in sorted(folder.glob('events-*.jsonl')):
                rows.extend(_records(path))
        except OSError:
            continue
    return rows


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stats(values):
    values = [v for v in values if _number(v)]
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def _rate(rows, key):
    values = [int(row[key]) for row in rows if isinstance(row.get(key), bool)]
    return sum(values) / len(values) if values else None


def _summary(rows):
    families = {}
    for family in sorted({row.get('task_type') or '?' for row in rows}):
        known = [r for r in rows if (r.get('task_type') or '?') == family and isinstance(r.get('success'), bool)]
        families[family] = [sum(r['success'] for r in known), len(known)]
    token_values = [r['trajectory_tokens'] for r in rows if _number(r.get('trajectory_tokens'))]
    return {'n': len(rows), 'success': sum(r.get('success') is True for r in rows),
            'success_known': sum(isinstance(r.get('success'), bool) for r in rows),
            'families': families, 'terminal': dict(Counter(r.get('status', 'unknown') for r in rows)),
            'steps': sum(r.get('actions') or r.get('turns') or 0 for r in rows),
            'invalid': sum((r.get('actions') or 0) - (r.get('valid_actions') or 0) for r in rows if _number(r.get('valid_actions'))),
            'tokens_p50': statistics.median(token_values) if token_values else None,
            'tokens_max': max(token_values) if token_values else None}


def _metadata(run, events=None):
    cfg = _json(run / 'effective_args.json')
    env = {}
    for event in events or []:
        if event.get('kind') == 'config':
            cfg = {**cfg, **(event.get('config') or {})}
            env.update(event.get('environment') or {})
    exit_info = _json(run / 'exit.json')
    status = 'unknown'  # Never infer remote-process liveness from this host's PID namespace.
    if exit_info.get('exit_code') is not None:
        status = 'completed' if exit_info['exit_code'] == 0 else 'failed'
    return {'run': run.name, 'short': run.name, 'stage': env.get('ARM') or env.get('OPSD_METHOD') or cfg.get('custom_loss_function_path') or 'TRAIN',
            'harness': 'opsd-slime', 'curriculum': '-', 'status': status,
            'updates_applied': [], 'started': exit_info.get('started_epoch'),
            'elapsed': exit_info.get('elapsed_seconds'), 'npu_hours': exit_info.get('allocated_npu_hours'),
            'eval_only': False, 'expected_updates': cfg.get('num_rollout'), 'config': cfg}


def summarize(run):
    run = canonical_run(run)
    events = sorted(_events(run), key=lambda e: e.get('time') or 0)
    common = _records(run / 'common_metrics.jsonl')
    srlvr = _records(run / 'srlvr_metrics.jsonl')
    logs = _records(run / 'train.log', text_log=True)
    points = defaultdict(dict)
    sources = defaultdict(dict)
    order = 0

    def put(key, step, value, priority, timestamp=0, source='derived'):
        nonlocal order
        if not _number(step) or not _number(value):
            return
        order += 1
        rank = (priority, timestamp or 0, order)
        old = points[key].get(step)
        if old is None or rank >= old[0]:
            points[key][step] = (rank, value)
            sources[key][step] = source

    for event in common:
        step, ts = event.get('rollout'), event.get('unix_time', 0)
        phase = event.get('phase')
        if phase == 'rollout':
            mapping = {'success_rate': 'episodes/train_success', 'action_valid_rate': 'episodes/action_valid_rate',
                       'actor_tokens': 'batch/generated_tokens', 'environment_turns': 'batch/turns',
                       'rollout_seconds': 'perf/rollout_time', 'real_rollouts': 'batch/episodes'}
            for field, key in mapping.items():
                put(key, step, event.get(field), 20, ts, 'common_metrics.jsonl')
            if not _number(event.get('success_rate')) and _number(event.get('real_rollouts')) and event['real_rollouts'] > 0 and _number(event.get('successes')):
                put('episodes/train_success', step, event['successes']/event['real_rollouts'], 20, ts, 'common_metrics.jsonl total/count')
            if _number(event.get('action_valid_rate')):
                put('episodes/invalid_action_rate', step, 1-event['action_valid_rate'], 20, ts, 'common_metrics.jsonl')
            count = event.get('real_rollouts')
            if _number(count) and count > 0:
                # Means may be derived from explicit totals/counts; min/max cannot.
                for field, key in [('actor_tokens','episodes/generated_tokens/mean'), ('environment_turns','episodes/steps_per_episode')]:
                    if _number(event.get(field)):
                        put(key, step, event[field]/count, 20, ts, 'common_metrics.jsonl total/count')
        elif phase == 'train_batch':
            for field, key in {'batch_wall_seconds':'perf/batch_wall_time', 'actor_phase_seconds':'perf/actor_phase_time',
                               'sync_communication_seconds':'perf/sync_communication_time', 'checkpoint_seconds':'perf/checkpoint_time',
                               'rollout_seconds':'perf/rollout_time'}.items():
                put(key, step, event.get(field), 20, ts, 'common_metrics.jsonl')

    # The SRLVR ledger is a phase breakdown, NOT a replacement for generic timer
    # semantics. In particular actor_phase includes ref/scoring/calibration.
    for record in srlvr:
        step = record.get('rollout')
        for field in ('reference_seconds', 'score_seconds', 'calibration_seconds', 'actor_seconds'):
            put('srlvr/' + field, step, record.get(field), 20, source='srlvr_metrics.jsonl')
        for field in ('b', 's', 'lambda_value', 'C_hat', 'D_hat'):
            put('srlvr/' + field, step, record.get(field), 20, source='srlvr_metrics.jsonl')

    for priority, records, source in [(30, logs, 'train.log'), (40, events, 'telemetry')]:
        for record in records:
            if record.get('kind') != 'metrics':
                continue
            step = record.get('step')
            values = record.get('metrics') or {}
            step_key = record.get('step_key', '')
            is_eval = step_key.startswith('eval/') or record.get('phase') == 'eval'
            for key, value in values.items():
                if key.endswith('/step'):
                    continue
                if is_eval and not key.startswith('eval/'):
                    key = 'eval/' + key
                put(key, step, value, priority, record.get('time', 0), source)

    # Keep one generation per rollout after resume; repeated copies of one ID
    # are not extra episodes. Select newest batch ID independently for eval/train.
    latest_batches = {}
    for event in events:
        if event.get('kind') in {'batch', 'attempt'} and _number(event.get('step')):
            batch_id = event.get('batch_id') if event['kind'] == 'attempt' else event.get('id')
            if batch_id:
                latest_batches[(event.get('phase', 'train'), event['step'])] = batch_id
    attempts_by_id = {}
    for index, event in enumerate(events):
        if event.get('kind') != 'attempt':
            continue
        key = (event.get('phase', 'train'), event.get('step'))
        if event.get('batch_id') and event['batch_id'] != latest_batches.get(key):
            continue
        attempts_by_id[event.get('id') or ('unkeyed', index)] = event
    attempts = list(attempts_by_id.values())
    train = [r for r in attempts if r.get('phase', 'train') != 'eval']
    evaluation = [r for r in attempts if r.get('phase') == 'eval']
    grouped = defaultdict(list)
    for event in attempts:
        if _number(event.get('step')):
            grouped[(event.get('phase', 'train'), event['step'])].append(event)
    batch_rows, eval_blocks, step_rows = [], [], []
    for (phase, step), rows in sorted(grouped.items()):
        prefix = 'episodes/' if phase != 'eval' else 'eval/episodes/'
        for key, field in [('train_success', 'success'), ('completion_rate','completed')]:
            put(prefix+key, step, _rate(rows, field), 50)
        valid = [r for r in rows if _number(r.get('valid_actions')) and _number(r.get('actions'))]
        actions = sum(r['actions'] for r in valid)
        if actions:
            rate = sum(r['valid_actions'] for r in valid)/actions
            put(prefix+'action_valid_rate', step, rate, 50)
            put(prefix+'invalid_action_rate', step, 1-rate, 50)
        for field, metric in [('duration_seconds','duration_seconds'), ('turns','turns'),
                              ('generated_tokens','generated_tokens'), ('trajectory_tokens','trajectory_tokens')]:
            stats = _stats([r.get(field) for r in rows])
            for stat in ('mean','min','max'):
                put(prefix+metric+'/'+stat, step, stats[stat], 50)
            if field == 'turns':
                put(prefix+'steps_per_episode', step, stats['mean'], 50)
        summary = _summary(rows)
        summary.update(batch=step, complete=True)
        if phase == 'eval':
            summary['label'] = f'rollout {step}'
            eval_blocks.append(summary)
        else:
            batch_rows.append(summary)
            step_rows.append({'step':step, 'count':len(rows)})

    # Explicit aliases only where semantics agree. Never coerce SRLVR sum losses
    # into GRPO means or episode lengths into training-row response lengths.
    aliases = {'train/v2/pg_loss_mean':'train/pg_loss', 'train/v2/reference_kl_mean':'train/kl_loss',
               'train/v2/ppo_kl':'train/ppo_kl', 'train/v2/pg_clipfrac':'train/pg_clipfrac'}
    for original, target in aliases.items():
        for step, (rank, value) in list(points.get(original, {}).items()):
            put(target, step, value, 10, rank[1], original)
    series = {key: [[step, entry[1]] for step, entry in sorted(values.items())] for key, values in points.items() if values}
    meta = _metadata(run, events)
    updates = sorted({step for key, rows in series.items() if key.startswith('train/') for step, _ in rows})
    meta['updates_applied'] = updates
    common_by_step = {}
    for row in common:
        if row.get('phase') == 'rollout' and _number(row.get('rollout')):
            common_by_step[row['rollout']] = row
    known_common = [r for step,r in common_by_step.items() if ('train',step) not in grouped]
    common_count = sum(r.get('real_rollouts') or 0 for r in known_common)
    known = [r for r in train if isinstance(r.get('success'),bool)]
    common_known = [r for r in known_common if _number(r.get('real_rollouts')) and (_number(r.get('successes')) or _number(r.get('success_rate')))]
    denom = len(known)+sum(r['real_rollouts'] for r in common_known)
    common_successes = sum(r['successes'] if _number(r.get('successes')) else r['success_rate']*r['real_rollouts'] for r in common_known)
    success = (sum(r['success'] for r in known)+common_successes)/denom if denom else _rate(train,'success')
    valid = [r for r in train if _number(r.get('actions')) and _number(r.get('valid_actions'))]
    total_actions = sum(r['actions'] for r in valid)+sum(r.get('environment_turns') or 0 for r in known_common)
    total_valid = sum(r['valid_actions'] for r in valid)+sum(r.get('valid_actions') or 0 for r in known_common)
    latest_times = [e['time'] for e in events+logs if _number(e.get('time'))]
    last_time = max(latest_times) if latest_times else None
    return {'run':str(run), 'events':len(events), 'attempts':len(train)+common_count,
            'batches':batch_rows, 'step_rows':step_rows, 'completion_rate':_rate(train,'completed'),
            'success_rate':success, 'action_validity_rate':total_valid/total_actions if total_actions else None,
            'duration_seconds':_stats([e.get('duration_seconds') for e in train]),
            'turns':_stats([e.get('turns') for e in train]),
            'trajectory_tokens':_stats([e.get('trajectory_tokens') for e in train]),
            'generated_tokens':_stats([e.get('generated_tokens') for e in train]),
            'prompt_tokens_processed':sum(e.get('prompt_tokens_processed') or 0 for e in train),
            'unknown_success':sum(e.get('success') is None for e in train),
            'series':series, 'series_sources':dict(sources), 'metrics':[e for e in events if e.get('kind')=='metrics'][-500:],
            'recent_attempts':attempts[-200:], 'meta':meta,
            'progress':{'updates_done':len(updates), 'updates_expected':meta.get('expected_updates'),
                        'current_batch_episodes':step_rows[-1]['count'] if step_rows else 0,
                        'batch_size':meta['config'].get('rollout_batch_size') or 0,
                        'train_episodes':len(train)+common_count, 'eval_episodes':len(evaluation),
                        'last_metric_time':datetime.fromtimestamp(last_time).isoformat(sep=' ') if last_time else None,
                        'log_age_seconds':None},
            'train_summary':_summary(train) if train else None, 'eval_blocks':eval_blocks,
            'gates':[], 'gate_threshold':None, 'gate_threshold_label':'', 'curriculum':[], 'checkpoint':None,
            'log_tail':[e['_line'] for e in logs[-30:]], 'raw_tail':[]}


def discover(root):
    """Discover shared runs without descending into model shards or telemetry."""
    root = Path(root).expanduser().resolve()
    candidates = set()
    for base, dirs, files in os.walk(root, followlinks=True):
        path = Path(base)
        dirs[:] = [d for d in dirs if d not in _ARTIFACTS and not d.startswith(('.', '_')) and not d.startswith('iter_')]
        depth = len(path.relative_to(root).parts)
        if depth >= 3:
            dirs[:] = []
        if path == root:
            continue
        if depth == 1 and path.name != 'eval_staging' or any(_exists(path/marker) for marker in _MARKERS):
            candidates.add(path)
    result = []
    for run in sorted(candidates):
        try:
            stat = run.stat()
        except OSError:
            continue
        meta = _metadata(run)
        times = [stat.st_mtime]
        for path in (run/'train.log', run/'common_metrics.jsonl', run/'srlvr_metrics.jsonl', run/'dashboard', run/'checkpoints/dashboard'):
            try:
                times.append(path.stat().st_mtime)
            except OSError:
                pass
        rel = str(run.relative_to(root))
        result.append({**meta, 'name':run.name, 'run':rel, 'relative_path':rel, 'path':str(run),
                       'mtime':max(times), 'modified':max(times), 'created':getattr(stat,'st_birthtime',stat.st_ctime),
                       'has_metrics':any(_exists(run/m) for m in _MARKERS[:4]) or _exists(run/'checkpoints/dashboard'),
                       'episodes':0})
    return result
