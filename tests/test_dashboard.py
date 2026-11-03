import json
from opsd_dashboard.aggregate import summarize

def test_summary_rates_extrema_and_unknown(tmp_path):
 d=tmp_path/'dashboard';d.mkdir();p=d/'events-x.jsonl'
 rows=[{'kind':'attempt','id':'1','completed':True,'success':True,'valid_actions':2,'actions':2,'duration_seconds':2,'turns':2,'trajectory_tokens':10,'generated_tokens':4}, {'kind':'attempt','id':'2','completed':False,'success':None,'valid_actions':1,'actions':2,'duration_seconds':4,'turns':4,'trajectory_tokens':20,'generated_tokens':8}]
 p.write_text(''.join(json.dumps(x)+'\n' for x in rows));s=summarize(tmp_path)
 assert s['completion_rate']==.5 and s['success_rate']==1 and s['action_validity_rate']==.75
 assert s['duration_seconds']['mean']==3 and s['duration_seconds']['min']==2 and s['duration_seconds']['max']==4
 assert s['turns']['mean']==3 and s['unknown_success']==1

def test_default_emitter_is_append_only_and_tolerates_partial_line(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from opsd_core.telemetry import emit
    monkeypatch.delenv('OPSD_DASHBOARD', raising=False)
    emit(SimpleNamespace(save=str(tmp_path/'run')), {'kind':'metrics','step':1})
    path=next((tmp_path/'run'/'dashboard').glob('events-*.jsonl'))
    with path.open('a') as f: f.write('{"partial":')
    assert summarize(tmp_path/'run')['events'] == 2

def test_discover_returns_runs_with_sortable_metadata(tmp_path):
    from opsd_dashboard.aggregate import discover
    (tmp_path/'z-run'/'dashboard').mkdir(parents=True)
    (tmp_path/'a-run'/'dashboard').mkdir(parents=True)
    runs=discover(tmp_path)
    assert {r['name'] for r in runs} == {'z-run','a-run'}
    assert all(r['path'].endswith('/dashboard') is False for r in runs)
    assert all(isinstance(r['modified'], float) for r in runs)


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def test_checkpoint_identity_and_independent_nested_experiments(tmp_path):
    from opsd_dashboard.aggregate import discover
    parent=tmp_path/'train-a'
    _jsonl(parent/'common_metrics.jsonl', [])
    _jsonl(parent/'checkpoints/dashboard/events-host.jsonl', [
        {'kind':'metrics','step':0,'metrics':{'train/grad_norm':2}}])
    _jsonl(tmp_path/'eval_staging/independent/dashboard/events-x.jsonl', [])
    # Shards / rollout payloads must never become experiments.
    _jsonl(parent/'checkpoints/iter_0001/effective_config.json', [])
    _jsonl(parent/'rollouts/subdir/effective_args.json', [])
    rows=discover(tmp_path)
    assert {row['run'] for row in rows}=={'train-a','eval_staging/independent'}
    assert summarize(parent)['series']==summarize(parent/'checkpoints')['series']
    assert summarize(parent/'checkpoints')['run']==str(parent)
    assert all(row['status']=='unknown' for row in rows)


def test_merge_same_step_keeps_episode_row_and_batch_units_separate(tmp_path):
    _jsonl(tmp_path/'common_metrics.jsonl', [
        {'phase':'rollout','rollout':3,'real_rollouts':2,'successes':1,
         'success_rate':0.5,'actor_tokens':30000,'environment_turns':60}])
    _jsonl(tmp_path/'checkpoints/dashboard/events-a.jsonl', [
        {'kind':'attempt','id':'one','step':3,'phase':'train','success':True,
         'generated_tokens':10000,'trajectory_tokens':11000,'turns':20},
        {'kind':'attempt','id':'two','step':3,'phase':'train','success':False,
         'generated_tokens':20000,'trajectory_tokens':21000,'turns':40},
        {'kind':'metrics','time':10,'step':3,'metrics':{'rollout/response_len/mean':900}},
        {'kind':'metrics','time':11,'step':3,'metrics':{'rollout/response_len/mean':500}},
    ])
    series=summarize(tmp_path)['series']
    assert series['rollout/response_len/mean']==[[3,500]]
    assert series['episodes/trajectory_tokens/mean']==[[3,16000]]
    assert series['episodes/generated_tokens/mean']==[[3,15000]]
    assert series['batch/generated_tokens']==[[3,30000]]
    assert all(len(rows)==len({row[0] for row in rows}) for rows in series.values())


def test_batch_legacy_fraction_is_not_fake_episode(tmp_path):
    _jsonl(tmp_path/'common_metrics.jsonl', [
        {'phase':'rollout','rollout':0,'real_rollouts':32,'success_rate':0.15625,
         'actor_tokens':320000,'environment_turns':1600,'rollout_seconds':600,
         'valid_actions':800}])
    result=summarize(tmp_path)
    assert result['success_rate']==0.15625
    assert result['series']['episodes/train_success']==[[0,0.15625]]
    assert result['series']['episodes/generated_tokens/mean']==[[0,10000]]
    assert 'episodes/trajectory_tokens/mean' not in result['series']
    assert result['duration_seconds']['count']==0
    assert result['turns']['count']==0
    assert 'episodes/duration_seconds/min' not in result['series']
    assert result['attempts']==32


def test_native_and_log_actor_and_srlvr_timers(tmp_path):
    (tmp_path/'train.log').write_text(
        "\x1b[36m[2026-10-28 10:00:00] model.py:728 - step 2: {'train/v2/pg_loss_mean': -0.1, 'train/v2/ppo_kl': 0.002, 'train/grad_norm': 10}\x1b[0m\n"
        "[2026-10-28 10:00:01] rollout.py:1373 - perf 2: {'perf/rollout_time': 600, 'rollout/response_len/mean': 400}\n"
    )
    _jsonl(tmp_path/'checkpoints/dashboard/events-x.jsonl',[
        {'kind':'metrics','step':2,'time':1,'metrics':{'train/grad_norm':12, 'train/srlvr_loss_sum':700}}
    ])
    _jsonl(tmp_path/'common_metrics.jsonl',[
        {'phase':'train_batch','rollout':2,'batch_wall_seconds':1100,'actor_phase_seconds':500}])
    _jsonl(tmp_path/'srlvr_metrics.jsonl',[
        {'rollout':2,'reference_seconds':30,'score_seconds':100,'calibration_seconds':200,'actor_seconds':160},
        {'rollout':2,'reference_seconds':31,'score_seconds':101,'calibration_seconds':201,'actor_seconds':161}])
    s=summarize(tmp_path)['series']
    assert s['train/pg_loss']==[[2,-0.1]]
    assert s['train/grad_norm']==[[2,12]]  # native telemetry wins over text duplicate
    assert s['train/srlvr_loss_sum']==[[2,700]]
    assert s['perf/batch_wall_time']==[[2,1100]]
    assert s['perf/actor_phase_time']==[[2,500]]
    assert s['srlvr/score_seconds']==[[2,101]]
    assert s['srlvr/actor_seconds']==[[2,161]]
    assert 'perf/actor_train_time' not in s  # no invented sum / mislabeled timer


def test_phases_and_restart_select_latest_batch(tmp_path):
    _jsonl(tmp_path/'dashboard/events-x.jsonl',[
        {'kind':'attempt','id':'old','batch_id':'old-batch','time':1,'step':1,'success':True,'phase':'train'},
        {'kind':'attempt','id':'new','batch_id':'new-batch','time':2,'step':1,'success':False,'phase':'train'},
        {'kind':'batch','id':'new-batch','time':3,'step':1,'phase':'train'},
        {'kind':'attempt','id':'eval','time':4,'step':1,'success':True,'phase':'eval'},
        {'kind':'metrics','time':5,'step':1,'step_key':'eval/step','metrics':{'rollout/response_len/mean':50}},
    ])
    s=summarize(tmp_path)
    assert s['attempts']==1 and s['success_rate']==0
    assert s['series']['episodes/train_success']==[[1,0]]
    assert s['series']['eval/episodes/train_success']==[[1,1]]
    assert s['series']['eval/rollout/response_len/mean']==[[1,50]]
    assert 'rollout/response_len/mean' not in s['series']


def test_incremental_partial_lines_and_truncation(tmp_path):
    path=tmp_path/'dashboard/events-x.jsonl'
    _jsonl(path,[{'kind':'metrics','step':0,'metrics':{'train/grad_norm':1}}])
    with path.open('a') as f: f.write('{"kind":"metrics","step":1,"metrics":{"train/grad_norm":')
    assert summarize(tmp_path)['series']['train/grad_norm']==[[0,1]]
    with path.open('a') as f: f.write('2}}\n')
    assert summarize(tmp_path)['series']['train/grad_norm']==[[0,1],[1,2]]
    _jsonl(path,[{'kind':'metrics','step':3,'metrics':{'train/grad_norm':9}}])
    assert summarize(tmp_path)['series']['train/grad_norm']==[[3,9]]


def test_save_directory_canonicalization(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from opsd_core.telemetry import data_dir
    monkeypatch.delenv('OPSD_DASHBOARD_DIR',raising=False)
    assert data_dir(SimpleNamespace(save=str(tmp_path/'run/checkpoints')))==tmp_path/'run/dashboard'
    monkeypatch.setenv('OPSD_DASHBOARD_DIR',str(tmp_path/'custom'))
    assert data_dir(SimpleNamespace(save=str(tmp_path/'run/checkpoints')))==tmp_path/'custom/dashboard'


def test_http_and_frontend_contract(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import urlopen
    from urllib.parse import quote
    from opsd_dashboard.server import make_handler
    run=tmp_path/'experiment'
    _jsonl(run/'checkpoints/dashboard/events-x.jsonl',[
        {'kind':'metrics','step':3,'metrics':{'train/srlvr_loss_sum':2}}])
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(tmp_path))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        rows=json.load(urlopen(base+'/api/runs'))
        assert [r['run'] for r in rows]==['experiment']
        for path in ['experiment','experiment/checkpoints']:
            data=json.load(urlopen(base+'/api/run/'+quote(path,safe='')))
            assert data['series']['train/srlvr_loss_sum']==[[3,2]]
        page=urlopen(base+'/').read().decode()
        assert '300000' in page and '自动刷新 5 分钟' in page
        assert 'train/srlvr_loss_sum' in page and 'srlvr/score_seconds' in page
        assert 'episodes/trajectory_tokens/mean' in page
    finally:
        server.shutdown();server.server_close();worker.join()
