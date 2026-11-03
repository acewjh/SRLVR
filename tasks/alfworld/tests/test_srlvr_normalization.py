"""Trajectory-token means through actor, calibration, history and framework IO."""
import ast
import json
import math
import pickle
import sys
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods import srlvr, srlvr_hsi as method, srlvr_compact
from tasks.alfworld import loss as shared_loss
from test_srlvr_hsi import _framework_loss_function

ROOT = Path(__file__).resolve().parents[3]


def _definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[name]


def _sample(position, turn, mask, prompt=50):
    response = len(mask)
    return SimpleNamespace(tokens=[0]*prompt + list(range(response)), response_length=response,
        loss_mask=mask, rollout_log_probs=[-.5]*response, reward=float(position != 1),
        status=SimpleNamespace(name='COMPLETED'), turn_ids=[turn]*response,
        metadata={'srlvr': dict(position=position, turn=turn)})


def _args(cfg):
    return SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=3,
        rollout_top_p=1., rollout_top_k=-1, rollout_temperature=.8, micro_batch_size=1,
        actor_num_nodes=1, actor_num_gpus_per_node=4, tensor_model_parallel_size=2)


def test_global_denominators_survive_real_dp_split_and_micro_iterator(monkeypatch):
    cfg = method.Settings('v2', .99, calibration=2)
    monkeypatch.setattr(method.Settings, 'from_env', classmethod(lambda cls: cfg))
    samples = [_sample(0, 0, [1, 0, 1], prompt=4000), _sample(0, 1, [1]),
               _sample(1, 0, [1, 1, 0, 1]), _sample(2, 0, [1]),
               _sample(2, 1, [1, 1]), _sample(2, 2, [1, 0, 1])]
    data = method.convert(_args(cfg), samples)
    expected = {0: 3, 1: 3, 2: 5}
    assert len(data['tokens']) == 8  # two zero-mask padding rows
    assert data['srlvr_batch']['N'] == 3
    assert data['srlvr_batch']['reduction'] == method.REDUCTION
    fw = ROOT/'frameworks/slime-ascend/slime'
    split = _definition(fw/'ray/rollout.py', '_split_train_data_by_dp',
        dict(Box=lambda x: x, ray=SimpleNamespace(put=lambda x: x)))
    iterator_cls = _definition(fw/'backends/megatron_utils/data.py', 'DataIterator', {})
    # Use the actual train forward's requested keys: adding metadata to convert
    # alone would not transport it through this boundary.
    tree = ast.parse((fw/'backends/megatron_utils/model.py').read_text())
    key_lists = [n for n in ast.walk(tree) if isinstance(n, ast.List)
                 and any(isinstance(v, ast.Constant) and v.value == 'srlvr_rows' for v in n.elts)]
    assert len(key_lists) == 1
    keys = ast.literal_eval(key_lists[0])
    partitions = split(SimpleNamespace(args=SimpleNamespace(balance_data=False)), data, 4)
    real_rows = 0
    for part in partitions:
        iterator = iterator_cls(part, micro_batch_size=1)
        for _ in part['tokens']:
            row = iterator.get_next(keys)
            meta = row['srlvr_rows'][0]
            assert meta['actor_token_count'] == expected[meta['position']]
            if meta.get('padding'):
                assert sum(row['loss_masks'][0]) == 0
            else:
                real_rows += 1
    assert real_rows == 6


def test_zero_actor_token_trajectory_rejected(monkeypatch):
    cfg = method.Settings('v2', .99, calibration=2)
    monkeypatch.setattr(method.Settings, 'from_env', classmethod(lambda cls: cfg))
    with pytest.raises(ValueError, match='no effective actor tokens'):
        method.convert(_args(cfg), [_sample(i, 0, [0] if i == 1 else [1]) for i in range(3)])


@pytest.mark.parametrize('selective', [False, True])
def test_actor_ppo_and_nonzero_kl_mean_once_per_trajectory(monkeypatch, selective):
    reducer = _framework_loss_function()
    monkeypatch.setattr(srlvr_compact, 'enabled', lambda _: selective)
    monkeypatch.setattr(shared_loss, 'log_probs_and_entropy',
                        lambda logits, **kw: (None, {'log_probs': [logits]}))
    monkeypatch.setattr(srlvr_compact, 'log_probs_from_hidden',
                        lambda args, batch, logits: {'log_probs': [logits]})
    args = SimpleNamespace(vocab_size=None, srlvr_calibration=False, eps_clip=.2, eps_clip_high=.2,
        srlvr_loss_scale=3., loss_type='custom_loss', custom_loss_function_path='methods.srlvr_hsi.actor_loss',
        calculate_per_token_loss=False, qkv_format='bshd', recompute_loss_function=False,
        allgather_cp=False, global_batch_size=6, use_kl_loss=True, kl_loss_coef=.17,
        kl_loss_type='low_var_kl', _srlvr_hidden=True)
    theta = torch.tensor([.23, -.14], requires_grad=True)
    xs = [torch.tensor([[1., 2.], [4., 9.]]), torch.tensor([[3., -1.], [.2, .1], [-.2, 1.]]),
          torch.tensor([[.1, 1.]]), torch.tensor([[.8, -.2], [.4, 1.]]),
          torch.tensor([[.2, .7], [-.3, .6]]), torch.tensor([[10., 10.]])]
    masks = [torch.tensor([1., 0.]), torch.ones(3), torch.ones(1),
             torch.ones(2), torch.ones(2), torch.zeros(1)]
    advantages = [.3, -.4, .2, -.1, .6, 0.]
    lengths = [5, 5, 5, 4, 4, 4]
    actual = theta.sum()*0
    expected = theta.sum()*0
    reported = 0.
    for x, mask, a, length in zip(xs, masks, advantages, lengths):
        lp = x@theta - 2
        old = lp.detach() + torch.linspace(-.4, .3, len(lp))
        ref = lp.detach() + .25
        batch = dict(unconcat_tokens=[], total_lengths=[100+len(lp)], response_lengths=[len(lp)],
                     loss_masks=[mask], rollout_log_probs=[old], opsd_advantages=[torch.full_like(lp, a)],
                     ref_log_probs=[ref], srlvr_rows=[dict(actor_token_count=length)])
        scaled, _, log = reducer(args, batch, 3, lp)
        reported += float(log['values'][1])/6
        actual = actual + scaled/3/2  # actual framework, schedule, DP scaling
        ratio = (lp-old).exp()
        token_loss = -torch.minimum(ratio*a, ratio.clamp(.8, 1.2)*a)
        delta = ref-lp
        token_loss = token_loss + .17*(delta.exp()-1-delta).clamp(-10, 10)
        expected = expected + (token_loss*mask).sum()/length/2
    assert torch.allclose(actual, expected, atol=1e-7)
    assert reported == pytest.approx(float(expected.detach()), abs=1e-7)
    assert torch.allclose(torch.autograd.grad(actual, theta, retain_graph=True)[0],
                          torch.autograd.grad(expected, theta)[0], atol=1e-7)


@pytest.mark.parametrize('chunk', [1, 2, 5])
@pytest.mark.parametrize('variant', ['v2', 'v3'])
@pytest.mark.parametrize('micro_batch_size', [1, 2])
@pytest.mark.parametrize('ablation', ['standard', 'lambda0-history', 'lambda0-batch', 'uniform_b'])
def test_train_batch_probes_moments_and_baseline_use_mean_score(monkeypatch, tmp_path, chunk, variant, micro_batch_size, ablation):
    import types
    cfg = method.Settings(variant, .73, calibration=2)
    if variant != 'v2' and ablation != 'standard':
        pytest.skip('ablations are v2-only')
    cfg = replace(cfg, actor_lambda_zero=ablation.startswith('lambda0'),
                  trajectory_uniform_b=ablation == 'uniform_b')
    mode = 'batch' if ablation == 'lambda0-batch' else 'history'
    monkeypatch.setenv('SRLVR_ADVANTAGE_MODE', mode)
    eff_b, eff_s, _ = method.effective_advantage_stats(mode, [1., 0., 1.], .3, .7, cfg)
    args = _args(cfg)
    args.actor_num_gpus_per_node = 1
    args.tensor_model_parallel_size = 1
    args.micro_batch_size = micro_batch_size
    args.vocab_size = 30
    args.data_pad_size_multiplier = 1
    args.eps_clip = args.eps_clip_high = .2
    args.use_kl_loss = False
    args.srlvr_dp_verified = True
    args.save = str(tmp_path/'checkpoints')
    args.srlvr_state = method.prepare_history(None, cfg, 0)
    args.srlvr_state.update(b=.3, s=.7, lambda_raw=.08)
    prior = deepcopy(args.srlvr_state)
    monkeypatch.setenv('SRLVR_PROBE_CHUNK', str(chunk))
    monkeypatch.setattr(method.Settings, 'from_env', classmethod(lambda cls: cfg))
    monkeypatch.setattr(srlvr_compact, 'enabled', lambda _: False)
    mpu = SimpleNamespace(get_data_parallel_world_size=lambda: 1,
                         get_data_parallel_rank=lambda: 0, get_data_parallel_group_gloo=lambda: None)
    monkeypatch.setitem(sys.modules, 'megatron.core', SimpleNamespace(mpu=mpu))
    config = SimpleNamespace(no_sync_func=None, grad_sync_func=None, param_sync_func=None)
    monkeypatch.setitem(sys.modules, 'megatron.core.utils', SimpleNamespace(get_model_config=lambda _: config))
    monkeypatch.setattr(torch.distributed, 'all_gather_object', lambda out, value, **kw: out.__setitem__(0, deepcopy(value)))
    datamodule = types.ModuleType('slime.backends.megatron_utils.data')
    datamodule.get_data_iterator = lambda args, model, batch: ([batch], [len(batch['tokens'])//args.micro_batch_size])
    monkeypatch.setitem(sys.modules, datamodule.__name__, datamodule)
    modelmodule = types.ModuleType('slime.backends.megatron_utils.model')
    monkeypatch.setitem(sys.modules, modelmodule.__name__, modelmodule)
    parameter = torch.nn.Parameter(torch.tensor([.2, -.1]))
    monkeypatch.setattr(srlvr, '_calibration_entries', lambda _: ([(parameter, 1.)], 1, 0, None))
    monkeypatch.setattr(srlvr, '_reconstruct_dp_gradients', lambda _: None)
    monkeypatch.setattr(shared_loss, 'log_probs_and_entropy',
                        lambda logits, **kw: (None, {'log_probs': [logits]}))
    captured, actor_calls = [], []

    def vectors(meta, n):
        return torch.tensor([[1.+meta['position'], .2+meta['turn']+j] for j in range(n)])

    def train(rollout, model, optimizer, scheduler, iterator, micro):
        batch = iterator[0]
        value = parameter.sum()*0
        assert args.srlvr_state == prior  # update happens after the sole actor step
        for i, meta in enumerate(batch['srlvr_rows']):
            x = vectors(meta, len(batch['loss_masks'][i]))
            lp = x@parameter - 3
            row = {key: [batch[key][i]] for key in
                   ('loss_masks', 'rollout_log_probs', 'opsd_advantages', 'srlvr_rows')}
            row.update(unconcat_tokens=[], total_lengths=[len(lp)+100], response_lengths=[len(lp)])
            value = value + method.actor_loss(args, row, lp, None)[0]/args.srlvr_loss_scale
        parameter.main_grad = torch.autograd.grad(value, parameter)[0]
        if getattr(args, 'srlvr_gradient_probe', None) is not None:
            args.srlvr_gradient_probe(model)
        else:
            lam = 0. if cfg.actor_lambda_zero else .08
            for meta, adv in zip(batch['srlvr_rows'], batch['opsd_advantages']):
                assert torch.allclose(adv, torch.full_like(adv,
                    (meta['terminal_reward'] - eff_b - lam * meta['B']) / eff_s))
            actor_calls.append(rollout)
            args.srlvr_actual_optimizer_steps = getattr(args, 'srlvr_actual_optimizer_steps', 0)+1

    modelmodule.train = train
    original_moments = method._probe_moments

    def moments(z, y, *rest):
        captured.append((z[0].clone(), y[0].clone() if y is not None else torch.zeros_like(z[0])))
        return original_moments(z, y, *rest)

    monkeypatch.setattr(method, '_probe_moments', moments)
    monkeypatch.setattr(method, 'score_row', lambda model, tokens, mask, *a, **kw:
        dict(mu=[0.]*len(mask), f_chosen=[.2]*len(mask), e=[.2]*len(mask), student_logp=[-.5]*len(mask)))
    # Lengths 3, 6, 1. Y drops final B=0 rows but must keep these full lengths.
    masks = [[1, 0, 1], [1], [1, 1, 0, 1], [1, 0, 1], [1], [1]]
    positions_turns = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0)]
    metadata = [dict(position=p, turn=t, policy_version=0, terminal_reward=float(p != 1),
                     teacher_view=([1, 2, 3], 1, 2)) for p, t in positions_turns]
    data = dict(tokens=[torch.tensor([i+10]* (100+len(mask))) for i, mask in enumerate(masks)],
                loss_masks=[torch.tensor(mask) for mask in masks], srlvr_rows=metadata,
                rollout_log_probs=[torch.full((len(mask),), -.5) for mask in masks],
                opsd_advantages=[torch.zeros(len(mask)) for mask in masks],
                total_lengths=[100+len(mask) for mask in masks], response_lengths=list(map(len, masks)),
                dynamic_global_batch_size=6, srlvr_batch=dict(N=3, rows=6, mode=mode, config=asdict(cfg)))
    actor = SimpleNamespace(args=args, model=[object()], optimizer=object(), opt_param_scheduler=object(),
                            weights_backuper=SimpleNamespace(backup=lambda _: None))
    method.train_batch(actor, 0, data)
    cs, ds, weights = [], [], []
    for position in range(2):
        selected = [(m, mask) for m, mask in zip(metadata, masks) if m['position'] == position]
        length = sum(sum(mask) for _, mask in selected)
        raw_score = sum((vectors(m, len(mask))*torch.tensor(mask)[:, None]).sum(0) for m, mask in selected)
        # Y's supplied advantage is +B/s; the loss gradient is -score*B/s.
        raw_y = -sum((vectors(m, len(mask))*torch.tensor(mask)[:, None]).sum(0)*m['B'] for m, mask in selected)/eff_s
        raw_z = (float(position != 1)-eff_b)*raw_score/eff_s
        z, y = captured[position]
        assert torch.allclose(z, raw_z/length, atol=1e-6)
        assert torch.allclose(y, raw_y/length, atol=1e-6)
        cs.append(float(raw_z@raw_y)/length**2)
        ds.append(float(raw_y@raw_y)/length**2)
        weights.append(float((raw_score/length).square().sum()))
    state = args.srlvr_state
    assert state['history'][0][1] == pytest.approx(weights[0], rel=1e-6)
    assert state['history'][1][1] == pytest.approx(weights[1], rel=1e-6)
    assert state['b'] == pytest.approx(weights[0]/sum(weights), rel=1e-6)
    assert state['s'] == pytest.approx(math.sqrt(.1**2+((1-eff_b)**2+eff_b**2)/2))
    assert state['lambda_raw'] == pytest.approx(sum(cs)/sum(ds), rel=1e-6)
    report = json.loads((tmp_path/'srlvr_metrics.jsonl').read_text())
    assert report['C_hat'] == pytest.approx(sum(cs)/2, rel=1e-6)
    assert report['D_hat'] == pytest.approx(sum(ds)/2, rel=1e-6)
    assert actor_calls == [0] and args.srlvr_actual_optimizer_steps == 1
    assert state['reduction'] == method.REDUCTION


def test_history_version_reset_roundtrip_and_future_rejection():
    cfg = method.Settings('v2', .99, calibration=2)
    old = dict(b=.2, s=.6, history=[[1., 900., .4]], lambda_raw=2., lambda_next=.1,
               last_rollout=12, actor_optimizer_steps=13, variant='v2', gamma=.99)
    saved = deepcopy(old)
    state = method.prepare_history(old, cfg, 13)
    assert old == saved
    assert (state['b'], state['s'], state['history'], state['lambda_raw']) == (.5, 1., [], 0.)
    assert state['last_rollout'] == 12 and state['actor_optimizer_steps'] == 13
    assert method.prepare_history(state, cfg, 13) is state
    new = method.next_history(state, [[1., 2., .5], [0., 1., .5]], cfg, .01, .1, 13)
    restored = pickle.loads(pickle.dumps(new))
    assert method.prepare_history(restored, cfg, 14) is restored
    assert restored == new and restored['actor_optimizer_steps'] == 14
    with pytest.raises(RuntimeError, match='unknown SRLVR history reduction'):
        method.prepare_history(dict(new, reduction='trajectory_token_mean_v99'), cfg, 14)
    with pytest.raises(RuntimeError, match='consecutive'):
        method.prepare_history(restored, cfg, 15)


@pytest.mark.parametrize('reduction', [None, method.REDUCTION])
def test_checkpoint_restore_migrates_only_legacy_statistics(monkeypatch, tmp_path, reduction):
    cfg = method.Settings('v2', .99, calibration=2)
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    monkeypatch.setattr(method.Settings, 'from_env', classmethod(lambda cls: cfg))
    state = dict(b=.2, s=.6, history=[[1., 9., .4]], lambda_raw=.2,
                 last_rollout=4, actor_optimizer_steps=5, variant='v2', gamma=.99)
    if reduction:
        state['reduction'] = reduction
    common = dict(args=SimpleNamespace(srlvr_state=state))
    monkeypatch.setitem(sys.modules, 'megatron.core', SimpleNamespace(
        dist_checkpointing=SimpleNamespace(load_common_state_dict=lambda _: common)))
    (tmp_path/'latest_checkpointed_iteration.txt').write_text('4')
    args = SimpleNamespace(load=str(tmp_path))
    srlvr.restore_history(args)
    assert args.srlvr_state['last_rollout'] == 4
    assert args.srlvr_state['actor_optimizer_steps'] == 5
    assert args.srlvr_state['reduction'] == method.REDUCTION
    if reduction:
        assert args.srlvr_state is state
    else:
        assert args.srlvr_state['history'] == [] and args.srlvr_state['b'] == .5
