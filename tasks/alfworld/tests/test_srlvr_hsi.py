"""Acceptance checks restricted to the SRLVR-v2/v3 estimator and integration."""
import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods import hsi2_jsd
from methods import srlvr_hsi as method
from tasks.alfworld import loss as shared_loss
from conftest import make_trajectory


@pytest.mark.parametrize('variant', ['v2', 'v3'])
def test_exact_student_centering_endpoints_and_detach(variant):
    p = torch.tensor([[.55, .15, .3, 0.], [1., 0., 0., 0.]], dtype=torch.float64)
    q = torch.tensor([[.1, .5, .4, 0.], [0., 1., 0., 0.]], dtype=torch.float64)
    lp, lq = p.log().requires_grad_(), q.log().requires_grad_()
    mu, picked, error = method.centered_features(lp, lq, torch.tensor([2, 0]), variant)
    a = torch.where(p+q > 0, p/(p+q), .5)
    if variant == 'v2':
        f = a
    else:
        f = 1 + (torch.special.xlogy(a, a) + torch.special.xlogy(1-a, 1-a))/method.LOG2
    assert torch.allclose(mu, (p*f).sum(-1))
    assert torch.allclose(error, picked-mu)
    assert torch.isfinite(method.feature(lp, lq, variant)).all()
    assert not any(t.requires_grad for t in (mu, picked, error))
    assert torch.allclose((p*(f-mu[:, None])).sum(-1), torch.zeros(2, dtype=p.dtype), atol=1e-14)
    if variant == 'v3':
        m = (p+q)/2
        jsd = (torch.special.xlogy(p, p/m) + torch.special.xlogy(q, q/m)).nan_to_num().sum(-1)/2/method.LOG2
        assert torch.allclose((m*f).sum(-1), jsd)
        assert not torch.allclose(mu[0], jsd[0])
        assert torch.allclose(f, method.feature(lq, lp, variant), atol=1e-14)
    else:
        assert torch.allclose(f+method.feature(lq, lp, variant), torch.ones_like(f))


@pytest.mark.parametrize('variant, expected', [('v2', .5), ('v3', 0.)])
def test_agreeing_views(variant, expected):
    lp = torch.randn(4, 29).log_softmax(-1)
    mu, picked, e = method.centered_features(lp, lp, torch.tensor([0, 3, 4, 1]), variant)
    assert torch.allclose(mu, torch.full_like(mu, expected), atol=1e-7)
    assert torch.allclose(picked, torch.full_like(picked, expected), atol=1e-7)
    assert e.abs().max() < 1e-7


def test_runtime_accepts_tp4_when_actor_topology_divides(monkeypatch):
    monkeypatch.setattr(method.Settings, "from_env", lambda: SimpleNamespace(calibration=2))
    monkeypatch.delenv("SGLANG_RETURN_ORIGINAL_LOGPROB", raising=False)
    monkeypatch.delenv("OPSD_LORA_DROPOUT", raising=False)
    args = SimpleNamespace(
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        tensor_model_parallel_size=4,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        n_samples_per_prompt=1,
        rollout_batch_size=4,
        rollout_top_p=1,
        rollout_top_k=-1,
        rollout_temperature=1,
        attention_dropout=0,
        hidden_dropout=0,
        entropy_coef=0,
        calculate_per_token_loss=False,
        micro_batch_size=1,
        use_dynamic_batch_size=False,
        update_weights_interval=1,
        keep_old_actor=False,
    )
    assert method.validate_runtime(args).calibration == 2


class ToyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 4)
        self.output_layer = torch.nn.Linear(4, 20, bias=False)

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids).cumsum(1).tanh().transpose(0, 1)
        return self.output_layer(hidden)


def test_probe_chunk_accumulation_matches_full_gradient_moments(monkeypatch):
    from methods import srlvr
    monkeypatch.setattr(srlvr, '_reconstruct_dp_gradients', lambda model: None)
    parameters = [torch.nn.Parameter(torch.zeros(7)), torch.nn.Parameter(torch.zeros(3))]
    entries = [(p, 1.) for p in parameters]
    chunks_z = [[torch.arange(7.) / 13, torch.arange(3.) / 7],
                [torch.arange(7.).flip(0) / 17, -torch.arange(3.) / 11]]
    chunks_y = [[-g / 3 for g in part] for part in chunks_z]
    def accumulate(chunks):
        total = None
        for part in chunks:
            for p, g in zip(parameters, part):
                p.main_grad = g.clone()
            total = method._accumulate_probe_gradient(None, entries, total)
        for p in parameters:
            p.main_grad.zero_()
        return total
    z, y = accumulate(chunks_z), accumulate(chunks_y)
    full_z = torch.cat([sum(part[i].double() for part in chunks_z) for i in range(2)])
    full_y = torch.cat([sum(part[i].double() for part in chunks_y) for i in range(2)])
    expected = (full_z @ full_z, full_z @ full_y, full_y @ full_y)
    assert method._probe_moments(z, y, torch.ones(2), 1, None) == pytest.approx(
        [v.item() for v in expected], rel=2e-6)
    zz, zy, yy = method._probe_moments(z, None, torch.ones(2), 1, None)
    assert zz == pytest.approx(expected[0].item(), rel=2e-6)
    assert zy == yy == 0.


@pytest.mark.parametrize('variant', ['v2', 'v3'])
def test_chunked_projection_matches_dense_full_vocab_and_uses_actor_mask(monkeypatch, variant):
    torch.manual_seed(21)
    actor = ToyActor().train()
    normal = [1, 2, 3, 4, 5, 6, 7]
    teacher = [8, 9] + normal
    mask = [1, 0, 1, 1, 1]
    cfg = method.Settings(variant, .73, position_chunk=2)
    hn = hsi2_jsd._hidden_states(actor, actor.output_layer, normal)[1:6]
    hq = hsi2_jsd._hidden_states(actor, actor.output_layer, teacher)[3:8]
    lp = (hn @ actor.output_layer.weight[:17].t()/.67).log_softmax(-1)
    lq = (hq @ actor.output_layer.weight[:17].t()/.67).log_softmax(-1)
    expected = method.centered_features(lp, lq, torch.tensor(normal[-5:]), variant)
    counts = []
    original = method.F.linear

    def counted(hidden, weight, bias=None):
        counts.append((hidden.shape[0], weight.shape[0]))
        return original(hidden, weight, bias)

    monkeypatch.setattr(method.F, 'linear', counted)
    result = method.score_row([actor], normal, mask, (teacher, 4, 5), cfg, .67, 17)
    assert counts == [(2, 17)] * 4
    for key, reference in zip(('mu', 'f_chosen', 'e'), expected):
        for index, active in enumerate(mask):
            assert result[key][index] == pytest.approx(reference[index].item() if active else 0., abs=1e-7)
    assert actor.training
    assert all(param.grad is None for param in actor.parameters())


def test_full_teacher_view_appears_only_after_rollout(tokenizer, monkeypatch):
    from tasks.alfworld.episode import EpisodeResult, TurnSample
    from tasks.alfworld.prompts import TaskView, build_turn_prompt_ids
    from tasks.alfworld.rollout import OutcomeAnchor, build_group_samples, _prompt_ids_for
    from tasks.alfworld.privilege import render_hpi_block, has_future
    from tasks.alfworld.hsi_shared import render_outcome_descriptor
    monkeypatch.setenv('OPSD_SRLVR', '1')
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    trajectory = make_trajectory(num_turns=2)
    turns = []
    for index, source in enumerate(trajectory.turns):
        view = TaskView(trajectory.task_description, source.observation, source.admissible_actions, [])
        prompt = build_turn_prompt_ids(tokenizer, view)
        turns.append(TurnSample(0, 0, trajectory.task_id, index, prompt,
            source.generated_token_ids, source.generated_text, view,
            generated_log_probs=[-.1] * len(source.generated_token_ids)))
    episode = EpisodeResult(trajectory, turns)
    rows = build_group_samples(tokenizer, None, [episode], OutcomeAnchor([0.], [1.]),
                               0, with_hpi=False, future_gap=1)
    for row, turn in zip(rows, turns):
        assert row.tokens == turn.prompt_token_ids + turn.generated_token_ids
        block = render_hpi_block(trajectory, turn.turn_idx,
            with_future=has_future(trajectory, turn.turn_idx, 1), future_gap=1,
            outcome_text=render_outcome_descriptor(trajectory.success))
        prompt = _prompt_ids_for(tokenizer, turn.view, block)
        teacher, plen, n = row.metadata['hsi_teacher_view']
        assert teacher[:plen] == prompt
        assert teacher[plen:] == row.tokens[-n:]
        assert teacher[:plen] != row.tokens[:-n]
    # The last turn still has the natural-language outcome, with no future trace.
    assert '(not available)' in block
    assert render_outcome_descriptor(True) in block


def test_full_teacher_keeps_each_eligible_raw_transition_and_student_stays_causal(tokenizer, monkeypatch):
    from tasks.alfworld.episode import EpisodeResult, TurnSample
    from tasks.alfworld.prompts import TaskView, build_turn_prompt_ids
    from tasks.alfworld.rollout import OutcomeAnchor, build_group_samples, _prompt_ids_for
    from tasks.alfworld.privilege import render_hpi_block

    monkeypatch.setenv('OPSD_SRLVR', '1')
    monkeypatch.setenv('SRLVR_VARIANT', 'v2')
    monkeypatch.setenv('SRLVR_COMPACT_FUTURE', '0')
    trajectory = make_trajectory(num_turns=4)
    # Equal adjacent transitions are intentionally repeated; Full must retain both.
    trajectory.turns[2].action_text = trajectory.turns[1].action_text
    trajectory.turns[2].next_observation = trajectory.turns[1].next_observation
    turns = []
    for index, source in enumerate(trajectory.turns):
        view = TaskView(trajectory.task_description, source.observation, source.admissible_actions, [])
        prompt = build_turn_prompt_ids(tokenizer, view)
        turns.append(TurnSample(0, 0, trajectory.task_id, index, prompt,
            source.generated_token_ids, source.generated_text, view,
            generated_log_probs=[-.1] * len(source.generated_token_ids)))
    rows = build_group_samples(tokenizer, None, [EpisodeResult(trajectory, turns)],
        OutcomeAnchor([0.], [1.]), 0, with_hpi=False, future_gap=1)
    row = rows[0]
    block = render_hpi_block(trajectory, 0, with_future=True, future_gap=1)
    assert block.count('[Future transition 1]') == 1
    assert block.count('[Future transition 2]') == 1
    assert block.count('[Future transition 3]') == 1
    assert 'Executed action: act1' in block and 'Executed action: act1' in block.split('[Future transition 2]', 1)[1]
    assert 'Resulting environment observation: obs2' in block
    assert 'line delta from turn' not in block
    assert row.tokens == turns[0].prompt_token_ids + turns[0].generated_token_ids
    teacher, prompt_len, target_len = row.metadata['hsi_teacher_view']
    assert teacher[-target_len:] == row.tokens[-target_len:]
    assert teacher[:prompt_len] != row.tokens[:-target_len]
    assert row.loss_mask == [1] * len(turns[0].generated_token_ids)
    assert row.rollout_log_probs == turns[0].generated_log_probs


def test_reverse_scan_excludes_current_turn_padding_and_other_trajectories():
    def row(pos, turn, e, mask):
        return dict(meta=dict(position=pos, turn=turn, e=e), mask=mask)
    rows = [row(0, 2, [.4], [1]), row(1, 0, [-.9], [1]),
            row(0, 0, [.8, -.3], [1, 1]), row(0, 1, [-.2, 1., .5], [1, 0, 1])]
    trajectories = method.fill_future(rows, .7)
    first = trajectories[0]
    assert first[0]['meta']['B'] == pytest.approx(.3*(-.2 + .7*.5 + .7**2*.4))
    assert first[1]['meta']['B'] == pytest.approx(.3*.4)
    assert first[-1]['meta']['B'] == 0.
    assert trajectories[1][0]['meta']['B'] == 0.


def test_lambda_freezes_from_prior_moments_and_history_uses_prior_b():
    cfg = method.Settings('v2', .73, calibration=2)
    state = dict(b=.5, s=1., history=[])
    frozen = method.frozen_history(state, cfg, 0)
    assert frozen == (.5, 1., 0.)
    pending = method.next_history(state, [[1., 2., .5], [0., 1., .5]], cfg, 1., .01, 0)
    assert method.frozen_history(state, cfg, 0) == frozen
    b, s, lam = method.frozen_history(pending, cfg, 1)
    assert b == pytest.approx(2/3)
    assert s == pytest.approx((.01+.25)**.5)
    assert lam == pytest.approx(cfg.rho/3)
    assert method.next_history(state, [], cfg, 1., 0., 0)['lambda_raw'] == 0.
    with pytest.raises(RuntimeError, match='consecutive'):
        method.frozen_history(pending, cfg, 0)
    with pytest.raises(RuntimeError, match='variant'):
        method.frozen_history(pending, replace(cfg, variant='v3'), 1)


def _framework_loss_function():
    # Execute the actual framework reducer without importing the accelerator
    # runtime. Only its external services are supplied as test seams.
    path = Path(__file__).parents[3] / 'frameworks/slime-ascend/slime/backends/megatron_utils/loss.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_function')
    namespace = dict(torch=torch, load_function=lambda _: method.actor_loss,
        get_sum_of_sample_mean=lambda *a, **k: None,
        mpu=SimpleNamespace(get_data_parallel_world_size=lambda **k: 2, get_context_parallel_world_size=lambda: 1))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['loss_function']


@pytest.mark.parametrize('probe', [False, True])
def test_real_framework_reduction_is_trajectory_token_mean(monkeypatch, probe):
    reducer = _framework_loss_function()
    theta = torch.tensor(.23, requires_grad=True)
    args = SimpleNamespace(vocab_size=None, srlvr_calibration=probe, eps_clip=.2, eps_clip_high=.2,
        srlvr_loss_scale=3., loss_type='custom_loss', custom_loss_function_path='methods.srlvr_hsi.actor_loss',
        calculate_per_token_loss=False, qkv_format='bshd', recompute_loss_function=False,
        allgather_cp=False, global_batch_size=6)
    monkeypatch.setattr(shared_loss, 'log_probs_and_entropy', lambda logits, **k: (None, {'log_probs': [logits.flatten()]}))
    rows = [(torch.tensor([1., 2.]), .3, torch.tensor([1., 0.])),
            (torch.tensor([3., -1., .2]), -.4, torch.ones(3)),
            (torch.tensor([-.5]), .2, torch.ones(1)),
            (torch.tensor([.8, -.2]), -.1, torch.ones(2)),
            (torch.tensor([.2, .7]), .6, torch.ones(2)),
            (torch.tensor([1.]), 0., torch.zeros(1))]
    # DP=2, three microbatches per rank; the final row is padding. N=2 real
    # trajectories, with unequal turn/token counts. Probe duplicates the same
    # three rows on both ranks (N irrelevant; scaling changed to sum, not mean).
    if probe:
        rows = rows[:3] * 2
    denominators = [5, 5, 5, 4, 4, 4] if not probe else [5] * 6
    actual = theta * 0
    for (x, advantage, mask), length in zip(rows, denominators):
        logits = theta*x
        batch = dict(unconcat_tokens=[], total_lengths=[len(x)], response_lengths=[len(x)],
            loss_masks=[mask], rollout_log_probs=[logits.detach()],
            opsd_advantages=[torch.full_like(x, advantage)],
            srlvr_rows=[dict(actor_token_count=length)])
        loss, _, _ = reducer(args, batch, 3, logits)
        actual = actual + loss/3/2  # Megatron microbatch and DP averaging.
    reference_rows = rows[:3] if probe else rows
    expected = -sum(a*(theta*x*mask).sum()/length
                    for (x, a, mask), length in zip(reference_rows, denominators))/(1 if probe else 2)
    assert torch.allclose(torch.autograd.grad(actual, theta)[0], torch.autograd.grad(expected, theta)[0], atol=1e-6)


def test_variant_entries_differ_only_by_feature_selection():
    from opsd_core.registry import resolve
    a, b = (resolve('alfworld', name)[1] for name in ('srlvr_v2', 'srlvr_v3'))
    assert a.loss == b.loss == 'methods.srlvr_hsi.actor_loss'
    assert a.convert == b.convert == 'methods.srlvr_hsi.convert'
    assert a.num_rollout == b.num_rollout == 25
    assert replace(a, name=b.name, slug=b.slug, extra_env=b.extra_env) == b


def test_confirmed_gamma_and_old_c_is_irrelevant(monkeypatch):
    monkeypatch.setenv('SRLVR_VARIANT', 'v3')
    monkeypatch.delenv('SRLVR_GAMMA', raising=False)
    assert method.Settings.from_env().gamma == .99
    monkeypatch.setenv('SRLVR_GAMMA', '.73')
    monkeypatch.setenv('SRLVR_C', 'not-a-number')
    assert method.Settings.from_env().variant == 'v3'


@pytest.mark.parametrize('variant', ['v2', 'v3'])
def test_train_batch_100_updates_uses_prior_lambda_on_every_real_trajectory(monkeypatch, tmp_path, variant):
    import sys
    import types
    from dataclasses import asdict
    cfg = method.Settings(variant, .73, calibration=2)
    args = SimpleNamespace(n_samples_per_prompt=1, rollout_batch_size=3,
        rollout_top_p=1., rollout_top_k=-1, rollout_temperature=.8, micro_batch_size=1,
        vocab_size=20, data_pad_size_multiplier=1, eps_clip=.2, eps_clip_high=.2,
        save=str(tmp_path/'checkpoints'), srlvr_dp_verified=True, use_kl_loss=True)
    monkeypatch.setattr(method.Settings, 'from_env', classmethod(lambda cls: cfg))
    mpu = SimpleNamespace(get_data_parallel_world_size=lambda: 1,
                         get_data_parallel_rank=lambda: 0, get_data_parallel_group_gloo=lambda: None)
    config = SimpleNamespace(no_sync_func=None, grad_sync_func=None, param_sync_func=None)
    monkeypatch.setitem(sys.modules, 'megatron.core', SimpleNamespace(mpu=mpu))
    monkeypatch.setitem(sys.modules, 'megatron.core.utils', SimpleNamespace(get_model_config=lambda _: config))
    monkeypatch.setattr(torch.distributed, 'all_gather_object', lambda out, value, **kw: out.__setitem__(0, deepcopy(value)))
    datamodule = types.ModuleType('slime.backends.megatron_utils.data')
    datamodule.get_data_iterator = lambda args, model, batch: ([batch], [len(batch['tokens'])])
    monkeypatch.setitem(sys.modules, datamodule.__name__, datamodule)
    modelmodule = types.ModuleType('slime.backends.megatron_utils.model')
    calls = []
    current_grad = []
    prior_state = []
    frozen = []

    def train(rollout_id, model, optimizer, scheduler, iterator, micro):
        batch = iterator[0]
        assert args.srlvr_state is prior_state[0]
        if getattr(args, 'srlvr_gradient_probe', None) is not None:
            # The raw probe's supplied coefficients carry the actual Z and Y
            # definitions; simulate a two-dimensional score vector.
            coeffs = torch.tensor([float(a[0])/m['actor_token_count']
                                   for a, m in zip(batch['opsd_advantages'], batch['srlvr_rows'])])
            grad = -torch.stack([coeffs.sum(), (coeffs * torch.arange(1, len(coeffs)+1)).sum()])
            current_grad[:] = [grad]
            config.no_sync_func = object()
            args.srlvr_gradient_probe(model)
            calls.append(('probe', rollout_id))
        else:
            b, s, lam = frozen[0]
            for meta, advantages in zip(batch['srlvr_rows'], batch['opsd_advantages']):
                expected = (meta['terminal_reward']-b-lam*meta['B'])/s
                assert torch.allclose(advantages, torch.full_like(advantages, expected))
            calls.append(('actor', rollout_id))
            args.srlvr_actual_optimizer_steps = getattr(args, 'srlvr_actual_optimizer_steps', 0) + 1

    modelmodule.train = train
    monkeypatch.setitem(sys.modules, modelmodule.__name__, modelmodule)
    monkeypatch.setattr(method, 'capture_full_gradient', lambda _: current_grad[0])
    monkeypatch.setattr('methods.srlvr._calibration_entries',
                        lambda _: ([(None, 1.)], 1, 0, None))
    monkeypatch.setattr(method, '_accumulate_probe_gradient',
                        lambda model, entries, total: [current_grad[0].clone()]
                        if total is None else [total[0] + current_grad[0]])

    def score(actor, tokens, mask, view, cfg, temperature, vocab, context_limit=None):
        # Two different trajectories carry opposite realized feature directions.
        error = .2 if tokens[0] != 11 else -.2
        return dict(mu=[.3, .3], f_chosen=[.3+error]*2, e=[error]*2,
                    student_logp=[-.7, -.8])

    monkeypatch.setattr(method, 'score_row', score)
    actor = SimpleNamespace(args=args, model=[SimpleNamespace()], optimizer=object(), opt_param_scheduler=object(),
                            weights_backuper=SimpleNamespace(backup=lambda _: None))
    reference_switches = []
    actor._switch_model = lambda tag: reference_switches.append(tag)
    def reference_forward(iterator, micro, store_prefix):
        assert reference_switches[-1] == 'ref'
        assert store_prefix == 'ref_'
        return dict(ref_log_probs=[torch.full_like(lp, -.9) for lp in iterator[0]['rollout_log_probs']])
    actor.compute_log_prob = reference_forward
    for k in range(100):
        meta = [dict(position=i, turn=t, calibration=i<2, policy_version=k,
                     terminal_reward=float(i!=1), teacher_view=([10+i, 2, 3, 4], 2, 2))
                for i in range(3) for t in range(2)]
        data = dict(tokens=[torch.tensor([10+m['position'], 2, 3, 4]) for m in meta],
            rollout_log_probs=[torch.tensor([-.7, -.8]) for _ in meta],
            loss_masks=[torch.ones(2) for _ in meta], opsd_advantages=[torch.zeros(2) for _ in meta],
            srlvr_rows=meta, total_lengths=[4]*6, response_lengths=[2]*6,
            dynamic_global_batch_size=6, srlvr_batch=dict(N=3, rows=6, mode='history', config=asdict(cfg)))
        args.srlvr_state = method.prepare_history(getattr(args, 'srlvr_state', None), cfg, k)
        prior_state[:] = [args.srlvr_state]
        frozen[:] = [method.frozen_history(args.srlvr_state, cfg, k)]
        if k == 0:
            assert frozen[0][2] == 0.
        else:
            assert frozen[0][2] != 0.
        method.train_batch(actor, k, data)
        assert config.no_sync_func is None
        assert args.srlvr_state['last_rollout'] == k
    assert [k for what, k in calls if what == 'actor'] == list(range(100))
    assert len([1 for what, _ in calls if what == 'probe']) == 100*2*cfg.calibration
    assert args.srlvr_actual_optimizer_steps == args.srlvr_state['actor_optimizer_steps'] == 100


@pytest.mark.parametrize('probe', [False, True])
def test_initial_reference_kl_enters_actor_gradient_but_not_raw_probe(monkeypatch, probe):
    args = SimpleNamespace(vocab_size=None, srlvr_calibration=probe, eps_clip=.2,
        eps_clip_high=.2, srlvr_loss_scale=2., use_kl_loss=True,
        kl_loss_type='low_var_kl', kl_loss_coef=.001, ref_load='initial-model')
    lp = torch.tensor([-.2, -.8, -.5], requires_grad=True)
    ref = torch.tensor([-.7, -.3, -.9], requires_grad=True)
    mask = torch.tensor([1., 0., 1.])
    monkeypatch.setattr(shared_loss, 'log_probs_and_entropy',
        lambda logits, **kw: (None, {'log_probs': [logits]}))
    batch = dict(unconcat_tokens=[], total_lengths=[3], response_lengths=[3],
        loss_masks=[mask], rollout_log_probs=[lp.detach()],
        opsd_advantages=[torch.full_like(lp, .4)], ref_log_probs=[ref],
        srlvr_rows=[dict(actor_token_count=7)])
    actual, metrics = method.actor_loss(args, batch, lp, None)
    delta = ref.detach() - lp
    kl = (delta.exp()-1-delta).clamp(-10, 10)
    expected = -(lp*.4*mask).sum() if probe else -(torch.exp(lp-lp.detach())*.4*mask).sum()+.001*(kl*mask).sum()
    expected = expected*2/7
    assert torch.allclose(torch.autograd.grad(actual, lp, retain_graph=True)[0],
                          torch.autograd.grad(expected, lp)[0])
    assert torch.autograd.grad(actual, ref, allow_unused=True)[0] is None
    assert metrics['srlvr_reference_kl_sum'] == pytest.approx(0. if probe else float((kl.detach()*mask).sum()*2/7))


# --- stale-policy (asynchronous rollout) contract -------------------------

def _rows_with_versions(*versions):
    return [dict(meta=dict(policy_version=v)) for v in versions]


def test_resolve_policy_gap_accepts_one_step_stale():
    # Arrange: an async batch generated one weight version behind rollout 5.
    rows = _rows_with_versions(4, 4)
    # Act
    behavior, gap = method.resolve_policy_gap(rows, rollout_id=5, max_stale=1)
    # Assert
    assert (behavior, gap) == (4, 1)


def test_resolve_policy_gap_accepts_on_policy_batch():
    rows = _rows_with_versions(7, 7, 7)
    assert method.resolve_policy_gap(rows, rollout_id=7, max_stale=1) == (7, 0)


def test_resolve_policy_gap_rejects_mixed_snapshots():
    rows = _rows_with_versions(4, 5)
    with pytest.raises(RuntimeError, match="mixed rollout snapshots"):
        method.resolve_policy_gap(rows, rollout_id=5, max_stale=1)


def test_resolve_policy_gap_rejects_gap_beyond_max_stale():
    # A two-version gap is outside the truncated-IS contract.
    with pytest.raises(RuntimeError, match="exceeds max_stale"):
        method.resolve_policy_gap(_rows_with_versions(3), rollout_id=5, max_stale=1)


def test_resolve_policy_gap_rejects_future_policy():
    with pytest.raises(RuntimeError, match="future policy"):
        method.resolve_policy_gap(_rows_with_versions(6), rollout_id=5, max_stale=1)


def test_settings_reject_deeper_than_one_step_staleness():
    with pytest.raises(ValueError, match="stale policy"):
        replace(method.Settings("v2", 0.99), stale_lag=2, max_stale=2).validate()
    with pytest.raises(ValueError, match="stale policy"):
        replace(method.Settings("v2", 0.99), stale_lag=2, max_stale=1).validate()
    # One-step lag is the only supported asynchronous protocol.
    replace(method.Settings("v2", 0.99), stale_lag=1, max_stale=1).validate()


def test_attach_rows_stamps_generation_version(monkeypatch):
    # Arrange: async lag stamps the batch with rollout_id - 1.
    monkeypatch.setenv("SRLVR_VARIANT", "v2")
    monkeypatch.setenv("SRLVR_STALE_LAG", "1")
    rows = [SimpleNamespace(response_length=2, loss_mask=[1, 1],
                            metadata={"hsi_teacher_view": ([1, 2], 1, 1)})]
    episode = SimpleNamespace(turns=[SimpleNamespace(turn_idx=0, episode_id="e")],
                              reward=1.0)
    # Act
    method.attach_rows(rows, episode, group_id=0, rollout_id=5)
    # Assert
    assert rows[0].metadata["srlvr"]["policy_version"] == 4
    assert rows[0].metadata["srlvr"]["trajectory_id"] == "5:e"


def test_attach_rows_first_async_rollout_stays_on_policy(monkeypatch):
    # The first async rollout is generated before any weight update, so even
    # with the lag armed it must be stamped with its own version.
    monkeypatch.setenv("SRLVR_VARIANT", "v2")
    monkeypatch.setenv("SRLVR_STALE_LAG", "1")
    rows = [SimpleNamespace(response_length=2, loss_mask=[1, 1],
                            metadata={"hsi_teacher_view": ([1, 2], 1, 1)})]
    episode = SimpleNamespace(turns=[SimpleNamespace(turn_idx=0, episode_id="e")],
                              reward=1.0)
    method.attach_rows(rows, episode, group_id=0, rollout_id=0, start_rollout_id=0)
    assert rows[0].metadata["srlvr"]["policy_version"] == 0


# --- output-head selection and probe scheduling ---------------------------

def test_lm_head_weight_prefers_output_layer_when_untied():
    from types import SimpleNamespace
    from methods.hsi2_jsd import _lm_head_weight
    head = SimpleNamespace(weight=torch.zeros(4, 2))
    embed = torch.ones(4, 2)
    model = SimpleNamespace(shared_embedding_or_output_weight=lambda: embed)
    assert _lm_head_weight(model, head) is head.weight


def test_lm_head_weight_falls_back_to_shared_embedding_when_tied():
    from types import SimpleNamespace
    from methods.hsi2_jsd import _lm_head_weight
    embed = torch.ones(4, 2)
    model = SimpleNamespace(shared_embedding_or_output_weight=lambda: embed)
    assert _lm_head_weight(model, SimpleNamespace()) is embed


def _probe_traj(half_tokens, has_y=True):
    rows = [dict(tokens=[0] * half_tokens, mask=[1], meta=dict(B=1.)) for _ in range(2)]
    rows[-1]["meta"]["B"] = 1. if has_y else 0.
    return rows


def test_balanced_probe_assignment_picks_lowest_peak():
    from methods.srlvr_hsi import assign_balanced_probe_positions
    # workloads 2*[100, 10, 10, 100]; pairing (03)(12) is worst (200 vs 40)
    trajectories = {i: _probe_traj(w // 2) for i, w in enumerate([100, 10, 10, 100])}
    assert assign_balanced_probe_positions(trajectories, 4, 0, 2) == [0, 1]
    assert assign_balanced_probe_positions(trajectories, 4, 1, 2) == [2, 3]


def test_balanced_probe_assignment_falls_back_for_other_dp():
    from methods.srlvr_hsi import assign_balanced_probe_positions
    trajectories = {i: _probe_traj(10) for i in range(4)}
    assert assign_balanced_probe_positions(trajectories, 4, 1, 4) == [1]
