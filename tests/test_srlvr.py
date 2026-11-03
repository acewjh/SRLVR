import math
from types import SimpleNamespace

import pytest
import torch

from methods.srlvr import (Settings, centered_features, future_corrections,
    choose_lambda, calibration_statistics, history_update, actor_loss)


@pytest.mark.parametrize("temperature", [.4, 1., 2.])
def test_exact_conditional_center_with_zero_probability_and_mask(temperature):
    logits = torch.tensor([[1., 2., -float("inf"), -3.]], dtype=torch.float64)
    p = torch.softmax(logits / temperature, -1)
    mu, chosen, e = centered_features(p, torch.tensor([1]), 2.)
    f = (p.log()/2).clamp(-1, 0)
    assert f[0, 2] == -1
    assert abs((p*(f-mu[:, None])).sum().item()) < 1e-14
    assert torch.allclose(chosen-mu, e)


def test_strictly_future_whole_suffix_and_last_turn():
    features = [[.9], [.2, -.4], [.3]]
    actual = future_corrections(features)
    value = 0.
    for e in [.2, -.4, .3]:
        value += .5*(1-abs(value))*e
    assert actual == pytest.approx([value, .15, 0.])
    assert future_corrections([[-.9], features[1], features[2]]) == actual


@pytest.mark.parametrize("reward", [0., 1.])
@pytest.mark.parametrize("b", [.01, .5, .99])
def test_nonzero_signed_coefficients_and_weight_bounds(reward, b):
    cfg = Settings(mode="fixed", fixed=.9*min(b,1-b))
    lam, _ = choose_lambda(cfg, b, 2, 0, 0, 0)
    for B in [-1., -.2, 0., .7, 1.]:
        a = reward-b-lam*B
        w = 1-lam*B/(reward-b)
        assert a*(2*reward-1) > 0
        assert 1-cfg.rho-1e-12 <= w <= 1+cfg.rho+1e-12


def test_certified_inputs_and_explicit_fixed_required():
    with pytest.raises(ValueError, match="population"):
        Settings(mode="certified").validate()
    with pytest.raises(ValueError, match="explicit"):
        Settings(mode="fixed").validate()
    with pytest.raises(ValueError, match="exceeds"):
        choose_lambda(Settings(mode="fixed", fixed=.46), .5, 2, 0,0,0)
    cfg = Settings(mode="certified", gz=1, gy=1, bound_basis="test-only analytical bounded toy model")
    lam, info = choose_lambda(cfg, .5, 2, .1, .1, .1)
    assert lam == 0 and info["zero_reason"] == "LQ_zero"
    lam, info = choose_lambda(cfg, .5, 100000, .3, .2, .5)
    assert 0 < lam <= .45


def test_empirical_lambda_uses_calibration_ratio_with_coefficient_bound():
    cfg = Settings(mode="empirical")
    lam, info = choose_lambda(cfg, .2, 8, .3, .1, 1.)
    assert lam == pytest.approx(.18)
    assert info["zero_reason"] == ""
    assert choose_lambda(cfg, .2, 8, -.03, .1, 1.)[0] == pytest.approx(-.18)
    assert choose_lambda(cfg, .2, 8, 0., .1, 1.)[1]["zero_reason"] == "C_zero"


def test_history_uses_own_historical_baselines_and_score_norms():
    state = dict(b=.5, s=1., history=[])
    cfg = Settings(window=3)
    next_state = history_update(state, [[1., 2., .5], [0., 1., .5]], cfg)
    assert state == dict(b=.5, s=1., history=[])
    assert next_state["b"] == pytest.approx(2/3)
    assert next_state["s"] == pytest.approx(math.sqrt(.01+.25))
    # Zero score weight preserves the last effective reference.
    assert history_update(dict(b=.3,s=1.,history=[]), [[1.,0.,.3]], cfg)["b"] == .3


def toy_turns(theta):
    # Token vectors include an environment token whose mask must zero its gradient.
    x = [torch.tensor([[1.,2.],[3.,-1.],[.2, .7]], dtype=torch.float64),
         torch.tensor([[.2,1.],[-2.,.5]], dtype=torch.float64)]
    return [(v @ theta).log_softmax(0) for v in x]


def test_full_gradients_calibration_and_microbatch_accumulation():
    theta = torch.tensor([.3, -.2], dtype=torch.float64, requires_grad=True)
    masks = [torch.tensor([1.,0.,1.]), torch.ones(2)]
    B = [.4, 0.]
    zs, ys = [], []
    for reward in [0., 1.]:
        lp = toy_turns(theta)
        z = torch.autograd.grad((reward-.5)*sum((l*m).sum() for l,m in zip(lp,masks)), theta, retain_graph=True)[0]
        y = torch.autograd.grad(sum(b*(l*m).sum() for b,l,m in zip(B,lp,masks)), theta)[0]
        zs.append(z); ys.append(y)
    c,d,q = calibration_statistics(zs,ys)
    assert c == pytest.approx(sum(torch.dot(z,y).item() for z,y in zip(zs,ys))/2)
    assert d == pytest.approx(sum(torch.dot(y,y).item() for y in ys)/2)
    assert q == pytest.approx(torch.dot(zs[0],zs[1]).item())
    # Same sum/N reduction in one graph and streaming microbatch backward.
    a = [.2, -.5]
    lp = toy_turns(theta)
    whole = -sum(v*(l*m).sum() for v,l,m in zip(a,lp,masks))/3
    reference = torch.autograd.grad(whole, theta)[0]
    accumulated = torch.zeros_like(theta)
    for t in range(2):
        lp = toy_turns(theta)
        accumulated += torch.autograd.grad(-a[t]*(lp[t]*masks[t]).sum()/3,theta)[0]
    assert torch.allclose(reference, accumulated, atol=1e-14)


@pytest.mark.parametrize("lam", [0., .1])
def test_real_loss_gradient_turn_coefficients_and_framework_scaling(monkeypatch,lam):
    import tasks.alfworld.loss as shared
    logits = torch.randn(1,5,4, dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)[0,:,0]
    monkeypatch.setattr(shared,"log_probs_and_entropy",lambda *a,**k:(None,{"log_probs":[lp[:3],lp[3:]]}))
    a = [(.5-lam*.4), -.5]
    masks = [torch.tensor([1.,0.,1.]),torch.ones(2)]
    args = SimpleNamespace(srlvr_loss_scale=4/3)
    batch = dict(unconcat_tokens=[],total_lengths=[3,2],response_lengths=[3,2],
                 loss_masks=masks,rollout_log_probs=[lp[:3].detach(),lp[3:].detach()],
                 opsd_advantages=[torch.full((3,),a[0]),torch.full((2,),a[1])])
    loss,_ = actor_loss(args,batch,logits,None)
    actual = torch.autograd.grad(loss,logits,retain_graph=True)[0]
    reference = -sum(v*(l*m).sum() for v,l,m in zip(a,[lp[:3],lp[3:]],masks))*4/3
    expected = torch.autograd.grad(reference,logits)[0]
    assert torch.allclose(actual,expected,atol=1e-7)
    assert actual[0,1].abs().sum() == 0


def test_ppo_ratio_is_tokenwise_clipped_and_has_correct_gradient(monkeypatch):
    import tasks.alfworld.loss as shared
    logits = torch.tensor([[[0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)[0, :, 0]
    old = lp.detach() + torch.tensor([-.6, .2, .4])
    monkeypatch.setattr(shared, "log_probs_and_entropy", lambda *a, **k: (None, {"log_probs": [lp]}))
    args = SimpleNamespace(srlvr_loss_scale=1., srlvr_calibration=False, rollout_temperature=1., rollout_top_p=1., rollout_top_k=-1, save="/tmp")
    batch = dict(unconcat_tokens=[torch.tensor([1,2,3])], total_lengths=[3], response_lengths=[3],
                 loss_masks=[torch.ones(3)], rollout_log_probs=[old],
                 opsd_advantages=[torch.ones(3)])
    loss, _ = actor_loss(args, batch, logits, None)
    ratios = torch.exp(lp.detach() - old)
    expected = -torch.minimum(ratios, ratios.clamp(.8, 1.2)).sum()
    assert loss.item() == pytest.approx(expected.item())
    assert args.srlvr_ratio_mean / args.srlvr_ratio_count == pytest.approx(ratios.mean().item())
    assert args.srlvr_clip_count == 1

    batch["rollout_log_probs"] = [lp.detach() - 1.0]
    clipped_loss, _ = actor_loss(args, batch, logits, None)
    assert args.srlvr_ratio_mean / args.srlvr_ratio_count > 1.2
    assert args.srlvr_clip_count > 0
    grad = torch.autograd.grad(clipped_loss, logits)[0]
    assert torch.isfinite(grad).all()


def test_alignment_mismatch_is_diagnostic_when_ratio_is_enabled(monkeypatch, caplog):
    import tasks.alfworld.loss as shared
    logits = torch.zeros(1, 2, 4, dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)[0, :, 0]
    monkeypatch.setattr(shared, "log_probs_and_entropy", lambda *a, **k: (None, {"log_probs": [lp]}))
    args = SimpleNamespace(srlvr_loss_scale=1., srlvr_calibration=False, rollout_temperature=1., rollout_top_p=1., rollout_top_k=-1, save="/tmp")
    batch = dict(unconcat_tokens=[torch.tensor([1,2])], total_lengths=[2], response_lengths=[2],
                 loss_masks=[torch.ones(2)], rollout_log_probs=[lp.detach() + 1.0],
                 opsd_advantages=[torch.ones(2)])
    with caplog.at_level("WARNING"):
        actor_loss(args, batch, logits, None)
    assert "token PPO actor surrogate" in caplog.text


def test_matrix_commands_share_entry_and_never_collide():
    from scripts.batch_experiments import commands
    combos = list(commands(["grpo","gspo","srlvr"],["alfworld"],[42,43],50,32,"stamp","empirical"))
    assert len({env["RUN_NAME"] for env,_ in combos}) == 6
    assert len({tuple(argv) for _,argv in combos}) == 1
    assert {env["NUM_ROLLOUT"] for env,_ in combos} == {"50"}
    assert {env["ROLLOUT_BS"] for env,_ in combos} == {"4", "32"}
    assert {env["SRLVR_MODE"] for env,_ in combos} == {"empirical"}


def test_reference_kl_enters_actor_gradient_but_not_calibration(monkeypatch):
    import tasks.alfworld.loss as shared
    logits = torch.tensor([[[.1, -.4], [.3, -.1]]], requires_grad=True)
    lp = logits.log_softmax(-1)[0, :, 0]
    ref = lp.detach() - .4
    monkeypatch.setattr(shared, "log_probs_and_entropy",
                        lambda *a, **k: (None, {"log_probs": [lp]}))
    args = SimpleNamespace(srlvr_loss_scale=1., use_kl_loss=True,
                           kl_loss_coef=.001, kl_loss_type="low_var_kl")
    batch = dict(unconcat_tokens=[], total_lengths=[2], response_lengths=[2],
                 loss_masks=[torch.ones(2)], rollout_log_probs=[lp.detach()],
                 opsd_advantages=[torch.zeros(2)], ref_log_probs=[ref])
    actor, _ = actor_loss(args, batch, logits, None)
    expected = .001 * shared.reference_kl(args, lp, ref).sum()
    assert actor.item() == pytest.approx(expected.item())
    assert torch.allclose(torch.autograd.grad(actor, logits, retain_graph=True)[0],
                          torch.autograd.grad(expected, logits, retain_graph=True)[0])
    args.srlvr_calibration = True
    batch.pop("ref_log_probs")
    calibration, _ = actor_loss(args, batch, logits, None)
    assert calibration.item() == 0
    assert torch.autograd.grad(calibration, logits)[0].abs().sum() == 0
