"""CPU regressions for the sampler/save boundaries without loading NPU runtimes.

Extract actual definitions from the vendored frameworks: their module imports
otherwise initialize unavailable CUDA/NPU dependencies on a CPU test host.
"""
import ast
import asyncio
import io
import logging
import os
import pickle
import re
import subprocess
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from opsd_core.token_cache import TokenCache, encode_prompt, token_cache_scope

ROOT = Path(__file__).resolve().parents[1]
SAMPLER = ROOT / "frameworks/sglang/python/sglang/srt/layers/sampler.py"
CHECKPOINT = ROOT / "frameworks/Megatron-LM/megatron/training/checkpointing.py"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def sampler_namespace():
    return definitions(SAMPLER, ["Sampler", "categorical_with_seed_cdf"], dict(
        torch=torch, nn=torch.nn, os=os, time=time,
        logger=logging.getLogger(__name__), _CDF_SEARCHSORTED_OK=True,
        SYNC_TOKEN_IDS_ACROSS_TP=False, SGLANG_RETURN_ORIGINAL_LOGPROB=False,
    ))


@pytest.mark.parametrize("variant", ["v1", "v2", "v3"])
@pytest.mark.parametrize("aux", ["none", "top", "ids"])
def test_sampler_selected_logprob_and_auxiliary_paths(monkeypatch, variant, aux):
    monkeypatch.setenv("OPSD_SRLVR", "1")
    monkeypatch.setenv("SRLVR_VARIANT", variant)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    ns = sampler_namespace()
    chosen = torch.tensor([0, 2], dtype=torch.int32)
    ns["sampling_from_probs_torch"] = lambda p, **kw: chosen
    ns["get_top_logprobs"] = lambda lp, nums, **kw: torch.topk(lp, max(nums), dim=-1)
    ns["get_token_ids_logprobs"] = lambda lp, ids, **kw: (lp[:, [1]], ids)
    sampler = ns["Sampler"].__new__(ns["Sampler"])
    torch.nn.Module.__init__(sampler)
    sampler.use_nan_detection = False
    sampler.tp_sync_group = None
    info = SimpleNamespace(is_all_greedy=False, need_top_p_sampling=False,
        need_top_k_sampling=False, need_min_p_sampling=False, has_custom_logit_processor=False,
        grammars=None, logit_bias=None, vocab_mask=None, acc_additive_penalties=None,
        acc_scaling_penalties=None, penalizer_orchestrator=None,
        temperatures=torch.tensor([[.7], [1.2]]), sampling_seed=None, device="cpu")
    logits = torch.tensor([[2., -4., 0., -float("inf")], [0., 1., -3., -float("inf")]])
    output = SimpleNamespace(next_token_logits=logits)
    if aux == "none":
        def no_vocab_logprob(*args):
            raise AssertionError("selected-token request constructed vocabulary logprobs")
        sampler._attach_logprobs_to_output = no_vocab_logprob
    result = sampler.forward(output, info, True,
        [1, 1] if aux == "top" else [0, 0],
        [[1], [1]] if aux == "ids" else [None, None], torch.tensor([0, 1]))
    p = torch.softmax(logits / info.temperatures, -1)
    expected = p[torch.arange(2), chosen.long()].log()
    assert torch.equal(result, chosen)
    assert torch.equal(output.next_token_logprobs, expected)
    assert torch.equal(output.customized_info["srlvr_logp"], expected)
    if variant == "v1":
        feature = (p.log() / 10).clamp(-1, 0)
        assert torch.equal(output.customized_info["srlvr_mu"], (p * feature).sum(-1))
    if aux == "top":
        assert torch.equal(output.next_token_top_logprobs_val, p.log().topk(1).values)
    if aux == "ids":
        assert torch.equal(output.next_token_token_ids_logprobs_val, p[:, [1]].log())
    info.need_top_p_sampling = True
    with pytest.raises(RuntimeError, match="full-softmax"):
        sampler.forward(output, info, False, [0, 0], [None, None], torch.tensor([0, 1]))


def test_cdf_zero_mass_boundaries_and_fallback():
    ns = sampler_namespace()
    ns["murmur_hash32"] = lambda *args: torch.tensor([[0], [2**32 - 1]], dtype=torch.int64)
    p = torch.tensor([[0., .25, .75, 0.], [0., .25, .75, 0.]])
    seeds, positions = torch.tensor([1, 2]), torch.tensor([0, 0])
    sample = ns["categorical_with_seed_cdf"]
    expected = torch.tensor([1, 2], dtype=torch.int32)
    assert torch.equal(sample(p, seeds, positions), expected)
    ns["_CDF_SEARCHSORTED_OK"] = False
    assert torch.equal(sample(p, seeds, positions), expected)


class CannotPickleGroup:
    def __reduce__(self):
        raise TypeError("cannot pickle 'ReloadableProcessGroup' object")


def test_step5_save_roundtrip_keeps_optimizer_scheduler_rng_and_live_head():
    ns = definitions(CHECKPOINT, ["_checkpoint_args", "generate_state_dict"], dict(
        Namespace=Namespace, mpu=SimpleNamespace(get_data_parallel_group=lambda **kw: None)))
    class Model(torch.nn.Linear):
        def sharded_state_dict(self, **kwargs):
            return self.state_dict()
    class Optimizer:
        is_stub_optimizer = False
        def sharded_state_dict(self, *args, **kwargs):
            return optimizer.state_dict()
    model = Model(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    def step():
        optimizer.zero_grad()
        model(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        scheduler.step()
    for _ in range(5):
        step()
    head = SimpleNamespace(tp_group=CannotPickleGroup())
    args = Namespace(ckpt_format="torch_dist", no_save_optim=False, no_save_rng=False,
        seed=42, _srlvr_head=head, _srlvr_hidden=True, _srlvr_full_loss_masks=torch.ones(2))
    with pytest.raises(TypeError, match="ReloadableProcessGroup"):
        pickle.dumps(args)
    state = ns["generate_state_dict"](args, [model], Optimizer(), scheduler,
        {"torch": torch.get_rng_state()}, iteration=5)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert args._srlvr_head is head  # save did not mutate live training handles
    assert not hasattr(restored["args"], "_srlvr_head")
    assert restored["iteration"] == 5 and restored["args"].seed == 42
    assert torch.equal(restored["rng_state"]["torch"], torch.get_rng_state())
    step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["opt_param_scheduler"])
    step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


class Tokenizer:
    calls = 0
    def encode(self, text, **kwargs):
        self.calls += 1
        return list(text.encode())


def test_cache_identity_eviction_copy_and_oversize():
    tok, other = Tokenizer(), Tokenizer()
    cache = TokenCache(max_entries=2, max_tokens=5, max_chars=5)
    result = cache.encode(tok, "hi")
    result[0] = -1
    assert cache.encode(tok, "hi") == list(b"hi")
    assert tok.calls == 1
    cache.encode(other, "hi")
    assert other.calls == 1
    cache.encode(tok, "abc")
    assert len(cache.entries) <= 2 and cache.tokens <= 5 and cache.chars <= 5
    cache.encode(tok, "oversize")
    assert not any(key[1] == "oversize" for key in cache.entries)
    cache.encode(tok, "hi")
    assert tok.calls == 4  # previous hi was evicted


def test_cache_batch_scope_concurrent_inheritance_and_teardown():
    tok = Tokenizer()
    async def run():
        async def one():
            return encode_prompt(tok, "same")
        with token_cache_scope() as cache:
            assert await asyncio.gather(*[one() for _ in range(8)]) == [list(b"same")] * 8
            assert cache.hits == 7 and cache.saved_tokens == 28
        assert not cache.entries
        with token_cache_scope() as next_batch:
            assert encode_prompt(tok, "same") == list(b"same")
            assert next_batch.hits == 0
    asyncio.run(run())
    assert tok.calls == 2


def test_gdn_removes_inherited_shim_and_stub_fails_early():
    shim = str(ROOT / "compat/sgl_kernel_npu_shim")
    command = 'source train_scripts/lib/npu_env.sh\nopsd_remove_gdn_kernel_shim "$PWD"\nprintf "%s" "$PYTHONPATH"'
    env = dict(os.environ, OPSD_QWEN35_GDN="1", PYTHONPATH=f"{shim}:/runtime:{shim}/")
    result = subprocess.run(["bash", "-uc", command], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    assert result.stdout == "/runtime"
    ns = {"__name__": "stub_test"}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("OPSD_QWEN35_GDN", "1")
        with pytest.raises(ImportError, match="requires real"):
            exec((Path(shim) / "sgl_kernel_npu/__init__.py").read_text(), ns)


def test_core_imports_do_not_depend_on_method_layer():
    violations = []
    for path in (ROOT / "opsd_core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            if any(m == "methods" or m.startswith("methods.") for m in modules):
                violations.append(f"{path.name}:{node.lineno}")
    assert not violations


@pytest.mark.parametrize("enabled,expected", [
    ("0", ["--sglang-disable-cuda-graph"]),
    ("1", ["--sglang-cuda-graph-max-bs", "8"]),
])
def test_graph_opt_in_argv(enabled, expected):
    script = (ROOT / "tasks/alfworld/scripts/_common_v2.sh").read_text()
    block = script[script.index("GRAPH_ARGS=("):script.index("TRAIN_ARGS=(")]
    env = dict(os.environ, SGLANG_ENABLE_NPU_GRAPH=enabled, SGLANG_CUDA_GRAPH_MAX_BS="8")
    result = subprocess.run(["bash", "-uc", block + '\nprintf "%s\\n" "${GRAPH_ARGS[@]}"'],
                            env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == expected


def test_effective_settings_recording_without_method_import(tmp_path, monkeypatch):
    from opsd_core.experiment_metrics import resolved
    import json
    monkeypatch.setenv("OPSD_EXPERIMENT_METRICS", "1")
    monkeypatch.setenv("OPSD_SRLVR", "1")
    monkeypatch.setenv("SRLVR_VARIANT", "v3")
    monkeypatch.setenv("SRLVR_MODE", "empirical")
    resolved(Namespace(save=str(tmp_path / "checkpoints"), seed=42))
    values = json.loads((tmp_path / "effective_args.json").read_text())
    assert values["srlvr_hsi"]["variant"] == "v3"
    assert values["srlvr_hsi"]["calibration"] == 8


def _tp_scoring_worker(rank, rendezvous, vocab_size):
    import torch.distributed as dist
    from methods import srlvr_hsi, tp_utils
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
        teacher = hidden.flip(1) + .1
        weight = torch.arange(56, dtype=torch.float32).reshape(14, 4) / 50
        targets = torch.tensor([0, vocab_size - 1, vocab_size // 2])
        start = rank * 7
        shard = weight[start:start + 7]
        lp = srlvr_hsi._local_global_log_probs(hidden, shard, None, .7,
                                              start, vocab_size, dist.group.WORLD)
        lq = srlvr_hsi._local_global_log_probs(teacher, shard, None, .7,
                                              start, vocab_size, dist.group.WORLD)
        dense_lp = (hidden @ weight[:vocab_size].t() / .7).log_softmax(-1)
        dense_lq = (teacher @ weight[:vocab_size].t() / .7).log_softmax(-1)
        calls = []
        original = tp_utils.tp_sum
        def counted(value, *args, **kwargs):
            calls.append(value.shape)
            return original(value, *args, **kwargs)
        tp_utils.tp_sum = counted
        for variant in ("v2", "v3"):
            actual = srlvr_hsi._centered_features_sharded(
                lp, lq, targets, variant, start, vocab_size, dist.group.WORLD)
            expected = (*srlvr_hsi.centered_features(dense_lp, dense_lq, targets, variant),
                        dense_lp[torch.arange(3), targets])
            for result, reference in zip(actual, expected):
                assert torch.allclose(result.double(), reference.double(), atol=1e-6)
        assert calls == [torch.Size([4, 3]), torch.Size([4, 3])]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("vocab_size", [11, 6])
def test_tp2_scoring_matches_dense_with_partial_or_fully_padded_shard(tmp_path, vocab_size):
    torch.multiprocessing.spawn(_tp_scoring_worker,
        args=(f"file://{tmp_path}/gloo", vocab_size), nprocs=2, join=True)


@pytest.mark.parametrize("tp,expected_actor,expected_rollout", [
    ("1", "1", "7"),
    ("2", "2", "6"),
])
def test_eval_split_gives_the_rest_of_the_node_to_the_engines(tp, expected_actor, expected_rollout):
    """An eval runs no training step, so the actor must take only its TP floor.

    The block used to default to 4/4 while its own comment called for
    "7 rollout + 1 actor"; on 2026-10-28 that ran a real ALFWorld eval on 4 of 8
    cards. The split is derived from TP now, and a caller can no longer pass a
    split that leaves cards idle without also passing it explicitly.
    """
    block = _extract_eval_only_block()
    env = {k: v for k, v in os.environ.items()
           if k not in ("ACTOR_GPUS", "ROLLOUT_GPUS")}
    env.update(EVAL_ONLY="1", TP=tp, NUM_GPUS_PER_NODE="8", PROJ=str(ROOT), MAX_RESPONSE="4096")
    env.pop("COLOCATE", None)  # non-colocated: the node is split
    result = subprocess.run(
        ["bash", "-uc", block + '\nprintf "%s %s\\n" "$ACTOR_GPUS" "$ROLLOUT_GPUS"'],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    actor, rollout = result.stdout.split()
    assert (actor, rollout) == (expected_actor, expected_rollout)
    assert int(actor) + int(rollout) == 8, "the whole node must be generating"


def test_eval_split_colocated_shares_the_whole_node():
    """With --colocate the actor and engines share cards, so both take the node."""
    block = _extract_eval_only_block()
    env = {k: v for k, v in os.environ.items() if k not in ("ACTOR_GPUS", "ROLLOUT_GPUS")}
    env.update(EVAL_ONLY="1", TP="2", NUM_GPUS_PER_NODE="8", COLOCATE="1",
               PROJ=str(ROOT), MAX_RESPONSE="4096")
    result = subprocess.run(
        ["bash", "-uc", block + '\nprintf "%s %s\\n" "$ACTOR_GPUS" "$ROLLOUT_GPUS"'],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["8", "8"]


@pytest.mark.parametrize("tp,actor,rollout,message", [
    ("2", "3", "5", "not divisible by TP"),
    ("2", "6", "6", "exceeds the 8 GPUs"),
    ("1", "0", "8", "eval split actor=0"),
])
def test_eval_split_fails_closed_on_impossible_topologies(tp, actor, rollout, message):
    """A split that cannot fit the node must abort, not silently under-use it."""
    block = _extract_eval_only_block()
    env = dict(os.environ, EVAL_ONLY="1", TP=tp, NUM_GPUS_PER_NODE="8",
               ACTOR_GPUS=actor, ROLLOUT_GPUS=rollout)
    env.pop("COLOCATE", None)
    result = subprocess.run(["bash", "-uc", block], env=env, text=True, capture_output=True)
    assert result.returncode == 2
    assert message in result.stderr

def _extract_eval_only_block() -> str:
    """Return the EVAL_ONLY branch of _common_v2.sh as a standalone snippet.

    Slicing on the literal ``else`` is brittle: the block contains nested
    if/else, so the first ``else`` after EVAL_ARGS= can be an inner one. Track
    if/fi nesting instead and copy through the matching fi.
    """
    script = (ROOT / "tasks/alfworld/scripts/_common_v2.sh").read_text()
    start = script.index('if [ "$EVAL_ONLY" = "1" ]; then')
    depth = 0
    for line in script[start:].splitlines(keepends=True):
        stripped = line.strip()
        if re.match(r"^if\b", stripped):
            depth += 1
        elif re.match(r"^fi\b", stripped):
            depth -= 1
            if depth == 0:
                parts = [line]
                break
    else:
        raise AssertionError("unterminated EVAL_ONLY block")
    end = script.index("".join(parts), start + len(line)) + len("".join(parts)) if False else None
    # Rebuild by walking again and stopping at the matching fi.
    out, depth = [], 0
    for line in script[start:].splitlines(keepends=True):
        out.append(line)
        stripped = line.strip()
        if re.match(r"^if\b", stripped):
            depth += 1
        elif re.match(r"^fi\b", stripped):
            depth -= 1
            if depth == 0:
                return "".join(out)
    raise AssertionError("unterminated EVAL_ONLY block")
