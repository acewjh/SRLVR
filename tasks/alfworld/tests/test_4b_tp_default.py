"""Qwen3-4B ALFWorld launchers must default to TP=2 on 32 GB cards.

TP=1 full RL does not fit on the 910B4 cards this node runs: actor init requests
14.99 GiB with 13.42 GiB free (measured 2026-10-29). TP=2 completes the smoke
end to end. The default lives in ``_4b_backbone.sh`` so every 4B wrapper inherits
it; a wrapper that sets TP itself would silently bypass the fix.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess

import pytest

PROJ = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = PROJ / "tasks/alfworld" / "scripts"
BACKBONE = SCRIPTS / "_4b_backbone.sh"
REPEAT_STOP = SCRIPTS / "run_alfworld_v2_repeat_stop_4b.sh"
EVAL_4B = SCRIPTS / "run_alfworld_v2_eval_4b.sh"


def _resolved_tp(override: str | None = None) -> str:
    prelude = f"export TP={shlex.quote(override)};" if override is not None else "unset TP;"
    script = f"""
set -uo pipefail
{prelude}
source {shlex.quote(str(BACKBONE))}
printf '%s' "$TP"
"""
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.unit
def test_backbone_defaults_to_tp2():
    assert _resolved_tp() == "2"


@pytest.mark.unit
def test_explicit_tp_still_wins():
    assert _resolved_tp("4") == "4"


@pytest.mark.unit
@pytest.mark.parametrize(
    "script", sorted(SCRIPTS.glob("run_alfworld_v2_*4b*.sh")), ids=lambda p: p.name
)
def test_every_4b_wrapper_sources_the_backbone(script: pathlib.Path):
    text = script.read_text()
    assert "_4b_backbone.sh" in text, f"{script.name} does not source _4b_backbone.sh"
    if script == REPEAT_STOP:
        # Frozen arm with its own teacher/chunking layout: it deliberately pins
        # TP=1 after the backbone, and that explicit override must stay visible.
        assert 'export TP="${TP:-1}"' in text
        return
    assert not re.search(r"(?m)^\s*(?:export\s+)?TP=", text), (
        f"{script.name} sets TP itself, bypassing the 32 GB-card default in "
        "_4b_backbone.sh; either delete the override or add it to this test's "
        "allowlist with a measured reason"
    )


@pytest.mark.unit
def test_shared_defaults_are_size_independent():
    text = (SCRIPTS / "_v2_defaults.sh").read_text()
    for key, value in {
        "NUM_ROLLOUT": "100",
        "SAVE_INTERVAL": "25",
        "G": "8",
        "ROLLOUT_BS": "4",
        "GBS_OVERRIDE": "32",
        "MAX_TURNS": "50",
        "MAX_RESPONSE": "4096",
        "OPSD_LORA_R": "256",
        "OPSD_LORA_ALPHA": "256",
        "OPSD_EXPERIMENT_METRICS": "1",
    }.items():
        assert f'export {key}="${{{key}:-{value}}}"' in text


@pytest.mark.unit
def test_environment_overrides_public_defaults():
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f"export NUM_ROLLOUT=7 G=3; source {SCRIPTS / '_v2_defaults.sh'}; printf '%s %s' \"$NUM_ROLLOUT\" \"$G\"",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "7 3"


@pytest.mark.unit
def test_4b_colocate_serving_override_and_8b_split():
    four_b = (SCRIPTS / "_4b_backbone.sh").read_text()
    eight_b = (SCRIPTS / "_8b_fast_rollout.sh").read_text()
    defaults = (SCRIPTS / "_v2_defaults.sh").read_text()
    # NPU graph is a shared, size-independent default (on), deliberately absent
    # from the 4B card so no source order can turn it back off.
    assert 'export SGLANG_ENABLE_NPU_GRAPH="${SGLANG_ENABLE_NPU_GRAPH:-1}"' in defaults
    assert "SGLANG_ENABLE_NPU_GRAPH=" not in four_b
    assert 'SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.30}"' in four_b
    assert 'SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-32768}"' in four_b
    assert 'COLOCATE="${COLOCATE:-0}"' in eight_b
    assert 'ACTOR_GPUS="${ACTOR_GPUS:-2}"' in eight_b
    assert 'ROLLOUT_GPUS="${ROLLOUT_GPUS:-6}"' in eight_b


def _resolved_graph(route: str, override: str | None = None) -> str:
    """Replay each size route's real source order and print the graph argv knobs."""
    prelude = (
        f"export SGLANG_ENABLE_NPU_GRAPH={shlex.quote(override)};"
        if override is not None else "unset SGLANG_ENABLE_NPU_GRAPH;"
    )
    prelude += " unset SGLANG_CUDA_GRAPH_MAX_BS;"
    sources = {
        "1p7b": [
            "_runtime_defaults.sh", "_v2_defaults.sh",
        ],
        "4b": [
            "_4b_backbone.sh", "_runtime_defaults.sh", "_v2_defaults.sh",
        ],
        "8b": [
            "_8b_backbone.sh", "_8b_fast_rollout.sh", "_runtime_defaults.sh",
            "_v2_defaults.sh",
        ],
    }[route]
    body = " ".join(f"source {shlex.quote(str(SCRIPTS / name))};" for name in sources)
    script = f"""
set -uo pipefail
{prelude}
{body}
printf '%s %s' "$SGLANG_ENABLE_NPU_GRAPH" "$SGLANG_CUDA_GRAPH_MAX_BS"
"""
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.unit
@pytest.mark.parametrize("route", ["1p7b", "4b", "8b"])
def test_graph_is_on_for_every_size_route(route: str):
    assert _resolved_graph(route) == "1 8"


@pytest.mark.unit
@pytest.mark.parametrize("route", ["1p7b", "4b", "8b"])
def test_explicit_graph_zero_survives_every_size_route(route: str):
    assert _resolved_graph(route, override="0") == "0 8"


@pytest.mark.unit
def test_eval_disables_experiment_metrics():
    text = (SCRIPTS / "_common_v2.sh").read_text()
    assert "export OPSD_EXPERIMENT_METRICS=0" in text
