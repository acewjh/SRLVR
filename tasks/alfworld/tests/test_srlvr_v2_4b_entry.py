"""SRLVR-v2 4B thin entry: inherit the shared recipe, pin only what the arm decides.

The 2026-10-30 defaults moved protocol/placement/budget/serving into
``_v2_defaults.sh`` and the 4B hardware overrides into ``_4b_backbone.sh``
(TP=2, 0.30, 32768; NPU graph is the shared default and stays on). A wrapper
that re-exports any of those values
would silently fork the recipe the next time it changes, so this test pins the
inheritance and the one source-order hazard: ``_v2_defaults.sh`` defaults
G=8/ROLLOUT_BS=4 for GRPO *after* the wrapper has set G=1/BS=32 -- ``${VAR:-}``
must keep the SRLVR values.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import subprocess

PROJ = pathlib.Path(__file__).resolve().parents[3]
SCRIPTS = PROJ / "tasks/alfworld" / "scripts"
WRAPPER = SCRIPTS / "run_alfworld_v2_srlvr_v2_4b.sh"
BACKBONE = SCRIPTS / "_4b_backbone.sh"
DEFAULTS = SCRIPTS / "_v2_defaults.sh"

FORKED = (
    "ACTOR_GPUS", "ROLLOUT_GPUS", "COLOCATE", "MAX_TURNS", "MAX_RESPONSE",
    "SEQ_LENGTH", "HISTORY_LENGTH", "NUM_ROLLOUT", "SAVE_INTERVAL", "SEED",
    "LR", "KL_COEF", "EPS_CLIP", "EPS_CLIP_HIGH", "CLIP_GRAD",
)


def _run(script: str) -> str:
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_wrapper_does_not_fork_shared_placement_or_protocol():
    text = WRAPPER.read_text()
    for key in FORKED:
        assert not re.search(rf"(?m)^\s*(?:export\s+)?{key}=", text), (
            f"{WRAPPER.name} re-exports {key}; inherit _v2_defaults.sh instead"
        )


def test_wrapper_keeps_algorithm_specific_values():
    text = WRAPPER.read_text()
    assert "export G=1" in text
    assert 'ROLLOUT_BS="${ROLLOUT_BS:-32}"' in text
    assert 'SRLVR_CALIBRATION="${SRLVR_CALIBRATION:-4}"' in text
    assert 'SRLVR_GAMMA="${SRLVR_GAMMA:-0.99}"' in text
    assert 'SRLVR_ADVANTAGE_MODE="${SRLVR_ADVANTAGE_MODE:-history}"' in text
    assert "export OPSD_LORA=0" in text


def test_srlvr_g1_bs32_survive_the_shared_defaults_source_order():
    """Wrapper exports first, then _v2_defaults.sh (GRPO G=8/BS=4) must not win."""
    out = _run(
        "export G=1 ROLLOUT_BS=32; "
        f"source {shlex.quote(str(DEFAULTS))}; "
        'printf "%s %s %s %s %s" "$G" "$ROLLOUT_BS" "$ACTOR_GPUS" "$ROLLOUT_GPUS" "$COLOCATE"'
    )
    assert out == "1 32 8 8 1"


def test_backbone_4b_placement_and_serving_survive_the_defaults():
    out = _run(
        f"source {shlex.quote(str(BACKBONE))}; source {shlex.quote(str(DEFAULTS))}; "
        'printf "%s %s %s %s %s" "$TP" "$SGLANG_ENABLE_NPU_GRAPH" '
        '"$SGLANG_MEM_FRACTION_STATIC" "$SGLANG_MAX_TOTAL_TOKENS" "$OPSD_LORA"'
    )
    assert out == "2 1 0.30 32768 0"


def test_backbone_4b_explicit_graph_zero_still_wins():
    out = _run(
        "export SGLANG_ENABLE_NPU_GRAPH=0; "
        f"source {shlex.quote(str(BACKBONE))}; source {shlex.quote(str(DEFAULTS))}; "
        'printf "%s" "$SGLANG_ENABLE_NPU_GRAPH"'
    )
    assert out == "0"
