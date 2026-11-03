"""Regression guard for the THD attention bug (Gate-1 null control, 2026-09-14).

The MindSpeed npu_fusion_attention TND/packed path (--qkv-format thd) computes
wrong attention on this stack: identical-context HF-vs-Megatron KL is 5.7-6.9
nats with thd vs 0.0006 nats with bshd (reports/outcome_pi_phase2_summary.md
section A). The launcher default must therefore stay bshd; anyone who flips it
back without a null control breaks every downstream training run silently.

Run:  python -m pytest tests/test_qkv_default.py -q
      (or: python tests/test_qkv_default.py)
"""

import pathlib
import re
import sys

PROJ = pathlib.Path(__file__).resolve().parent.parent
LAUNCHER = PROJ / "train_scripts" / "run_math_opsd_slime_npu.sh"


def test_launcher_default_qkv_format_is_bshd() -> None:
    text = LAUNCHER.read_text()
    match = re.search(r'--qkv-format "\$\{OPSD_QKV_FORMAT:-(\w+)\}"', text)
    assert match is not None, (
        f"{LAUNCHER.name}: no --qkv-format default found; the launcher must pin "
        "the qkv format explicitly"
    )
    default = match.group(1)
    assert default == "bshd", (
        f"launcher default --qkv-format is '{default}'; 'thd' routes the Megatron "
        "student through the broken MindSpeed TND/packed attention path (null KL "
        "5.7-6.9 nats). Restore bshd, or re-run scripts/run_null_control.sh and "
        "update reports/outcome_pi_phase2_summary.md if you have a proven fix."
    )


def test_bshd_null_control_artifact_present() -> None:
    summary = PROJ / "reports" / "outcome_pi_phase2_summary.md"
    assert summary.exists(), "Gate-1 null-control report missing"
    text = summary.read_text()
    assert "0.00064" in text or "0.0006" in text, (
        "Gate-1 bshd null-control result (KL ~6e-4) not recorded in the phase report"
    )


if __name__ == "__main__":
    test_launcher_default_qkv_format_is_bshd()
    test_bshd_null_control_artifact_present()
    print("test_qkv_default: 2/2 PASS")
    sys.exit(0)
