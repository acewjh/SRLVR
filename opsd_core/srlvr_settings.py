"""SRLVR v2/v3 configuration shared by methods and experiment recording.

This module has no model, torch or method dependencies.
"""
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    variant: str
    gamma: float
    calibration: int = 8
    rho: float = .9
    window: int = 128
    eps_b0: float = .01
    delta_s: float = .1
    position_chunk: int = 64
    actor_lambda_zero: bool = False
    trajectory_uniform_b: bool = False
    teacher_outcome_only: bool = False

    @classmethod
    def from_env(cls):
        if os.environ.get("SRLVR_MODE", "empirical") != "empirical":
            raise ValueError("SRLVR-v2/v3 require empirical second-moment calibration")
        values = dict(variant=os.environ.get("SRLVR_VARIANT", ""), gamma=float(os.environ.get("SRLVR_GAMMA") or "0.99"))
        for name, typ in (("calibration", int), ("rho", float), ("window", int),
                          ("eps_b0", float), ("delta_s", float)):
            if os.environ.get("SRLVR_" + name.upper()):
                values[name] = typ(os.environ["SRLVR_" + name.upper()])
        values["position_chunk"] = int(os.environ.get("SRLVR_POSITION_CHUNK") or os.environ.get("OPSD_HSI2_POS_CHUNK") or 64)
        for name in ("actor_lambda_zero", "trajectory_uniform_b", "teacher_outcome_only"):
            value = os.environ.get("SRLVR_" + name.upper(), "0")
            if value not in ("0", "1"):
                raise ValueError(f"SRLVR_{name.upper()} must be 0 or 1")
            values[name] = value == "1"
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        if self.variant not in ("v2", "v3"):
            raise ValueError("SRLVR_VARIANT must be v2 or v3")
        if self.variant != "v2" and any(self.ablations().values()):
            raise ValueError("SRLVR ablations require variant v2")
        if not (math.isfinite(self.gamma) and 0 < self.gamma < 1):
            raise ValueError("SRLVR_GAMMA must satisfy 0 < gamma < 1")
        if not (self.calibration >= 2 and 0 < self.rho < 1 and self.window >= 1
                and 0 < self.eps_b0 < .5 and self.delta_s > 0 and self.position_chunk >= 1):
            raise ValueError("invalid SRLVR-v2/v3 calibration, history, or position chunk settings")

    def ablations(self):
        return {name: getattr(self, name) for name in
                ("actor_lambda_zero", "trajectory_uniform_b", "teacher_outcome_only")}



#: Where the SRLVR actor advantage takes its baseline/scale from.
#:   history -- the calibration history window (default; the method as published)
#:   batch   -- the CURRENT batch of trajectories (REINFORCE++-style)
ADVANTAGE_MODES = ("history", "batch")


def advantage_mode() -> str:
    """Read ``SRLVR_ADVANTAGE_MODE`` and fail closed on anything unknown."""
    mode = os.environ.get("SRLVR_ADVANTAGE_MODE", "history").lower()
    if mode not in ADVANTAGE_MODES:
        raise ValueError(
            f"SRLVR_ADVANTAGE_MODE must be one of {ADVANTAGE_MODES}, got {mode!r}"
        )
    return mode


def batch_baseline_and_scale(rewards, b_history, s_history, delta_s, eps_b0=0.01):
    """REINFORCE++-style baseline and scale from THIS batch instead of history.

    The SRLVR actor advantage is ``A = (R - b - 1_U * lambda * B) / s``. The
    method as published takes ``b`` (a reward mean) and ``s`` (an RMS-deviation
    scale floored by ``delta_s``) from the calibration history window. With one
    trajectory per task position there is no group to normalise against, so a
    per-batch alternative is the only REINFORCE++-style option available.

    This keeps the SAME functional form and the same ``delta_s`` floor; only the
    data the statistics come from changes. That matters: swapping in a plain
    ``std`` would silently change the meaning of ``s`` and with it the effective
    learning rate, so the comparison would not be a fair A/B of the estimator.

    Falls back to the historical values when the batch cannot support a
    statistic (no rows, or every reward identical), and says so rather than
    dividing by a near-zero scale.
    """
    values = [float(r) for r in rewards]
    if not values:
        return float(b_history), float(s_history), "empty_batch"
    if len(values) < 2 or max(values) == min(values):
        # A degenerate batch carries no relative information; the history values
        # are the only defensible baseline, and the caller must still see it was
        # not a real batch estimate.
        return float(b_history), float(s_history), "degenerate_batch"
    n = len(values)
    mean = sum(values) / n
    b = min(1.0 - eps_b0, max(eps_b0, mean))
    s = math.sqrt(delta_s ** 2 + sum((v - mean) ** 2 for v in values) / n)
    if not (math.isfinite(s) and s > 0):
        return float(b_history), float(s_history), "nonfinite"
    return b, s, "batch"
