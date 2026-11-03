"""Per-token credit-weight normalisation. R3: ``compute_hpi_weights`` / ``compute_hpi_v2_weights`` /
``compute_hpi_v15_weights`` all ended with the same four lines — mean-normalise,
clamp, stopgrad, with a degenerate-mean guard. Only the raw weight expression
differed. The normalisation is now shared; the weighting *method* supplies `raw`.
"""

from __future__ import annotations

import torch

WEIGHT_MIN = 0.70
WEIGHT_MAX = 1.30
SCALE_FLOOR = 0.01


def normalize_weights(
    raw: torch.Tensor,
    *,
    min_w: float = WEIGHT_MIN,
    max_w: float = WEIGHT_MAX,
) -> torch.Tensor:
    """Mean-1 normalisation, clamped, detached.

    ``alpha == 0`` makes `raw` identically 1, hence mean 1, hence this returns
    exactly 1 everywhere — the weighting degenerates to the unweighted objective
    bit-for-bit. The stopgrad is load-bearing: the weight is a credit-allocation
    coefficient, never a gradient path.
    """
    mean = raw.mean()
    if not torch.isfinite(mean) or mean <= 0:
        return torch.ones_like(raw)
    return (raw / mean).clamp(min_w, max_w).detach()


def weight_stats(w: torch.Tensor) -> dict[str, float]:
    if w.numel() == 0:
        return {"p10": 1.0, "p50": 1.0, "p90": 1.0, "mean": 1.0}
    q = torch.quantile(w.float(), torch.tensor([0.10, 0.50, 0.90]))
    return {
        "p10": float(q[0]),
        "p50": float(q[1]),
        "p90": float(q[2]),
        "mean": float(w.float().mean()),
    }
