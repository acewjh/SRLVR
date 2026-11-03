"""Contiguous span partitioning.

Extracted verbatim from ``math_opsd/hpi.py`` (was ``split_chunks``), which the
math, ALFWorld and code lines all imported. Single source now.
"""

from __future__ import annotations

DEFAULT_K = 4


def split_chunks(rlen: int, k: int = DEFAULT_K) -> list[tuple[int, int]]:
    """Contiguous near-equal partition of [0, rlen) into k chunks.

    Always returns exactly k entries (possibly empty for rlen < k) so downstream
    shapes stay uniform; callers skip zero-length targets.
    """
    if rlen <= 0:
        return [(0, 0)] * k
    bounds = [(i * rlen) // k for i in range(k + 1)]
    bounds[-1] = rlen
    return [(bounds[i], bounds[i + 1]) for i in range(k)]


def chunk_count(rlen: int, k: int, *, min_len: int = 1) -> int:
    """Number of chunks that carry at least ``min_len`` tokens (diagnostics)."""
    return sum(1 for s, e in split_chunks(rlen, k) if e - s >= min_len)
