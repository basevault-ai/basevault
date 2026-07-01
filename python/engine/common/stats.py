"""Tiny content-free statistics helpers shared across the pipeline.

These are the few summary statistics the diagnostic rollups need —
nearest-rank percentile + arithmetic mean — extracted so they aren't
re-implemented per call-site. Both operate on plain float lists; no
state, no external dependencies."""
from __future__ import annotations

from math import ceil


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile. ``q`` in ``[0, 1]``;
    rank = ``ceil(q * n)``, 1-indexed. Returns ``None`` for an empty
    list.

    Linear interpolation is overkill for diagnostic latency buckets —
    the nearest-rank variant is robust + stable across run sizes."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    rank = max(1, min(n, ceil(q * n))) if q > 0 else 1
    return s[rank - 1]


def mean(values: list[float]) -> float | None:
    """Arithmetic mean. ``None`` for an empty list (so the caller
    can distinguish \"no samples\" from \"zero mean\")."""
    if not values:
        return None
    return sum(values) / len(values)
