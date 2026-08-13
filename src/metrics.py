"""Shared summary statistics.

Small, but it exists because it did not. Latency percentiles were computed in
two places with two different index conventions, so the same run reported a p95
of 406 ms in one table and 402 ms in another. Neither figure was wrong on its
own terms and publishing both under one name was.
"""

from __future__ import annotations

from math import ceil


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest observation at or above p of the data.

    Always a value that actually occurred, rather than an interpolation between
    two, which matters when reporting a latency somebody might try to reproduce.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(ceil(p * len(ordered)) - 1, 0), len(ordered) - 1)
    return ordered[index]


def median(values: list[float]) -> float:
    return percentile(values, 0.5)
