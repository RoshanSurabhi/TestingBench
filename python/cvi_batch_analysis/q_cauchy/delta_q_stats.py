"""Robust *sigma*(Δ*q*) and *Q_w* from a raw *q* series (percentile-spread path)."""
from __future__ import annotations

import math
import statistics
from typing import Tuple

MIN_OBS = 10


def percentile_linear(xs: list[float], p: float) -> float:
    """Linear interpolation percentile, *p* in [0, 100]."""
    if not xs:
        return float("nan")
    if p <= 0:
        return min(xs)
    if p >= 100:
        return max(xs)
    ys = sorted(xs)
    m = len(ys)
    k = (m - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def sigma_dq_from_dq_percentiles(
    dq: list[float],
    p_lo: float,
    p_hi: float,
) -> float:
    if p_lo >= p_hi or p_lo <= 0.0 or p_hi >= 100.0:
        raise ValueError("require 0 < p_lo < p_hi < 100")
    lo = percentile_linear(dq, p_lo)
    hi = percentile_linear(dq, p_hi)
    z_lo = statistics.NormalDist().inv_cdf(p_lo / 100.0)
    z_hi = statistics.NormalDist().inv_cdf(p_hi / 100.0)
    denom = z_hi - z_lo
    if denom <= 0.0:
        return 1e-12
    return max((hi - lo) / denom, 1e-12)


def _dq_stdev_sample(dq: list[float]) -> float:
    if len(dq) < 2:
        return 0.0
    try:
        s = statistics.stdev(dq)
    except statistics.StatisticsError:
        return 0.0
    if not math.isfinite(s) or s < 0.0:
        return 0.0
    return float(s)


def robust_sigma_Q_w_from_q(
    q_vals: list[float],
    *,
    dq_pct_lo: float = 25.0,
    dq_pct_hi: float = 75.0,
    range_floor_frac: float = 0.03,
) -> Tuple[float, float, float, int, float] | None:
    """
    Gaussian-equivalent *sigma*(Δ*q*) from inter-percentile spread; *Q_w* = *sigma*² with range floor.

    Returns ``(sigma_dq, Q_w, q_range, n, dq_stdev_sample)`` or ``None`` if too few points.
    """
    n = len(q_vals)
    if n < MIN_OBS:
        return None
    dq = [q_vals[i] - q_vals[i - 1] for i in range(1, n)]
    sigma_dq = sigma_dq_from_dq_percentiles(dq, dq_pct_lo, dq_pct_hi)
    Q_w = sigma_dq**2
    q_range = max(q_vals) - min(q_vals)
    Q_w = max(Q_w, (q_range * range_floor_frac) ** 2)
    dq_std = _dq_stdev_sample(dq)
    return sigma_dq, Q_w, q_range, n, dq_std
