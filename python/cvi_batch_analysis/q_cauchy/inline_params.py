"""Inline *Q_w* / *gamma* for compare-style calibration (percentile Δ*q* + *K** or MAD legacy)."""
from __future__ import annotations

import math

from .delta_q_stats import robust_sigma_Q_w_from_q


def Q_w_from_percentile_delta_q(
    q_clean: list[float],
    *,
    dq_pct_lo: float = 25.0,
    dq_pct_hi: float = 75.0,
) -> float:
    core = robust_sigma_Q_w_from_q(
        q_clean, dq_pct_lo=dq_pct_lo, dq_pct_hi=dq_pct_hi
    )
    if core is None:
        return 1e-8
    return core[1]


def gamma_from_k_star_gain(Q_w: float, k_star: float) -> float:
    k = min(max(float(k_star), 1e-6), 1.0 - 1e-6)
    return max(math.sqrt(Q_w * (1.0 - k) / k), 1e-8)


def gamma_mad_legacy_ratio(
    q_raw: list[float],
    *,
    dq_pct_lo: float = 25.0,
    dq_pct_hi: float = 75.0,
) -> float:
    core = robust_sigma_Q_w_from_q(
        q_raw, dq_pct_lo=dq_pct_lo, dq_pct_hi=dq_pct_hi
    )
    if core is None:
        return 0.01
    sigma_dq, _, _, _, _ = core
    return max(sigma_dq / 1.4826, 1e-6)
