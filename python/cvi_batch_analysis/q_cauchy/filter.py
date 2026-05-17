"""DP-MAP Cauchy filter (Yoosefian & Lessard 2025, scalar Eq. 14)."""
from __future__ import annotations

import math
from typing import List, Optional, Union

Floatish = Union[float, None]


def dp_cauchy_filter(
    q_raw: List[Optional[float]],
    Q_w: float,
    gamma: float,
    *,
    level_shift_theta: Optional[float] = 10.0,
    level_shift_n_streak: int = 3,
) -> List[Optional[float]]:
    """
    Scalar random-walk state, Cauchy measurement noise. Same recursion for
    calibrated or inline *Q_w*, *gamma*.
    """
    n = len(q_raw)
    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out

    first = next(
        (i for i, x in enumerate(q_raw) if x is not None and math.isfinite(float(x))),
        None,
    )
    if first is None:
        return out

    g2 = max(gamma * gamma, 1e-18)
    mu = float(q_raw[first])
    P = g2
    out[first] = mu
    streak = 0
    n_streak = max(1, int(level_shift_n_streak)) if level_shift_theta is not None else 0

    for t in range(first + 1, n):
        y = q_raw[t]
        P_pred = P + Q_w
        mu_pred = mu

        if y is None or not math.isfinite(float(y)):
            mu = mu_pred
            P = P_pred
            out[t] = mu
            streak = 0
            continue

        v_bar = float(y) - mu_pred

        if n_streak > 0:
            thresh = float(level_shift_theta) * gamma
            if abs(v_bar) > thresh:
                streak += 1
            else:
                streak = 0
            if streak >= n_streak:
                mu = float(y)
                P = g2
                out[t] = mu
                streak = 0
                continue

        denom = v_bar * v_bar + g2
        M_r = 2.0 / denom
        score = 2.0 * v_bar / denom
        P = 1.0 / (1.0 / max(P_pred, 1e-18) + M_r)
        mu = mu_pred + P * score
        out[t] = mu

    return out
