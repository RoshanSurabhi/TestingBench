"""Implied forward *F* from smoothed *q* using row-wise carry from raw (*F*, *q*, *r*, *volTime*)."""
from __future__ import annotations

import math
from typing import List, Optional


def implied_forward_f_hat_series(
    F: List[float],
    q: List[float],
    r: List[float],
    vol_time: List[float],
    q_hat: List[Optional[float]],
) -> List[float]:
    """
    Per row: ``S_raw = F * exp(-(r - q) * volTime)``,
    ``F_hat = S_raw * exp((r - q_hat) * volTime)``.
    """
    n = len(F)
    out: List[float] = []
    for i in range(n):
        qh = q_hat[i]
        Ti = float(vol_time[i])
        ri = float(r[i])
        Fi = float(F[i])
        qi = float(q[i])
        if qh is None or not math.isfinite(float(qh)):
            out.append(float("nan"))
        elif not math.isfinite(Ti) or Ti <= 0.0:
            out.append(float("nan"))
        elif not (math.isfinite(ri) and math.isfinite(Fi) and math.isfinite(qi)):
            out.append(float("nan"))
        else:
            s_raw = Fi * math.exp(-(ri - qi) * Ti)
            out.append(s_raw * math.exp((ri - float(qh)) * Ti))
    return out
