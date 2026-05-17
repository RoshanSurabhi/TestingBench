"""Geometric-mean pooling of per-expiry (*Q_w*, *gamma*) JSON blobs."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

KSS_TARGET = 2.0
KSS_LO = 1.0
KSS_HI = 3.0


def _geomean(vals: list[float]) -> float:
    if not vals:
        raise ValueError("empty list")
    log_sum = sum(math.log(v) for v in vals)
    return math.exp(log_sum / len(vals))


def steady_state_kalman_gain(Q_w: float, gamma: float) -> float:
    """K_ss = sqrt(2 * Q_w) / gamma (diagnostic for this parameterization)."""
    return math.sqrt(2.0 * Q_w) / gamma if gamma > 0 else float("inf")


def clamp_Q_w_for_kss(
    Q_w: float,
    gamma: float,
    kss_lo: float = KSS_LO,
    kss_hi: float = KSS_HI,
    kss_target: float = KSS_TARGET,
) -> tuple[float, float, bool]:
    kss = steady_state_kalman_gain(Q_w, gamma)
    if kss_lo <= kss <= kss_hi:
        return Q_w, kss, False
    Q_w_new = (kss_target * gamma) ** 2 / 2.0
    return Q_w_new, kss_target, True


def pool_params_geometric(
    param_dicts: list[dict[str, Any]],
    kss_lo: float = KSS_LO,
    kss_hi: float = KSS_HI,
    kss_target: float = KSS_TARGET,
) -> dict[str, Any]:
    if not param_dicts:
        raise ValueError("need at least one param dict")

    cal_days = [d.get("calibration_day", "?") for d in param_dicts]

    all_expiries: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"gammas": [], "Qws": []}
    )
    for pd_ in param_dicts:
        for exp, vals in pd_.get("by_expiry", {}).items():
            g = float(vals.get("gamma", 0) or 0)
            q = float(vals.get("Q_w", 0) or 0)
            if g > 0 and q > 0:
                all_expiries[exp]["gammas"].append(g)
                all_expiries[exp]["Qws"].append(q)

    by_expiry: dict[str, dict[str, Any]] = {}
    all_pooled_Qw: list[float] = []
    all_pooled_gamma: list[float] = []

    for exp in sorted(all_expiries.keys()):
        gammas = all_expiries[exp]["gammas"]
        Qws = all_expiries[exp]["Qws"]

        gamma_pooled = _geomean(gammas)
        Qw_pooled = _geomean(Qws)

        Qw_final, kss_final, was_clamped = clamp_Q_w_for_kss(
            Qw_pooled, gamma_pooled, kss_lo, kss_hi, kss_target
        )

        by_expiry[exp] = {
            "Q_w": Qw_final,
            "gamma": gamma_pooled,
            "K_ss": round(kss_final, 4),
            "K_ss_clamped": was_clamped,
            "n_days_pooled": len(gammas),
            "source_gammas": gammas,
            "source_Qws": Qws,
        }
        all_pooled_Qw.append(Qw_final)
        all_pooled_gamma.append(gamma_pooled)

    default_gamma = _geomean(all_pooled_gamma) if all_pooled_gamma else 1e-6
    default_Qw = _geomean(all_pooled_Qw) if all_pooled_Qw else 1e-10
    default_Qw, default_kss, _ = clamp_Q_w_for_kss(
        default_Qw, default_gamma, kss_lo, kss_hi, kss_target
    )

    batch_dirs = list({d.get("batch_dir", "") for d in param_dicts})

    return {
        "calibration_days": cal_days,
        "pooling_method": "geometric_mean",
        "kss_target": kss_target,
        "kss_range": [kss_lo, kss_hi],
        "batch_dir": batch_dirs[0] if len(batch_dirs) == 1 else batch_dirs,
        "default": {
            "Q_w": default_Qw,
            "gamma": default_gamma,
            "K_ss": round(default_kss, 4),
        },
        "by_expiry": by_expiry,
    }
