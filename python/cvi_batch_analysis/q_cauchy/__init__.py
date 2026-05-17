"""
Library for DP-Cauchy *q* smoothing on CVI batch data.

**1. Parameters for *q* (calibration)** — MAD / pooled JSON lives in
:mod:`cvi_batch_analysis.calibrate_cauchy_params` (CLI). Shared statistics:
:mod:`cvi_batch_analysis.q_cauchy.delta_q_stats`. Geometric pooling of JSON
params: :func:`q_cauchy.pool.pool_params_geometric`. Inline *Q_w* / *gamma*
for compare-style plots: :mod:`q_cauchy.inline_params`.

**2. Apply filter on a day** — :func:`q_cauchy.filter.dp_cauchy_filter` and
:func:`q_cauchy.repricing.implied_forward_f_hat_series`.

**3. Combine params (geometric mean)** — :func:`q_cauchy.pool.pool_params_geometric`.

**4. Call-price delta residual / decoupling** — implementations stay in
``cvi_batch_plot``; use :mod:`q_cauchy.decoupling` for stable imports.
"""
from __future__ import annotations

from .decoupling import build_call_price_delta_payload, build_decoupling_payload
from .delta_q_stats import robust_sigma_Q_w_from_q
from .filter import dp_cauchy_filter
from .inline_params import gamma_from_k_star_gain, gamma_mad_legacy_ratio, Q_w_from_percentile_delta_q
from .pool import pool_params_geometric
from .repricing import implied_forward_f_hat_series
from .time_series import calendar_date, insert_null_between_calendar_days

__all__ = [
    "build_call_price_delta_payload",
    "build_decoupling_payload",
    "calendar_date",
    "dp_cauchy_filter",
    "gamma_from_k_star_gain",
    "gamma_mad_legacy_ratio",
    "implied_forward_f_hat_series",
    "insert_null_between_calendar_days",
    "pool_params_geometric",
    "Q_w_from_percentile_delta_q",
    "robust_sigma_Q_w_from_q",
]
