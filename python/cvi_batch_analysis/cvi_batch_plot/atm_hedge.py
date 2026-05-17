"""ATM hedge residuals, Klassen metrics, grid CSV export."""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .fundamentals import *
from .batch_payloads import (
    build_ln_scatter_payload,
    fit_ssr_full_day_from_points,
    infer_num_expiries_from_batch,
    prior_trading_day,
    _build_ln_scatter_snaps,
    _ln_transition_points_from_snaps,
)
from .snapshots_decouple import _build_snapshot_state, _find_forward_idx

def write_call_delta_prediction_csvs(payload: dict, out_dir: Path, file_prefix: str) -> list[Path]:
    """Write per-horizon CSVs with realized/predicted delta and naive delta."""
    groups = payload.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise SystemExit("No grouped samples available for delta CSV export.")

    rows_by_h: dict[int, list[dict[str, object]]] = defaultdict(list)
    for _g, rows in groups.items():
        if not isinstance(rows, list):
            continue
        for r in rows:
            try:
                h = int(round(float(r.get("horizon_min"))))
            except (TypeError, ValueError):
                continue
            row = {
                "target": r.get("target"),
                "horizon_min": h,
                "from_t": r.get("from_t"),
                "to_t": r.get("to_t"),
                "strike": r.get("strike"),
                "z_t": r.get("z_t"),
                "dF": r.get("dF"),
                "dlnF": r.get("dlnF"),
                "delta_realized": r.get("delta_realized"),
                "delta_pred_rolling_reg": r.get("delta_pred_klassen_linearized"),
                "delta_naive": r.get("delta_bs_t"),
                "delta_pred_sticky_strike": r.get("delta_pred_sticky_strike"),
                "delta_pred_sticky_moneyness": r.get("delta_pred_sticky_moneyness"),
                "delta_resid_rolling_reg": r.get("delta_resid_klassen_linearized"),
                "delta_resid_naive": (
                    (_to_float_or_none(r.get("delta_realized")) - _to_float_or_none(r.get("delta_bs_t")))
                    if _to_float_or_none(r.get("delta_realized")) is not None
                    and _to_float_or_none(r.get("delta_bs_t")) is not None
                    else None
                ),
                "running_delta_mse_rolling_reg": r.get("running_delta_mse_klassen_linearized"),
                "running_delta_mse_naive": None,
            }
            rows_by_h[h].append(row)

    wrote: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for h, rows in sorted(rows_by_h.items()):
        if not rows:
            continue
        # running naive MSE per (target, strike) stream in timestamp order
        key_streams: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
        for r in rows:
            tk = str(r.get("target"))
            k = float(r.get("strike")) if _to_float_or_none(r.get("strike")) is not None else float("nan")
            key_streams[(tk, k)].append(r)
        for _k, stream in key_streams.items():
            stream.sort(key=lambda x: str(x.get("to_t")))
            n = 0
            sse = 0.0
            for r in stream:
                e = _to_float_or_none(r.get("delta_resid_naive"))
                if e is None:
                    continue
                n += 1
                sse += e * e
                r["running_delta_mse_naive"] = (sse / n) if n > 0 else None

        rows.sort(key=lambda x: (str(x.get("target")), float(x.get("strike") or 0.0), str(x.get("to_t"))))
        out_csv = out_dir / f"{file_prefix}_h{h}m.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        wrote.append(out_csv)
    if not wrote:
        raise SystemExit("No delta CSV files were written.")
    return wrote


def write_focused_delta_compare_csvs(
    payload: dict,
    out_dir: Path,
    file_prefix: str,
    *,
    delta_floor_abs_df: float = 0.05,
) -> tuple[list[Path], Path]:
    """
    Focused naive vs rolling-regression delta compare:
    - CVI target only
    - hedge-error loss: (delta_real - delta_pred)^2 * dF^2 (all rows)
    - delta-error loss: (delta_real - delta_pred)^2 for |dF| >= floor
    """
    groups = payload.get("groups", {})
    if not isinstance(groups, dict) or not groups:
        raise SystemExit("No grouped samples available for focused delta compare.")

    by_h: dict[int, list[dict[str, object]]] = defaultdict(list)
    for _g, rows in groups.items():
        if not isinstance(rows, list):
            continue
        for r in rows:
            if str(r.get("target")) != "cvi":
                continue
            h = int(round(float(r.get("horizon_min", float("nan")))))
            if h not in {1, 5, 15, 30, 60, 120}:
                continue
            dF = _to_float_or_none(r.get("dF"))
            d_real = _to_float_or_none(r.get("delta_realized"))
            d_naive = _to_float_or_none(r.get("delta_bs_t"))
            d_roll = _to_float_or_none(r.get("delta_pred_klassen_linearized"))
            dc_real = _to_float_or_none(r.get("dc_realized"))
            if dF is None or d_real is None or d_naive is None or d_roll is None or dc_real is None:
                continue
            err_naive = d_real - d_naive
            err_roll = d_real - d_roll
            hedge_loss_naive = (err_naive * err_naive) * (dF * dF)
            hedge_loss_roll = (err_roll * err_roll) * (dF * dF)
            keep_floor = abs(dF) >= float(delta_floor_abs_df)
            by_h[h].append(
                {
                    "horizon_min": h,
                    "from_t": r.get("from_t"),
                    "to_t": r.get("to_t"),
                    "strike": r.get("strike"),
                    "z_t": r.get("z_t"),
                    "dF": dF,
                    "dC_realized": dc_real,
                    "delta_realized": d_real,
                    "delta_naive": d_naive,
                    "delta_pred_rolling_reg": d_roll,
                    "hedge_loss_naive": hedge_loss_naive,
                    "hedge_loss_rolling": hedge_loss_roll,
                    "delta_loss_naive_floor": (err_naive * err_naive) if keep_floor else None,
                    "delta_loss_rolling_floor": (err_roll * err_roll) if keep_floor else None,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_paths: list[Path] = []
    summary_rows: list[dict[str, float | int | None]] = []
    for h in (1, 5, 15, 30, 60, 120):
        rows = by_h.get(h, [])
        if not rows:
            continue
        rows.sort(key=lambda x: (float(x["strike"]), str(x["to_t"])))
        p = out_dir / f"{file_prefix}_h{h}m.csv"
        write_path = p
        try:
            fh = write_path.open("w", newline="", encoding="utf-8")
        except PermissionError:
            write_path = out_dir / f"{file_prefix}_h{h}m_new.csv"
            fh = write_path.open("w", newline="", encoding="utf-8")
        with fh as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        detail_paths.append(write_path)

        n_total = len(rows)
        n_floor = sum(1 for r in rows if r.get("delta_loss_naive_floor") is not None)
        sse_h_naive = sum(float(r["hedge_loss_naive"]) for r in rows)
        sse_h_roll = sum(float(r["hedge_loss_rolling"]) for r in rows)
        sse_d_naive = sum(float(r["delta_loss_naive_floor"]) for r in rows if r.get("delta_loss_naive_floor") is not None)
        sse_d_roll = sum(float(r["delta_loss_rolling_floor"]) for r in rows if r.get("delta_loss_rolling_floor") is not None)
        mse_h_naive = sse_h_naive / n_total if n_total > 0 else None
        mse_h_roll = sse_h_roll / n_total if n_total > 0 else None
        mse_d_naive = sse_d_naive / n_floor if n_floor > 0 else None
        mse_d_roll = sse_d_roll / n_floor if n_floor > 0 else None
        summary_rows.append(
            {
                "horizon_min": h,
                "n_total": n_total,
                "n_delta_floor": n_floor,
                "delta_floor_abs_df": float(delta_floor_abs_df),
                "mse_hedge_naive": mse_h_naive,
                "mse_hedge_rolling": mse_h_roll,
                "improve_hedge_vs_naive": (
                    (1.0 - mse_h_roll / mse_h_naive) if mse_h_naive is not None and mse_h_naive > 1e-30 and mse_h_roll is not None else None
                ),
                "mse_delta_naive_floor": mse_d_naive,
                "mse_delta_rolling_floor": mse_d_roll,
                "improve_delta_vs_naive_floor": (
                    (1.0 - mse_d_roll / mse_d_naive) if mse_d_naive is not None and mse_d_naive > 1e-30 and mse_d_roll is not None else None
                ),
            }
        )
    if not detail_paths:
        raise SystemExit("No focused detail CSVs written (no valid CVI rows).")
    summary_path = out_dir / f"{file_prefix}_summary.csv"
    try:
        fh = summary_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        summary_path = out_dir / f"{file_prefix}_summary_new.csv"
        fh = summary_path.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    return detail_paths, summary_path


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) * (x - mx) for x in xs)
    syy = sum((y - my) * (y - my) for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _beta_y_on_x(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) * (x - mx) for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def klassen_dsigma_calibration_metrics(
    rows: list[dict[str, object]], method: str
) -> dict[str, float | bool | int | None]:
    """
    From previous-day / rolling SSR methodology (σ-prediction diagnostics):

    - ρ = corr(dσ̂, Δσ_real)
    - r = SD(dσ̂) / SD(Δσ_real)  (population SD, n divisor)
    - β_cal = OLS slope of Δσ_real on dσ̂ (= ρ/r when variances well-behaved)
    - VarRed_approx = 2ρr − r² (identity under ε_naive ≈ ν·Δσ approx.)
    - Beats naive (predicted) iff ρ > r/2
    """
    col_p = f"dSigma_pred_{method}"
    pred: list[float] = []
    real: list[float] = []
    for r in rows:
        p = _to_float_or_none(r.get(col_p))
        y = _to_float_or_none(r.get("dSigma_real"))
        if p is None or y is None:
            continue
        pred.append(float(p))
        real.append(float(y))
    n = len(pred)
    base: dict[str, float | bool | int | None] = {
        f"n_dsigma_pairs_{method}": n,
        f"rho_dsigma_{method}": None,
        f"r_sd_ratio_dsigma_{method}": None,
        f"beta_cal_dsigma_{method}": None,
        f"var_red_two_rho_r_minus_r2_{method}": None,
        f"rho_gt_half_r_{method}": None,
    }
    if n < 2:
        return base
    rho = _corr(pred, real)
    mp = sum(pred) / n
    mr = sum(real) / n
    vp = sum((x - mp) * (x - mp) for x in pred) / n
    vr = sum((x - mr) * (x - mr) for x in real) / n
    sp = math.sqrt(vp) if vp > 1e-30 else None
    sr = math.sqrt(vr) if vr > 1e-30 else None
    r_ratio = (sp / sr) if sp is not None and sr is not None else None
    beta_cal = _beta_y_on_x(pred, real)
    var_red_theo = None
    if rho is not None and r_ratio is not None and math.isfinite(rho) and math.isfinite(r_ratio):
        var_red_theo = 2.0 * rho * r_ratio - r_ratio * r_ratio
    beats = None
    if rho is not None and r_ratio is not None and math.isfinite(rho) and math.isfinite(r_ratio):
        beats = bool(rho > 0.5 * r_ratio)
    base[f"rho_dsigma_{method}"] = float(rho) if rho is not None else None
    base[f"r_sd_ratio_dsigma_{method}"] = float(r_ratio) if r_ratio is not None else None
    base[f"beta_cal_dsigma_{method}"] = float(beta_cal) if beta_cal is not None else None
    base[f"var_red_two_rho_r_minus_r2_{method}"] = float(var_red_theo) if var_red_theo is not None else None
    base[f"rho_gt_half_r_{method}"] = beats
    return base


def _ssr_by_ts_from_ln_payload(ln_payload: dict) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for r in ln_payload.get("pred_series", []):
        ts = str(r.get("timestamp") or "")
        if not ts:
            continue
        out[ts] = {
            "ssr_ols": _to_float_or_none(r.get("ssr_ols")),
            "ssr_wls": _to_float_or_none(r.get("ssr_wls")),
            "ssr_huber": _to_float_or_none(r.get("ssr_huber")),
            "ssr_lad": _to_float_or_none(r.get("ssr_lad")),
            "ssr_ts": _to_float_or_none(r.get("ssr_ts")),
        }
    return out


def load_prior_ssr_from_full_day_ssr_json_doc(
    doc: dict,
    *,
    fit_day: str,
    expiry_index: int,
) -> tuple[dict[str, float | None], dict[str, float | int | str | None]]:
    """
    Read fixed full-day SSRs from a document produced by ``build_full_day_ssr_json_document``
    (same keys as ``--full-day-ssr-json`` output).
    """
    days = doc.get("days")
    if not isinstance(days, dict):
        raise SystemExit("prior-ssr-json: missing or invalid top-level 'days'")
    day_rec = days.get(fit_day)
    if day_rec is None:
        raise SystemExit(f"prior-ssr-json: no day {fit_day!r} in document (keys sample: {list(days)[:5]!r})")
    expiries = day_rec.get("expiries")
    if not isinstance(expiries, dict):
        raise SystemExit(f"prior-ssr-json: no 'expiries' for day {fit_day}")
    erec = expiries.get(str(int(expiry_index)))
    if erec is None:
        raise SystemExit(f"prior-ssr-json: no expiry_index {expiry_index} for day {fit_day}")
    if not erec.get("ok"):
        err = erec.get("error", "ok=false")
        raise SystemExit(f"prior-ssr-json: day {fit_day} e{expiry_index} not ok: {err}")
    ssr_block = erec.get("ssr")
    if not isinstance(ssr_block, dict):
        raise SystemExit(f"prior-ssr-json: missing 'ssr' for day {fit_day} e{expiry_index}")
    fixed: dict[str, float | None] = {
        "ssr_ols": _to_float_or_none(ssr_block.get("ssr_ols")),
        "ssr_wls": _to_float_or_none(ssr_block.get("ssr_wls")),
        "ssr_huber": _to_float_or_none(ssr_block.get("ssr_huber")),
        "ssr_lad": _to_float_or_none(ssr_block.get("ssr_lad")),
        "ssr_ts": _to_float_or_none(ssr_block.get("ssr_ts")),
    }
    if not any(v is not None for v in fixed.values()):
        raise SystemExit(f"prior-ssr-json: all SSR values null for day {fit_day} e{expiry_index}")
    fit_diag: dict[str, float | int | str | None] = {"ssr_source": "full_day_ssr_json"}
    fd = erec.get("fit_diagnostics")
    if isinstance(fd, dict):
        for k in (
            "n_points_used",
            "mean_skew",
            "snapshot_spacing_min",
            "n_transitions_after_move_trim",
            "move_trim_frac",
        ):
            if k in fd:
                fit_diag[k] = fd[k]
    fit_diag["doc_snapshot_spacing_min"] = _to_float_or_none(doc.get("snapshot_spacing_min"))
    fit_diag["doc_move_trim_frac"] = _to_float_or_none(doc.get("move_trim_frac"))
    return fixed, fit_diag


def compute_atm_hedge_2h_detail(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    ln_payload: dict | None = None,
    horizon_min: float = 120.0,
    ssr_mode: str = "rolling",
    sigma_mode: str = "avg3",
    prior_day_fallback: str = "same_day",
    min_entry_spacing_min: float = 0.0,
    entry_time_window: str = "all",
    prior_full_day_ssr_json: dict | None = None,
) -> tuple[list[dict[str, object]], dict[str, float | int | str | None], dict[str, int | float | str | None]]:
    """
    ATM hedge residuals ε = ΔC − δ·ΔF (naive BS δ vs SSR-adjusted δ).

    ``ssr_mode``:
      - ``rolling``: SSR at each snapshot from ln-scatter ``pred_series`` (1h rolling regression).
      - ``prior_day``: one SSR per method from the **previous** trading day's full-day Δln transitions;
        held fixed for all hedge cycles on ``day``. No intraday SSR warmup.

    ``horizon_min``: holding period in minutes (e.g. 30, 60, 120).

    ``min_entry_spacing_min``: if > 0, only start a new hedge when at least this many minutes have
    passed since the **previous kept** entry snapshot (approximates trading every N minutes instead of
    every batch snapshot).

    ``entry_time_window``: ``all`` (default), ``open_2h`` (09:30–11:30 naive), or ``close_2h``
    (last ~2h of feasible entry times ending at ``t_last − horizon_min``, from 12:00) — only hedges
    whose **entry** time falls in the band are kept.

    First batch day: if ``prior_day_fallback`` is ``skip``, raises; if ``same_day``, fits SSR from
    ``day``'s full-day transitions (labeled ``same_day_fallback`` in summary — look-ahead bias).

    If ``prior_full_day_ssr_json`` is set and ``ssr_mode`` is ``prior_day``, SSRs are taken from that
    document for ``(ssr_fit_day, expiry_index)`` instead of refitting from batch snapshots (e.g. to
    match a ``--full-day-ssr-json`` run built with ``--full-day-ssr-snapshot-spacing-min 5``).
    """
    if ssr_mode not in ("rolling", "prior_day"):
        raise SystemExit("ssr_mode must be 'rolling' or 'prior_day'.")
    if prior_day_fallback not in ("skip", "same_day"):
        raise SystemExit("prior_day_fallback must be 'skip' or 'same_day'.")

    summary = read_summary(batch_dir, day=day)
    if not summary:
        raise SystemExit(f"No successful snapshots found for date {day}.")

    fixed_ssr: dict[str, float | None] | None = None
    hedge_ssr_source = "rolling_timestamp"
    ssr_fit_day: str | None = None
    fit_diag: dict[str, float | int | str | None] = {}

    if ssr_mode == "prior_day":
        prior = prior_trading_day(batch_dir, day)
        if prior is None:
            if prior_day_fallback == "skip":
                raise SystemExit(
                    f"No prior trading day in batch before {day}: cannot use prior_day SSR (use "
                    "--atm-hedge-prior-fallback same_day or pick a later day)."
                )
            fit_day = day
            hedge_ssr_source = "same_day_fallback"
        else:
            fit_day = prior
            hedge_ssr_source = "prior_day"
        ssr_fit_day = fit_day
        if prior_full_day_ssr_json is not None:
            fixed_ssr, fit_diag = load_prior_ssr_from_full_day_ssr_json_doc(
                prior_full_day_ssr_json, fit_day=fit_day, expiry_index=expiry_index
            )
            hedge_ssr_source = "prior_day_json" if prior is not None else "same_day_json"
        else:
            snaps_fit, _, _, _ = _build_ln_scatter_snaps(
                batch_dir, day=fit_day, expiry_index=expiry_index, sigma_mode=sigma_mode
            )
            pts_fit = _ln_transition_points_from_snaps(snaps_fit)
            ssr_map, fit_diag = fit_ssr_full_day_from_points(pts_fit)
            if not any(v is not None for v in ssr_map.values()):
                err = fit_diag.get("error", "fit_failed")
                raise SystemExit(f"Prior-day SSR fit produced no SSRs for {fit_day} (e={expiry_index}): {err}")
            fixed_ssr = {
                "ssr_ols": ssr_map["ssr_ols"],
                "ssr_wls": ssr_map["ssr_wls"],
                "ssr_huber": ssr_map["ssr_huber"],
                "ssr_lad": ssr_map["ssr_lad"],
                "ssr_ts": ssr_map["ssr_ts"],
            }
        ssr_by_ts: dict[str, dict[str, float | None]] = {}
    else:
        if ln_payload is None:
            try:
                ln_payload = build_ln_scatter_payload(
                    batch_dir, day=day, expiry_index=expiry_index, sigma_mode=sigma_mode
                )
            except Exception:  # noqa: BLE001
                ln_payload = {"pred_series": []}
        ssr_by_ts = _ssr_by_ts_from_ln_payload(ln_payload)

    snaps: list[dict] = []
    for row in summary:
        st = _build_snapshot_state(batch_dir, row, expiry_index, ssr_by_ts, fixed_ssr=fixed_ssr)
        if st is not None and st.get("time") is not None:
            snaps.append(st)
    if len(snaps) < 3:
        raise SystemExit("Not enough snapshots with option data for ATM hedge experiment.")

    times = [s["time"] for s in snaps]
    rows: list[dict[str, object]] = []
    skipped_missing = 0
    skipped_small_df = 0
    skipped_entry_spacing = 0
    skipped_entry_window = 0
    last_entry_time: datetime | None = None
    methods = ("ols", "wls", "huber", "lad", "ts")
    ew = (entry_time_window or "all").strip().lower()

    for i in range(len(snaps) - 1):
        ti_entry = times[i]
        if (
            float(min_entry_spacing_min) > 1e-9
            and last_entry_time is not None
            and ti_entry is not None
        ):
            if (ti_entry - last_entry_time) < timedelta(minutes=float(min_entry_spacing_min)):
                skipped_entry_spacing += 1
                continue
        if ew == "open_2h":
            if ti_entry is None or not _rth_open_two_hour_allows(ti_entry):
                skipped_entry_window += 1
                continue
        elif ew == "close_2h":
            ts_valid = [t for t in times if t is not None]
            if not ts_valid or ti_entry is None:
                skipped_entry_window += 1
                continue
            t_last = max(ts_valid)
            if not _close_session_feasible_two_hour_allows(
                ti_entry, t_last=t_last, horizon_min=float(horizon_min)
            ):
                skipped_entry_window += 1
                continue
        elif ew != "all":
            raise SystemExit(f"unknown entry_time_window {ew!r}")
        j = _find_forward_idx(times, i, float(horizon_min))
        if j is None:
            continue
        s0 = snaps[i]
        s1 = snaps[j]
        f0 = float(s0["fwd"])
        f1 = float(s1["fwd"])
        if not (f0 > 0.0 and f1 > 0.0):
            skipped_missing += 1
            continue
        dF = f1 - f0
        if abs(dF) <= 1e-12:
            skipped_small_df += 1
            continue
        dlnf = math.log(f1) - math.log(f0)

        zs0 = [float(z) for z in s0["zs"]]
        atm_idx = min(range(len(zs0)), key=lambda ii: abs(zs0[ii]))
        k = float(s0["strikes"][atm_idx])
        sigma0 = interp_linear(s0["strikes"], s0["fitted_vols"], k)
        sigma1 = interp_linear(s1["strikes"], s1["fitted_vols"], k)
        if sigma0 is None or sigma1 is None or sigma0 <= 0.0 or sigma1 <= 0.0:
            skipped_missing += 1
            continue
        r0 = float(s0.get("r", 0.0))
        c0 = _bs_call_price_from_fwd(f0, k, sigma0, float(s0["vol_time"]), r=r0)
        c1 = _bs_call_price_from_fwd(f1, k, sigma1, float(s1["vol_time"]), r=r0)
        g = _bs_call_greeks_from_fwd(f0, k, sigma0, float(s0["vol_time"]), r=r0)
        if c0 is None or c1 is None or g is None:
            skipped_missing += 1
            continue
        s_atf_snap = _to_float_or_none(s0.get("s_atf_norm"))
        if s_atf_snap is None:
            skipped_missing += 1
            continue
        ssrs = {
            "ols": _to_float_or_none(s0.get("ssr_ols")),
            "wls": _to_float_or_none(s0.get("ssr_wls")),
            "huber": _to_float_or_none(s0.get("ssr_huber")),
            "lad": _to_float_or_none(s0.get("ssr_lad")),
            "ts": _to_float_or_none(s0.get("ssr_ts")),
        }
        if not any(v is not None for v in ssrs.values()):
            skipped_missing += 1
            continue

        dsigma_real = float(sigma1) - float(sigma0)
        dln_sigma_real = math.log(float(sigma1) / float(sigma0)) if sigma0 > 0.0 else float("nan")
        dc_real = float(c1) - float(c0)
        delta_naive = float(g["delta"])
        gamma_t = float(g["gamma"])
        vega_t = float(g["vega"])
        eps_naive = dc_real - delta_naive * dF
        # Match legacy single-method convention: extra 0.5 on snapshot skew in σ prediction path.
        s_atf_used = 0.5 * float(s_atf_snap)

        row: dict[str, object] = {
            "from_t": str(s0["timestamp"]),
            "to_t": str(s1["timestamp"]),
            "hedge_ssr_source": hedge_ssr_source,
            "K": k,
            "dF": dF,
            "dlnF": dlnf,
            "dln_sigma_real": dln_sigma_real,
            "dSigma_real": dsigma_real,
            "delta_naive": delta_naive,
            "gamma_t": gamma_t,
            "vega_t": vega_t,
            "s_atf_volspace_snapshot": float(s_atf_snap),
            "s_atf_used_for_sigma_pred": s_atf_used,
            "dC_real": dc_real,
            "eps_naive": eps_naive,
            "gamma_bleed": 0.5 * gamma_t * dF * dF,
            "vega_term_real": vega_t * dsigma_real,
        }
        for m in methods:
            ssr_m = ssrs[m]
            row[f"ssr_{m}"] = ssr_m
            if ssr_m is None:
                row[f"delta_rolling_{m}"] = None
                row[f"dSigma_pred_{m}"] = None
                row[f"vega_term_pred_{m}"] = None
                row[f"eps_hedge_{m}"] = None
                continue
            dln_sigma_pred = float(ssr_m) * s_atf_used * dlnf
            dsigma_pred = _dsigma_from_sigma0_dln_sigma(float(sigma0), dln_sigma_pred)
            delta_roll = delta_naive + (vega_t * dsigma_pred / dF)
            eps_h = dc_real - delta_roll * dF
            row[f"delta_rolling_{m}"] = delta_roll
            row[f"dSigma_pred_{m}"] = dsigma_pred
            row[f"vega_term_pred_{m}"] = vega_t * dsigma_pred
            row[f"eps_hedge_{m}"] = eps_h

        # Backward-compatible aliases (WLS = primary rolling hedge in prior exports)
        row["ssr_used"] = row["ssr_wls"]
        row["delta_rolling"] = row["delta_rolling_wls"]
        row["dSigma_pred"] = row["dSigma_pred_wls"]
        row["vega_term_pred"] = row["vega_term_pred_wls"]
        row["eps_rolling"] = row["eps_hedge_wls"]
        rows.append(row)
        if ti_entry is not None:
            last_entry_time = ti_entry

    if not rows:
        raise SystemExit("No valid rows for ATM hedge experiment.")

    eps_n = [float(r["eps_naive"]) for r in rows]
    n = len(rows)
    mean_eps_n = sum(eps_n) / n
    var_eps_n = sum((x - mean_eps_n) * (x - mean_eps_n) for x in eps_n) / n
    std_eps_n = math.sqrt(var_eps_n)

    dlnf_corr: list[float] = []
    dlns_corr: list[float] = []
    for r in rows:
        dx = _to_float_or_none(r.get("dlnF"))
        dy = _to_float_or_none(r.get("dln_sigma_real"))
        if dx is not None and dy is not None:
            dlnf_corr.append(float(dx))
            dlns_corr.append(float(dy))
    corr_dlns_dlnf = _corr(dlnf_corr, dlns_corr) if len(dlnf_corr) == len(dlns_corr) and len(dlnf_corr) > 1 else None

    sampling_lbl = (
        f"min_entry_spacing_{float(min_entry_spacing_min):g}min"
        if float(min_entry_spacing_min) > 1e-9
        else "overlapping_1min_stride"
    )
    summary_row: dict[str, float | int | str | None] = {
        "strike_scope": "ATM",
        "horizon_min": float(horizon_min),
        "min_entry_spacing_min": float(min_entry_spacing_min),
        "entry_time_window": ew,
        "ssr_mode": ssr_mode,
        "hedge_ssr_source": hedge_ssr_source,
        "ssr_fit_day": ssr_fit_day,
        "sigma_mode": sigma_mode,
        "sampling": sampling_lbl,
        "n": n,
        "mean_eps_naive": mean_eps_n,
        "std_eps_naive": std_eps_n,
        "var_eps_naive": var_eps_n,
        "corr_dln_sigma_real_dln_f": corr_dlns_dlnf,
    }
    if ssr_mode == "prior_day":
        summary_row["n_ssr_fit_transitions"] = fit_diag.get("n_points_used")
        summary_row["mean_skew_ssr_fit"] = fit_diag.get("mean_skew")
    vt_pred_wls = [float(r["vega_term_pred_wls"]) for r in rows if r.get("vega_term_pred_wls") is not None]
    vt_real_wls = [float(r["vega_term_real"]) for r in rows if r.get("vega_term_pred_wls") is not None]
    if len(vt_pred_wls) == len(vt_real_wls) and len(vt_pred_wls) > 1:
        summary_row["corr_vega_term_pred_real_wls"] = _corr(vt_pred_wls, vt_real_wls)
        summary_row["beta_vega_real_on_pred_wls"] = _beta_y_on_x(vt_pred_wls, vt_real_wls)
    else:
        summary_row["corr_vega_term_pred_real_wls"] = None
        summary_row["beta_vega_real_on_pred_wls"] = None

    for m in methods:
        eps_m = [
            float(r[f"eps_hedge_{m}"])
            for r in rows
            if r.get(f"eps_hedge_{m}") is not None and math.isfinite(float(r[f"eps_hedge_{m}"]))
        ]
        if not eps_m:
            summary_row[f"n_eps_{m}"] = 0
            summary_row[f"mean_eps_hedge_{m}"] = None
            summary_row[f"std_eps_hedge_{m}"] = None
            summary_row[f"var_eps_hedge_{m}"] = None
            summary_row[f"var_ratio_{m}_vs_naive"] = None
            continue
        nm = len(eps_m)
        mm = sum(eps_m) / nm
        vm = sum((x - mm) * (x - mm) for x in eps_m) / nm
        sm = math.sqrt(vm)
        summary_row[f"n_eps_{m}"] = nm
        summary_row[f"mean_eps_hedge_{m}"] = mm
        summary_row[f"std_eps_hedge_{m}"] = sm
        summary_row[f"var_eps_hedge_{m}"] = vm
        vr = (vm / var_eps_n) if var_eps_n > 1e-30 else None
        summary_row[f"var_ratio_{m}_vs_naive"] = vr
        summary_row[f"var_red_{m}_vs_naive"] = (1.0 - vr) if vr is not None and math.isfinite(vr) else None

    for m in methods:
        summary_row.update(klassen_dsigma_calibration_metrics(rows, m))

    # Legacy summary fields (WLS rolling)
    summary_row["mean_eps_rolling"] = summary_row.get("mean_eps_hedge_wls")
    summary_row["std_eps_rolling"] = summary_row.get("std_eps_hedge_wls")
    summary_row["var_ratio"] = summary_row.get("var_ratio_wls_vs_naive")

    out_diag: dict[str, int | float | str | None] = {
        "n_rows": len(rows),
        "n_snapshots_used": len(snaps),
        "n_skipped_missing": skipped_missing,
        "n_skipped_small_df": skipped_small_df,
        "n_skipped_entry_spacing": skipped_entry_spacing,
        "n_skipped_entry_window": skipped_entry_window,
        "entry_time_window": ew,
        "min_entry_spacing_min": float(min_entry_spacing_min),
        "horizon_min": float(horizon_min),
        "ssr_mode": ssr_mode,
        "hedge_ssr_source": hedge_ssr_source,
    }
    if ssr_mode == "prior_day":
        out_diag["ssr_fit_day"] = ssr_fit_day
        out_diag.update({f"ssr_fit_{k}": v for k, v in fit_diag.items()})
    return rows, summary_row, out_diag


def render_hedge_residual_distributions_html(
    rows: list[dict[str, object]],
    summary: dict[str, float | int | str | None],
    title: str,
) -> str:
    """Histogram overlays of hedge residuals ε = ΔC - δ·ΔF by method."""
    payload = {"rows": rows, "summary": summary}
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    h1 {{ font-size: 1.1rem; margin-bottom: 8px; }}
    .plot {{ width: 100%; height: 560px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    p.hint {{ color: #444; font-size: 0.92rem; max-width: 920px; line-height: 1.45; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>ATM 2h hedge residual distributions</h1>
  <p class="hint">
    Hedge residual per row: <b>ε = ΔC − δ<sub>hedge</sub>·ΔF</b> (BS call prices, ATM strike).
    <b>Naive</b> uses δ<sub>BS</sub> at t; each <b>rolling SSR</b> method adjusts δ by ν·Δσ̂/ΔF with Δσ̂ from that method’s regression.
    For P&amp;L over the window, you generally want the distribution <b>tight</b> (low dispersion) and <b>centered near zero</b>
    (little systematic bleed). Overlapping windows make errors serially correlated — interpret dispersion as descriptive.
  </p>
  <div id="hist_eps" class="plot"></div>
  <div id="ts_eps" class="plot"></div>
  <h3>Summary (population variance uses n divisor per row set)</h3>
  <pre id="sum"></pre>
  <script>
    const P = {data_json};
    const rows = P.rows || [];
    const summ = P.summary || {{}};
    const methods = [
      {{ key: "naive", col: "eps_naive", label: "Naive δ", color: "#7f7f7f" }},
      {{ key: "ols", col: "eps_hedge_ols", label: "Rolling OLS SSR", color: "#2ca02c" }},
      {{ key: "wls", col: "eps_hedge_wls", label: "Rolling WLS SSR", color: "#d62728" }},
      {{ key: "huber", col: "eps_hedge_huber", label: "Rolling Huber SSR", color: "#ff7f0e" }},
      {{ key: "lad", col: "eps_hedge_lad", label: "Rolling LAD SSR", color: "#8c564b" }},
      {{ key: "ts", col: "eps_hedge_ts", label: "Rolling Theil-Sen SSR", color: "#17becf" }},
    ];
    const histTraces = methods.map(m => {{
      const vals = rows.map(r => r[m.col]).filter(v => v != null && Number.isFinite(v));
      return {{
        type: "histogram",
        name: m.label,
        x: vals,
        opacity: 0.55,
        marker: {{ color: m.color }},
        histnorm: "probability density",
        nbinsx: 45,
      }};
    }});
    Plotly.newPlot("hist_eps", histTraces, {{
      title: "Distribution of hedge residuals ε",
      xaxis: {{ title: "ε = ΔC − δ·ΔF" }},
      yaxis: {{ title: "density" }},
      barmode: "overlay",
      margin: {{ l: 60, r: 30, t: 48, b: 50 }},
    }}, {{ responsive: true }});

    const x = rows.map(r => r.to_t);
    Plotly.newPlot("ts_eps", methods.map(m => ({{
      type: "scatter",
      mode: "lines+markers",
      x,
      y: rows.map(r => r[m.col] ?? null),
      name: m.label,
      line: {{ width: 1.4, color: m.color }},
      marker: {{ size: 4 }},
      hovertemplate: "%{{x}}<br>%{{y:.6f}}<extra></extra>",
    }})), {{
      title: "Hedge residual time series",
      xaxis: {{ title: "to_t", tickangle: -45 }},
      yaxis: {{ title: "ε" }},
      shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#888" }} }}],
      margin: {{ l: 60, r: 30, t: 48, b: 100 }},
    }}, {{ responsive: true }});

    document.getElementById("sum").textContent = JSON.stringify(summ, null, 2);
  </script>
</body>
</html>
"""


def render_klassen_ssr_methodology_html(
    summary_row: dict[str, float | int | str | bool | None],
    title: str,
) -> str:
    """
    Static reference + Plotly charts for Klassen-style σ-calibration metrics (ρ, r, VarRed≈2ρr−r²)
    and measured hedge residual variance reduction.
    """
    safe_summary = json.loads(json.dumps(summary_row, default=str))
    data_json = json.dumps(safe_summary, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; color: #1a1a18; }}
    h1 {{ font-size: 1.15rem; margin-bottom: 8px; }}
    h2 {{ font-size: 1rem; margin: 24px 0 8px; border-bottom: 1px solid #d8d6cf; padding-bottom: 4px; }}
    .plot {{ width: 100%; height: 420px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 12px 0; }}
    .ref {{ background: #f7f6f3; border-radius: 8px; padding: 14px 18px; margin: 12px 0; font-size: 0.92rem; line-height: 1.55; max-width: 920px; }}
    .ref code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.88em; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 960px; margin: 12px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    .ok {{ color: #0f6e56; font-weight: 600; }}
    .bad {{ color: #a32d2d; font-weight: 600; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="ref">
    <b>Variance reduction (hedge residuals).</b> Measured:
    <code>VarRed_meas = 1 − Var(ε_roll)/Var(ε_naive)</code> with <code>ε = ΔC − δ·ΔF</code>.<br/>
    <b>σ-prediction diagnostics</b> (same rows, ATM 2h window):
    <code>ρ = corr(dσ̂, Δσ)</code>,
    <code>r = SD(dσ̂)/SD(Δσ)</code>,
    OLS <code>β_cal</code> of <code>Δσ</code> on <code>dσ̂</code> (≈ <code>ρ/r</code>).
    Approximate identity: <code>VarRed ≈ 2ρr − r²</code> when <code>ε_naive ≈ ν·Δσ</code>.
    <b>Winning condition (predicted):</b> <code>ρ &gt; r/2</code> ⇒ positive VarRed from this decomposition.
  </div>
  <h2>Measured vs approximate variance reduction</h2>
  <div id="chart_varred" class="plot"></div>
  <h2>ρ, r, and calibration</h2>
  <div id="chart_rho_r" class="plot"></div>
  <h2>Summary table</h2>
  <div id="tbl"></div>
  <h2>Raw summary (CSV row)</h2>
  <pre id="raw"></pre>
  <script>
    const S = {data_json};
    const methods = [
      {{ key: "ols", label: "OLS" }},
      {{ key: "wls", label: "WLS" }},
      {{ key: "huber", label: "Huber" }},
      {{ key: "lad", label: "LAD" }},
      {{ key: "ts", label: "Theil–Sen" }},
    ];
    const labels = methods.map(m => m.label);
    const varMeas = methods.map(m => {{
      const v = S["var_red_" + m.key + "_vs_naive"];
      return (v != null && Number.isFinite(+v)) ? +v : null;
    }});
    const varTheo = methods.map(m => {{
      const v = S["var_red_two_rho_r_minus_r2_" + m.key];
      return (v != null && Number.isFinite(+v)) ? +v : null;
    }});
    Plotly.newPlot("chart_varred", [
      {{ type: "bar", name: "VarRed measured (ε)", x: labels, y: varMeas, marker: {{ color: "#185fa5" }} }},
      {{ type: "bar", name: "2ρr−r² (σ approx)", x: labels, y: varTheo, marker: {{ color: "#888" }} }},
    ], {{
      title: "Variance reduction: hedge residuals vs σ-decomposition approx.",
      yaxis: {{ title: "VarRed" }},
      barmode: "group",
      margin: {{ t: 40, b: 60, l: 55, r: 25 }},
      shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#333" }} }}],
    }}, {{ responsive: true }});

    const rho = methods.map(m => {{
      const v = S["rho_dsigma_" + m.key];
      return (v != null && Number.isFinite(+v)) ? +v : null;
    }});
    const rsd = methods.map(m => {{
      const v = S["r_sd_ratio_dsigma_" + m.key];
      return (v != null && Number.isFinite(+v)) ? +v : null;
    }});
    const halfR = rsd.map(rv => (rv != null && Number.isFinite(rv)) ? 0.5 * rv : null);
    Plotly.newPlot("chart_rho_r", [
      {{ type: "bar", name: "ρ = corr(dσ̂,Δσ)", x: labels, y: rho, marker: {{ color: "#2ca02c" }} }},
      {{ type: "bar", name: "r = SD(dσ̂)/SD(Δσ)", x: labels, y: rsd, marker: {{ color: "#d62728" }} }},
      {{ type: "bar", name: "r/2", x: labels, y: halfR, marker: {{ color: "#ffbb78" }} }},
    ], {{
      title: "ρ vs r (predicted improvement when ρ > r/2)",
      yaxis: {{ title: "value" }},
      barmode: "group",
      margin: {{ t: 40, b: 60, l: 55, r: 25 }},
      shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#333" }} }}],
    }}, {{ responsive: true }});

    let html = "<table><thead><tr><th>Method</th><th>n pairs</th><th>ρ</th><th>r</th><th>ρ&gt;r/2</th><th>β_cal</th><th>2ρr−r²</th><th>VarRed_meas</th></tr></thead><tbody>";
    methods.forEach(m => {{
      const n = S["n_dsigma_pairs_" + m.key];
      const r = S["rho_dsigma_" + m.key];
      const rr = S["r_sd_ratio_dsigma_" + m.key];
      const win = S["rho_gt_half_r_" + m.key];
      const bc = S["beta_cal_dsigma_" + m.key];
      const vt = S["var_red_two_rho_r_minus_r2_" + m.key];
      const vm = S["var_red_" + m.key + "_vs_naive"];
      const wcls = (win === true || win === "True") ? "ok" : ((win === false || win === "False") ? "bad" : "");
      html += "<tr><td>" + m.label + "</td><td>" + (n ?? "") + "</td><td>" + (r ?? "") + "</td><td>" + (rr ?? "") + "</td><td class='" + wcls + "'>" + win + "</td><td>" + (bc ?? "") + "</td><td>" + (vt ?? "") + "</td><td>" + (vm ?? "") + "</td></tr>";
    }});
    html += "</tbody></table>";
    document.getElementById("tbl").innerHTML = html;
    document.getElementById("raw").textContent = JSON.stringify(S, null, 2);
  </script>
</body>
</html>
"""


def atm_hedge_file_prefix(
    *,
    expiry_index: int,
    day: str,
    horizon_min: float,
    ssr_mode: str,
    entry_spacing_min: float = 0.0,
) -> str:
    """Stable output basename for ATM hedge CSV/HTML artifacts."""
    es = float(entry_spacing_min)
    sp = ""
    if es > 1e-9:
        sp = f"_s{int(es)}m" if abs(es - round(es)) < 1e-9 else f"_s{es}m"
    if ssr_mode == "rolling" and abs(float(horizon_min) - 120.0) < 1e-9 and not sp:
        return f"atm_hedge_2h_e{expiry_index}_{day}"
    hm = int(horizon_min) if float(horizon_min).is_integer() else horizon_min
    tag = "priorSSR" if ssr_mode == "prior_day" else "rollingSSR"
    return f"atm_hedge_{tag}_h{hm}m{sp}_e{expiry_index}_{day}"


def write_atm_hedge_2h_artifacts(
    batch_dir: Path,
    file_prefix: str,
    rows: list[dict[str, object]],
    summary_row: dict[str, float | int | str | None],
) -> tuple[Path, Path, Path, Path]:
    """Write detail CSV, summary CSV, hedge residual HTML, and Klassen σ-methodology HTML."""
    out_dir = batch_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{file_prefix}_detail.csv"
    try:
        fh_detail = detail_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        detail_path = out_dir / f"{file_prefix}_detail_new.csv"
        fh_detail = detail_path.open("w", newline="", encoding="utf-8")
    with fh_detail as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_path = out_dir / f"{file_prefix}_summary.csv"
    try:
        fh_summary = summary_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        summary_path = out_dir / f"{file_prefix}_summary_new.csv"
        fh_summary = summary_path.open("w", newline="", encoding="utf-8")
    with fh_summary as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        w.writeheader()
        w.writerow(summary_row)

    dist_path = out_dir / f"{file_prefix}_hedge_residuals.html"
    try:
        dist_path.write_text(
            render_hedge_residual_distributions_html(
                rows,
                summary_row,
                f"ATM 2h hedge residuals — {file_prefix}",
            ),
            encoding="utf-8",
        )
    except PermissionError:
        dist_path = out_dir / f"{file_prefix}_hedge_residuals_new.html"
        dist_path.write_text(
            render_hedge_residual_distributions_html(
                rows,
                summary_row,
                f"ATM 2h hedge residuals — {file_prefix}",
            ),
            encoding="utf-8",
        )

    klassen_path = out_dir / f"{file_prefix}_klassen_ssr_methodology.html"
    try:
        klassen_path.write_text(
            render_klassen_ssr_methodology_html(
                summary_row,
                f"Klassen / previous-day SSR diagnostics — {file_prefix}",
            ),
            encoding="utf-8",
        )
    except PermissionError:
        klassen_path = out_dir / f"{file_prefix}_klassen_ssr_methodology_new.html"
        klassen_path.write_text(
            render_klassen_ssr_methodology_html(
                summary_row,
                f"Klassen / previous-day SSR diagnostics — {file_prefix}",
            ),
            encoding="utf-8",
        )

    return detail_path, summary_path, dist_path, klassen_path


def write_atm_hedge_2h_csvs(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    file_prefix: str,
    horizon_min: float = 120.0,
    ssr_mode: str = "rolling",
    sigma_mode: str = "avg3",
    prior_day_fallback: str = "same_day",
    min_entry_spacing_min: float = 0.0,
    entry_time_window: str = "all",
    prior_full_day_ssr_json: dict | None = None,
) -> tuple[Path, Path, Path, Path, dict[str, int | float | str | None]]:
    """
    ATM hedge residual experiment (overlapping windows, ~1-min stride in batch unless entry spacing set).
    Writes detail CSV, summary CSV, hedge residual HTML, and Klassen methodology HTML.
    """
    rows, summary_row, diag = compute_atm_hedge_2h_detail(
        batch_dir,
        day=day,
        expiry_index=expiry_index,
        ln_payload=None,
        horizon_min=horizon_min,
        ssr_mode=ssr_mode,
        sigma_mode=sigma_mode,
        prior_day_fallback=prior_day_fallback,
        min_entry_spacing_min=min_entry_spacing_min,
        entry_time_window=entry_time_window,
        prior_full_day_ssr_json=prior_full_day_ssr_json,
    )
    d0, d1, d2, d3 = write_atm_hedge_2h_artifacts(batch_dir, file_prefix, rows, summary_row)
    return d0, d1, d2, d3, diag


_KLASSEN_GRID_METHODS: tuple[tuple[str, str], ...] = (
    ("ols", "OLS"),
    ("wls", "WLS"),
    ("huber", "Huber"),
    ("lad", "LAD"),
    ("ts", "Theil–Sen"),
)


def _fetch_prior_day_atm_hedge_summary(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    horizon_min: float,
    sigma_mode: str,
    prior_day_fallback: str,
    min_entry_spacing_min: float,
    entry_time_window: str = "all",
    prior_full_day_ssr_json: dict | None = None,
) -> dict[str, float | int | str | None] | None:
    """ATM hedge summary row for prior_day SSR, or None if the experiment cannot run."""
    try:
        _rows, summary_row, _diag = compute_atm_hedge_2h_detail(
            batch_dir,
            day=day,
            expiry_index=expiry_index,
            ln_payload=None,
            horizon_min=horizon_min,
            ssr_mode="prior_day",
            sigma_mode=sigma_mode,
            prior_day_fallback=prior_day_fallback,
            min_entry_spacing_min=min_entry_spacing_min,
            entry_time_window=entry_time_window,
            prior_full_day_ssr_json=prior_full_day_ssr_json,
        )
        return summary_row
    except SystemExit:
        return None


def _mse_from_mean_var(mean: object, var: object) -> float | None:
    """MSE = Var(ε) + Mean(ε)² when both finite."""
    m = _to_float_or_none(mean)
    v = _to_float_or_none(var)
    if m is None or v is None:
        return None
    if not (math.isfinite(m) and math.isfinite(v)):
        return None
    return float(v) + float(m) * float(m)


def _rmse_from_mse(mse: float | None) -> float | None:
    if mse is None or not math.isfinite(mse) or mse < 0.0:
        return None
    return math.sqrt(float(mse))


def write_atm_hedge_klassen_grid_csv(
    batch_dir: Path,
    out_path: Path,
    *,
    days: list[str],
    expiry_indices: list[int] | None,
    horizon_min: float,
    sigma_mode: str,
    prior_day_fallback: str,
    min_entry_spacing_min: float,
    entry_time_windows: list[str] | None = None,
    prior_full_day_ssr_json: dict | None = None,
) -> Path:
    """
    One CSV of Klassen-style σ-prediction metrics (per regression method) for each (date, expiry).

    SSR is always **prior_day** (previous calendar day in the batch, or fallback per
    ``prior_day_fallback``), optionally loaded from ``prior_full_day_ssr_json`` per
    ``(ssr_fit_day, expiry)``. Metrics match the Klassen HTML table: ρ, r, ρ>r/2, β_cal,
    2ρr−r², and measured var reduction vs naive hedge.

    Also reports hedge residual **mean**, **var**, **MSE** = Var + Mean², **RMSE** = sqrt(MSE), and
    fractional **MSE improvement vs naive**: (MSE_naive − MSE_method) / MSE_naive.

    ``entry_time_windows``: e.g. ``[\"all\"]`` (default), or ``[\"open_2h\", \"close_2h\"]`` for
    open morning vs last feasible ~2h band (see ``entry_time_window`` on ``compute_atm_hedge_2h_detail``).
    """
    wins = entry_time_windows if entry_time_windows is not None else ["all"]
    fieldnames = [
        "date",
        "expiry_index",
        "entry_window",
        "horizon_min",
        "min_entry_spacing_min",
        "hedge_ssr_source",
        "ssr_fit_day",
        "n_hedge_rows",
        "method",
        "n_pairs",
        "rho",
        "r",
        "rho_gt_half_r",
        "beta_cal",
        "two_rho_r_minus_r2",
        "var_red_meas",
        "mean_eps_naive",
        "var_eps_naive",
        "mse_naive",
        "rmse_naive",
        "mean_eps",
        "var_eps",
        "mse",
        "rmse",
        "mse_imp_vs_naive",
        "rmse_imp_vs_naive",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for day in days:
        if expiry_indices is None:
            try:
                m = infer_num_expiries_from_batch(batch_dir, day=day)
                e_iter = list(range(m))
            except SystemExit:
                print(f"klassen-grid: skip day {day} (cannot infer expiries)", file=sys.stderr)
                continue
        else:
            e_iter = list(expiry_indices)
        for e in e_iter:
            for win in wins:
                summary_row = _fetch_prior_day_atm_hedge_summary(
                    batch_dir,
                    day=day,
                    expiry_index=e,
                    horizon_min=horizon_min,
                    sigma_mode=sigma_mode,
                    prior_day_fallback=prior_day_fallback,
                    min_entry_spacing_min=min_entry_spacing_min,
                    entry_time_window=win,
                    prior_full_day_ssr_json=prior_full_day_ssr_json,
                )
                if summary_row is None:
                    print(f"klassen-grid: skip {day} e{e} win={win} (hedge failed)", file=sys.stderr)
                    continue
                hsrc = summary_row.get("hedge_ssr_source")
                sfit = summary_row.get("ssr_fit_day")
                nhr = summary_row.get("n")
                mn0 = summary_row.get("mean_eps_naive")
                vn0 = summary_row.get("var_eps_naive")
                mse0 = _mse_from_mean_var(mn0, vn0)
                rmse0 = _rmse_from_mse(mse0)
                for mid, mlabel in _KLASSEN_GRID_METHODS:
                    mm = summary_row.get(f"mean_eps_hedge_{mid}")
                    vm = summary_row.get(f"var_eps_hedge_{mid}")
                    mse_m = _mse_from_mean_var(mm, vm)
                    rmse_m = _rmse_from_mse(mse_m)
                    mse_imp = None
                    rmse_imp = None
                    if mse0 is not None and mse_m is not None and mse0 > 1e-30:
                        mse_imp = (mse0 - mse_m) / mse0
                    if rmse0 is not None and rmse_m is not None and rmse0 > 1e-30:
                        rmse_imp = (rmse0 - rmse_m) / rmse0
                    all_rows.append(
                        {
                            "date": day,
                            "expiry_index": e,
                            "entry_window": win,
                            "horizon_min": summary_row.get("horizon_min"),
                            "min_entry_spacing_min": summary_row.get("min_entry_spacing_min"),
                            "hedge_ssr_source": hsrc,
                            "ssr_fit_day": sfit,
                            "n_hedge_rows": nhr,
                            "method": mlabel,
                            "n_pairs": summary_row.get(f"n_dsigma_pairs_{mid}"),
                            "rho": summary_row.get(f"rho_dsigma_{mid}"),
                            "r": summary_row.get(f"r_sd_ratio_dsigma_{mid}"),
                            "rho_gt_half_r": summary_row.get(f"rho_gt_half_r_{mid}"),
                            "beta_cal": summary_row.get(f"beta_cal_dsigma_{mid}"),
                            "two_rho_r_minus_r2": summary_row.get(f"var_red_two_rho_r_minus_r2_{mid}"),
                            "var_red_meas": summary_row.get(f"var_red_{mid}_vs_naive"),
                            "mean_eps_naive": mn0,
                            "var_eps_naive": vn0,
                            "mse_naive": mse0,
                            "rmse_naive": rmse0,
                            "mean_eps": mm,
                            "var_eps": vm,
                            "mse": mse_m,
                            "rmse": rmse_m,
                            "mse_imp_vs_naive": mse_imp,
                            "rmse_imp_vs_naive": rmse_imp,
                        }
                    )
    if not all_rows:
        raise SystemExit("klassen-grid: no rows written (all cells failed)")
    try:
        fh = out_path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        out_path = out_path.with_name(out_path.stem + "_new" + out_path.suffix)
        fh = out_path.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    return out_path

