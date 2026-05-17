"""Default q/vol payload, SSR & ln-scatter, full-day SSR, ln details CSV."""
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
from .fundamentals import _read_single_row_csv

def build_payload(batch_dir: Path) -> dict:
    summary = read_summary(batch_dir)
    if not summary:
        raise SystemExit("No ok rows in batch_cvi_summary.csv")

    expiry_labels: list[tuple[int, str]] | None = None
    times: list[str] = []
    short_times: list[str] = []
    q_matrix: dict[int, list[float | None]] = defaultdict(list)  # expiry -> [q per t]
    f0_series: list[float] = []

    vol_snapshots: list[dict[tuple[int, float], tuple[float, float]]] = []

    for row in summary:
        sub = batch_dir / row["subfolder"]
        efq = sub / "expiry_fwd_q.csv"
        opt = sub / "option_fit_comparison.csv"
        if not efq.is_file() or not opt.is_file():
            continue
        order, qmap, _fmap, f0 = read_expiry_fwd_q(efq)
        if expiry_labels is None:
            expiry_labels = order
        times.append(row["timestamp"])
        st = row["timestamp"].strip().split()
        short_times.append(st[1][:8] if len(st) > 1 else row["timestamp"])
        f0_series.append(f0)
        for idx, _lab in expiry_labels:
            q_matrix[idx].append(qmap.get(idx))
        vol_snapshots.append(read_option_fit(opt))

    if not times or expiry_labels is None:
        raise SystemExit("Missing expiry_fwd_q / option_fit in batch folders.")

    ex_order = [x[0] for x in expiry_labels]
    ex_labs = [f"{i} — {lab}" for i, lab in expiry_labels]
    expiry_label_by_idx = {str(i): f"{i} — {lab}" for i, lab in expiry_labels}

    # q_heatmap: rows = expiry order, cols = time
    q_heat = [[q_matrix[j][ti] for ti in range(len(times))] for j in ex_order]

    all_strikes = sorted({k for vs in vol_snapshots for (_e, k) in vs})
    # strike_key -> expiry_idx str -> WLS block with line endpoints
    strike_scatter: dict[str, dict[str, dict]] = {}
    for k in all_strikes:
        kstr = fmt_strike_key(k)
        per_exp: dict[str, dict] = {}
        for e in ex_order:
            pts: list[dict] = []
            for t in range(1, len(times)):
                d_s = pct_change(f0_series[t], f0_series[t - 1])
                if d_s is None or not math.isfinite(d_s):
                    continue
                prev = vol_snapshots[t - 1]
                cur = vol_snapshots[t]
                key = (e, k)
                if key not in prev or key not in cur:
                    continue
                _z0, v0 = prev[key]
                _z1, v1 = cur[key]
                d_vol = pct_change(v1, v0)
                if d_vol is None or not math.isfinite(d_vol):
                    continue
                pts.append(
                    {
                        "x": d_s,
                        "y": d_vol,
                        "t_label": f"{short_times[t - 1]} → {short_times[t]}",
                    }
                )
            if len(pts) < 2:
                continue
            xs = [p["x"] for p in pts]
            ys = [p["y"] for p in pts]
            ws = wls_weights_inverse_bivariate_move_sq(xs, ys)
            ob = wls_alpha_beta(xs, ys, ws)
            if ob is None:
                continue
            alpha, beta, n = ob
            xmin, xmax = min(xs), max(xs)
            pad = (xmax - xmin) * 0.06 + 1e-9
            x0, x1 = xmin - pad, xmax + pad
            per_exp[str(e)] = {
                "alpha": alpha,
                "beta": beta,
                "n": n,
                "line_x": [x0, x1],
                "line_y": [alpha + beta * x0, alpha + beta * x1],
                "pts": pts,
            }
        if per_exp:
            strike_scatter[kstr] = per_exp

    strikes_list = sorted(strike_scatter.keys(), key=lambda s: float(s))
    best_strike = strikes_list[0] if strikes_list else ""
    best_n = -1
    for ks, inner in strike_scatter.items():
        n = sum(int(block.get("n", 0)) for block in inner.values())
        if n > best_n:
            best_n = n
            best_strike = ks

    # expiry_idx str -> rows sorted by strike: β across strikes for WLS at fixed expiry
    beta_by_expiry: dict[str, list[dict]] = defaultdict(list)
    for kstr, inner in strike_scatter.items():
        try:
            kf = float(kstr)
        except ValueError:
            kf = 0.0
        for estr, block in inner.items():
            if "beta" not in block:
                continue
            beta_by_expiry[estr].append(
                {
                    "strike": kf,
                    "strike_key": kstr,
                    "alpha": block["alpha"],
                    "beta": block["beta"],
                    "n": block["n"],
                }
            )
    for estr in beta_by_expiry:
        beta_by_expiry[estr].sort(key=lambda r: r["strike"])
    beta_by_expiry_out = {k: v for k, v in beta_by_expiry.items()}

    return {
        "times": times,
        "short_times": short_times,
        "expiry_indices": ex_order,
        "expiry_labels": ex_labs,
        "expiry_label_by_idx": expiry_label_by_idx,
        "q_heatmap": q_heat,
        "f0_series": f0_series,
        "spot_note": (
            "S is approximated by forward F at expiry index 0 (same-day) from expiry_fwd_q.csv each snapshot — "
            "so X is 100·(Sₜ−Sₜ₋₁)/Sₜ₋₁ between consecutive batch snapshots. Y is 100·(σₜ−σₜ₋₁)/σₜ₋₁ for fitted vol "
            "at the selected strike and each expiry."
        ),
        "strike_scatter": strike_scatter,
        "strikes_list": strikes_list,
        "default_strike": best_strike,
        "beta_by_expiry": beta_by_expiry_out,
    }


def build_ssr_scatter_payload(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    min_abs_skew: float = 1e-4,
) -> dict:
    summary = read_summary(batch_dir, day=day)
    if not summary:
        raise SystemExit(f"No successful snapshots found for date {day}.")

    snaps: list[dict] = []
    n_missing_expiry = 0
    n_missing_optfit = 0
    n_missing_skew = 0
    for row in summary:
        sub = batch_dir / row["subfolder"]
        efq = sub / "expiry_fwd_q.csv"
        opt = sub / "option_fit_comparison.csv"
        if not efq.is_file() or not opt.is_file():
            n_missing_optfit += 1
            continue
        e = read_expiry_row(efq, expiry_index)
        if e is None:
            n_missing_expiry += 1
            continue
        by_exp = read_option_fit_by_expiry(opt)
        zvol = by_exp.get(expiry_index, [])
        s_info = estimate_s_atf_norm_with_details(zvol, e["sigma_star"])
        s = s_info.get("s_atf_norm")
        if s is None:
            n_missing_skew += 1
        snaps.append(
            {
                "timestamp": row["timestamp"],
                "F": e["F"],
                "sigma_star": e["sigma_star"],
                "s_atf_norm": s,
            }
        )

    if len(snaps) < 2:
        raise SystemExit("Not enough snapshots to form transitions.")

    points: list[dict] = []
    skipped_bad_data = 0
    skipped_small_skew = 0
    for i in range(1, len(snaps)):
        prev = snaps[i - 1]
        cur = snaps[i]
        f0 = float(prev["F"])
        f1 = float(cur["F"])
        s0 = float(prev["sigma_star"])
        s1 = float(cur["sigma_star"])
        sk = cur.get("s_atf_norm")
        if sk is None:
            skipped_bad_data += 1
            continue
        skf = float(sk)
        if (
            not math.isfinite(f0)
            or not math.isfinite(f1)
            or not math.isfinite(s0)
            or not math.isfinite(s1)
            or f0 <= 0.0
            or f1 <= 0.0
            or s0 <= 0.0
            or s1 <= 0.0
            or not math.isfinite(skf)
        ):
            skipped_bad_data += 1
            continue
        if abs(skf) < min_abs_skew:
            skipped_small_skew += 1
            continue
        x = math.log(f1) - math.log(f0)
        dln_sigma = math.log(s1) - math.log(s0)
        y = dln_sigma / skf
        if not (math.isfinite(x) and math.isfinite(y)):
            skipped_bad_data += 1
            continue
        p = {
            "x": x,
            "y": y,
            "s_atf_norm": skf,
            "from_t": prev["timestamp"],
            "to_t": cur["timestamp"],
        }
        points.append(p)

    if not points:
        raise SystemExit("No valid SSR scatter points after filtering.")

    sum_xx = sum(p["x"] * p["x"] for p in points)
    sum_xy = sum(p["x"] * p["y"] for p in points)
    ssr_hat = (sum_xy / sum_xx) if sum_xx > 1e-20 else None

    skew_vals = [float(s["s_atf_norm"]) for s in snaps if s.get("s_atf_norm") is not None]
    skew_preview = skew_vals[:12]
    diagnostics = {
        "date": day,
        "expiry_index": expiry_index,
        "n_snapshots": len(snaps),
        "n_points_used": len(points),
        "n_skipped_bad_data": skipped_bad_data,
        "n_skipped_small_skew": skipped_small_skew,
        "n_missing_expiry_row": n_missing_expiry,
        "n_missing_option_fit": n_missing_optfit,
        "n_missing_skew_estimate": n_missing_skew,
        "min_abs_skew_threshold": min_abs_skew,
        "ssr_hat_origin": ssr_hat,
        "skew_preview": skew_preview,
    }

    return {"points": points, "diagnostics": diagnostics}


def list_batch_trading_days(batch_dir: Path) -> list[str]:
    """Sorted unique calendar dates (YYYY-MM-DD) with ok=1 rows in batch_cvi_summary.csv."""
    p = batch_dir / "batch_cvi_summary.csv"
    if not p.is_file():
        return []
    days: set[str] = set()
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row.get("ok", "0") or 0) != 1:
                continue
            ts = (row.get("timestamp") or "").strip()
            if not ts:
                continue
            days.add(ts.split()[0])
    return sorted(days)


def prior_trading_day(batch_dir: Path, day: str) -> str | None:
    """Previous calendar date in batch summary order that has data before ``day``."""
    days = list_batch_trading_days(batch_dir)
    try:
        i = days.index(day)
    except ValueError:
        return None
    return days[i - 1] if i > 0 else None


def _build_ln_scatter_snaps(
    batch_dir: Path, *, day: str, expiry_index: int, sigma_mode: str
) -> tuple[list[dict], int, int, int]:
    """
    Snapshots for one day / expiry (same construction as ln-scatter).
    Returns (snaps, n_missing, n_nonpositive, n_missing_skew).
    """
    summary = read_summary(batch_dir, day=day)
    if not summary:
        raise SystemExit(f"No successful snapshots found for date {day}.")

    snaps: list[dict] = []
    n_missing = 0
    n_nonpositive = 0
    n_missing_skew = 0
    for row in summary:
        sub = batch_dir / row["subfolder"]
        efq = sub / "expiry_fwd_q.csv"
        opt = sub / "option_fit_comparison.csv"
        if not efq.is_file():
            n_missing += 1
            continue
        e = read_expiry_row(efq, expiry_index)
        if e is None:
            n_missing += 1
            continue
        fwd = float(e["F"])
        sigma_row = float(e["sigma_star"])
        if fwd <= 0.0 or sigma_row <= 0.0:
            n_nonpositive += 1
            continue

        by_exp: dict[int, list[dict[str, float]]] = {}
        if opt.is_file():
            by_exp = read_option_fit_by_expiry(opt)
        zvol = by_exp.get(expiry_index, [])

        if sigma_mode == "z0":
            sigma_info = sigma_star_z0_point(zvol)
        else:
            sigma_info = sigma_star_avg3_nearest_z(zvol)
        sigma_used = sigma_info.get("sigma_star_used")
        if sigma_used is None:
            sigma_used = sigma_row
        sigma_used_f = float(sigma_used)
        if not (math.isfinite(sigma_used_f) and sigma_used_f > 0.0):
            n_nonpositive += 1
            continue

        s_atf = _basis_s_atf_norm_from_snapshot(
            sub,
            expiry_index=expiry_index,
            v_star=float(e["v_star"]),
        )
        if s_atf is None:
            n_missing_skew += 1
        skew_method = "basis_vol_space" if s_atf is not None else "basis_missing"
        snaps.append(
            {
                "timestamp": row["timestamp"],
                "F": fwd,
                "sigma_star_row": sigma_row,
                "sigma_star_used": sigma_used_f,
                "s_atf_norm": s_atf,
                "source_expiry_idx": expiry_index,
                "source_file": "expiry_fwd_q.csv",
                "source_strike_for_F_sigma": "N/A (term-level row, no strike)",
                "method": skew_method,
                "left_z": None,
                "left_vol": None,
                "left_strike": None,
                "atf_z": 0.0 if s_atf is not None else None,
                "atf_vol": sigma_used_f if s_atf is not None else None,
                "atf_strike": None,
                "right_z": None,
                "right_vol": None,
                "right_strike": None,
                **sigma_info,
            }
        )

    if len(snaps) < 2:
        raise SystemExit("Not enough snapshots to build Δln transitions.")
    return snaps, n_missing, n_nonpositive, n_missing_skew


def _subsample_snaps_by_min_time_spacing(
    snaps: list[dict],
    *,
    min_spacing_min: float,
) -> tuple[list[dict], dict[str, float | int | str | None]]:
    """
    Keep snapshots in chronological order so consecutive **kept** timestamps are at least
    ``min_spacing_min`` minutes apart (first snapshot always kept). Skips intermediate
    ~1-minute batch snapshots when a larger spacing is requested (e.g. 5 minutes).
    If any timestamp is unparseable, returns the original list and does not subsample.
    """
    ms = float(min_spacing_min)
    if ms <= 1e-9 or len(snaps) < 2:
        return list(snaps), {
            "snapshot_spacing_min": ms,
            "n_snapshots_before_spacing": len(snaps),
            "n_snapshots_after_spacing": len(snaps),
            "n_skipped_snapshot_spacing": 0,
        }
    for s in snaps:
        if parse_snapshot_ts(str(s.get("timestamp") or "")) is None:
            return list(snaps), {
                "snapshot_spacing_min": ms,
                "n_snapshots_before_spacing": len(snaps),
                "n_snapshots_after_spacing": len(snaps),
                "n_skipped_snapshot_spacing": 0,
                "snapshot_spacing_note": "unparseable_timestamp_no_spacing",
            }

    def sort_key(ent: tuple[int, dict]) -> tuple[datetime, int]:
        i, s = ent
        t = parse_snapshot_ts(str(s["timestamp"]))
        assert t is not None
        return (t, i)

    indexed = list(enumerate(snaps))
    indexed.sort(key=lambda ii: sort_key((ii[0], ii[1])))
    ordered = [snaps[i] for i, _ in indexed]
    dt = timedelta(minutes=ms)
    out: list[dict] = [ordered[0]]
    skipped = 0
    last_t = parse_snapshot_ts(str(ordered[0]["timestamp"]))
    assert last_t is not None
    for s in ordered[1:]:
        t = parse_snapshot_ts(str(s["timestamp"]))
        assert t is not None
        if t - last_t >= dt:
            out.append(s)
            last_t = t
        else:
            skipped += 1
    return out, {
        "snapshot_spacing_min": ms,
        "n_snapshots_before_spacing": len(snaps),
        "n_snapshots_after_spacing": len(out),
        "n_skipped_snapshot_spacing": skipped,
    }


def _ln_transition_points_from_snaps(snaps: list[dict]) -> list[dict]:
    points: list[dict] = []
    for i in range(1, len(snaps)):
        prev = snaps[i - 1]
        cur = snaps[i]
        x = math.log(float(cur["F"])) - math.log(float(prev["F"]))
        y = math.log(float(cur["sigma_star_used"])) - math.log(float(prev["sigma_star_used"]))
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        points.append(
            {
                "from_t": prev["timestamp"],
                "to_t": cur["timestamp"],
                "dlnF": x,
                "dlnSigma": y,
                "sigma_prev": float(prev["sigma_star_used"]),
                "sigma_cur": float(cur["sigma_star_used"]),
                "skew_cur": cur.get("s_atf_norm"),
                "sigma_method_prev": prev.get("sigma_method"),
                "sigma_method_cur": cur.get("sigma_method"),
            }
        )
    return points


def _trim_transitions_by_abs_move_quantiles(
    points: list[dict],
    *,
    trim_frac: float,
) -> tuple[list[dict], dict[str, float | int | list[float] | None]]:
    """
    Symmetric tail trim on |ΔlnF| and |Δlnσ| **independently**: keep a transition only if
    both absolute moves lie between the empirical ``trim_frac`` and ``1 - trim_frac``
    quantiles (same ``k`` = floor(n * trim_frac) dropped from each tail on each margin).

    ``trim_frac <= 0`` disables trimming (returns a copy of ``points``).
    """
    base: dict[str, float | int | list[float] | None] = {
        "move_trim_frac": float(trim_frac),
        "n_transitions_before_move_trim": len(points),
        "move_trim_k_per_tail": None,
        "abs_dlnF_quantile_band": None,
        "abs_dln_sigma_quantile_band": None,
        "n_trimmed_abs_move": 0,
        "n_transitions_after_move_trim": len(points),
    }
    if trim_frac <= 0.0 or not points:
        return list(points), base
    n = len(points)
    k = int(math.floor(float(n) * float(trim_frac) + 1e-15))
    if k <= 0:
        return list(points), base
    if 2 * k >= n:
        k = max(0, (n - 1) // 2)
        if k <= 0:
            return list(points), {**base, "move_trim_k_per_tail": 0}
    af = [abs(float(p["dlnF"])) for p in points]
    asig = [abs(float(p["dlnSigma"])) for p in points]
    sf = sorted(af)
    ss = sorted(asig)
    lo_f, hi_f = sf[k], sf[n - 1 - k]
    lo_s, hi_s = ss[k], ss[n - 1 - k]
    out: list[dict] = []
    n_drop = 0
    for p in points:
        aif = abs(float(p["dlnF"]))
        ais = abs(float(p["dlnSigma"]))
        if lo_f <= aif <= hi_f and lo_s <= ais <= hi_s:
            out.append(p)
        else:
            n_drop += 1
    return out, {
        "move_trim_frac": float(trim_frac),
        "n_transitions_before_move_trim": n,
        "move_trim_k_per_tail": k,
        "abs_dlnF_quantile_band": [float(lo_f), float(hi_f)],
        "abs_dln_sigma_quantile_band": [float(lo_s), float(hi_s)],
        "n_trimmed_abs_move": n_drop,
        "n_transitions_after_move_trim": len(out),
    }


def fit_ssr_full_day_from_points(points: list[dict]) -> tuple[dict[str, float | None], dict[str, float | int | str | None]]:
    """
    One SSR per estimator from all transitions in ``points`` (origin-constrained),
    same scaling as rolling: SSR = beta / mean(skew_cur) over points with finite skew.
    """
    xs: list[float] = []
    ys: list[float] = []
    sks: list[float] = []
    for p in points:
        sk = p.get("skew_cur")
        if sk is None or not math.isfinite(float(sk)):
            continue
        xs.append(float(p["dlnF"]))
        ys.append(float(p["dlnSigma"]))
        sks.append(float(sk))
    n = len(xs)
    diag: dict[str, float | int | str | None] = {"n_points_used": n}
    empty_ssr = {"ssr_ols": None, "ssr_wls": None, "ssr_huber": None, "ssr_lad": None, "ssr_ts": None}
    if n < 2:
        diag["error"] = "need_at_least_2_transitions_with_skew"
        return empty_ssr, diag
    mean_skew = float(sum(sks) / len(sks))
    diag["mean_skew"] = mean_skew
    if not math.isfinite(mean_skew) or abs(mean_skew) <= 1e-12:
        diag["error"] = "invalid_mean_skew"
        return empty_ssr, diag
    ws = wls_weights_inverse_bivariate_move_sq(xs, ys)
    ols = wls_beta_origin(xs, ys, None)
    wls = wls_beta_origin(xs, ys, ws)
    hub = robust_beta_origin_irls(xs, ys, base_weights=ws, method="huber")
    lad = robust_beta_origin_irls(xs, ys, base_weights=ws, method="lad")
    beta_ts = theil_sen_beta_pairwise(xs, ys)
    diag["beta_ols"] = float(ols[0]) if ols is not None else None
    diag["beta_wls"] = float(wls[0]) if wls is not None else None
    diag["beta_huber"] = float(hub["beta"]) if hub is not None else None
    diag["beta_lad"] = float(lad["beta"]) if lad is not None else None
    diag["beta_ts"] = float(beta_ts) if beta_ts is not None else None
    out: dict[str, float | None] = {
        "ssr_ols": float(ols[0]) / mean_skew if ols is not None else None,
        "ssr_wls": float(wls[0]) / mean_skew if wls is not None else None,
        "ssr_huber": float(hub["beta"]) / mean_skew if hub is not None else None,
        "ssr_lad": float(lad["beta"]) / mean_skew if lad is not None else None,
        "ssr_ts": float(beta_ts) / mean_skew if beta_ts is not None else None,
    }
    return out, diag


def _guess_ticker_from_batch_path(batch_dir: Path) -> str | None:
    """Best-effort: parent folder name like ``.../AAPL/<batch>``."""
    p = batch_dir.parent.name.strip()
    if p and p.isupper() and 1 <= len(p) <= 5 and p.isalpha():
        return p
    return None


def infer_num_expiries_from_batch(batch_dir: Path, *, day: str) -> int:
    """Number of expiries ``m`` from CVI_dims.csv (same as hedge/ln-scatter)."""
    summary = read_summary(batch_dir, day=day)
    if not summary:
        raise SystemExit(f"No snapshots for date {day}; cannot infer expiry count.")
    dims_p = batch_dir / summary[0]["subfolder"] / "CVI_dims.csv"
    if not dims_p.is_file():
        raise SystemExit(f"Missing CVI_dims.csv in first snapshot for {day}.")
    dims = _read_single_row_csv(dims_p)
    m = int(float(dims.get("m", "nan")))
    if m < 1:
        raise SystemExit(f"Invalid m={m} in CVI_dims for {day}.")
    return m


def compute_full_day_ssr_record(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    sigma_mode: str,
    move_trim_frac: float = 0.0,
    snapshot_spacing_min: float = 0.0,
) -> dict[str, object]:
    """
    Full-day Δln SSR (OLS / WLS / Huber / LAD / Theil–Sen) for one calendar day and expiry.
    Returns a JSON-friendly dict (no raise on failure — ``ok`` flag + ``error``).
    """
    try:
        snaps, n_missing, n_nonpositive, n_missing_skew = _build_ln_scatter_snaps(
            batch_dir, day=day, expiry_index=expiry_index, sigma_mode=sigma_mode
        )
        snaps, spacing_diag = _subsample_snaps_by_min_time_spacing(
            snaps, min_spacing_min=float(snapshot_spacing_min)
        )
        if len(snaps) < 2:
            raise SystemExit(
                f"After {snapshot_spacing_min}-minute snapshot spacing, fewer than 2 snapshots remain "
                f"for {day} expiry {expiry_index}."
            )
        points = _ln_transition_points_from_snaps(snaps)
        trimmed, move_diag = _trim_transitions_by_abs_move_quantiles(
            points, trim_frac=float(move_trim_frac)
        )
        ssrs, fit_diag = fit_ssr_full_day_from_points(trimmed)
        merged_diag: dict[str, float | int | str | list[float] | None] = {
            **spacing_diag,
            **move_diag,
            **fit_diag,
        }
        ok_fit = any(v is not None for v in ssrs.values())
        lo_f = hi_f = lo_s = hi_s = None
        if move_diag.get("abs_dlnF_quantile_band") is not None:
            bf = move_diag["abs_dlnF_quantile_band"]
            if isinstance(bf, list) and len(bf) == 2:
                lo_f, hi_f = float(bf[0]), float(bf[1])
        if move_diag.get("abs_dln_sigma_quantile_band") is not None:
            bs = move_diag["abs_dln_sigma_quantile_band"]
            if isinstance(bs, list) and len(bs) == 2:
                lo_s, hi_s = float(bs[0]), float(bs[1])
        scatter: list[dict[str, float | str | bool]] = []
        for p in points:
            sk = p.get("skew_cur")
            has_skew = sk is not None and math.isfinite(float(sk))
            passes_move = True
            if float(move_trim_frac) > 0.0 and lo_f is not None and hi_f is not None and lo_s is not None and hi_s is not None:
                aif = abs(float(p["dlnF"]))
                ais = abs(float(p["dlnSigma"]))
                passes_move = bool(lo_f <= aif <= hi_f and lo_s <= ais <= hi_s)
            used_in_fit = bool(has_skew and passes_move)
            scatter.append(
                {
                    "dlnF": float(p["dlnF"]),
                    "dlnSigma": float(p["dlnSigma"]),
                    "from_t": str(p["from_t"]),
                    "to_t": str(p["to_t"]),
                    "has_skew": bool(has_skew),
                    "passes_move_trim": bool(passes_move),
                    "used_in_ssr_fit": used_in_fit,
                    "in_skew_regression": used_in_fit,
                }
            )
        return {
            "day": day,
            "expiry_index": expiry_index,
            "ok": ok_fit,
            "n_snapshots": len(snaps),
            "n_transition_points": len(points),
            "snap_build": {
                "n_missing_or_invalid": n_missing,
                "n_nonpositive": n_nonpositive,
                "n_missing_skew_estimate": n_missing_skew,
                "snapshot_spacing_min": spacing_diag.get("snapshot_spacing_min"),
                "n_snapshots_before_spacing": spacing_diag.get("n_snapshots_before_spacing"),
                "n_skipped_snapshot_spacing": spacing_diag.get("n_skipped_snapshot_spacing"),
                "snapshot_spacing_note": spacing_diag.get("snapshot_spacing_note"),
            },
            "scatter": scatter,
            "ssr": {k: ssrs[k] for k in sorted(ssrs.keys())},
            "fit_diagnostics": merged_diag,
            "error": merged_diag.get("error") if not ok_fit else None,
        }
    except SystemExit as e:
        return {
            "day": day,
            "expiry_index": expiry_index,
            "ok": False,
            "scatter": [],
            "error": str(e),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "day": day,
            "expiry_index": expiry_index,
            "ok": False,
            "scatter": [],
            "error": repr(e),
        }


def build_full_day_ssr_json_document(
    batch_dir: Path,
    *,
    days: list[str],
    expiry_indices: list[int],
    sigma_mode: str,
    move_trim_frac: float = 0.0,
    snapshot_spacing_min: float = 0.0,
) -> dict[str, object]:
    """Structured document: days × expiries → full-day SSR table."""
    batch_dir = batch_dir.resolve()
    out_days: dict[str, object] = {}
    for d in days:
        by_e: dict[str, object] = {}
        for e in expiry_indices:
            by_e[str(e)] = compute_full_day_ssr_record(
                batch_dir,
                day=d,
                expiry_index=e,
                sigma_mode=sigma_mode,
                move_trim_frac=move_trim_frac,
                snapshot_spacing_min=snapshot_spacing_min,
            )
        out_days[d] = {"expiries": by_e}
    if float(move_trim_frac) > 0.0:
        defn_tail = (
            "lines use full-day betas from transitions with finite skew after symmetric move trimming: "
            f"drop the lowest/highest {100.0 * float(move_trim_frac):.3g}% of |ΔlnF| and the same for |Δlnσ*| "
            "(independently), then intersect. "
        )
    else:
        defn_tail = "lines use full-day betas from transitions with finite skew (no move trim). "
    if float(snapshot_spacing_min) > 1e-9:
        defn_tail += (
            f"Snapshots are subsampled to at least {float(snapshot_spacing_min):g}-minute spacing before forming Δ transitions "
            "(first tick of each day kept, then next kept when Δt ≥ spacing). "
        )
    else:
        defn_tail += "Δ transitions use consecutive OK batch snapshots (~1-minute stride when the batch is 1-minute). "
    return {
        "batch_dir": str(batch_dir),
        "batch_folder_name": batch_dir.name,
        "ticker_guess": _guess_ticker_from_batch_path(batch_dir),
        "sigma_mode": sigma_mode,
        "move_trim_frac": float(move_trim_frac),
        "snapshot_spacing_min": float(snapshot_spacing_min),
        "sigma_star_note": "sigma* = mean of fitted vols at the 3 strikes with z closest to 0 (avg3); not a single z=0 strike.",
        "regression_methods": ["ols", "wls", "huber", "lad", "theil_sen"],
        "definition": "Origin-constrained Y~beta*X on Delta ln(sigma*), X=Delta ln(F); SSR_m = beta_m / mean(skew_cur) "
        "over transitions with finite basis ATF skew (vol-space norm). Scatter plots: Δln between snapshots after optional "
        "time spacing filter; "
        + defn_tail,
        "days": out_days,
    }


def render_full_day_ssr_html(doc: dict[str, object], title: str) -> str:
    """SSR vs day (per expiry) plus ΔlnF–Δlnσ* scatters with full-day origin fits (ln-scatter style)."""
    data_json = json.dumps(doc, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    h1 {{ font-size: 1.15rem; margin-bottom: 6px; }}
    p.meta {{ color: #444; font-size: 0.9rem; max-width: 960px; line-height: 1.45; }}
    .plot {{ width: 100%; height: 380px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 20px; }}
    .plot-scatter {{ width: 100%; height: 480px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 24px; }}
    h2 {{ font-size: 1.05rem; margin: 20px 0 8px; color: #222; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
    h3 {{ font-size: 0.95rem; margin: 14px 0 6px; color: #444; }}
    h4 {{ font-size: 0.88rem; margin: 10px 0 4px; color: #555; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta" id="meta"></p>
  <h2>Full-day SSR vs calendar date</h2>
  <div id="plots"></div>
  <h2>Δln(F) vs Δln(σ*) — full-day transitions (mean of 3 vols nearest z=0)</h2>
  <p class="meta" id="scatterMeta"></p>
  <div id="scatters"></div>
  <script>
    const D = {data_json};
    const methodColors = {{
      ssr_ols: "#2ca02c",
      ssr_wls: "#d62728",
      ssr_huber: "#ff7f0e",
      ssr_lad: "#8c564b",
      ssr_ts: "#17becf",
    }};
    const methodLabels = {{
      ssr_ols: "OLS",
      ssr_wls: "WLS",
      ssr_huber: "Huber",
      ssr_lad: "LAD",
      ssr_ts: "Theil–Sen",
    }};
    const methods = ["ssr_ols", "ssr_wls", "ssr_huber", "ssr_lad", "ssr_ts"];
    const betaSpec = [
      {{ key: "beta_ols", color: methodColors.ssr_ols, label: "OLS" }},
      {{ key: "beta_wls", color: methodColors.ssr_wls, label: "WLS" }},
      {{ key: "beta_huber", color: methodColors.ssr_huber, label: "Huber" }},
      {{ key: "beta_lad", color: methodColors.ssr_lad, label: "LAD" }},
      {{ key: "beta_ts", color: methodColors.ssr_ts, label: "Theil–Sen" }},
    ];
    const dayKeys = Object.keys(D.days || {{}}).sort();
    let expiryKeys = [];
    if (dayKeys.length > 0) {{
      const ex0 = D.days[dayKeys[0]].expiries || {{}};
      expiryKeys = Object.keys(ex0).sort((a, b) => +a - +b);
    }}
    document.getElementById("meta").textContent =
      (D.ticker_guess ? "Ticker (path): " + D.ticker_guess + ". " : "") +
      "Batch: " + (D.batch_folder_name || "") +
      ". σ mode: " + (D.sigma_mode || "") +
      ". " + (D.sigma_star_note || "") +
      " Days: " + dayKeys.join(", ") +
      ". Expiries: " + expiryKeys.length + " indices (" + expiryKeys.slice(0, 6).join(", ") + (expiryKeys.length > 6 ? ", …" : "") + ")." +
      ((D.move_trim_frac != null && +D.move_trim_frac > 0)
        ? " Move trim: central " + (100 - 200 * +D.move_trim_frac).toFixed(2) + "% of |ΔlnF| and |Δlnσ*| (each margin)."
        : "") +
      ((D.snapshot_spacing_min != null && +D.snapshot_spacing_min > 0)
        ? " Snapshot spacing: ≥ " + D.snapshot_spacing_min + " min between kept ticks before Δ transitions."
        : "");

    function _hasSkew(p) {{
      if (p.has_skew != null) return !!p.has_skew;
      return !!p.in_skew_regression;
    }}
    function _usedInSsrFit(p) {{
      if (p.used_in_ssr_fit != null) return !!p.used_in_ssr_fit;
      return !!p.in_skew_regression;
    }}
    document.getElementById("scatterMeta").textContent =
      "Blue: transitions used in SSR fit (finite ATF skew"
      + ((D.move_trim_frac != null && +D.move_trim_frac > 0) ? ", after |ΔlnF|/|Δlnσ*| tail trim" : "")
      + "). Orange: finite skew but excluded by move trim. Gray: skew missing. "
      + ((D.snapshot_spacing_min != null && +D.snapshot_spacing_min > 0)
        ? "Δ transitions use ~" + D.snapshot_spacing_min + "-minute (or wider) snapshot spacing. "
        : "")
      + "Lines: y = βx through origin (β from Python; OLS/WLS/Huber/LAD/Theil–Sen).";

    const host = document.getElementById("plots");
    expiryKeys.forEach((ek) => {{
      const h = document.createElement("h3");
      h.textContent = "Expiry index " + ek + " — SSR time series";
      host.appendChild(h);
      const div = document.createElement("div");
      div.className = "plot";
      div.id = "plot_e_" + ek;
      host.appendChild(div);

      const traces = methods.map((m) => ({{
        type: "scatter",
        mode: "lines+markers",
        name: methodLabels[m] || m,
        x: dayKeys,
        y: dayKeys.map((dk) => {{
          const rec = (D.days[dk].expiries || {{}})[ek];
          if (!rec || !rec.ok || !rec.ssr) return null;
          const v = rec.ssr[m];
          return (v != null && Number.isFinite(v)) ? v : null;
        }}),
        line: {{ width: 2, color: methodColors[m] }},
        marker: {{ size: 8, color: methodColors[m] }},
      }}));

      Plotly.newPlot(div.id, traces, {{
        title: "SSR from full-day regression (β / mean skew)",
        xaxis: {{ title: "Date", tickangle: -35 }},
        yaxis: {{ title: "SSR" }},
        legend: {{ orientation: "h", y: -0.2 }},
        margin: {{ l: 55, r: 25, t: 45, b: 80 }},
        hovermode: "x unified",
      }}, {{ responsive: true }});
    }});

    const scHost = document.getElementById("scatters");
    dayKeys.forEach((dk) => {{
      const hDay = document.createElement("h3");
      hDay.textContent = "Session " + dk;
      scHost.appendChild(hDay);
      expiryKeys.forEach((ek) => {{
        const rec = (D.days[dk].expiries || {{}})[ek];
        const sc = (rec && rec.scatter) ? rec.scatter : [];
        const h4 = document.createElement("h4");
        h4.textContent = "Expiry " + ek + (rec && rec.ok ? "" : " (fit incomplete)");
        scHost.appendChild(h4);
        const div = document.createElement("div");
        div.className = "plot-scatter";
        div.id = "sc_" + dk + "_e" + ek;
        scHost.appendChild(div);

        const inFit = sc.filter(_usedInSsrFit);
        const moveTrimmed = sc.filter((p) => _hasSkew(p) && !_usedInSsrFit(p));
        const noSkew = sc.filter((p) => !_hasSkew(p));
        const traces = [];
        if (inFit.length > 0) {{
          traces.push({{
            type: "scatter",
            mode: "markers",
            name: "Δ transitions (SSR fit)",
            x: inFit.map((p) => p.dlnF),
            y: inFit.map((p) => p.dlnSigma),
            text: inFit.map((p) => p.from_t + " → " + p.to_t),
            marker: {{ size: 8, color: "#1f77b4", opacity: 0.85 }},
            hovertemplate: "%{{text}}<br>ΔlnF=%{{x:.6f}}<br>Δlnσ*=%{{y:.6f}}<extra></extra>",
          }});
        }}
        if (moveTrimmed.length > 0) {{
          traces.push({{
            type: "scatter",
            mode: "markers",
            name: "Δ transitions (skew ok, move-trimmed out)",
            x: moveTrimmed.map((p) => p.dlnF),
            y: moveTrimmed.map((p) => p.dlnSigma),
            text: moveTrimmed.map((p) => p.from_t + " → " + p.to_t),
            marker: {{ size: 7, color: "#ff7f0e", opacity: 0.85 }},
            hovertemplate: "%{{text}}<br>ΔlnF=%{{x:.6f}}<br>Δlnσ*=%{{y:.6f}}<extra></extra>",
          }});
        }}
        if (noSkew.length > 0) {{
          traces.push({{
            type: "scatter",
            mode: "markers",
            name: "Δ transitions (skew missing)",
            x: noSkew.map((p) => p.dlnF),
            y: noSkew.map((p) => p.dlnSigma),
            text: noSkew.map((p) => p.from_t + " → " + p.to_t),
            marker: {{ size: 6, color: "#bbbbbb", opacity: 0.75 }},
            hovertemplate: "%{{text}}<br>ΔlnF=%{{x:.6f}}<br>Δlnσ*=%{{y:.6f}}<extra></extra>",
          }});
        }}

        let xmin = 0, xmax = 0, span = 1e-12;
        if (sc.length > 0) {{
          const xs = sc.map((p) => p.dlnF);
          xmin = Math.min(...xs);
          xmax = Math.max(...xs);
          span = (xmax - xmin) * 0.08 + 1e-12;
        }}
        const x0 = xmin - span;
        const x1 = xmax + span;
        const fd = (rec && rec.fit_diagnostics) ? rec.fit_diagnostics : {{}};
        betaSpec.forEach((b) => {{
          const beta = fd[b.key];
          if (beta == null || !Number.isFinite(beta)) return;
          traces.push({{
            type: "scatter",
            mode: "lines",
            name: b.label + ": y=" + beta.toFixed(6) + "·x",
            x: [x0, x1],
            y: [beta * x0, beta * x1],
            line: {{ width: 2.2, color: b.color }},
            hovertemplate: b.label + " β=" + beta + "<extra></extra>",
          }});
        }});

        Plotly.newPlot(div.id, traces, {{
          title: "Δln(F) vs Δln(σ*) — " + dk + " e" + ek,
          xaxis: {{ title: "Δln(F)" }},
          yaxis: {{ title: "Δln(σ*)" }},
          shapes: [
            {{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#aaa" }} }},
            {{ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: {{ dash: "dot", color: "#aaa" }} }},
          ],
          legend: {{ orientation: "h", y: -0.25 }},
          margin: {{ l: 60, r: 25, t: 40, b: 100 }},
          hovermode: "closest",
        }}, {{ responsive: true }});
      }});
    }});
  </script>
</body>
</html>
"""


def build_ln_scatter_payload(
    batch_dir: Path, *, day: str, expiry_index: int, sigma_mode: str = "avg3"
) -> dict:
    snaps, n_missing, n_nonpositive, n_missing_skew = _build_ln_scatter_snaps(
        batch_dir, day=day, expiry_index=expiry_index, sigma_mode=sigma_mode
    )
    points = _ln_transition_points_from_snaps(snaps)
    if not points:
        raise SystemExit("No valid Δln(F), Δln(sigma_star) points after filtering.")

    # Rolling 1-hour regression on transition points (updated each transition).
    # Origin-constrained fits: Y ~ beta*X (no intercept).
    # SSR per window end is computed as beta / avg_skew_window.
    rolling: list[dict[str, float | int | str]] = []
    point_times = [parse_snapshot_ts(str(p["to_t"])) for p in points]
    for i, ti in enumerate(point_times):
        if ti is None:
            continue
        t0 = ti - timedelta(hours=1)
        xs: list[float] = []
        ys: list[float] = []
        sks: list[float] = []
        for j, tj in enumerate(point_times):
            if tj is None:
                continue
            if tj <= ti and tj > t0:
                xs.append(float(points[j]["dlnF"]))
                ys.append(float(points[j]["dlnSigma"]))
                s_j = snaps[j + 1].get("s_atf_norm")
                if s_j is not None and math.isfinite(float(s_j)):
                    sks.append(float(s_j))
        if len(xs) < 2:
            continue
        ols = wls_beta_origin(xs, ys, None)
        ws = wls_weights_inverse_bivariate_move_sq(xs, ys)
        wls = wls_beta_origin(xs, ys, ws)
        hub = robust_beta_origin_irls(xs, ys, base_weights=ws, method="huber")
        lad = robust_beta_origin_irls(xs, ys, base_weights=ws, method="lad")
        beta_ts = theil_sen_beta_pairwise(xs, ys)
        if ols is None and wls is None and hub is None and lad is None and beta_ts is None:
            continue
        avg_skew = None
        if sks:
            avg_skew = float(sum(sks) / len(sks))
        row: dict[str, float | int | str] = {
            "timestamp": str(points[i]["to_t"]),
            "n_window": len(xs),
        }
        if avg_skew is not None:
            row["avg_skew"] = avg_skew
        if ols is not None:
            row["beta_ols"] = float(ols[0])
            if avg_skew is not None and abs(avg_skew) > 1e-12:
                row["ssr_ols"] = float(ols[0]) / avg_skew
        if wls is not None:
            row["beta_wls"] = float(wls[0])
            if avg_skew is not None and abs(avg_skew) > 1e-12:
                row["ssr_wls"] = float(wls[0]) / avg_skew
        if hub is not None:
            row["beta_huber"] = float(hub["beta"])
            row["huber_iters"] = int(hub["iters"])
            row["huber_converged"] = bool(hub["converged"])
            if avg_skew is not None and abs(avg_skew) > 1e-12:
                row["ssr_huber"] = float(hub["beta"]) / avg_skew
        if lad is not None:
            row["beta_lad"] = float(lad["beta"])
            row["lad_iters"] = int(lad["iters"])
            row["lad_converged"] = bool(lad["converged"])
            if avg_skew is not None and abs(avg_skew) > 1e-12:
                row["ssr_lad"] = float(lad["beta"]) / avg_skew
        if beta_ts is not None:
            row["beta_ts"] = float(beta_ts)
            if avg_skew is not None and abs(avg_skew) > 1e-12:
                row["ssr_ts"] = float(beta_ts) / avg_skew
        rolling.append(row)

    # One-step-ahead predicted sigma series from rolling SSR.
    pred_series: list[dict[str, float | str | None]] = []
    sse_naive = 0.0
    sse_ols = 0.0
    sse_wls = 0.0
    sse_huber = 0.0
    sse_lad = 0.0
    sse_ts = 0.0
    n_err_ols = 0
    n_err_wls = 0
    n_err_huber = 0
    n_err_lad = 0
    n_err_ts = 0
    for i, ti in enumerate(point_times):
        if ti is None:
            continue
        t0 = ti - timedelta(hours=1)
        xs_win: list[float] = []
        ys_win: list[float] = []
        sks_win: list[float] = []
        # Use only prior transitions to predict current point i.
        for j, tj in enumerate(point_times):
            if tj is None:
                continue
            if tj < ti and tj > t0:
                xs_win.append(float(points[j]["dlnF"]))
                ys_win.append(float(points[j]["dlnSigma"]))
                s_j = points[j].get("skew_cur")
                if s_j is not None and math.isfinite(float(s_j)):
                    sks_win.append(float(s_j))
        if len(xs_win) < 2 or not sks_win:
            continue
        ols_p = wls_beta_origin(xs_win, ys_win, None)
        wts_p = wls_weights_inverse_bivariate_move_sq(xs_win, ys_win)
        wls_p = wls_beta_origin(xs_win, ys_win, wts_p)
        hub_p = robust_beta_origin_irls(xs_win, ys_win, base_weights=wts_p, method="huber")
        lad_p = robust_beta_origin_irls(xs_win, ys_win, base_weights=wts_p, method="lad")
        beta_ts_p = theil_sen_beta_pairwise(xs_win, ys_win)
        avg_skew_win = float(sum(sks_win) / len(sks_win))
        if not math.isfinite(avg_skew_win) or abs(avg_skew_win) <= 1e-12:
            continue
        dln_f_cur = float(points[i]["dlnF"])
        s_cur = points[i].get("skew_cur")
        s_cur_f = float(s_cur) if s_cur is not None and math.isfinite(float(s_cur)) else avg_skew_win
        sigma_prev = float(points[i]["sigma_prev"])
        sigma_actual = float(points[i]["sigma_cur"])
        rowp: dict[str, float | str | None] = {
            "timestamp": str(points[i]["to_t"]),
            "sigma_actual": sigma_actual,
            "sigma_prev": sigma_prev,
            "sigma_pred_naive": sigma_prev,
            "dlnF": dln_f_cur,
            "dlnSigma_actual": float(points[i]["dlnSigma"]),
            "dlnSigma_pred_naive": 0.0,
            "avg_skew_window": avg_skew_win,
            "skew_cur": s_cur_f,
            "n_window": len(xs_win),
        }
        e_naive = sigma_actual - sigma_prev
        sse_naive += e_naive * e_naive
        if ols_p is not None:
            beta_ols = float(ols_p[0])
            ssr_ols = beta_ols / avg_skew_win
            dln_pred_ols = ssr_ols * s_cur_f * dln_f_cur
            sigma_pred_ols = _sigma_prev_times_exp_dln(sigma_prev, dln_pred_ols)
            rowp["ssr_ols"] = ssr_ols
            rowp["dlnSigma_pred_ols"] = dln_pred_ols
            rowp["sigma_pred_ols"] = sigma_pred_ols
            e_ols = sigma_actual - sigma_pred_ols
            sse_ols += e_ols * e_ols
            n_err_ols += 1
        if wls_p is not None:
            beta_wls = float(wls_p[0])
            ssr_wls = beta_wls / avg_skew_win
            dln_pred_wls = ssr_wls * s_cur_f * dln_f_cur
            sigma_pred_wls = _sigma_prev_times_exp_dln(sigma_prev, dln_pred_wls)
            rowp["ssr_wls"] = ssr_wls
            rowp["dlnSigma_pred_wls"] = dln_pred_wls
            rowp["sigma_pred_wls"] = sigma_pred_wls
            e_wls = sigma_actual - sigma_pred_wls
            sse_wls += e_wls * e_wls
            n_err_wls += 1
        if hub_p is not None:
            beta_huber = float(hub_p["beta"])
            ssr_huber = beta_huber / avg_skew_win
            dln_pred_huber = ssr_huber * s_cur_f * dln_f_cur
            sigma_pred_huber = _sigma_prev_times_exp_dln(sigma_prev, dln_pred_huber)
            rowp["ssr_huber"] = ssr_huber
            rowp["dlnSigma_pred_huber"] = dln_pred_huber
            rowp["sigma_pred_huber"] = sigma_pred_huber
            rowp["huber_iters"] = int(hub_p["iters"])
            rowp["huber_converged"] = bool(hub_p["converged"])
            e_huber = sigma_actual - sigma_pred_huber
            sse_huber += e_huber * e_huber
            n_err_huber += 1
        if lad_p is not None:
            beta_lad = float(lad_p["beta"])
            ssr_lad = beta_lad / avg_skew_win
            dln_pred_lad = ssr_lad * s_cur_f * dln_f_cur
            sigma_pred_lad = _sigma_prev_times_exp_dln(sigma_prev, dln_pred_lad)
            rowp["ssr_lad"] = ssr_lad
            rowp["dlnSigma_pred_lad"] = dln_pred_lad
            rowp["sigma_pred_lad"] = sigma_pred_lad
            rowp["lad_iters"] = int(lad_p["iters"])
            rowp["lad_converged"] = bool(lad_p["converged"])
            e_lad = sigma_actual - sigma_pred_lad
            sse_lad += e_lad * e_lad
            n_err_lad += 1
        if beta_ts_p is not None:
            beta_ts_f = float(beta_ts_p)
            ssr_ts = beta_ts_f / avg_skew_win
            dln_pred_ts = ssr_ts * s_cur_f * dln_f_cur
            sigma_pred_ts = _sigma_prev_times_exp_dln(sigma_prev, dln_pred_ts)
            rowp["ssr_ts"] = ssr_ts
            rowp["dlnSigma_pred_ts"] = dln_pred_ts
            rowp["sigma_pred_ts"] = sigma_pred_ts
            e_ts = sigma_actual - sigma_pred_ts
            sse_ts += e_ts * e_ts
            n_err_ts += 1
        pred_series.append(rowp)

    skew_series = [
        {"timestamp": s["timestamp"], "s_atf_norm": s["s_atf_norm"]}
        for s in snaps
        if s.get("s_atf_norm") is not None and math.isfinite(float(s["s_atf_norm"]))
    ]

    diagnostics = {
        "date": day,
        "expiry_index": expiry_index,
        "sigma_mode": sigma_mode,
        "n_snapshots": len(snaps),
        "n_transition_points_used": len(points),
        "n_skew_points": len(skew_series),
        "n_rolling_points": len(rolling),
        "n_pred_points": len(pred_series),
        "n_missing_or_invalid": n_missing,
        "n_nonpositive": n_nonpositive,
        "n_missing_skew_estimate": n_missing_skew,
    }
    diagnostics["sse_naive"] = sse_naive
    diagnostics["sse_ols"] = sse_ols if n_err_ols > 0 else None
    diagnostics["sse_wls"] = sse_wls if n_err_wls > 0 else None
    diagnostics["sse_huber"] = sse_huber if n_err_huber > 0 else None
    diagnostics["sse_lad"] = sse_lad if n_err_lad > 0 else None
    diagnostics["sse_ts"] = sse_ts if n_err_ts > 0 else None
    diagnostics["r2_vs_naive_ols"] = (
        (1.0 - sse_ols / sse_naive) if n_err_ols > 0 and sse_naive > 1e-30 else None
    )
    diagnostics["r2_vs_naive_wls"] = (
        (1.0 - sse_wls / sse_naive) if n_err_wls > 0 and sse_naive > 1e-30 else None
    )
    diagnostics["r2_vs_naive_huber"] = (
        (1.0 - sse_huber / sse_naive) if n_err_huber > 0 and sse_naive > 1e-30 else None
    )
    diagnostics["r2_vs_naive_lad"] = (
        (1.0 - sse_lad / sse_naive) if n_err_lad > 0 and sse_naive > 1e-30 else None
    )
    diagnostics["r2_vs_naive_ts"] = (
        (1.0 - sse_ts / sse_naive) if n_err_ts > 0 and sse_naive > 1e-30 else None
    )
    paired_ts_wls: list[tuple[float, float]] = []
    for r in rolling:
        b_ts = _to_float_or_none(r.get("beta_ts"))
        b_w = _to_float_or_none(r.get("beta_wls"))
        if b_ts is not None and b_w is not None:
            paired_ts_wls.append((float(b_ts), float(b_w)))
    if paired_ts_wls:
        diagnostics["median_abs_beta_ts_minus_beta_wls"] = float(
            statistics.median(abs(a - b) for a, b in paired_ts_wls)
        )
        diagnostics["frac_abs_beta_ts_gt_abs_wls"] = float(
            sum(1 for a, b in paired_ts_wls if abs(a) > abs(b)) / len(paired_ts_wls)
        )
    else:
        diagnostics["median_abs_beta_ts_minus_beta_wls"] = None
        diagnostics["frac_abs_beta_ts_gt_abs_wls"] = None
    return {
        "points": points,
        "rolling": rolling,
        "pred_series": pred_series,
        "skew_series": skew_series,
        "snapshots": snaps,
        "diagnostics": diagnostics,
    }


def write_ln_scatter_details_csv(payload: dict, out_csv: Path) -> None:
    """Write one row per transition with all source fields used to compute point."""
    snaps = payload.get("snapshots", [])
    if not isinstance(snaps, list) or len(snaps) < 2:
        raise SystemExit("Payload has insufficient snapshots for CSV export.")
    rows: list[dict[str, object]] = []
    for i in range(1, len(snaps)):
        prev = snaps[i - 1]
        cur = snaps[i]
        try:
            dln_f = float(cur["F"])
            dln_f = math.log(float(cur["F"])) - math.log(float(prev["F"]))
            dln_sigma = math.log(float(cur["sigma_star_used"])) - math.log(float(prev["sigma_star_used"]))
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "from_timestamp": prev.get("timestamp"),
                "to_timestamp": cur.get("timestamp"),
                "expiry_index_used": cur.get("source_expiry_idx"),
                "F_sigma_source_file": cur.get("source_file"),
                "F_sigma_source_note": cur.get("source_strike_for_F_sigma"),
                "F_prev": prev.get("F"),
                "F_cur": cur.get("F"),
                "sigma_star_prev": prev.get("sigma_star_used"),
                "sigma_star_cur": cur.get("sigma_star_used"),
                "sigma_star_row_prev": prev.get("sigma_star_row"),
                "sigma_star_row_cur": cur.get("sigma_star_row"),
                "sigma_method_prev": prev.get("sigma_method"),
                "sigma_method_cur": cur.get("sigma_method"),
                "dlnF": dln_f,
                "dlnSigma": dln_sigma,
                "s_atf_prev": prev.get("s_atf_norm"),
                "s_atf_cur": cur.get("s_atf_norm"),
                "skew_method_prev": prev.get("method"),
                "skew_method_cur": cur.get("method"),
                "sigma_avg3_strike_1_prev": prev.get("sigma_avg3_strike_1"),
                "sigma_avg3_z_1_prev": prev.get("sigma_avg3_z_1"),
                "sigma_avg3_vol_1_prev": prev.get("sigma_avg3_vol_1"),
                "sigma_avg3_strike_2_prev": prev.get("sigma_avg3_strike_2"),
                "sigma_avg3_z_2_prev": prev.get("sigma_avg3_z_2"),
                "sigma_avg3_vol_2_prev": prev.get("sigma_avg3_vol_2"),
                "sigma_avg3_strike_3_prev": prev.get("sigma_avg3_strike_3"),
                "sigma_avg3_z_3_prev": prev.get("sigma_avg3_z_3"),
                "sigma_avg3_vol_3_prev": prev.get("sigma_avg3_vol_3"),
                "sigma_avg3_strike_1_cur": cur.get("sigma_avg3_strike_1"),
                "sigma_avg3_z_1_cur": cur.get("sigma_avg3_z_1"),
                "sigma_avg3_vol_1_cur": cur.get("sigma_avg3_vol_1"),
                "sigma_avg3_strike_2_cur": cur.get("sigma_avg3_strike_2"),
                "sigma_avg3_z_2_cur": cur.get("sigma_avg3_z_2"),
                "sigma_avg3_vol_2_cur": cur.get("sigma_avg3_vol_2"),
                "sigma_avg3_strike_3_cur": cur.get("sigma_avg3_strike_3"),
                "sigma_avg3_z_3_cur": cur.get("sigma_avg3_z_3"),
                "sigma_avg3_vol_3_cur": cur.get("sigma_avg3_vol_3"),
                "sigma_z0_strike_prev": prev.get("sigma_z0_strike"),
                "sigma_z0_z_prev": prev.get("sigma_z0_z"),
                "sigma_z0_vol_prev": prev.get("sigma_z0_vol"),
                "sigma_z0_strike_cur": cur.get("sigma_z0_strike"),
                "sigma_z0_z_cur": cur.get("sigma_z0_z"),
                "sigma_z0_vol_cur": cur.get("sigma_z0_vol"),
                "left_strike_prev": prev.get("left_strike"),
                "left_z_prev": prev.get("left_z"),
                "left_vol_prev": prev.get("left_vol"),
                "atf_strike_prev": prev.get("atf_strike"),
                "atf_z_prev": prev.get("atf_z"),
                "atf_vol_prev": prev.get("atf_vol"),
                "right_strike_prev": prev.get("right_strike"),
                "right_z_prev": prev.get("right_z"),
                "right_vol_prev": prev.get("right_vol"),
                "left_strike_cur": cur.get("left_strike"),
                "left_z_cur": cur.get("left_z"),
                "left_vol_cur": cur.get("left_vol"),
                "atf_strike_cur": cur.get("atf_strike"),
                "atf_z_cur": cur.get("atf_z"),
                "atf_vol_cur": cur.get("atf_vol"),
                "right_strike_cur": cur.get("right_strike"),
                "right_z_cur": cur.get("right_z"),
                "right_vol_cur": cur.get("right_vol"),
            }
        )
    if not rows:
        raise SystemExit("No transition rows available for CSV export.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


