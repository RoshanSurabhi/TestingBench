"""Snapshot state, forward indexing, decoupling diagnostic."""
from __future__ import annotations

import argparse
import csv
import functools
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .fundamentals import *
from .fundamentals import (
    _basis_dot,
    _basis_eval_skew_row,
    _first_positive_from_row,
    _lr_call_greeks_from_fwd,
    _lr_call_price_from_fwd,
    _lr_call_theta_from_fwd,
    _market_price_from_row,
    _read_single_row_csv,
    _read_vector_csv_values,
    _to_float_or_none,
)

def _interp_on_pairs(pairs: list[tuple[float, float]], xq: float) -> float | None:
    if not pairs:
        return None
    pairs_s = sorted(pairs, key=lambda x: x[0])
    xs = [p[0] for p in pairs_s]
    ys = [p[1] for p in pairs_s]
    return interp_linear(xs, ys, xq)


def _market_price_at_strike(
    rows: list[dict[str, float]],
    strike: float,
    fwd: float,
    texp: float,
) -> tuple[float | None, str]:
    for col in MARKET_PRICE_CANDIDATE_COLUMNS:
        pairs = [(float(r["strike"]), float(r[col])) for r in rows if col in r]
        p = _interp_on_pairs(pairs, strike)
        if p is not None:
            return p, col
    for col in MARKET_VOL_CANDIDATE_COLUMNS:
        pairs = [(float(r["strike"]), float(r[col])) for r in rows if col in r and float(r[col]) > 0.0]
        v = _interp_on_pairs(pairs, strike)
        if v is None:
            continue
        p = _lr_call_price_from_fwd(fwd, strike, v, texp, r=0.0, q=0.0)
        if p is not None:
            return p, f"{col}_to_lr_price"
    return None, "unavailable"


def _build_snapshot_state(
    batch_dir: Path,
    row: dict,
    expiry_index: int,
    ssr_by_ts: dict[str, dict[str, float]],
    fixed_ssr: dict[str, float | None] | None = None,
) -> dict | None:
    sub = batch_dir / row["subfolder"]
    efq = sub / "expiry_fwd_q.csv"
    opt = sub / "option_fit_comparison.csv"
    dims_p = sub / "CVI_dims.csv"
    knot_p = sub / "knot_vector.csv"
    xsol_p = sub / "x_solution.csv"
    required = [efq, opt, dims_p, knot_p, xsol_p]
    missing = [p for p in required if not p.is_file()]
    if missing:
        miss = ", ".join(str(p) for p in missing)
        raise SystemExit(f"Missing required CVI artifacts for snapshot {row['subfolder']}: {miss}")
    exp_row = read_expiry_row(efq, expiry_index)
    if exp_row is None:
        return None
    fwd = float(exp_row["F"])
    vol_time = float(exp_row.get("volTime", float("nan")))
    if not (math.isfinite(fwd) and fwd > 0.0 and math.isfinite(vol_time) and vol_time > 0.0):
        return None
    by_exp = read_option_fit_by_expiry(opt)
    rows = by_exp.get(expiry_index) or []
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: float(r["strike"]))
    strikes = [float(r["strike"]) for r in rows]
    zs = [float(r["z"]) for r in rows]
    fitted = [float(r["fitted_vol"]) for r in rows]
    if len(strikes) < 2:
        return None

    dims = _read_single_row_csv(dims_p)
    nb = int(float(dims.get("num_basis", "nan")))
    n_v_orig = int(float(dims.get("n_v_orig", "nan")))
    m = int(float(dims.get("m", "nan")))
    knots = _read_vector_csv_values(knot_p)
    x_values = _read_vector_csv_values(xsol_p)
    if len(knots) < 8:
        raise SystemExit(f"knot_vector.csv too short in snapshot {row['subfolder']}")
    if len(knots) - 4 != nb:
        raise SystemExit(
            f"num_basis mismatch with knot vector in snapshot {row['subfolder']}: num_basis={nb}, len(knots)-4={len(knots)-4}"
        )
    if n_v_orig != m * nb:
        raise SystemExit(f"Invalid CVI_dims n_v_orig in snapshot {row['subfolder']}")
    if len(x_values) < n_v_orig:
        raise SystemExit(
            f"x_solution.csv too short in snapshot {row['subfolder']}: got {len(x_values)}, need >= {n_v_orig}"
        )
    if not (0 <= expiry_index < m):
        raise SystemExit(f"expiry-index {expiry_index} out of range [0,{m-1}] for snapshot {row['subfolder']}")
    base = expiry_index * nb
    alpha = [float(x_values[base + i]) for i in range(nb)]
    v_star = float(exp_row.get("v_star", float("nan")))
    if not (math.isfinite(v_star) and v_star > 0.0):
        return None
    s_row_atf = _basis_eval_skew_row(knots, 0.0, v_star, nb)
    s_atf_basis_raw = _basis_dot(s_row_atf, alpha)  # raw basis: (1/v*) dv/dz
    s_atf_norm = 0.5 * s_atf_basis_raw  # vol-space normalization used by SSR path

    return {
        "timestamp": row["timestamp"],
        "time": parse_snapshot_ts(str(row["timestamp"])),
        "fwd": fwd,
        "vol_time": vol_time,
        "rows": rows,
        "strikes": strikes,
        "zs": zs,
        "fitted_vols": fitted,
        "s_atf_norm": s_atf_norm,
        "basis_num": nb,
        "alpha": alpha,
        "knots": knots,
        "v_star": v_star,
        "r": float(exp_row.get("r", 0.0)),
        "q": float(exp_row.get("q", 0.0)),
        "ssr_ols": (
            fixed_ssr.get("ssr_ols") if fixed_ssr is not None else ssr_by_ts.get(str(row["timestamp"]), {}).get("ssr_ols")
        ),
        "ssr_wls": (
            fixed_ssr.get("ssr_wls") if fixed_ssr is not None else ssr_by_ts.get(str(row["timestamp"]), {}).get("ssr_wls")
        ),
        "ssr_huber": (
            fixed_ssr.get("ssr_huber")
            if fixed_ssr is not None
            else ssr_by_ts.get(str(row["timestamp"]), {}).get("ssr_huber")
        ),
        "ssr_lad": (
            fixed_ssr.get("ssr_lad") if fixed_ssr is not None else ssr_by_ts.get(str(row["timestamp"]), {}).get("ssr_lad")
        ),
        "ssr_ts": (
            fixed_ssr.get("ssr_ts") if fixed_ssr is not None else ssr_by_ts.get(str(row["timestamp"]), {}).get("ssr_ts")
        ),
    }


def _find_forward_idx(times: list[datetime | None], i: int, horizon_min: float, max_slip_min: float = 2.0) -> int | None:
    ti = times[i]
    if ti is None:
        return None
    target = ti + timedelta(minutes=float(horizon_min))
    limit = target + timedelta(minutes=float(max_slip_min))
    for j in range(i + 1, len(times)):
        tj = times[j]
        if tj is None:
            continue
        if tj < target:
            continue
        if tj <= limit:
            return j
        return None
    return None


def _pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 1e-18 or syy <= 1e-18:
        return None
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    out = sxy / math.sqrt(sxx * syy)
    return float(out) if math.isfinite(out) else None


def _build_decoupling_snapshot_state(batch_dir: Path, row: dict, expiry_index: int) -> dict | None:
    sub = batch_dir / row["subfolder"]
    efq = sub / "expiry_fwd_q.csv"
    opt = sub / "option_fit_comparison.csv"
    price_cmp = sub / "price_comparison.csv"
    if not efq.is_file() or not opt.is_file():
        return None
    exp_row = read_expiry_row(efq, expiry_index)
    if exp_row is None:
        return None
    fwd = float(exp_row["F"])
    vol_time = float(exp_row.get("volTime", float("nan")))
    if not (math.isfinite(fwd) and fwd > 0.0 and math.isfinite(vol_time) and vol_time > 0.0):
        return None
    by_exp = read_option_fit_by_expiry(opt)
    rows = by_exp.get(expiry_index) or []
    if not rows:
        return None
    price_by_exp = read_price_comparison_by_expiry(price_cmp)
    price_rows = price_by_exp.get(expiry_index) or {}
    out_rows: dict[str, dict[str, float | str | None]] = {}
    for rr in rows:
        k = float(rr["strike"])
        kkey = fmt_strike_key(k)
        sigma = _to_float_or_none(rr.get("fitted_vol"))
        z = _to_float_or_none(rr.get("z"))
        if sigma is None or sigma <= 0.0:
            continue
        c_mkt, c_src = _market_price_from_row(rr, fwd, vol_time)
        pc_row = price_rows.get(kkey) or {}
        c_bid = _to_float_or_none(pc_row.get("market_call_bid"))
        c_ask = _to_float_or_none(pc_row.get("market_call_ask"))
        c_mid = _to_float_or_none(pc_row.get("market_call_mid"))
        if c_mid is None and c_bid is not None and c_ask is not None:
            c_mid = 0.5 * (c_bid + c_ask)
        bid_vol, bid_vol_src = _first_positive_from_row(rr, BID_VOL_CANDIDATE_COLUMNS)
        ask_vol, ask_vol_src = _first_positive_from_row(rr, ASK_VOL_CANDIDATE_COLUMNS)
        mid_vol, mid_vol_src = _first_positive_from_row(rr, MARKET_VOL_CANDIDATE_COLUMNS)
        c_bid_lr = (
            _lr_call_price_from_fwd(fwd, k, bid_vol, vol_time)
            if bid_vol is not None
            else None
        )
        c_ask_lr = (
            _lr_call_price_from_fwd(fwd, k, ask_vol, vol_time)
            if ask_vol is not None
            else None
        )
        c_mid_lr = (
            _lr_call_price_from_fwd(fwd, k, mid_vol, vol_time)
            if mid_vol is not None
            else None
        )
        # Many CVI batches omit ask_impl_vol (and sometimes bid is zero); use quoted prices when IV→LR is missing.
        bid_quote_src: str | None = None
        if c_bid_lr is None and c_bid is not None:
            c_bid_lr = float(c_bid)
            bid_quote_src = "market_call_bid"
        ask_quote_src: str | None = None
        if c_ask_lr is None and c_ask is not None:
            c_ask_lr = float(c_ask)
            ask_quote_src = "market_call_ask"
        mid_quote_src: str | None = None
        if c_mid_lr is None and c_mid is not None:
            c_mid_lr = float(c_mid)
            mid_quote_src = "market_call_mid"
        c_fit = _to_float_or_none(pc_row.get("fitted_call"))
        c_fit_src = "price_comparison.fitted_call" if c_fit is not None else "fitted_vol_to_lr_price"
        if c_fit is None:
            c_fit = _lr_call_price_from_fwd(
                fwd,
                k,
                sigma,
                vol_time,
                r=float(exp_row.get("r", 0.0)),
                q=float(exp_row.get("q", 0.0)),
            )
        greeks = _lr_call_greeks_from_fwd(
            fwd,
            k,
            sigma,
            vol_time,
            r=float(exp_row.get("r", 0.0)),
            q=float(exp_row.get("q", 0.0)),
        )
        theta = _lr_call_theta_from_fwd(
            fwd,
            k,
            sigma,
            vol_time,
            r=float(exp_row.get("r", 0.0)),
            q=float(exp_row.get("q", 0.0)),
        )
        out_rows[kkey] = {
            "strike": k,
            "z": z,
            "sigma": sigma,
            "c_market": c_mkt,
            "c_market_source": c_src,
            "c_bid": c_bid,
            "c_ask": c_ask,
            "c_mid": c_mid,
            "c_bid_lr": c_bid_lr,
            "c_ask_lr": c_ask_lr,
            "c_mid_lr": c_mid_lr,
            "bid_impl_vol": bid_vol,
            "ask_impl_vol": ask_vol,
            "mid_impl_vol": mid_vol,
            "bid_price_source": (
                bid_quote_src
                if bid_quote_src is not None
                else (f"{bid_vol_src}_to_lr_price" if bid_vol_src is not None else "unavailable")
            ),
            "ask_price_source": (
                ask_quote_src
                if ask_quote_src is not None
                else (f"{ask_vol_src}_to_lr_price" if ask_vol_src is not None else "unavailable")
            ),
            "mid_price_source": (
                mid_quote_src
                if mid_quote_src is not None
                else (f"{mid_vol_src}_to_lr_price" if mid_vol_src is not None else "unavailable")
            ),
            "c_fit": c_fit,
            "c_fit_source": c_fit_src if c_fit is not None else "unavailable",
            "delta_bs": (float(greeks["delta"]) if greeks is not None else None),
            "gamma_bs": (float(greeks["gamma"]) if greeks is not None else None),
            "vega_bs": (float(greeks["vega"]) if greeks is not None else None),
            "theta_bs": theta,
        }
    if not out_rows:
        return None
    return {
        "timestamp": row["timestamp"],
        "time": parse_snapshot_ts(str(row["timestamp"])),
        "date": str(row["timestamp"]).split()[0],
        "fwd": fwd,
        "vol_time": vol_time,
        "r": float(exp_row.get("r", 0.0)),
        "q": float(exp_row.get("q", 0.0)),
        "rows": out_rows,
    }


def _locate_decoupling_engine_exe() -> Path | None:
    env = os.environ.get("CVI_DECOUPLING_CPP_EXE")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_file() else None
    repo_root = Path(__file__).resolve().parents[3]
    # Prefer the standalone decoupling tool now owned by ResearchBench.
    candidates = [
        repo_root.parent / "ResearchBench" / "x64" / "Release" / "cvi_decoupling_engine.exe",
        repo_root.parent / "ResearchBench" / "x64" / "Debug" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "x64" / "Release" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "x64" / "Debug" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "Release" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "Debug" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "Win32" / "Release" / "cvi_decoupling_engine.exe",
        repo_root / "native" / "Win32" / "Debug" / "cvi_decoupling_engine.exe",
        repo_root / "x64" / "Release" / "cvi_decoupling_engine.exe",
        repo_root / "x64" / "Debug" / "cvi_decoupling_engine.exe",
        repo_root / "Release" / "cvi_decoupling_engine.exe",
        repo_root / "Debug" / "cvi_decoupling_engine.exe",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _run_decoupling_engine_subprocess(exe: Path, json_in: str) -> subprocess.CompletedProcess:
    """Run engine: standalone exe needs no argv; full TrDBClient uses env CVI_RUN_DECOUPLING_ENGINE=1."""
    env = os.environ.copy()
    if exe.name.lower() == "cvi_decoupling_engine.exe":
        cmd = [str(exe)]
    else:
        env["CVI_RUN_DECOUPLING_ENGINE"] = "1"
        cmd = [str(exe)]
    return subprocess.run(
        cmd,
        input=json_in,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
        env=env,
    )


def _snapshot_for_cpp(s: dict) -> dict[str, object]:
    rows = []
    src_rows = s.get("rows", {})
    if isinstance(src_rows, dict):
        for kkey, r in src_rows.items():
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "key": str(kkey),
                    "strike": r.get("strike"),
                    "z": r.get("z"),
                    "sigma": r.get("sigma"),
                    "c_market": r.get("c_market"),
                    "c_market_source": r.get("c_market_source"),
                    "c_bid": r.get("c_bid"),
                    "c_ask": r.get("c_ask"),
                    "c_mid": r.get("c_mid"),
                    "c_bid_lr": r.get("c_bid_lr"),
                    "c_ask_lr": r.get("c_ask_lr"),
                    "c_mid_lr": r.get("c_mid_lr"),
                    "bid_impl_vol": r.get("bid_impl_vol"),
                    "ask_impl_vol": r.get("ask_impl_vol"),
                    "mid_impl_vol": r.get("mid_impl_vol"),
                    "bid_price_source": r.get("bid_price_source"),
                    "ask_price_source": r.get("ask_price_source"),
                    "mid_price_source": r.get("mid_price_source"),
                    "c_fit": r.get("c_fit"),
                    "c_fit_source": r.get("c_fit_source"),
                    "delta_bs": r.get("delta_bs"),
                    "gamma_bs": r.get("gamma_bs"),
                    "vega_bs": r.get("vega_bs"),
                    "theta_bs": r.get("theta_bs"),
                }
            )
    rows.sort(key=lambda rr: float(rr.get("strike")) if _to_float_or_none(rr.get("strike")) is not None else float("inf"))
    t = s.get("time")
    t_sec = float(t.timestamp()) if isinstance(t, datetime) else None
    return {
        "timestamp": str(s.get("timestamp")),
        "time_sec": t_sec,
        "fwd": s.get("fwd"),
        "rows": rows,
    }


def _build_decoupling_payload_cpp(
    batch_dir: Path,
    *,
    days: list[str],
    expiry_indices: list[int],
    window_min: float,
    snapshot_spacing_min: float,
    min_abs_df_frac: float,
) -> dict[str, object]:
    exe = _locate_decoupling_engine_exe()
    if exe is None:
        raise RuntimeError(
            "C++ decoupling engine not found (expected cvi_decoupling_engine.exe under "
            "ResearchBench/x64/Release or similar). Build the ResearchBenchDecoupling project in Visual Studio, "
            "or set CVI_DECOUPLING_CPP_EXE to the engine executable path."
        )

    groups: list[dict[str, object]] = []
    for day in days:
        summary = read_summary(batch_dir, day=day)
        if not summary:
            continue
        for expiry_index in expiry_indices:
            snaps: list[dict] = []
            for row in summary:
                st = _build_decoupling_snapshot_state(batch_dir, row, expiry_index)
                if st is not None and st.get("time") is not None:
                    snaps.append(st)
            if len(snaps) < 2:
                continue
            snaps.sort(key=lambda s: s["time"])
            groups.append(
                {
                    "date": day,
                    "expiry_index": int(expiry_index),
                    "snaps": [_snapshot_for_cpp(s) for s in snaps],
                }
            )

    diag_empty = {
        "n_days": len(days),
        "n_expiries": len(expiry_indices),
        "n_rows_total": 0,
        "n_rows_valid": 0,
        "n_skipped_small_df": 0,
        "n_skipped_missing_price": 0,
        "n_skipped_missing_sigma": 0,
        "n_skipped_missing_greeks": 0,
        "n_skipped_missing_exit_strike": 0,
        "n_skipped_other": 0,
    }
    if not groups:
        return {"rows": [], "summary": [], "diagnostics": diag_empty}

    in_doc = {
        "n_days": len(days),
        "n_expiries": len(expiry_indices),
        "window_min": float(window_min),
        "snapshot_spacing_min": float(snapshot_spacing_min),
        "min_abs_df_frac": float(min_abs_df_frac),
        "eps_delta": 1e-8,
        "groups": groups,
    }
    proc = _run_decoupling_engine_subprocess(exe, json.dumps(in_doc, separators=(",", ":")))
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"C++ decoupling engine failed (code={proc.returncode}): {stderr or 'no stderr'}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"C++ decoupling engine returned non-JSON output: {exc}") from exc
    if not isinstance(out, dict):
        raise RuntimeError("C++ decoupling engine output is not an object.")
    if "rows" not in out or "summary" not in out or "diagnostics" not in out:
        raise RuntimeError("C++ decoupling engine output missing required keys.")
    if not isinstance(out.get("mid_wls_per_strike"), list):
        out["mid_wls_per_strike"] = build_decoupling_mid_wls_beta_rows_per_strike(out)
    d = out.get("diagnostics")
    if isinstance(d, dict):
        d.setdefault("n_days", len(days))
        d.setdefault("n_expiries", len(expiry_indices))
    return out


def _build_decoupling_payload_python(
    batch_dir: Path,
    *,
    days: list[str],
    expiry_indices: list[int],
    window_min: float,
    snapshot_spacing_min: float,
    min_abs_df_frac: float,
) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    diag = {
        "n_days": len(days),
        "n_expiries": len(expiry_indices),
        "n_rows_total": 0,
        "n_rows_valid": 0,
        "n_skipped_small_df": 0,
        "n_skipped_missing_price": 0,
        "n_skipped_missing_sigma": 0,
        "n_skipped_missing_greeks": 0,
        "n_skipped_missing_exit_strike": 0,
        "n_skipped_other": 0,
    }
    eps_delta = 1e-8
    for day in days:
        summary = read_summary(batch_dir, day=day)
        if not summary:
            continue
        for expiry_index in expiry_indices:
            snaps: list[dict] = []
            for row in summary:
                st = _build_decoupling_snapshot_state(batch_dir, row, expiry_index)
                if st is not None and st.get("time") is not None:
                    snaps.append(st)
            if len(snaps) < 2:
                continue
            snaps.sort(key=lambda s: s["time"])
            spot_vals = [float(s["fwd"]) for s in snaps if math.isfinite(float(s["fwd"]))]
            spot_mean = statistics.mean(spot_vals) if spot_vals else None
            spot_std_dev = statistics.pstdev(spot_vals) if len(spot_vals) >= 2 else None
            spot_min = min(spot_vals) if spot_vals else None
            spot_max = max(spot_vals) if spot_vals else None
            entry_idx: list[int] = []
            last_t: datetime | None = None
            spacing_td = timedelta(minutes=float(snapshot_spacing_min))
            for i, s in enumerate(snaps):
                ti = s["time"]
                if last_t is None or (ti - last_t) >= spacing_td:
                    entry_idx.append(i)
                    last_t = ti
            valid_rows_for_group: list[dict[str, object]] = []
            for i in entry_idx:
                j = _find_forward_idx([s["time"] for s in snaps], i, float(window_min))
                if j is None:
                    continue
                s0 = snaps[i]
                s1 = snaps[j]
                dt_year = (s1["time"] - s0["time"]).total_seconds() / (365.0 * 24.0 * 3600.0)
                dF = float(s1["fwd"]) - float(s0["fwd"])
                df_floor = float(min_abs_df_frac) * float(s0["fwd"])
                for kkey, r0 in s0["rows"].items():
                    row_out: dict[str, object] = {
                        "date": day,
                        "expiry_index": int(expiry_index),
                        "from_t": str(s0["timestamp"]),
                        "to_t": str(s1["timestamp"]),
                        "strike": r0.get("strike"),
                        "z_entry": r0.get("z"),
                        "F_entry": s0["fwd"],
                        "F_exit": s1["fwd"],
                        "dF": dF,
                        "dF_floor": df_floor,
                        "dt_years": dt_year,
                        "sigma_entry": r0.get("sigma"),
                        "sigma_exit": None,
                        "dSigma": None,
                        "c_bid_quote_entry": r0.get("c_bid"),
                        "c_bid_entry": r0.get("c_bid_lr"),
                        "c_bid_quote_exit": None,
                        "c_bid_exit": None,
                        "c_ask_quote_entry": r0.get("c_ask"),
                        "c_ask_entry": r0.get("c_ask_lr"),
                        "c_ask_quote_exit": None,
                        "c_ask_exit": None,
                        "c_mid_quote_entry": r0.get("c_mid"),
                        "c_mid_entry": r0.get("c_mid_lr"),
                        "c_mid_quote_exit": None,
                        "c_mid_exit": None,
                        "bid_impl_vol_entry": r0.get("bid_impl_vol"),
                        "bid_impl_vol_exit": None,
                        "ask_impl_vol_entry": r0.get("ask_impl_vol"),
                        "ask_impl_vol_exit": None,
                        "mid_impl_vol_entry": r0.get("mid_impl_vol"),
                        "mid_impl_vol_exit": None,
                        "c_entry": r0.get("c_market"),
                        "c_exit": None,
                        "c_fit_entry": r0.get("c_fit"),
                        "c_fit_exit": None,
                        "dC": None,
                        "dC_bid": None,
                        "dC_ask": None,
                        "dC_mid": None,
                        "dC_fit": None,
                        "delta_bs_entry": r0.get("delta_bs"),
                        "gamma_bs_entry": r0.get("gamma_bs"),
                        "vega_bs_entry": r0.get("vega_bs"),
                        "theta_bs_entry": r0.get("theta_bs"),
                        "dc_tilde": None,
                        "dc_tilde_bid": None,
                        "dc_tilde_ask": None,
                        "dc_tilde_mid": None,
                        "dc_tilde_fit": None,
                        "delta_realized": None,
                        "delta_realized_bid": None,
                        "delta_realized_ask": None,
                        "delta_realized_mid": None,
                        "delta_realized_fit": None,
                        "decoupling": None,
                        "decoupling_bid": None,
                        "decoupling_ask": None,
                        "decoupling_mid": None,
                        "decoupling_fit": None,
                        "decoupling_normalized": None,
                        "decoupling_normalized_bid": None,
                        "decoupling_normalized_ask": None,
                        "decoupling_normalized_mid": None,
                        "decoupling_normalized_fit": None,
                        "entry_price_source": r0.get("c_market_source"),
                        "exit_price_source": None,
                        "entry_bid_price_source": r0.get("bid_price_source"),
                        "exit_bid_price_source": None,
                        "entry_ask_price_source": r0.get("ask_price_source"),
                        "exit_ask_price_source": None,
                        "entry_mid_price_source": r0.get("mid_price_source"),
                        "exit_mid_price_source": None,
                        "entry_fit_price_source": r0.get("c_fit_source"),
                        "exit_fit_price_source": None,
                        "valid": 0,
                        "skip_reason": None,
                    }
                    diag["n_rows_total"] += 1
                    r1 = s1["rows"].get(kkey)
                    if r1 is None:
                        row_out["skip_reason"] = "missing_exit_strike"
                        diag["n_skipped_missing_exit_strike"] += 1
                        all_rows.append(row_out)
                        continue
                    row_out["sigma_exit"] = r1.get("sigma")
                    row_out["exit_price_source"] = r1.get("c_market_source")
                    row_out["c_bid_quote_exit"] = r1.get("c_bid")
                    row_out["c_bid_exit"] = r1.get("c_bid_lr")
                    row_out["c_ask_quote_exit"] = r1.get("c_ask")
                    row_out["c_ask_exit"] = r1.get("c_ask_lr")
                    row_out["c_mid_quote_exit"] = r1.get("c_mid")
                    row_out["c_mid_exit"] = r1.get("c_mid_lr")
                    row_out["bid_impl_vol_exit"] = r1.get("bid_impl_vol")
                    row_out["ask_impl_vol_exit"] = r1.get("ask_impl_vol")
                    row_out["mid_impl_vol_exit"] = r1.get("mid_impl_vol")
                    row_out["exit_bid_price_source"] = r1.get("bid_price_source")
                    row_out["exit_ask_price_source"] = r1.get("ask_price_source")
                    row_out["exit_mid_price_source"] = r1.get("mid_price_source")
                    row_out["c_fit_exit"] = r1.get("c_fit")
                    row_out["exit_fit_price_source"] = r1.get("c_fit_source")
                    c0 = _to_float_or_none(r0.get("c_market"))
                    c1 = _to_float_or_none(r1.get("c_market"))
                    if c0 is None or c1 is None:
                        row_out["skip_reason"] = "missing_market_price"
                        diag["n_skipped_missing_price"] += 1
                        all_rows.append(row_out)
                        continue
                    if abs(dF) < df_floor:
                        row_out["skip_reason"] = "small_dF"
                        diag["n_skipped_small_df"] += 1
                        all_rows.append(row_out)
                        continue
                    sigma0 = _to_float_or_none(r0.get("sigma"))
                    sigma1 = _to_float_or_none(r1.get("sigma"))
                    if sigma0 is None or sigma1 is None:
                        row_out["skip_reason"] = "missing_sigma"
                        diag["n_skipped_missing_sigma"] += 1
                        all_rows.append(row_out)
                        continue
                    delta0 = _to_float_or_none(r0.get("delta_bs"))
                    gamma0 = _to_float_or_none(r0.get("gamma_bs"))
                    vega0 = _to_float_or_none(r0.get("vega_bs"))
                    theta0 = _to_float_or_none(r0.get("theta_bs"))
                    if delta0 is None or gamma0 is None or vega0 is None or theta0 is None:
                        row_out["skip_reason"] = "missing_greeks"
                        diag["n_skipped_missing_greeks"] += 1
                        all_rows.append(row_out)
                        continue
                    d_sigma = float(sigma1 - sigma0)
                    d_c = float(c1 - c0)
                    dc_tilde = d_c - vega0 * d_sigma - 0.5 * gamma0 * dF * dF - theta0 * dt_year
                    delta_real = dc_tilde / dF
                    dec = delta_real - delta0
                    dec_norm = dec / max(abs(delta0), eps_delta)
                    row_out["dSigma"] = d_sigma
                    row_out["dC"] = d_c
                    row_out["c_exit"] = c1
                    row_out["dc_tilde"] = dc_tilde
                    row_out["delta_realized"] = delta_real
                    row_out["decoupling"] = dec
                    row_out["decoupling_normalized"] = dec_norm
                    for label, pkey in (("bid", "c_bid_lr"), ("ask", "c_ask_lr"), ("mid", "c_mid_lr")):
                        p0 = _to_float_or_none(r0.get(pkey))
                        p1 = _to_float_or_none(r1.get(pkey))
                        if p0 is None or p1 is None:
                            continue
                        d_c_side = float(p1 - p0)
                        dc_tilde_side = d_c_side - vega0 * d_sigma - 0.5 * gamma0 * dF * dF - theta0 * dt_year
                        delta_side = dc_tilde_side / dF
                        dec_side = delta_side - delta0
                        row_out[f"dC_{label}"] = d_c_side
                        row_out[f"dc_tilde_{label}"] = dc_tilde_side
                        row_out[f"delta_realized_{label}"] = delta_side
                        row_out[f"decoupling_{label}"] = dec_side
                        row_out[f"decoupling_normalized_{label}"] = dec_side / max(abs(delta0), eps_delta)
                    c0_fit = _to_float_or_none(r0.get("c_fit"))
                    c1_fit = _to_float_or_none(r1.get("c_fit"))
                    if c0_fit is not None and c1_fit is not None:
                        d_c_fit = float(c1_fit - c0_fit)
                        dc_tilde_fit = d_c_fit - vega0 * d_sigma - 0.5 * gamma0 * dF * dF - theta0 * dt_year
                        delta_fit = dc_tilde_fit / dF
                        dec_fit = delta_fit - delta0
                        row_out["dC_fit"] = d_c_fit
                        row_out["dc_tilde_fit"] = dc_tilde_fit
                        row_out["delta_realized_fit"] = delta_fit
                        row_out["decoupling_fit"] = dec_fit
                        row_out["decoupling_normalized_fit"] = dec_fit / max(abs(delta0), eps_delta)
                    row_out["valid"] = 1
                    diag["n_rows_valid"] += 1
                    valid_rows_for_group.append(row_out)
                    all_rows.append(row_out)
            vals_dec = [float(r["decoupling"]) for r in valid_rows_for_group if _to_float_or_none(r.get("decoupling")) is not None]
            vals_abs = [abs(v) for v in vals_dec]
            vals_dec_bid = [
                float(r["decoupling_bid"]) for r in valid_rows_for_group if _to_float_or_none(r.get("decoupling_bid")) is not None
            ]
            vals_dec_ask = [
                float(r["decoupling_ask"]) for r in valid_rows_for_group if _to_float_or_none(r.get("decoupling_ask")) is not None
            ]
            vals_dec_mid = [
                float(r["decoupling_mid"]) for r in valid_rows_for_group if _to_float_or_none(r.get("decoupling_mid")) is not None
            ]
            vals_dec_fit = [
                float(r["decoupling_fit"]) for r in valid_rows_for_group if _to_float_or_none(r.get("decoupling_fit")) is not None
            ]
            vals_dr = [float(r["delta_realized"]) for r in valid_rows_for_group if _to_float_or_none(r.get("delta_realized")) is not None]
            vals_db = [float(r["delta_bs_entry"]) for r in valid_rows_for_group if _to_float_or_none(r.get("delta_bs_entry")) is not None]
            vals_dr_bid = [
                float(r["delta_realized_bid"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_bid")) is not None
            ]
            vals_db_bid = [
                float(r["delta_bs_entry"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_bid")) is not None and _to_float_or_none(r.get("delta_bs_entry")) is not None
            ]
            vals_dr_ask = [
                float(r["delta_realized_ask"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_ask")) is not None
            ]
            vals_db_ask = [
                float(r["delta_bs_entry"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_ask")) is not None and _to_float_or_none(r.get("delta_bs_entry")) is not None
            ]
            vals_dr_mid = [
                float(r["delta_realized_mid"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_mid")) is not None
            ]
            vals_db_mid = [
                float(r["delta_bs_entry"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_mid")) is not None and _to_float_or_none(r.get("delta_bs_entry")) is not None
            ]
            vals_dr_fit = [
                float(r["delta_realized_fit"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_fit")) is not None
            ]
            vals_db_fit = [
                float(r["delta_bs_entry"])
                for r in valid_rows_for_group
                if _to_float_or_none(r.get("delta_realized_fit")) is not None and _to_float_or_none(r.get("delta_bs_entry")) is not None
            ]
            corr = _pearson_corr(vals_dr, vals_db) if len(vals_dr) == len(vals_db) else None
            corr_bid = _pearson_corr(vals_dr_bid, vals_db_bid) if len(vals_dr_bid) == len(vals_db_bid) else None
            corr_ask = _pearson_corr(vals_dr_ask, vals_db_ask) if len(vals_dr_ask) == len(vals_db_ask) else None
            corr_mid = _pearson_corr(vals_dr_mid, vals_db_mid) if len(vals_dr_mid) == len(vals_db_mid) else None
            corr_fit = _pearson_corr(vals_dr_fit, vals_db_fit) if len(vals_dr_fit) == len(vals_db_fit) else None
            summary_rows.append(
                {
                    "date": day,
                    "expiry_index": int(expiry_index),
                    "n_valid": len(valid_rows_for_group),
                    "n_total": sum(1 for r in all_rows if r.get("date") == day and int(r.get("expiry_index", -1)) == int(expiry_index)),
                    "spot_n": len(spot_vals),
                    "spot_mean": spot_mean,
                    "spot_std_dev": spot_std_dev,
                    "spot_min": spot_min,
                    "spot_max": spot_max,
                    "rms_decoupling": (math.sqrt(sum(v * v for v in vals_dec) / len(vals_dec)) if vals_dec else None),
                    "rms_decoupling_bid": (
                        math.sqrt(sum(v * v for v in vals_dec_bid) / len(vals_dec_bid)) if vals_dec_bid else None
                    ),
                    "rms_decoupling_ask": (
                        math.sqrt(sum(v * v for v in vals_dec_ask) / len(vals_dec_ask)) if vals_dec_ask else None
                    ),
                    "rms_decoupling_mid": (
                        math.sqrt(sum(v * v for v in vals_dec_mid) / len(vals_dec_mid)) if vals_dec_mid else None
                    ),
                    "rms_decoupling_fit": (
                        math.sqrt(sum(v * v for v in vals_dec_fit) / len(vals_dec_fit)) if vals_dec_fit else None
                    ),
                    "median_abs_decoupling": (statistics.median(vals_abs) if vals_abs else None),
                    "median_abs_decoupling_bid": (statistics.median([abs(v) for v in vals_dec_bid]) if vals_dec_bid else None),
                    "median_abs_decoupling_ask": (statistics.median([abs(v) for v in vals_dec_ask]) if vals_dec_ask else None),
                    "median_abs_decoupling_mid": (statistics.median([abs(v) for v in vals_dec_mid]) if vals_dec_mid else None),
                    "median_abs_decoupling_fit": (statistics.median([abs(v) for v in vals_dec_fit]) if vals_dec_fit else None),
                    "corr_delta_realized_vs_bs": corr,
                    "corr_delta_realized_bid_vs_bs": corr_bid,
                    "corr_delta_realized_ask_vs_bs": corr_ask,
                    "corr_delta_realized_mid_vs_bs": corr_mid,
                    "corr_delta_realized_fit_vs_bs": corr_fit,
                }
            )
    diag["n_skipped_other"] = max(
        int(diag["n_rows_total"]) - int(diag["n_rows_valid"]) - int(diag["n_skipped_small_df"]) - int(diag["n_skipped_missing_price"])
        - int(diag["n_skipped_missing_sigma"]) - int(diag["n_skipped_missing_greeks"]) - int(diag["n_skipped_missing_exit_strike"]),
        0,
    )
    out_py: dict[str, object] = {"rows": all_rows, "summary": summary_rows, "diagnostics": diag}
    out_py["mid_wls_per_strike"] = build_decoupling_mid_wls_beta_rows_per_strike(out_py)
    return out_py


def build_decoupling_payload(
    batch_dir: Path,
    *,
    days: list[str],
    expiry_indices: list[int],
    window_min: float,
    snapshot_spacing_min: float,
    min_abs_df_frac: float,
) -> dict[str, object]:
    use_cpp = str(os.environ.get("CVI_DECOUPLING_CPP", "1")).strip().lower() not in {"0", "false", "no", "off"}
    cpp_strict = str(os.environ.get("CVI_DECOUPLING_CPP_STRICT", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if use_cpp:
        try:
            return _build_decoupling_payload_cpp(
                batch_dir,
                days=days,
                expiry_indices=expiry_indices,
                window_min=window_min,
                snapshot_spacing_min=snapshot_spacing_min,
                min_abs_df_frac=min_abs_df_frac,
            )
        except Exception as exc:
            if cpp_strict:
                raise SystemExit(f"C++ decoupling required (CVI_DECOUPLING_CPP_STRICT) but failed: {exc}") from exc
            print(f"decoupling cpp unavailable, falling back to python: {exc}", file=sys.stderr)
    return _build_decoupling_payload_python(
        batch_dir,
        days=days,
        expiry_indices=expiry_indices,
        window_min=window_min,
        snapshot_spacing_min=snapshot_spacing_min,
        min_abs_df_frac=min_abs_df_frac,
    )


def write_decoupling_detail_csv(payload: dict[str, object], out_csv: Path) -> Path:
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("decoupling: no rows to write.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out_csv.open("w", newline="", encoding="utf-8")
    except PermissionError:
        out_csv = out_csv.with_name(out_csv.stem + "_new" + out_csv.suffix)
        fh = out_csv.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def write_decoupling_summary_csv(payload: dict[str, object], out_csv: Path) -> Path:
    rows = payload.get("summary", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("decoupling: no summary rows to write.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out_csv.open("w", newline="", encoding="utf-8")
    except PermissionError:
        out_csv = out_csv.with_name(out_csv.stem + "_new" + out_csv.suffix)
        fh = out_csv.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return out_csv


def _decoupling_mid_wls_alpha_beta_r2(xs: list[float], ys: list[float]) -> tuple[float | None, float | None, float | None]:
    """WLS fit Y ~ α + β X with inverse bivariate-move weights; returns (α, β, r²_wls)."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None, None, None
    ws = wls_weights_inverse_bivariate_move_sq(xs, ys)
    ab = wls_alpha_beta(xs, ys, ws)
    if ab is None:
        return None, None, None
    alpha, beta, _nn = ab
    w = [max(ws[i], 1e-15) for i in range(n)]
    sw = sum(w)
    my = sum(w[i] * ys[i] for i in range(n)) / sw
    y_hat = [alpha + beta * xs[i] for i in range(n)]
    ss_res = sum(w[i] * (ys[i] - y_hat[i]) ** 2 for i in range(n))
    ss_tot = sum(w[i] * (ys[i] - my) ** 2 for i in range(n))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else None
    return alpha, beta, r2


def build_decoupling_mid_wls_beta_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    """
    Pooled WLS across all strikes/windows: delta_realized_mid ~ alpha + beta * delta_bs(entry).

    Weights match :func:`wls_weights_inverse_bivariate_move_sq` on (x, y) = (δ_BS, δ_realized_mid).
    One output row per (date, expiry_index).
    """
    rows_in = payload.get("rows", [])
    if not isinstance(rows_in, list) or not rows_in:
        return []
    by_key: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for r in rows_in:
        if int(r.get("valid", 0)) != 1:
            continue
        x = _to_float_or_none(r.get("delta_bs_entry"))
        y = _to_float_or_none(r.get("delta_realized_mid"))
        if x is None or y is None:
            continue
        day = str(r.get("date", ""))
        ei = int(r["expiry_index"])
        by_key[(day, ei)].append((float(x), float(y)))

    out: list[dict[str, object]] = []
    for (day, ei), pairs in sorted(by_key.items()):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(xs)
        alpha, beta, r2 = _decoupling_mid_wls_alpha_beta_r2(xs, ys)
        row_base: dict[str, object] = {
            "date": day,
            "expiry_index": ei,
            "n": n,
            "alpha": alpha,
            "beta": beta,
            "r2_wls": r2,
            "weight_scheme": "w_i = 1/(eps + x_i^2 + y_i^2), eps=1e-8",
            "model": "delta_realized_mid ~ alpha + beta * delta_bs_entry",
        }
        out.append(row_base)
    return out


def build_decoupling_mid_wls_beta_rows_per_strike(payload: dict[str, object]) -> list[dict[str, object]]:
    """
    Per (date, expiry_index, strike): WLS on mid-implied-vol realized δ vs BS δ(entry).

    Same regression as :func:`build_decoupling_mid_wls_beta_rows`, but points are restricted to one strike.
    """
    rows_in = payload.get("rows", [])
    if not isinstance(rows_in, list) or not rows_in:
        return []
    by_key: dict[tuple[str, int, str], list[tuple[float, float]]] = defaultdict(list)
    z_by: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    strike_float_by: dict[tuple[str, int, str], float] = {}
    for r in rows_in:
        if int(r.get("valid", 0)) != 1:
            continue
        x = _to_float_or_none(r.get("delta_bs_entry"))
        y = _to_float_or_none(r.get("delta_realized_mid"))
        if x is None or y is None:
            continue
        sk = r.get("strike")
        if sk is None:
            continue
        fk = float(sk)
        kkey = fmt_strike_key(fk)
        day = str(r.get("date", ""))
        ei = int(r["expiry_index"])
        key = (day, ei, kkey)
        by_key[key].append((float(x), float(y)))
        strike_float_by[key] = fk
        ze = _to_float_or_none(r.get("z_entry"))
        if ze is not None:
            z_by[key].append(float(ze))

    out: list[dict[str, object]] = []
    for key in sorted(by_key.keys(), key=lambda k: (k[0], k[1], strike_float_by[k])):
        day, ei, _kkey = key
        pairs = by_key[key]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(xs)
        alpha, beta, r2 = _decoupling_mid_wls_alpha_beta_r2(xs, ys)
        zs = z_by.get(key, [])
        z_med = statistics.median(zs) if zs else None
        out.append(
            {
                "date": day,
                "expiry_index": ei,
                "strike": strike_float_by[key],
                "z_entry_median": z_med,
                "n": n,
                "alpha": alpha,
                "beta": beta,
                "r2_wls": r2,
                "weight_scheme": "w_i = 1/(eps + x_i^2 + y_i^2), eps=1e-8",
                "model": "delta_realized_mid ~ alpha + beta * delta_bs_entry (per strike)",
            }
        )
    return out


def render_decoupling_mid_wls_html(payload: dict[str, object], title: str) -> str:
    """
    Plotly page: pooled δ_realized_mid vs δ_BS(entry), colored by date, with one WLS line
    y = α + β x per (date, expiry_index) matching :func:`build_decoupling_mid_wls_beta_rows`.
    """
    rows_in = payload.get("rows", [])
    pts: list[dict[str, float | str | int]] = []
    if isinstance(rows_in, list):
        for r in rows_in:
            if int(r.get("valid", 0)) != 1:
                continue
            x = _to_float_or_none(r.get("delta_bs_entry"))
            y = _to_float_or_none(r.get("delta_realized_mid"))
            if x is None or y is None:
                continue
            pts.append(
                {
                    "date": str(r.get("date", "")),
                    "expiry_index": int(r["expiry_index"]),
                    "x": float(x),
                    "y": float(y),
                }
            )
    if not pts:
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{title}</title></head>
<body><p>No valid points with delta_realized_mid for mid WLS plot.</p></body></html>"""
    betas = build_decoupling_mid_wls_beta_rows(payload)
    pts_json = json.dumps(pts, separators=(",", ":"))
    betas_json = json.dumps(betas, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    #mid_wls {{ width: 100%; height: 640px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; font-size: 13px; }}
    p.note {{ color: #444; max-width: 960px; }}
  </style>
</head>
<body>
  <h1>Mid-track WLS: δ_realized_mid ~ α + β · δ_BS(entry)</h1>
  <p class="note">Markers = all valid windows (all strikes). Lines = WLS fit per calendar date × expiry.
  Weights <code>w_i = 1/(ε + x_i² + y_i²)</code>, ε = 1e−8. Same α, β as <code>decoupling_mid_wls_betas.csv</code>.</p>
  <div id="mid_wls"></div>
  <h3>WLS coefficients (intercept α, slope β)</h3>
  <pre id="beta_tbl"></pre>
  <script>
    const pts = {pts_json};
    const betas = {betas_json};
    const cat = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"];
    const traces = [];
    const dates = [...new Set(pts.map(p => p.date))].sort();
    const dateColor = {{}};
    dates.forEach((d, i) => {{ dateColor[d] = cat[i % cat.length]; }});
    dates.forEach((d) => {{
      const sub = pts.filter(p => p.date === d);
      traces.push({{
        type: "scatter", mode: "markers", name: d + " (mid)",
        x: sub.map(p => p.x), y: sub.map(p => p.y),
        marker: {{ size: 6, opacity: 0.65, color: dateColor[d] }},
        hovertemplate: "δ_BS=%{{x:.5f}}<br>δ_mid=%{{y:.5f}}<extra></extra>",
      }});
    }});
    betas.forEach((b) => {{
      if (b.alpha == null || b.beta == null || !Number.isFinite(b.alpha) || !Number.isFinite(b.beta)) return;
      const sub = pts.filter(p => p.date === b.date && p.expiry_index === b.expiry_index);
      if (sub.length === 0) return;
      const xs = sub.map(p => p.x);
      const xmin = Math.min(...xs);
      const xmax = Math.max(...xs);
      const xline = [xmin, xmax];
      const yline = [b.alpha + b.beta * xmin, b.alpha + b.beta * xmax];
      const col = dateColor[b.date] || "#333";
      traces.push({{
        type: "scatter", mode: "lines",
        name: `WLS e${{b.expiry_index}} ${{b.date}}`,
        x: xline, y: yline,
        line: {{ width: 2.5, dash: "solid", color: col }},
        hovertemplate: `α=${{b.alpha}} β=${{b.beta}} r²=${{b.r2_wls}}<extra></extra>`,
      }});
    }});
    const layout = {{
      title: "{title}",
      xaxis: {{ title: "δ_BS (entry)" }},
      yaxis: {{ title: "δ_realized (mid track)" }},
      margin: {{ l: 64, r: 28, t: 56, b: 56 }},
      hovermode: "closest",
      legend: {{ orientation: "v", x: 1.02, y: 1 }},
    }};
    Plotly.newPlot("mid_wls", traces, layout, {{ responsive: true }});
    document.getElementById("beta_tbl").textContent = JSON.stringify(betas, null, 2);
  </script>
</body>
</html>"""


def render_decoupling_mid_wls_per_strike_html(payload: dict[str, object], title: str) -> str:
    """
    Plotly: pick calendar date, expiry, and strike — scatter of δ_realized_mid vs δ_BS(entry)
    plus the per-strike WLS line (same α, β as CSV).
    """
    rows_in = payload.get("rows", [])
    betas_raw = payload.get("mid_wls_per_strike")
    if isinstance(betas_raw, list) and betas_raw:
        betas = betas_raw
    else:
        betas = build_decoupling_mid_wls_beta_rows_per_strike(payload)
    pts: list[dict[str, float | str | int]] = []
    if isinstance(rows_in, list):
        for r in rows_in:
            if int(r.get("valid", 0)) != 1:
                continue
            x = _to_float_or_none(r.get("delta_bs_entry"))
            y = _to_float_or_none(r.get("delta_realized_mid"))
            sk = _to_float_or_none(r.get("strike"))
            if x is None or y is None or sk is None:
                continue
            pts.append(
                {
                    "date": str(r.get("date", "")),
                    "expiry_index": int(r["expiry_index"]),
                    "strike": float(sk),
                    "x": float(x),
                    "y": float(y),
                }
            )
    if not pts or not betas:
        return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>{html.escape(title)}</title></head>
<body><p>No per-strike mid points or betas to plot.</p></body></html>"""
    pts_json = json.dumps(pts, separators=(",", ":"))
    betas_json = json.dumps(betas, separators=(",", ":"))
    title_esc = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title_esc}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    #plot {{ width: 100%; height: 560px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    .row {{ display: flex; gap: 12px; margin: 12px 0; align-items: center; flex-wrap: wrap; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; font-size: 13px; }}
    p.note {{ color: #444; max-width: 960px; }}
  </style>
</head>
<body>
  <h1>{title_esc}</h1>
  <p class="note">Each point is one decoupling window (mid implied-vol track). WLS: δ_realized_mid ~ α + β·δ_BS(entry),
  weights 1/(ε+x²+y²). Pick date, expiry, and strike to match <code>decoupling_mid_wls_betas_per_strike.csv</code>.</p>
  <div class="row">
    <label>Date <select id="pick_date"></select></label>
    <label>Expiry <select id="pick_exp"></select></label>
    <label>Strike <select id="pick_k"></select></label>
  </div>
  <div id="plot"></div>
  <h3>Selected row (from CSV)</h3>
  <pre id="beta_one"></pre>
  <script>
    const pts = {pts_json};
    const betas = {betas_json};
    function strikeClose(a, b) {{
      return Math.abs(Number(a) - Number(b)) <= 1e-5 + 1e-9 * Math.max(Math.abs(Number(a)), Math.abs(Number(b)));
    }}
    const selD = document.getElementById("pick_date");
    const selE = document.getElementById("pick_exp");
    const selK = document.getElementById("pick_k");
    const dates = [...new Set(betas.map(b => String(b.date)))].sort();
    dates.forEach((d, i) => {{
      const o = document.createElement("option");
      o.value = d; o.textContent = d;
      if (i === 0) o.selected = true;
      selD.appendChild(o);
    }});
    function expiriesForDate(d) {{
      const xs = [...new Set(betas.filter(b => String(b.date) === d).map(b => Number(b.expiry_index)))];
      return xs.sort((a,b) => a - b);
    }}
    function strikesFor(d, e) {{
      return betas.filter(b => String(b.date) === d && Number(b.expiry_index) === Number(e))
        .map(b => Number(b.strike)).sort((a,b) => a - b);
    }}
    function refillExp() {{
      const d = selD.value;
      selE.innerHTML = "";
      expiriesForDate(d).forEach((e, i) => {{
        const o = document.createElement("option");
        o.value = String(e); o.textContent = "e" + e;
        if (i === 0) o.selected = true;
        selE.appendChild(o);
      }});
      refillK();
    }}
    function refillK() {{
      const d = selD.value;
      const e = Number(selE.value);
      selK.innerHTML = "";
      const ks = strikesFor(d, e);
      ks.forEach((k, i) => {{
        const o = document.createElement("option");
        o.value = String(k); o.textContent = String(k);
        if (i === 0) o.selected = true;
        selK.appendChild(o);
      }});
      redraw();
    }}
    function redraw() {{
      const d = selD.value;
      const e = Number(selE.value);
      const k = Number(selK.value);
      const sub = pts.filter(p => String(p.date) === d && Number(p.expiry_index) === e && strikeClose(p.strike, k));
      const b = betas.find(x => String(x.date) === d && Number(x.expiry_index) === e && strikeClose(x.strike, k));
      document.getElementById("beta_one").textContent = b ? JSON.stringify(b, null, 2) : "(no beta row)";
      const traces = [];
      traces.push({{
        type: "scatter", mode: "markers", name: "windows",
        x: sub.map(p => p.x), y: sub.map(p => p.y),
        marker: {{ size: 8, opacity: 0.75 }},
        hovertemplate: "δ_BS=%{{x:.5f}}<br>δ_mid=%{{y:.5f}}<extra></extra>",
      }});
      if (b && b.alpha != null && b.beta != null && Number.isFinite(b.alpha) && Number.isFinite(b.beta) && sub.length) {{
        const xv = sub.map(p => p.x);
        const xmin = Math.min(...xv);
        const xmax = Math.max(...xv);
        traces.push({{
          type: "scatter", mode: "lines", name: "WLS",
          x: [xmin, xmax],
          y: [b.alpha + b.beta * xmin, b.alpha + b.beta * xmax],
          line: {{ width: 2.5, color: "#d62728" }},
          hovertemplate: `α=${{b.alpha}} β=${{b.beta}} r²=${{b.r2_wls}}<extra></extra>`,
        }});
      }}
      Plotly.react("plot", traces, {{
        title: `${{d}}  expiry ${{e}}  K=${{k}}  (n=${{sub.length}})`,
        xaxis: {{ title: "δ_BS (entry)" }},
        yaxis: {{ title: "δ_realized (mid)" }},
        margin: {{ l: 64, r: 28, t: 56, b: 56 }},
        shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0,
          line: {{ dash: "dot", color: "#bbb" }} }}],
      }}, {{ responsive: true }});
    }}
    selD.addEventListener("change", () => {{ refillExp(); }});
    selE.addEventListener("change", refillK);
    selK.addEventListener("change", redraw);
    refillExp();
  </script>
</body>
</html>"""


def write_decoupling_mid_wls_betas_per_strike_csv(payload: dict[str, object], out_csv: Path) -> Path:
    rows_raw = payload.get("mid_wls_per_strike")
    if isinstance(rows_raw, list) and rows_raw:
        rows = rows_raw
    else:
        rows = build_decoupling_mid_wls_beta_rows_per_strike(payload)
    if not rows:
        raise SystemExit(
            "decoupling mid WLS per strike: no (date,expiry,strike) groups with valid delta_bs_entry "
            "and delta_realized_mid."
        )
    fieldnames = [
        "date",
        "expiry_index",
        "strike",
        "z_entry_median",
        "n",
        "alpha",
        "beta",
        "r2_wls",
        "weight_scheme",
        "model",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out_csv.open("w", newline="", encoding="utf-8")
    except PermissionError:
        out_csv = out_csv.with_name(out_csv.stem + "_new" + out_csv.suffix)
        fh = out_csv.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out_csv


def write_decoupling_mid_wls_betas_csv(payload: dict[str, object], out_csv: Path) -> Path:
    rows = build_decoupling_mid_wls_beta_rows(payload)
    if not rows:
        raise SystemExit(
            "decoupling mid WLS: no (date,expiry) groups with valid delta_bs_entry and delta_realized_mid."
        )
    fieldnames = [
        "date",
        "expiry_index",
        "n",
        "alpha",
        "beta",
        "r2_wls",
        "weight_scheme",
        "model",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        fh = out_csv.open("w", newline="", encoding="utf-8")
    except PermissionError:
        out_csv = out_csv.with_name(out_csv.stem + "_new" + out_csv.suffix)
        fh = out_csv.open("w", newline="", encoding="utf-8")
    with fh as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return out_csv


def render_decoupling_html(payload: dict[str, object], title: str) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    .plot {{ width: 100%; height: 500px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; }}
    .row {{ display: flex; gap: 12px; margin: 10px 0; align-items: center; flex-wrap: wrap; }}
  </style>
</head>
<body>
  <h1>Option-spot decoupling diagnostic</h1>
  <div class="row">
    <label>Series <select id="series_pick"></select></label>
  </div>
  <div id="dec_ts" class="plot"></div>
  <div id="rms_ts" class="plot"></div>
  <div id="delta_scatter" class="plot"></div>
  <h3>Summary (date × expiry)</h3>
  <pre id="summary"></pre>
  <h3>Diagnostics</h3>
  <pre id="diag"></pre>
  <script>
    const P = {data_json};
    const rows = (P.rows || []).filter(r => Number(r.valid) === 1);
    const groups = {{}};
    for (const r of rows) {{
      const key = `e${{r.expiry_index}}|K${{Number(r.strike).toFixed(4)}}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push(r);
    }}
    const keys = Object.keys(groups).sort();
    const sel = document.getElementById("series_pick");
    keys.forEach((k, i) => {{
      const o = document.createElement("option");
      o.value = k;
      o.textContent = k;
      if (i === 0) o.selected = true;
      sel.appendChild(o);
    }});
    function rollingRms(arr, win) {{
      const out = [];
      for (let i = 0; i < arr.length; i++) {{
        const lo = Math.max(0, i - win + 1);
        const sub = arr.slice(lo, i + 1).filter(v => Number.isFinite(v));
        if (sub.length === 0) {{ out.push(null); continue; }}
        const m = sub.reduce((a, b) => a + b * b, 0) / sub.length;
        out.push(Math.sqrt(m));
      }}
      return out;
    }}
    function redraw() {{
      const k = sel.value;
      const s = (groups[k] || []).slice().sort((a, b) => String(a.to_t).localeCompare(String(b.to_t)));
      const x = s.map(r => r.to_t);
      const dec = s.map(r => r.decoupling);
      const decBid = s.map(r => r.decoupling_bid);
      const decAsk = s.map(r => r.decoupling_ask);
      const decMid = s.map(r => r.decoupling_mid);
      const decN = s.map(r => r.decoupling_normalized);
      const rms = rollingRms(dec, 6);
      Plotly.react("dec_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: dec, name: "cvi fit" }},
        {{ type: "scatter", mode: "lines+markers", x, y: decBid, name: "bid" }},
        {{ type: "scatter", mode: "lines+markers", x, y: decAsk, name: "ask" }},
        {{ type: "scatter", mode: "lines+markers", x, y: decMid, name: "mid" }},
        {{ type: "scatter", mode: "lines+markers", x, y: decN, name: "decoupling_normalized", yaxis: "y2" }},
      ], {{
        title: "Per-window decoupling",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "δ_realized - δ_BS" }},
        yaxis2: {{ title: "normalized", overlaying: "y", side: "right", showgrid: false }},
        margin: {{ l: 70, r: 70, t: 48, b: 100 }},
        shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#888" }} }}]
      }}, {{ responsive: true }});
      Plotly.react("rms_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: rms, name: "rolling_rms_decoupling(6)" }},
      ], {{
        title: "Rolling RMS of decoupling",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "RMS" }},
        margin: {{ l: 70, r: 40, t: 48, b: 100 }},
      }}, {{ responsive: true }});
      const xDelta = s.map(r => r.delta_bs_entry);
      const yDelta = s.map(r => r.delta_realized);
      const yDeltaBid = s.map(r => r.delta_realized_bid);
      const yDeltaAsk = s.map(r => r.delta_realized_ask);
      const yDeltaMid = s.map(r => r.delta_realized_mid);
      function finitePairs(xs, ys) {{
        const xf = [];
        const yf = [];
        for (let i = 0; i < xs.length; i++) {{
          const a = xs[i];
          const b = ys[i];
          if (Number.isFinite(a) && Number.isFinite(b)) {{ xf.push(a); yf.push(b); }}
        }}
        return {{ x: xf, y: yf }};
      }}
      const fitXY = finitePairs(xDelta, yDelta);
      const bidXY = finitePairs(xDelta, yDeltaBid);
      const askXY = finitePairs(xDelta, yDeltaAsk);
      const midXY = finitePairs(xDelta, yDeltaMid);
      const finite = [...fitXY.x, ...fitXY.y, ...bidXY.y, ...askXY.y, ...midXY.y].filter(v => Number.isFinite(v));
      const lo = finite.length ? Math.min(...finite) : 0;
      const hi = finite.length ? Math.max(...finite) : 1;
      Plotly.react("delta_scatter", [
        {{ type: "scatter", mode: "markers", x: fitXY.x, y: fitXY.y, name: "cvi fit",
           marker: {{ size: 7, color: "#1f77b4", opacity: 0.75 }} }},
        {{ type: "scatter", mode: "markers", x: bidXY.x, y: bidXY.y, name: "bid",
           marker: {{ size: 7, symbol: "square", color: "#ff7f0e", opacity: 0.75 }} }},
        {{ type: "scatter", mode: "markers", x: askXY.x, y: askXY.y, name: "ask",
           marker: {{ size: 11, symbol: "diamond", color: "#2ca02c",
                      line: {{ width: 1.4, color: "#0d260d" }}, opacity: 0.9 }} }},
        {{ type: "scatter", mode: "markers", x: midXY.x, y: midXY.y, name: "mid",
           marker: {{ size: 7, symbol: "x", color: "#d62728", opacity: 0.75 }} }},
        {{ type: "scatter", mode: "lines", x: [lo, hi], y: [lo, hi], name: "y=x",
           line: {{ color: "#9467bd", width: 2 }} }},
      ], {{
        title: "δ_realized vs δ_BS(entry)",
        xaxis: {{ title: "δ_BS(entry)" }},
        yaxis: {{ title: "δ_realized" }},
        margin: {{ l: 70, r: 40, t: 48, b: 70 }},
      }}, {{ responsive: true }});
    }}
    if (keys.length > 0) {{
      sel.addEventListener("change", redraw);
      redraw();
    }}
    document.getElementById("summary").textContent = JSON.stringify(P.summary || [], null, 2);
    document.getElementById("diag").textContent = JSON.stringify(P.diagnostics || {{}}, null, 2);
  </script>
</body>
</html>"""

