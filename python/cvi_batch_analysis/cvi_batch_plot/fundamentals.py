"""Constants, CSV readers, LR/BS pricing, spline skew, robust regression."""
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

MARKET_PRICE_CANDIDATE_COLUMNS = (
    "call_mid",
    "call_price",
    "mid",
    "option_mid",
    "option_price",
)
MARKET_VOL_CANDIDATE_COLUMNS = (
    "option_impl_vol",
    "impl_vol",
    "mid_impl_vol",
)

BID_VOL_CANDIDATE_COLUMNS = (
    "bid_impl_vol",
    "call_bid_impl_vol",
    "bid_vol",
    "call_bid_vol",
)

ASK_VOL_CANDIDATE_COLUMNS = (
    "ask_impl_vol",
    "call_ask_impl_vol",
    "ask_vol",
    "call_ask_vol",
)


def _to_float_or_none(v: object) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for tok in (raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for tok in (raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t, 10))
    return vals


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_call_price_from_fwd(
    fwd: float,
    strike: float,
    sigma: float,
    texp: float,
    *,
    r: float = 0.0,
) -> float | None:
    if (
        not math.isfinite(fwd)
        or not math.isfinite(strike)
        or not math.isfinite(sigma)
        or not math.isfinite(texp)
        or fwd <= 0.0
        or strike <= 0.0
        or sigma <= 0.0
        or texp <= 0.0
    ):
        return None
    st = sigma * math.sqrt(texp)
    if st <= 0.0:
        return None
    d1 = (math.log(fwd / strike) + 0.5 * st * st) / st
    d2 = d1 - st
    disc = math.exp(-r * texp)
    return disc * (fwd * norm_cdf(d1) - strike * norm_cdf(d2))


def _bs_call_greeks_from_fwd(
    fwd: float,
    strike: float,
    sigma: float,
    texp: float,
    *,
    r: float = 0.0,
) -> dict[str, float] | None:
    price = _bs_call_price_from_fwd(fwd, strike, sigma, texp, r=r)
    if price is None:
        return None
    st = sigma * math.sqrt(texp)
    d1 = (math.log(fwd / strike) + 0.5 * st * st) / st
    disc = math.exp(-r * texp)
    pdf = norm_pdf(d1)
    delta = disc * norm_cdf(d1)
    gamma = disc * pdf / (fwd * st)
    vega = disc * fwd * pdf * math.sqrt(texp)
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
    }


def _pp_peizer_pratt(n: int, z: float) -> float:
    dn = float(n)
    denom = dn + (1.0 / 3.0) + (0.1 / (dn + 1.0))
    exp_term = -((z / denom) ** 2.0) * (dn + 1.0 / 6.0)
    out = 0.5 + (1.0 if z >= 0.0 else -1.0) * 0.5 * math.sqrt(max(0.0, 1.0 - math.exp(exp_term)))
    return min(max(out, 1e-10), 1.0 - 1e-10)


def _lr_adj_ud_step(r: float, q: float, dt: float, p: float, p_prime: float) -> tuple[float, float]:
    carry = math.exp((r - q) * dt)
    u = carry * (p_prime / p)
    d = carry * ((1.0 - p_prime) / (1.0 - p))
    return (u, d)


def _lr_d1(fwd: float, strike: float, r: float, vol_time: float, t: float, sigma: float, q: float) -> float:
    # Matches OptionPricerUtil::calcD1 overload with volTime/t separation.
    return (math.log(fwd / strike) + t * (r - q) + ((sigma * sigma) * vol_time / 2.0)) / (sigma * math.sqrt(vol_time))


def _lr_call_price_from_fwd(
    fwd: float,
    strike: float,
    sigma: float,
    texp: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    num_steps: int = 101,
) -> float | None:
    if (
        not math.isfinite(fwd)
        or not math.isfinite(strike)
        or not math.isfinite(sigma)
        or not math.isfinite(texp)
        or fwd <= 0.0
        or strike <= 0.0
        or sigma <= 0.0
        or texp <= 0.0
    ):
        return None
    n = int(num_steps)
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    dt = texp / float(n)
    if dt <= 0.0:
        return None

    d1 = _lr_d1(fwd, strike, r, texp, texp, sigma, q)
    d2 = d1 - sigma * math.sqrt(texp)
    p_prime = _pp_peizer_pratt(n, d1)
    p = _pp_peizer_pratt(n, d2)
    u, d = _lr_adj_ud_step(r, q, dt, p, p_prime)
    if not (math.isfinite(u) and math.isfinite(d) and u > 0.0 and d > 0.0):
        return None
    disc = math.exp(-r * dt)

    vals = [0.0] * (n + 1)
    stock = fwd * (u**n)
    for j in range(n + 1):
        vals[j] = max(stock - strike, 0.0)
        stock *= d / u

    for step in range(n - 1, -1, -1):
        stock = fwd * (u**step)
        for j in range(step + 1):
            cont = disc * (p * vals[j] + (1.0 - p) * vals[j + 1])
            exer = max(stock - strike, 0.0)  # LR pricer in CVI path is American by default.
            vals[j] = max(exer, cont)
            stock *= d / u
    return vals[0]


def _lr_call_greeks_from_fwd(
    fwd: float,
    strike: float,
    sigma: float,
    texp: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    num_steps: int = 101,
) -> dict[str, float] | None:
    if (
        not math.isfinite(fwd)
        or not math.isfinite(strike)
        or not math.isfinite(sigma)
        or not math.isfinite(texp)
        or fwd <= 0.0
        or strike <= 0.0
        or sigma <= 0.0
        or texp <= 0.0
    ):
        return None
    base = _lr_call_price_from_fwd(fwd, strike, sigma, texp, r=r, q=q, num_steps=num_steps)
    if base is None:
        return None
    h_f = max(0.01, abs(fwd) * 1e-4)
    h_s = max(1e-4, abs(sigma) * 1e-3)
    s_up = sigma + h_s
    s_dn = max(1e-6, sigma - h_s)
    f_up = fwd + h_f
    f_dn = max(1e-6, fwd - h_f)

    c_fu = _lr_call_price_from_fwd(f_up, strike, sigma, texp, r=r, q=q, num_steps=num_steps)
    c_fd = _lr_call_price_from_fwd(f_dn, strike, sigma, texp, r=r, q=q, num_steps=num_steps)
    c_su = _lr_call_price_from_fwd(fwd, strike, s_up, texp, r=r, q=q, num_steps=num_steps)
    c_sd = _lr_call_price_from_fwd(fwd, strike, s_dn, texp, r=r, q=q, num_steps=num_steps)
    c_fu_su = _lr_call_price_from_fwd(f_up, strike, s_up, texp, r=r, q=q, num_steps=num_steps)
    c_fu_sd = _lr_call_price_from_fwd(f_up, strike, s_dn, texp, r=r, q=q, num_steps=num_steps)
    c_fd_su = _lr_call_price_from_fwd(f_dn, strike, s_up, texp, r=r, q=q, num_steps=num_steps)
    c_fd_sd = _lr_call_price_from_fwd(f_dn, strike, s_dn, texp, r=r, q=q, num_steps=num_steps)
    needed = [c_fu, c_fd, c_su, c_sd, c_fu_su, c_fu_sd, c_fd_su, c_fd_sd]
    if any(v is None for v in needed):
        return None
    c_fu = float(c_fu)
    c_fd = float(c_fd)
    c_su = float(c_su)
    c_sd = float(c_sd)
    c_fu_su = float(c_fu_su)
    c_fu_sd = float(c_fu_sd)
    c_fd_su = float(c_fd_su)
    c_fd_sd = float(c_fd_sd)

    delta = (c_fu - c_fd) / (2.0 * h_f)
    gamma = (c_fu - 2.0 * base + c_fd) / (h_f * h_f)
    vega = (c_su - c_sd) / (s_up - s_dn)
    volga = (c_su - 2.0 * base + c_sd) / (((s_up - s_dn) * 0.5) ** 2)
    vanna = (c_fu_su - c_fu_sd - c_fd_su + c_fd_sd) / (4.0 * h_f * ((s_up - s_dn) * 0.5))
    return {
        "price": float(base),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "vanna": float(vanna),
        "volga": float(volga),
    }


def _lr_call_theta_from_fwd(
    fwd: float,
    strike: float,
    sigma: float,
    texp: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    num_steps: int = 101,
    dt_year: float = 1.0 / (365.0 * 24.0 * 60.0),
) -> float | None:
    """Finite-difference dC/dt where t is elapsed calendar time in years."""
    c_now = _lr_call_price_from_fwd(fwd, strike, sigma, texp, r=r, q=q, num_steps=num_steps)
    if c_now is None:
        return None
    if not math.isfinite(texp) or texp <= 0.0:
        return None
    h = min(float(dt_year), max(texp * 0.5, 1e-8))
    if h <= 0.0:
        return None
    t_next = max(texp - h, 1e-8)
    c_next = _lr_call_price_from_fwd(fwd, strike, sigma, t_next, r=r, q=q, num_steps=num_steps)
    if c_next is None:
        return None
    return float(c_next - c_now) / h


def _market_price_from_row(row: dict[str, float], fwd: float, texp: float) -> tuple[float | None, str]:
    for col in MARKET_PRICE_CANDIDATE_COLUMNS:
        v = _to_float_or_none(row.get(col))
        if v is not None:
            return float(v), col
    for col in MARKET_VOL_CANDIDATE_COLUMNS:
        v = _to_float_or_none(row.get(col))
        if v is None or v <= 0.0:
            continue
        p = _lr_call_price_from_fwd(fwd, float(row["strike"]), v, texp, r=0.0, q=0.0)
        if p is not None:
            return float(p), f"{col}_to_lr_price"
    fv = _to_float_or_none(row.get("fitted_vol"))
    if fv is not None and fv > 0.0:
        p = _lr_call_price_from_fwd(fwd, float(row["strike"]), fv, texp, r=0.0, q=0.0)
        if p is not None:
            return float(p), "fitted_vol_to_lr_price"
    return None, "unavailable"


def _first_positive_from_row(row: dict[str, float | str | None], columns: tuple[str, ...]) -> tuple[float | None, str | None]:
    for col in columns:
        v = _to_float_or_none(row.get(col))
        if v is not None and v > 0.0:
            return v, col
    return None, None


def _lr_call_implied_vol_from_price(
    fwd: float,
    strike: float,
    price: float,
    texp: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    num_steps: int = 101,
) -> float | None:
    if not (
        math.isfinite(fwd)
        and fwd > 0.0
        and math.isfinite(strike)
        and strike > 0.0
        and math.isfinite(price)
        and price > 0.0
        and math.isfinite(texp)
        and texp > 0.0
    ):
        return None
    lo = 1e-6
    hi = 8.0
    p_lo = _lr_call_price_from_fwd(fwd, strike, lo, texp, r=r, q=q, num_steps=num_steps)
    p_hi = _lr_call_price_from_fwd(fwd, strike, hi, texp, r=r, q=q, num_steps=num_steps)
    if p_lo is None or p_hi is None:
        return None
    while price > p_hi and hi < 20.0:
        hi *= 1.5
        p_hi = _lr_call_price_from_fwd(fwd, strike, hi, texp, r=r, q=q, num_steps=num_steps)
        if p_hi is None:
            return None
    if price < p_lo - 1e-8 or price > p_hi + 1e-8:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p_mid = _lr_call_price_from_fwd(fwd, strike, mid, texp, r=r, q=q, num_steps=num_steps)
        if p_mid is None:
            return None
        if p_mid < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if row is None:
            raise SystemExit(f"Required artifact is empty: {path}")
        return row


def _read_vector_csv_values(path: Path) -> list[float]:
    out: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        if "value" in fields:
            col = "value"
        elif "z" in fields:
            col = "z"
        else:
            raise SystemExit(f"Artifact missing expected value column: {path}")
        for row in r:
            v = _to_float_or_none(row.get(col))
            if v is None:
                raise SystemExit(f"Non-numeric {col} in artifact: {path}")
            out.append(float(v))
    return out


def _bspline_basis_degree(knots: list[float], z: float, degree: int) -> list[float]:
    n = len(knots) - degree - 1
    if n <= 0:
        return []
    N = [0.0] * n
    for i in range(n):
        if (knots[i] <= z < knots[i + 1]) or (z == knots[-1] and i == n - 1):
            N[i] = 1.0
    for p in range(1, degree + 1):
        Np = [0.0] * n
        for i in range(n):
            left = 0.0
            den_l = knots[i + p] - knots[i]
            if den_l > 0.0:
                left = ((z - knots[i]) / den_l) * N[i]
            right = 0.0
            if i + 1 < n:
                den_r = knots[i + p + 1] - knots[i + 1]
                if den_r > 0.0:
                    right = ((knots[i + p + 1] - z) / den_r) * N[i + 1]
            Np[i] = left + right
        N = Np
    return N


def _basis_eval_variance_row_in_support(knots: list[float], z: float, num_basis: int) -> list[float]:
    vals = _bspline_basis_degree(knots, z, 3)
    if len(vals) != num_basis:
        raise SystemExit("Basis size mismatch while evaluating variance row.")
    return vals


def _basis_eval_skew_row_in_support(knots: list[float], z: float, v_star: float, num_basis: int) -> list[float]:
    n2 = _bspline_basis_degree(knots, z, 2)
    if len(n2) != num_basis + 1:
        raise SystemExit("Basis size mismatch while evaluating skew row.")
    out = [0.0] * num_basis
    p = 3.0
    inv_v = 1.0 / v_star
    for i in range(num_basis):
        t0 = knots[i]
        t1 = knots[i + 3]
        t2 = knots[i + 1]
        t3 = knots[i + 4]
        a = (p / (t1 - t0)) * n2[i] if (t1 - t0) > 0.0 else 0.0
        b = (p / (t3 - t2)) * n2[i + 1] if (t3 - t2) > 0.0 else 0.0
        out[i] = (a - b) * inv_v
    return out


def _basis_eval_skew_row(knots: list[float], z: float, v_star: float, num_basis: int) -> list[float]:
    z0 = knots[0]
    zn1 = knots[-1]
    zz = z
    if zz < z0:
        zz = z0
    elif zz > zn1:
        zz = zn1
    return _basis_eval_skew_row_in_support(knots, zz, v_star, num_basis)


def _basis_dot(row: list[float], alpha: list[float]) -> float:
    return sum(row[i] * alpha[i] for i in range(len(alpha)))


def _basis_s_atf_norm_from_snapshot(
    snapshot_dir: Path,
    *,
    expiry_index: int,
    v_star: float,
) -> float | None:
    """
    Basis-derived ATF skew for one snapshot/expiry in vol-space normalization.

    Returns None for missing/malformed artifacts so ln-scatter mode can continue.
    """
    if not (math.isfinite(v_star) and v_star > 0.0):
        return None
    dims_p = snapshot_dir / "CVI_dims.csv"
    knot_p = snapshot_dir / "knot_vector.csv"
    xsol_p = snapshot_dir / "x_solution.csv"
    if not (dims_p.is_file() and knot_p.is_file() and xsol_p.is_file()):
        return None
    try:
        dims = _read_single_row_csv(dims_p)
        nb = int(float(dims.get("num_basis", "nan")))
        n_v_orig = int(float(dims.get("n_v_orig", "nan")))
        m = int(float(dims.get("m", "nan")))
        knots = _read_vector_csv_values(knot_p)
        x_values = _read_vector_csv_values(xsol_p)
        if len(knots) < 8:
            return None
        if len(knots) - 4 != nb:
            return None
        if n_v_orig != m * nb:
            return None
        if len(x_values) < n_v_orig:
            return None
        if not (0 <= expiry_index < m):
            return None
        base = expiry_index * nb
        alpha = [float(x_values[base + i]) for i in range(nb)]
        s_row_atf = _basis_eval_skew_row(knots, 0.0, v_star, nb)
        s_basis_raw = _basis_dot(s_row_atf, alpha)  # raw basis: (1/v*) dv/dz
        if not math.isfinite(s_basis_raw):
            return None
        return 0.5 * float(s_basis_raw)  # vol-space normalization
    except Exception:  # noqa: BLE001
        return None


def interp_linear(xs: list[float], ys: list[float], xq: float) -> float | None:
    if len(xs) != len(ys) or not xs:
        return None
    if xq <= xs[0]:
        return ys[0]
    if xq >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if abs(x1 - x0) <= 1e-14:
        return y0
    w = (xq - x0) / (x1 - x0)
    return y0 * (1.0 - w) + y1 * w


def bucket_z(z: float | None) -> str:
    if z is None or not math.isfinite(z):
        return "unknown_z"
    az = abs(z)
    if az <= 0.5:
        return "atf"
    if az <= 1.5:
        return "mid"
    return "wing"


def bucket_abs_dlnf(dlnf: float | None) -> str:
    if dlnf is None or not math.isfinite(dlnf):
        return "unknown_move"
    a = abs(dlnf)
    if a <= 0.002:
        return "small_move"
    if a <= 0.006:
        return "medium_move"
    return "large_move"


def update_sse_bucket(
    table: dict[str, dict[str, float]],
    key: str,
    method: str,
    err: float | None,
) -> None:
    if err is None or not math.isfinite(err):
        return
    row = table.setdefault(key, {"n": 0.0})
    row["n"] += 1.0
    row[f"sse_{method}"] = row.get(f"sse_{method}", 0.0) + err * err


def read_summary(batch_dir: Path, day: str | None = None) -> list[dict]:
    rows: list[dict] = []
    with (batch_dir / "batch_cvi_summary.csv").open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row.get("ok", "0") or 0) != 1:
                continue
            ts = row["timestamp"].strip()
            if day:
                ds = ts.split()[0] if ts else ""
                if ds != day:
                    continue
            rows.append(
                {
                    "subfolder": row["subfolder"].strip(),
                    "idx_in_bin": int(row["idx_in_bin"]),
                    "timestamp": ts,
                }
            )
    return rows


def read_expiry_fwd_q(path: Path) -> tuple[list[tuple[int, str]], dict[int, float], dict[int, float], float]:
    """Returns (ordered (idx,date) list, q_by_idx, F_by_idx, F0)."""
    qmap: dict[int, float] = {}
    fmap: dict[int, float] = {}
    order: list[tuple[int, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["expiry_idx"])
            lab = row.get("expiry_date", "").strip().strip('"')
            order.append((idx, lab))
            qmap[idx] = float(row["q"])
            fmap[idx] = float(row["F"])
    order.sort(key=lambda x: x[0])
    f0 = fmap.get(0, float("nan"))
    return order, qmap, fmap, f0


def read_option_fit(path: Path) -> dict[tuple[int, float], tuple[float, float]]:
    """(expiry, strike) -> (z, fitted_vol)"""
    out: dict[tuple[int, float], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row["expiry_index"])
            k = float(row["strike"])
            z = float(row["z"])
            fv = float(row["fitted_vol"])
            out[(e, k)] = (z, fv)
    return out


@functools.lru_cache(maxsize=4096)
def read_option_fit_by_expiry(path: Path) -> dict[int, list[dict[str, float]]]:
    """expiry -> rows with at least {z, fitted_vol, strike}, plus optional market cols."""
    out: dict[int, list[dict[str, float]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["expiry_index"])
                z = float(row["z"])
                fv = float(row["fitted_vol"])
                k = float(row["strike"])
            except (KeyError, ValueError, TypeError):
                continue
            if math.isfinite(z) and math.isfinite(fv) and math.isfinite(k):
                rec: dict[str, float] = {"z": z, "fitted_vol": fv, "strike": k}
                for c in MARKET_PRICE_CANDIDATE_COLUMNS + MARKET_VOL_CANDIDATE_COLUMNS + BID_VOL_CANDIDATE_COLUMNS + ASK_VOL_CANDIDATE_COLUMNS + ("vega",):
                    v = _to_float_or_none(row.get(c))
                    if v is not None:
                        rec[c] = v
                out[e].append(rec)
    for e in out:
        out[e].sort(key=lambda p: p["z"])
    return out


@functools.lru_cache(maxsize=4096)
def read_price_comparison_by_expiry(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """expiry -> strike-key -> price comparison fields such as fitted_call and market_call_mid."""
    out: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                e = int(row["expiry_index"])
                k = float(row["strike"])
            except (KeyError, ValueError, TypeError):
                continue
            rec: dict[str, float] = {"strike": k}
            for c in (
                "fitted_call",
                "fitted_put",
                "market_call_bid",
                "market_call_ask",
                "market_call_mid",
                "market_put_bid",
                "market_put_ask",
                "market_put_mid",
            ):
                v = _to_float_or_none(row.get(c))
                if v is not None:
                    rec[c] = float(v)
            out[e][fmt_strike_key(k)] = rec
    return out


@functools.lru_cache(maxsize=8192)
def read_expiry_row(path: Path, expiry_index: int) -> dict[str, float] | None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["expiry_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            if idx != expiry_index:
                continue
            try:
                fwd = float(row["F"])
                sigma_star = float(row["sigma_star"])
                vol_time = float(row.get("volTime", "nan"))
                v_star = float(row.get("v_star", "nan"))
                q = float(row.get("q", "nan"))
                r = float(row.get("r", "nan"))
            except (KeyError, ValueError, TypeError):
                return None
            if not (math.isfinite(fwd) and math.isfinite(sigma_star)):
                return None
            out: dict[str, float] = {"F": fwd, "sigma_star": sigma_star}
            if math.isfinite(v_star) and v_star > 0.0:
                out["v_star"] = v_star
            elif sigma_star > 0.0:
                out["v_star"] = sigma_star * sigma_star
            if math.isfinite(vol_time) and vol_time > 0.0:
                out["volTime"] = vol_time
            if math.isfinite(q):
                out["q"] = q
            if math.isfinite(r):
                out["r"] = r
            return out
    return None


def _format_anchor(anchor: dict[str, float] | None) -> dict[str, float | None]:
    if not anchor:
        return {"z": None, "fitted_vol": None, "strike": None}
    return {
        "z": float(anchor["z"]),
        "fitted_vol": float(anchor["fitted_vol"]),
        "strike": float(anchor["strike"]),
    }


def estimate_s_atf_norm_with_details(
    z_vol: list[dict[str, float]], sigma_star: float, z_tol: float = 1e-10
) -> dict[str, float | str | None]:
    """Estimate vol-space normalized ATF skew and return anchors used."""
    out: dict[str, float | str | None] = {
        "s_atf_norm": None,
        "method": None,
        "left_z": None,
        "left_vol": None,
        "left_strike": None,
        "atf_z": None,
        "atf_vol": None,
        "atf_strike": None,
        "right_z": None,
        "right_vol": None,
        "right_strike": None,
    }
    if not z_vol or not math.isfinite(sigma_star) or sigma_star <= 0.0:
        return out

    left = [p for p in z_vol if p["z"] < -z_tol]
    right = [p for p in z_vol if p["z"] > z_tol]
    atf_exact = [p for p in z_vol if abs(p["z"]) <= z_tol]
    atf_nearest = min(z_vol, key=lambda p: abs(p["z"])) if z_vol else None

    left_pt = left[-1] if left else None
    right_pt = right[0] if right else None
    atf_pt = atf_exact[0] if atf_exact else atf_nearest

    for side, pt in [("left", left_pt), ("atf", atf_pt), ("right", right_pt)]:
        m = _format_anchor(pt)
        out[f"{side}_z"] = m["z"]
        out[f"{side}_vol"] = m["fitted_vol"]
        out[f"{side}_strike"] = m["strike"]

    slope = None
    method = None
    if left_pt and right_pt:
        denom = float(right_pt["z"] - left_pt["z"])
        if abs(denom) > 1e-14:
            slope = float(right_pt["fitted_vol"] - left_pt["fitted_vol"]) / denom
            method = "central_lr"
    elif atf_pt and left_pt:
        denom = float(0.0 - left_pt["z"])
        if abs(denom) > 1e-14:
            slope = float(atf_pt["fitted_vol"] - left_pt["fitted_vol"]) / denom
            method = "one_sided_left_atf_exact" if atf_exact else "one_sided_left_atf_nearest"
    elif atf_pt and right_pt:
        denom = float(right_pt["z"] - 0.0)
        if abs(denom) > 1e-14:
            slope = float(right_pt["fitted_vol"] - atf_pt["fitted_vol"]) / denom
            method = "one_sided_atf_right_exact" if atf_exact else "one_sided_atf_right_nearest"

    if slope is None or not math.isfinite(slope):
        return out

    s_norm = slope / sigma_star
    if math.isfinite(s_norm):
        out["s_atf_norm"] = float(s_norm)
        out["method"] = method
    return out


def sigma_star_avg3_nearest_z(z_vol: list[dict[str, float]]) -> dict[str, float | str | None]:
    """Compute sigma* as simple mean of 3 nearest-to-zero z points."""
    out: dict[str, float | str | None] = {
        "sigma_star_used": None,
        "sigma_method": "fallback_expiry_row",
        "sigma_avg3_strike_1": None,
        "sigma_avg3_z_1": None,
        "sigma_avg3_vol_1": None,
        "sigma_avg3_strike_2": None,
        "sigma_avg3_z_2": None,
        "sigma_avg3_vol_2": None,
        "sigma_avg3_strike_3": None,
        "sigma_avg3_z_3": None,
        "sigma_avg3_vol_3": None,
    }
    if not z_vol:
        return out
    nearest = sorted(z_vol, key=lambda p: abs(p["z"]))[:3]
    if len(nearest) < 3:
        return out
    vols = [float(p["fitted_vol"]) for p in nearest]
    if not all(math.isfinite(v) and v > 0.0 for v in vols):
        return out
    out["sigma_star_used"] = sum(vols) / 3.0
    out["sigma_method"] = "avg3_nearest_z"
    for i, p in enumerate(nearest, start=1):
        out[f"sigma_avg3_strike_{i}"] = float(p["strike"])
        out[f"sigma_avg3_z_{i}"] = float(p["z"])
        out[f"sigma_avg3_vol_{i}"] = float(p["fitted_vol"])
    return out


def sigma_star_z0_point(z_vol: list[dict[str, float]], z_tol: float = 1e-10) -> dict[str, float | str | None]:
    """Compute sigma* from z=0 point; fallback to nearest-to-zero point."""
    out: dict[str, float | str | None] = {
        "sigma_star_used": None,
        "sigma_method": "fallback_expiry_row",
        "sigma_z0_strike": None,
        "sigma_z0_z": None,
        "sigma_z0_vol": None,
    }
    if not z_vol:
        return out
    exact = [p for p in z_vol if abs(float(p["z"])) <= z_tol]
    pick = exact[0] if exact else min(z_vol, key=lambda p: abs(float(p["z"])))
    v = float(pick["fitted_vol"])
    if not (math.isfinite(v) and v > 0.0):
        return out
    out["sigma_star_used"] = v
    out["sigma_method"] = "z0_exact" if exact else "z0_nearest"
    out["sigma_z0_strike"] = float(pick["strike"])
    out["sigma_z0_z"] = float(pick["z"])
    out["sigma_z0_vol"] = v
    return out


def pct_change(new: float, old: float) -> float | None:
    if old is None or abs(old) < 1e-14:
        return None
    return 100.0 * (new - old) / old


def fmt_strike_key(k: float) -> str:
    r = round(k, 8)
    if abs(r - round(r)) < 1e-6:
        return str(int(round(r)))
    return f"{k:.10g}"


def parse_snapshot_ts(ts: str) -> datetime | None:
    """
    Parse snapshot timestamp like:
      2026-04-07 09:30:34-04
    into naive local datetime (timezone part ignored for intraday rolling windows).
    """
    s = (ts or "").strip()
    m = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        return None
    y, mo, d = map(int, m.group(1).split("-"))
    hh, mm, ss = int(m.group(2)), int(m.group(3)), int(m.group(4))
    return datetime(y, mo, d, hh, mm, ss)


def _rth_open_two_hour_allows(t: datetime) -> bool:
    """First two RTH hours from 09:30 — ``[09:30, 11:30)`` on naive clock."""
    sec = t.hour * 3600 + t.minute * 60 + t.second
    open_lo = 9 * 3600 + 30 * 60
    open_hi = 11 * 3600 + 30 * 60
    return open_lo <= sec < open_hi


def _close_session_feasible_two_hour_allows(
    t_entry: datetime,
    *,
    t_last: datetime,
    horizon_min: float,
) -> bool:
    """
    **Closing** session entries: last ~2 clock hours of *feasible* hedge starts on this day,
    capped at noon from below: ``max(noon, t_last − horizon − 2h) ≤ t_entry ≤ t_last − horizon``.

    This avoids the empty intersection of a fixed 14:00–16:00 entry band with a 120m horizon when
    the batch ends near 16:00 (no snapshot at entry+120m).
    """
    dl = t_last - timedelta(minutes=float(horizon_min))
    if t_entry > dl:
        return False
    band_start = dl - timedelta(hours=2)
    noon = datetime(t_entry.year, t_entry.month, t_entry.day, 12, 0, 0)
    lo = max(band_start, noon)
    return lo <= t_entry <= dl


def wls_weights_inverse_bivariate_move_sq(xs: list[float], ys: list[float]) -> list[float]:
    """Weights ∝ 1/(ε + x² + y²) so large |ΔS| and |Δσ| together get less influence."""
    eps = 1e-8
    return [1.0 / (eps + xi * xi + yi * yi) for xi, yi in zip(xs, ys)]


def wls_alpha_beta(
    xs: list[float], ys: list[float], weights: list[float] | None = None
) -> tuple[float, float, int] | None:
    """Return (alpha, beta, n) for WLS fit Y ~ alpha + beta*X. Uniform weights => OLS."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            return None
        w = [max(weights[i], 1e-15) for i in range(n)]
    sw = sum(w)
    if sw <= 0:
        return None
    mx = sum(w[i] * xs[i] for i in range(n)) / sw
    my = sum(w[i] * ys[i] for i in range(n)) / sw
    sxx = sum(w[i] * (xs[i] - mx) ** 2 for i in range(n))
    if sxx <= 1e-18:
        return None
    sxy = sum(w[i] * (xs[i] - mx) * (ys[i] - my) for i in range(n))
    beta = sxy / sxx
    alpha = my - beta * mx
    return (alpha, beta, n)


def wls_beta_origin(
    xs: list[float], ys: list[float], weights: list[float] | None = None
) -> tuple[float, int] | None:
    """Return (beta, n) for origin-constrained fit Y ~ beta*X."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    if weights is None:
        w = [1.0] * n
    else:
        if len(weights) != n:
            return None
        w = [max(float(weights[i]), 1e-15) for i in range(n)]
    sxx = sum(w[i] * xs[i] * xs[i] for i in range(n))
    if sxx <= 1e-18:
        return None
    sxy = sum(w[i] * xs[i] * ys[i] for i in range(n))
    beta = sxy / sxx
    if not math.isfinite(beta):
        return None
    return (beta, n)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    m = len(arr) // 2
    if len(arr) % 2 == 1:
        return float(arr[m])
    return 0.5 * float(arr[m - 1] + arr[m])


def robust_beta_origin_irls(
    xs: list[float],
    ys: list[float],
    *,
    base_weights: list[float] | None = None,
    method: str = "huber",
    huber_c: float = 1.345,
    max_iter: int = 40,
    tol: float = 1e-10,
) -> dict[str, float | int | bool] | None:
    """
    Robust origin-constrained slope using IRLS.
    method:
      - huber: psi(u)=u if |u|<=1 else sign(u), with u=r/(c*scale)
      - lad: L1 approximation via w_r = 1/max(eps, |r|)
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    if base_weights is None:
        wb = [1.0] * n
    else:
        if len(base_weights) != n:
            return None
        wb = [max(float(base_weights[i]), 1e-15) for i in range(n)]
    init = wls_beta_origin(xs, ys, wb)
    if init is None:
        return None
    beta = float(init[0])
    converged = False
    iters = 0
    eps = 1e-12
    for it in range(max_iter):
        iters = it + 1
        residuals = [ys[i] - beta * xs[i] for i in range(n)]
        robust_w: list[float] = [1.0] * n
        if method == "huber":
            med = _median(residuals)
            if med is None:
                return None
            mad = _median([abs(r - med) for r in residuals])
            scale = max(1.4826 * float(mad or 0.0), 1e-8)
            denom = huber_c * scale
            for i, r in enumerate(residuals):
                u = abs(r) / max(denom, eps)
                robust_w[i] = 1.0 if u <= 1.0 else (1.0 / u)
        elif method == "lad":
            for i, r in enumerate(residuals):
                robust_w[i] = 1.0 / max(eps, abs(r))
        else:
            return None
        w_eff = [wb[i] * robust_w[i] for i in range(n)]
        nxt = wls_beta_origin(xs, ys, w_eff)
        if nxt is None:
            return None
        beta_new = float(nxt[0])
        if abs(beta_new - beta) <= tol * max(1.0, abs(beta)):
            beta = beta_new
            converged = True
            break
        beta = beta_new
    return {
        "beta": beta,
        "n": n,
        "iters": iters,
        "converged": converged,
    }


_EXP_ARG_CLAMP = 700.0  # math.exp(709) overflows; clamp predicted Δlnσ contributions


def _sigma_prev_times_exp_dln(sigma_prev: float, dln: float) -> float:
    """sigma_prev * exp(dln) with dln clamped so exp() does not overflow."""
    if not math.isfinite(sigma_prev) or not math.isfinite(dln):
        return float("nan")
    d = max(-_EXP_ARG_CLAMP, min(_EXP_ARG_CLAMP, float(dln)))
    return float(sigma_prev) * math.exp(d)


def _dsigma_from_sigma0_dln_sigma(sigma0: float, dln_sigma: float) -> float:
    """Δσ = σ₀(exp(Δlnσ)−1) with clamped exponent (ATM hedge Δσ̂ path)."""
    if not math.isfinite(sigma0) or not math.isfinite(dln_sigma):
        return float("nan")
    d = max(-_EXP_ARG_CLAMP, min(_EXP_ARG_CLAMP, float(dln_sigma)))
    return float(sigma0) * (math.exp(d) - 1.0)


def theil_sen_beta_pairwise(xs: list[float], ys: list[float]) -> float | None:
    """
    Theil–Sen slope through the origin: median of (y_j - y_i) / (x_j - x_i) over pairs
    with distinct x. Unweighted — large-move pairs contribute slopes at full strength.

    ``sklearn.linear_model.TheilSenRegressor`` is related but uses subsampling /
    subpopulation limits and can differ numerically; this matches the textbook
    pairwise-median definition (feasible for typical 1h window sizes here).
    """
    n = len(xs)
    if n < 2 or len(ys) != n:
        return None
    slopes: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = float(xs[j]) - float(xs[i])
            if abs(dx) <= 1e-18:
                continue
            dy = float(ys[j]) - float(ys[i])
            s = dy / dx
            if math.isfinite(s):
                slopes.append(s)
    if not slopes:
        return None
    return float(statistics.median(slopes))

