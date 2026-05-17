#!/usr/bin/env python3
"""
calibrate_cauchy_params.py — Estimate DP-Cauchy filter parameters.

Three modes:
  1. Single-day:  --day 2026-04-06
  2. Multi-day pooled (geometric mean):  --days 2026-04-06,2026-04-07
  3. From existing JSON files:  --pool-jsons params_day1.json params_day2.json

The multi-day modes compute per-expiry geometric means of gamma and Q_w
across calibration days, then apply the K_ss sanity check.

Single-day calibration (per expiry):
  Let d* = median(Δq), abs_dev = |Δq − d*|.
  gamma = quantile(abs_dev, p) with p=50 → classic MAD; p=75 → upper-quartile abs deviation.
  Q_w   = (1.4826 * gamma)², floored by (q_range * QW_FLOOR_FACTOR)²

Usage:
  # Single day
  python calibrate_cauchy_params.py <batch_dir> --day 2026-04-06

  # Multi-day pooled from raw data
  python calibrate_cauchy_params.py <batch_dir> --days 2026-04-06,2026-04-07

  # Pool from existing JSON param files
  python calibrate_cauchy_params.py --pool-jsons params_04-06.json params_04-07.json

  # All modes accept: [-o output.json] [--kss-target 2.0] [--kss-range 1.0,3.0]

Legacy (percentile σ on Δq) for plot_cvi_batch_qf_cauchy_compare.py:
  robust_sigma_Q_w_from_q(...) → (sigma_dq, Q_w, q_range, n, dq_stdev_sample)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from cvi_batch_analysis.q_cauchy.delta_q_stats import (
    MIN_OBS,
    percentile_linear,
    robust_sigma_Q_w_from_q,
    sigma_dq_from_dq_percentiles,
)
from cvi_batch_analysis.q_cauchy.pool import (
    KSS_HI,
    KSS_LO,
    KSS_TARGET,
    pool_params_geometric,
    steady_state_kalman_gain,
)

# ─── Constants ─────────────────────────────────────────────────────────────

MAD_TO_SIGMA = 1.4826  # MAD → Gaussian-equivalent σ
QW_FLOOR_FACTOR = 0.003  # Q_w floor: (q_range * factor)²


# ─── Helpers ───────────────────────────────────────────────────────────────


def _calendar_date(timestamp: str) -> str:
    ts = timestamp.strip()
    return ts.split()[0] if ts else ""


# ─── Single-day calibration ───────────────────────────────────────────────


def _robust_Q_w_gamma(
    q_vals: list[float],
    *,
    abs_dev_quantile_pct: float = 50.0,
) -> dict | None:
    """Compute Q_w and gamma from a q series using a percentile of |Δq − median(Δq)|.

    gamma = quantile(|Δq − median(Δq)|, abs_dev_quantile_pct); default 50 is MAD.
    Q_w   = (1.4826 * gamma)² with range floor (same σ coupling as the MAD default).

    With median abs-dev and no floor binding, K_ss = sqrt(2*Q_w)/gamma ≈ 2.097.
    """
    n = len(q_vals)
    if n < MIN_OBS:
        return None

    dq = [q_vals[i] - q_vals[i - 1] for i in range(1, n)]

    med = statistics.median(dq)
    abs_devs = [abs(d - med) for d in dq]
    scale = percentile_linear(abs_devs, abs_dev_quantile_pct) if abs_devs else 0.0

    gamma = max(scale, 1e-8)

    sigma_dq = scale * MAD_TO_SIGMA
    Q_w = sigma_dq**2

    q_range = max(q_vals) - min(q_vals)
    Q_w = max(Q_w, (q_range * QW_FLOOR_FACTOR) ** 2)

    q_mean = statistics.fmean(q_vals)

    return {
        "Q_w": Q_w,
        "gamma": gamma,
        "n_obs": n,
        "q_mean": round(q_mean, 8),
        "q_std_mad": round(sigma_dq, 8),
        "dq_mad": round(scale, 10),
        "q_range": round(q_range, 8),
    }


# ─── Data loading ─────────────────────────────────────────────────────────


def load_day_data(batch_dir: Path, day: str) -> dict[str, list[float]]:
    """Load per-expiry-date q series for a single day from the batch."""
    summary_path = batch_dir / "batch_cvi_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    by_expiry_date: dict[str, list[float]] = defaultdict(list)

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))

    for row in rows:
        ts = (row.get("timestamp") or "").strip()
        if _calendar_date(ts) != day:
            continue
        sub = batch_dir / (row.get("subfolder") or "").strip()
        efq = sub / "expiry_fwd_q.csv"
        if not efq.is_file():
            continue
        with efq.open(newline="", encoding="utf-8") as ef:
            for erow in csv.DictReader(ef):
                qv = float(erow["q"])
                exp_date = (erow.get("expiry_date") or "").strip().strip('"')[:10]
                if math.isfinite(qv) and exp_date:
                    by_expiry_date[exp_date].append(qv)

    return dict(by_expiry_date)


def calibrate_single_day(
    batch_dir: Path,
    day: str,
    *,
    abs_dev_quantile_pct: float = 50.0,
) -> dict:
    """Calibrate all expiries for a single day. Returns JSON-ready dict."""
    data = load_day_data(batch_dir, day)
    if not data:
        raise SystemExit(f"No data found for {day} in {batch_dir}")

    by_expiry: dict[str, dict] = {}
    all_Qw: list[float] = []
    all_gamma: list[float] = []

    for exp_date in sorted(data.keys()):
        q_vals = data[exp_date]
        result = _robust_Q_w_gamma(q_vals, abs_dev_quantile_pct=abs_dev_quantile_pct)
        if result is None:
            print(f"  {exp_date}: only {len(q_vals)} obs, skipping", file=sys.stderr)
            continue
        by_expiry[exp_date] = result
        all_Qw.append(result["Q_w"])
        all_gamma.append(result["gamma"])
        print(
            f"  {exp_date}: n={result['n_obs']:>4d}  "
            f"Q_w={result['Q_w']:.2e}  gamma={result['gamma']:.2e}  "
            f"K_ss={steady_state_kalman_gain(result['Q_w'], result['gamma']):.3f}",
            file=sys.stderr,
        )

    if not all_Qw:
        raise SystemExit("No expiries with enough data to calibrate")

    default_params = {
        "Q_w": float(statistics.median(all_Qw)),
        "gamma": float(statistics.median(all_gamma)),
    }

    return {
        "calibration_day": day,
        "batch_dir": str(batch_dir),
        "dq_abs_dev_quantile_pct": float(abs_dev_quantile_pct),
        "default": default_params,
        "by_expiry": by_expiry,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Calibrate DP-Cauchy filter params (single-day or multi-day pooled)"
    )
    ap.add_argument(
        "batch_dir",
        type=Path,
        nargs="?",
        default=None,
        help="CVI batch directory (required for --day/--days)",
    )
    ap.add_argument("--day", default=None, help="Single calibration day: YYYY-MM-DD")
    ap.add_argument(
        "--days",
        default=None,
        help="Comma-separated calibration days for pooling: YYYY-MM-DD,YYYY-MM-DD",
    )
    ap.add_argument(
        "--pool-jsons",
        nargs="+",
        default=None,
        help="Pool from existing JSON param files",
    )
    ap.add_argument("-o", "--output", default=None, help="Output JSON path")
    ap.add_argument(
        "--kss-target",
        type=float,
        default=KSS_TARGET,
        help=f"K_ss target when clamping (default: {KSS_TARGET})",
    )
    ap.add_argument(
        "--kss-range",
        default=f"{KSS_LO},{KSS_HI}",
        help=f"K_ss acceptable range lo,hi (default: {KSS_LO},{KSS_HI})",
    )
    ap.add_argument(
        "--mad-abs-quantile",
        type=float,
        default=50.0,
        metavar="PCT",
        dest="mad_abs_quantile",
        help=(
            "Percentile of |Δq−median(Δq)| for gamma/Q_w (default 50 = MAD). "
            "Example: 75 for 75th-percentile absolute deviation."
        ),
    )
    args = ap.parse_args()

    kss_lo, kss_hi = (float(x.strip()) for x in args.kss_range.split(","))
    mad_q = float(args.mad_abs_quantile)
    if not (0.0 < mad_q < 100.0):
        ap.error("--mad-abs-quantile must be strictly between 0 and 100")

    def _dump(path: Path, obj: dict) -> None:
        path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )

    if args.pool_jsons:
        param_dicts = []
        for jp in args.pool_jsons:
            p = Path(jp)
            with p.open(encoding="utf-8") as f:
                param_dicts.append(json.load(f))
            print(f"  Loaded {p}", file=sys.stderr)

        output = pool_params_geometric(
            param_dicts, kss_lo=kss_lo, kss_hi=kss_hi, kss_target=args.kss_target
        )

        out_path = (
            Path(args.output).resolve()
            if args.output
            else Path("cauchy_params_pooled.json").resolve()
        )
        _dump(out_path, output)
        print(f"\nWrote {out_path}", file=sys.stderr)
        print(
            f"  Pooled {len(output['by_expiry'])} expiries from {len(param_dicts)} days",
            file=sys.stderr,
        )
        print(f"  Days: {output['calibration_days']}", file=sys.stderr)
        n_clamped = sum(1 for v in output["by_expiry"].values() if v.get("K_ss_clamped"))
        if n_clamped:
            print(
                f"  {n_clamped} expiries had K_ss clamped to {args.kss_target}",
                file=sys.stderr,
            )
        return

    if args.batch_dir is None:
        ap.error("batch_dir is required for --day/--days modes")
    batch_dir = args.batch_dir.resolve()

    if args.days:
        days = [d.strip() for d in args.days.split(",") if d.strip()]
        print(f"Calibrating {len(days)} days: {days}", file=sys.stderr)

        param_dicts = []
        for day in days:
            print(f"\n--- {day} ---", file=sys.stderr)
            pd_ = calibrate_single_day(
                batch_dir, day, abs_dev_quantile_pct=mad_q
            )
            param_dicts.append(pd_)

        output = pool_params_geometric(
            param_dicts, kss_lo=kss_lo, kss_hi=kss_hi, kss_target=args.kss_target
        )

        day_tag = "_".join(days)
        out_path = (
            Path(args.output).resolve()
            if args.output
            else batch_dir / f"cauchy_params_pooled_{day_tag}.json"
        )
        _dump(out_path, output)
        print(f"\nWrote {out_path}", file=sys.stderr)
        print(
            f"  Pooled {len(output['by_expiry'])} expiries from {len(days)} days",
            file=sys.stderr,
        )
        n_clamped = sum(1 for v in output["by_expiry"].values() if v.get("K_ss_clamped"))
        if n_clamped:
            print(
                f"  {n_clamped} expiries had K_ss clamped to {args.kss_target}",
                file=sys.stderr,
            )
        return

    if args.day:
        day = args.day.strip()
        output = calibrate_single_day(
            batch_dir, day, abs_dev_quantile_pct=mad_q
        )

        out_path = (
            Path(args.output).resolve()
            if args.output
            else batch_dir / f"cauchy_params_{day}.json"
        )
        _dump(out_path, output)
        print(f"\nWrote {out_path}", file=sys.stderr)
        print(f"  {len(output['by_expiry'])} expiries calibrated", file=sys.stderr)
        print(
            f"  default Q_w={output['default']['Q_w']:.2e}  "
            f"gamma={output['default']['gamma']:.2e}",
            file=sys.stderr,
        )
        return

    ap.error("Specify --day, --days, or --pool-jsons")


if __name__ == "__main__":
    main()
