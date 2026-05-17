#!/usr/bin/env python3
"""
CVI batch — compare raw q/F vs DP-Cauchy filtered q and implied F.

Uses the Dynamic Programming MAP estimator from Yoosefian & Lessard (2025)
specialized to Cauchy measurement noise (Eq. 14):

  Scalar system:  x_t = x_{t-1} + w_t,   y_t = x_t + v_t
  Process noise:  w_t ~ N(0, Q_w)
  Measurement:    v_t ~ Cauchy(0, gamma)

  r(v) = log(v^2 + gamma^2)
  score  = nabla r(v_bar) = 2*v_bar / (v_bar^2 + gamma^2)
  M_r    = score / v_bar  = 2 / (v_bar^2 + gamma^2)      [Eq. 12]

  Time update:
    P_{t|t-1} = P_{t-1} + Q_w                              [Eq. 14a]
    mu_{t|t-1} = mu_{t-1}                                   [Eq. 14b, A=1, mu_w=0]

  Measurement update:
    P_t^{-1} = P_{t|t-1}^{-1} + M_r                        [Eq. 14c]
    mu_t = mu_{t|t-1} + P_t * score                         [Eq. 14d]

Outputs a standalone Plotly HTML (CDN) with 4 plots:
  1) raw q vs time (all expiries)
  2) raw F vs time (all expiries)
  3) DP-Cauchy filtered q_hat vs time (all expiries)
  4) implied F_hat per snapshot: F_hat = S_raw * exp((r_raw - q_hat) * volTime),
     with S_raw from F_raw = S_raw * exp((r_raw - q_raw) * volTime) (row r, volTime from CSV)

Usage:
  python plot_cvi_batch_qf_cauchy_compare.py <batch_dir> [out.html]
    Default out: <batch_dir>/q_f_cauchy_compare.html
  Gamma calibration (on-the-fly in this script; JSON from calibrate_cauchy_params.py
  uses MAD-based Q_w/gamma — different path):
    --gamma-mode k_star (default)   gamma from target K* and Q_w
    --gamma-mode mad_legacy         legacy gamma = σ_pct/1.4826 (MAD-equivalent width)
    Q_w from Gaussian-equiv σ(Δq) via percentile spread (default P25–P75).
"""

from __future__ import annotations

import csv
import base64
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from .q_cauchy.filter import dp_cauchy_filter
from .q_cauchy.repricing import implied_forward_f_hat_series
from .q_cauchy.inline_params import (
    gamma_from_k_star_gain,
    gamma_mad_legacy_ratio,
    Q_w_from_percentile_delta_q,
)
from .q_cauchy.time_series import calendar_date, insert_null_between_calendar_days


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(
    batch_dir: Path,
    k_star: float = 0.6,
    gamma_mode: str = "k_star",
    *,
    level_shift_theta: float | None = 10.0,
    level_shift_n_streak: int = 3,
    dq_pct_lo: float = 25.0,
    dq_pct_hi: float = 75.0,
) -> dict:
    return build_payload_for_day(
        batch_dir,
        only_day=None,
        k_star=k_star,
        gamma_mode=gamma_mode,
        level_shift_theta=level_shift_theta,
        level_shift_n_streak=level_shift_n_streak,
        dq_pct_lo=dq_pct_lo,
        dq_pct_hi=dq_pct_hi,
    )


def build_payload_for_day(
    batch_dir: Path,
    only_day: str | None,
    k_star: float = 0.6,
    gamma_mode: str = "k_star",
    *,
    level_shift_theta: float | None = 10.0,
    level_shift_n_streak: int = 3,
    dq_pct_lo: float = 25.0,
    dq_pct_hi: float = 75.0,
) -> dict:
    summary_path = batch_dir / "batch_cvi_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    by_exp: dict[str, dict[str, list]] = defaultdict(
        lambda: {"t": [], "q": [], "F": [], "volTime": [], "r": []}
    )
    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))

    for row in rows:
        sub = batch_dir / (row.get("subfolder") or "").strip()
        efq = sub / "expiry_fwd_q.csv"
        if not efq.is_file():
            continue
        ts = (row.get("timestamp") or "").strip()
        if only_day is not None and calendar_date(ts) != only_day:
            continue
        with efq.open(newline="", encoding="utf-8") as ef:
            for erow in csv.DictReader(ef):
                eidx = int(erow["expiry_idx"])
                d = (erow.get("expiry_date") or "").strip().strip('"')
                lab = f"{eidx}: {d}"
                qv = float(erow["q"])
                fv = float(erow["F"])
                vt = float(erow.get("volTime") or 0.0)
                rv = float(erow.get("r") or 0.0)
                if not (
                    math.isfinite(qv)
                    and math.isfinite(fv)
                    and math.isfinite(vt)
                    and math.isfinite(rv)
                ):
                    continue
                by_exp[lab]["t"].append(ts)
                by_exp[lab]["q"].append(qv)
                by_exp[lab]["F"].append(fv)
                by_exp[lab]["volTime"].append(vt)
                by_exp[lab]["r"].append(rv)

    exp_labels = sorted(by_exp.keys(), key=lambda s: int(s.split(":", 1)[0]))

    by_exp_out: dict[str, dict[str, list]] = {}
    for lab in exp_labels:
        raw = by_exp[lab]

        q_vals = [float(x) for x in raw["q"]]

        # Q_w from percentile σ(Δq); gamma legacy ratio or K* (see q_cauchy.inline_params)
        Q_w = Q_w_from_percentile_delta_q(
            q_vals, dq_pct_lo=dq_pct_lo, dq_pct_hi=dq_pct_hi
        )
        if gamma_mode == "mad_legacy":
            gamma = gamma_mad_legacy_ratio(
                q_vals, dq_pct_lo=dq_pct_lo, dq_pct_hi=dq_pct_hi
            )
        else:
            gamma = gamma_from_k_star_gain(Q_w, k_star)

        # Run DP-MAP Cauchy filter (Eq. 14)
        q_hat = dp_cauchy_filter(
            q_vals,
            Q_w=Q_w,
            gamma=gamma,
            level_shift_theta=level_shift_theta,
            level_shift_n_streak=level_shift_n_streak,
        )

        F_hat = implied_forward_f_hat_series(
            raw["F"], raw["q"], raw["r"], raw["volTime"], q_hat
        )

        t2, q2, f2 = insert_null_between_calendar_days(
            raw["t"], raw["q"], raw["F"]
        )
        _t3, qh2, fh2 = insert_null_between_calendar_days(
            raw["t"],
            [float(x) if x is not None else float("nan") for x in q_hat],
            F_hat,
        )

        by_exp_out[lab] = {
            "t": t2,
            "q": q2,
            "F": f2,
            "q_cauchy": qh2,
            "F_cauchy": fh2,
        }

    return {
        "batch_name": batch_dir.name,
        "exp_labels": exp_labels,
        "by_exp": by_exp_out,
    }


# ---------------------------------------------------------------------------
# HTML rendering (unchanged except title tweaks)
# ---------------------------------------------------------------------------

def _scrub_non_finite(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_non_finite(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _payload_b64(payload: dict) -> str:
    scrubbed = _scrub_non_finite(payload)
    raw = json.dumps(scrubbed, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return base64.b64encode(raw.encode("utf-8")).decode("ascii")


def _b64_chunk_array_literal(b64: str, chunk_size: int = 16384) -> str:
    parts = [b64[i : i + chunk_size] for i in range(0, len(b64), chunk_size)]
    lines = ["  " + json.dumps(part, ensure_ascii=True) for part in parts]
    return "[\n" + ",\n".join(lines) + "\n]"


def render_html(payload: dict, title: str) -> str:
    b64 = _payload_b64(payload)
    b64_parts_js = _b64_chunk_array_literal(b64)
    tpl = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>__TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 12px; }
    h1 { font-size: 1.05rem; margin: 0 0 10px 0; }
    #boot-err { color: #b00020; font-weight: 600; margin-bottom: 10px; white-space: pre-wrap; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .plot { width: 100%; height: 420px; border: 1px solid #e0e0e0; border-radius: 6px; }
    @media(max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<h1>__TITLE__</h1>
<div id="boot-err"></div>
<div class="grid">
  <div id="g1" class="plot"></div>
  <div id="g2" class="plot"></div>
  <div id="g3" class="plot"></div>
  <div id="g4" class="plot"></div>
</div>
<script>
const _B64_PARTS = __B64_PARTS__;
function _decodePayload() {
  const b64 = _B64_PARTS.join('');
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return JSON.parse(new TextDecoder('utf-8').decode(bytes));
}

let DATA;
try {
  DATA = _decodePayload();
} catch (e) {
  document.getElementById('boot-err').textContent = 'Failed to decode payload: ' + (e && e.message ? e.message : e);
  throw e;
}

(function() {
  const err = document.getElementById('boot-err');
  if (typeof Plotly === 'undefined') { err.textContent = 'Plotly failed to load.'; return; }
  const labels = DATA.exp_labels || [];
  const byExp = DATA.by_exp || {};
  const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
  const TRADING_RB = [
    { bounds: ['sat','mon'], pattern: 'day of week' },
    { bounds: [16, 9.5], pattern: 'hour' },
  ];

  function mkTraces(keyY) {
    return labels.map((lab, i) => ({
      type: 'scatter',
      mode: 'lines+markers',
      name: lab,
      x: byExp[lab].t,
      y: byExp[lab][keyY],
      connectgaps: false,
      line: { width: 1.4, color: colors[i % colors.length] },
      marker: { size: 3.5 },
    }));
  }

  const baseX = { title: 'timestamp', type: 'date', rangebreaks: TRADING_RB, tickangle: -45 };
  const leg = { orientation: 'v', x: 1.02, y: 1, font: { size: 9 } };
  const m = { r: 200, b: 120, t: 44, l: 60 };
  const CFG = {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: { format: 'png', filename: 'q_f_cauchy_compare', scale: 2 },
  };

  try {
    Plotly.newPlot('g1', mkTraces('q'), {
      title: 'q (raw) vs snapshot time', xaxis: baseX, yaxis: { title: 'q' },
      legend: leg, margin: m, hovermode: 'closest'
    }, CFG);
    Plotly.newPlot('g2', mkTraces('F'), {
      title: 'Forward F (raw) vs snapshot time', xaxis: baseX, yaxis: { title: 'F' },
      legend: leg, margin: m, hovermode: 'closest'
    }, CFG);
    Plotly.newPlot('g3', mkTraces('q_cauchy'), {
      title: 'q\\u0302 (DP-MAP Cauchy, Eq. 14) vs snapshot time', xaxis: baseX, yaxis: { title: 'q_hat' },
      legend: leg, margin: m, hovermode: 'closest'
    }, CFG);
    Plotly.newPlot('g4', mkTraces('F_cauchy'), {
      title: 'F\\u0302 = S_raw * exp((r_raw - q\\u0302)*volTime) vs snapshot time', xaxis: baseX, yaxis: { title: 'F_hat' },
      legend: leg, margin: m, hovermode: 'closest'
    }, CFG);
  } catch (e) {
    err.textContent = 'Plot error: ' + (e && e.message ? e.message : e);
  }
})();
</script>
</body>
</html>
"""
    return tpl.replace("__TITLE__", title).replace("__B64_PARTS__", b64_parts_js)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("out", nargs="?", default=None, help="Output HTML path (optional)")
    ap.add_argument("--day", dest="only_day", default=None, help="Filter to YYYY-MM-DD")
    ap.add_argument(
        "--gamma-mode",
        choices=("k_star", "mad_legacy"),
        default="k_star",
        help="gamma from K*+Q_w (default) or legacy sigma_pct/1.4826.",
    )
    ap.add_argument(
        "--k-star",
        type=float,
        default=0.6,
        dest="k_star",
        help="Used when --gamma-mode k_star. Default 0.6.",
    )
    ap.add_argument(
        "--no-level-shift",
        action="store_true",
        help="Disable consecutive large-innovation reset (default: theta=10, n=3).",
    )
    ap.add_argument(
        "--level-shift-theta",
        type=float,
        default=10.0,
        help="Innovation threshold in gamma units (default 10). Ignored with --no-level-shift.",
    )
    ap.add_argument(
        "--level-shift-n",
        type=int,
        default=3,
        dest="level_shift_n",
        help="Consecutive ticks above threshold to trigger reset (default 3).",
    )
    ap.add_argument(
        "--dq-pct-lo",
        type=float,
        default=25.0,
        dest="dq_pct_lo",
        help="Lower percentile of delta-q for Q_w / sigma scale (default 25).",
    )
    ap.add_argument(
        "--dq-pct-hi",
        type=float,
        default=75.0,
        dest="dq_pct_hi",
        help="Upper percentile of delta-q (default 75).",
    )
    args = ap.parse_args()

    if args.gamma_mode == "k_star" and not (0.0 < args.k_star < 1.0):
        raise SystemExit("--k-star must be strictly between 0 and 1 for k_star mode")
    dq_lo = float(args.dq_pct_lo)
    dq_hi = float(args.dq_pct_hi)
    if not (0.0 < dq_lo < dq_hi < 100.0):
        raise SystemExit("--dq-pct-lo and --dq-pct-hi must satisfy 0 < lo < hi < 100")

    batch_dir = args.batch_dir.resolve()
    only_day = (args.only_day or "").strip() or None
    k_star = float(args.k_star)
    gamma_mode = str(args.gamma_mode)
    default_name = (
        "q_f_cauchy_compare.html"
        if not only_day
        else f"q_f_cauchy_compare_{only_day}.html"
    )
    out = Path(args.out).resolve() if args.out else (batch_dir / default_name)

    ls_theta = None if args.no_level_shift else float(args.level_shift_theta)
    payload = build_payload_for_day(
        batch_dir,
        only_day=only_day,
        k_star=k_star,
        gamma_mode=gamma_mode,
        level_shift_theta=ls_theta,
        level_shift_n_streak=int(args.level_shift_n),
        dq_pct_lo=dq_lo,
        dq_pct_hi=dq_hi,
    )
    gtag = f"K*={k_star}" if gamma_mode == "k_star" else "gamma=legacy sigma_pct/1.4826"
    title = f"q/F raw vs DP-Cauchy ({gtag}) — {batch_dir.name}" + (
        f" — {only_day}" if only_day else ""
    )
    out.write_text(render_html(payload, title), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"file:///{str(out).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()