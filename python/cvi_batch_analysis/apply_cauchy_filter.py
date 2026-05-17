#!/usr/bin/env python3
"""
apply_cauchy_filter.py — Apply DP-Cauchy filter using pre-calibrated parameters.

Reads parameters from a JSON file (produced by calibrate_cauchy_params.py)
and applies the filter to a target day's data. No fitting to the target day.

Parameter lookup order for each expiry:
  1. Exact expiry_date match in params["by_expiry"]
  2. params["default"] (median across calibration-day expiries)

Usage:
  python apply_cauchy_filter.py <batch_dir> --day 2026-04-07 --params cauchy_params_2026-04-06.json [-o out.html]

Output: Plotly HTML with 5 panels (raw q, raw F, filtered q_hat, implied F_hat, spot S).
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from .q_cauchy.filter import dp_cauchy_filter
from .q_cauchy.repricing import implied_forward_f_hat_series
from .q_cauchy.time_series import calendar_date, insert_null_between_calendar_days


def build_payload_for_day(
    batch_dir: Path,
    target_day: str,
    params: dict,
    *,
    level_shift_theta: float | None = 10.0,
    level_shift_n_streak: int = 3,
) -> dict:
    summary_path = batch_dir / "batch_cvi_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    default_Qw = float(params["default"]["Q_w"])
    default_gamma = float(params["default"]["gamma"])
    by_expiry_params = params.get("by_expiry", {})

    by_exp: dict[str, dict[str, list]] = defaultdict(
        lambda: {"t": [], "q": [], "F": [], "volTime": [], "r": [], "expiry_date": ""}
    )
    spot_t: list[str] = []
    spot_s: list[float] = []

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))

    for row in rows:
        ts = (row.get("timestamp") or "").strip()
        if calendar_date(ts) != target_day:
            continue
        sub = batch_dir / (row.get("subfolder") or "").strip()
        efq = sub / "expiry_fwd_q.csv"
        if not efq.is_file():
            continue
        with efq.open(newline="", encoding="utf-8") as ef:
            erows = list(csv.DictReader(ef))
        if not erows:
            continue
        er0 = erows[0]
        try:
            q0 = float(er0["q"])
            f0 = float(er0["F"])
            vt0 = float(er0.get("volTime") or 0.0)
            r0 = float(er0.get("r") or 0.0)
        except (KeyError, ValueError, TypeError):
            q0 = f0 = vt0 = r0 = float("nan")
        if (
            math.isfinite(q0)
            and math.isfinite(f0)
            and math.isfinite(vt0)
            and vt0 > 0.0
            and math.isfinite(r0)
        ):
            s0 = f0 * math.exp(-(r0 - q0) * vt0)
            if math.isfinite(s0):
                spot_t.append(ts)
                spot_s.append(s0)
        for erow in erows:
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
            by_exp[lab]["expiry_date"] = d[:10]

    exp_labels = sorted(by_exp.keys(), key=lambda s: int(s.split(":", 1)[0]))

    param_source_log: list[str] = []
    by_exp_out: dict[str, dict[str, list]] = {}

    for lab in exp_labels:
        raw = by_exp[lab]
        exp_date = raw["expiry_date"]
        q_vals = [float(x) for x in raw["q"]]

        if exp_date in by_expiry_params:
            ep = by_expiry_params[exp_date]
            Q_w = float(ep["Q_w"])
            gamma = float(ep["gamma"])
            source = "matched"
        else:
            Q_w = default_Qw
            gamma = default_gamma
            source = "default"

        param_source_log.append(
            f"  {lab}: Q_w={Q_w:.2e}  gamma={gamma:.2e}  ({source})"
        )

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

    t_spot, s_spot = insert_null_between_calendar_days(spot_t, spot_s)

    return {
        "batch_name": batch_dir.name,
        "exp_labels": exp_labels,
        "by_exp": by_exp_out,
        "spot": {"t": t_spot, "S": s_spot},
        "_param_log": param_source_log,
    }


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


def render_html(payload: dict, title: str, params_info: str) -> str:
    b64 = _payload_b64(payload)
    b64_parts_js = _b64_chunk_array_literal(b64)
    params_info_esc = html.escape(params_info, quote=True)
    tpl = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>__TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body { font-family: system-ui, sans-serif; margin: 12px; }
    h1 { font-size: 1.05rem; margin: 0 0 4px 0; }
    .subtitle { font-size: 0.8rem; color: #666; margin: 0 0 10px 0; }
    #boot-err { color: #b00020; font-weight: 600; margin-bottom: 10px; white-space: pre-wrap; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .plot { width: 100%; height: 420px; border: 1px solid #e0e0e0; border-radius: 6px; }
    @media(max-width: 1100px) { .grid { grid-template-columns: 1fr; } }
    .spot-row { width: 100%; height: 400px; margin-top: 12px; border: 1px solid #e0e0e0; border-radius: 6px; }
    details { margin-top: 8px; font-size: 0.78rem; color: #888; }
    details pre { white-space: pre-wrap; margin: 4px 0; }
  </style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="subtitle">Parameters calibrated from prior day — no look-ahead</p>
<div id="boot-err"></div>
<div class="grid">
  <div id="g1" class="plot"></div>
  <div id="g2" class="plot"></div>
  <div id="g3" class="plot"></div>
  <div id="g4" class="plot"></div>
</div>
<div id="g5" class="plot spot-row"></div>
<details>
  <summary>Parameter sources</summary>
  <pre>__PARAMS_INFO__</pre>
</details>
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
try { DATA = _decodePayload(); }
catch (e) { document.getElementById('boot-err').textContent = 'Decode error: ' + (e&&e.message?e.message:e); throw e; }

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
      type: 'scatter', mode: 'lines+markers', name: lab,
      x: byExp[lab].t, y: byExp[lab][keyY], connectgaps: false,
      line: { width: 1.4, color: colors[i % colors.length] },
      marker: { size: 3.5 },
    }));
  }
  const baseX = { title: 'timestamp', type: 'date', rangebreaks: TRADING_RB, tickangle: -45 };
  const leg = { orientation: 'v', x: 1.02, y: 1, font: { size: 9 } };
  const m = { r: 200, b: 120, t: 44, l: 60 };
  const CFG = { responsive: true, displaylogo: false,
    toImageButtonOptions: { format: 'png', filename: 'q_f_cauchy_applied', scale: 2 } };
  try {
    Plotly.newPlot('g1', mkTraces('q'), {
      title: 'q (raw)', xaxis: baseX, yaxis: { title: 'q' },
      legend: leg, margin: m, hovermode: 'closest' }, CFG);
    Plotly.newPlot('g2', mkTraces('F'), {
      title: 'Forward F (raw)', xaxis: baseX, yaxis: { title: 'F' },
      legend: leg, margin: m, hovermode: 'closest' }, CFG);
    Plotly.newPlot('g3', mkTraces('q_cauchy'), {
      title: 'q\\u0302 (DP-MAP Cauchy, prior-day params)', xaxis: baseX, yaxis: { title: 'q_hat' },
      legend: leg, margin: m, hovermode: 'closest' }, CFG);
    Plotly.newPlot('g4', mkTraces('F_cauchy'), {
      title: 'F\\u0302 = S_raw * exp((r - q\\u0302) * volTime)', xaxis: baseX, yaxis: { title: 'F_hat' },
      legend: leg, margin: m, hovermode: 'closest' }, CFG);
    const spot = DATA.spot || {};
    const st = spot.t || [];
    const sS = spot.S || [];
    if (st.length && sS.length) {
      Plotly.newPlot('g5', [{
        type: 'scatter', mode: 'lines+markers', name: 'Spot S',
        x: st, y: sS, connectgaps: false,
        line: { width: 2, color: '#444' },
        marker: { size: 4 },
      }], {
        title: 'Spot S = F exp(\\u2212(r \\u2212 q) \\u00b7 volTime) — first expiry row per snapshot',
        xaxis: baseX, yaxis: { title: 'Spot' },
        showlegend: false, margin: m, hovermode: 'closest' }, CFG);
    }
  } catch (e) { err.textContent = 'Plot error: ' + (e&&e.message?e.message:e); }
})();
</script>
</body>
</html>
"""
    return (
        tpl.replace("__TITLE__", title)
        .replace("__B64_PARTS__", b64_parts_js)
        .replace("__PARAMS_INFO__", params_info_esc)
    )


def _params_source_label(params: dict, params_path: Path) -> str:
    """Human-readable label for pooled or single-day param JSONs."""
    d = params.get("calibration_day")
    if d:
        return str(d)
    if params.get("target_day"):
        return f"pooled→{params['target_day']}"
    cds = params.get("calibration_days")
    if isinstance(cds, list) and cds:
        return "pool[" + ",".join(str(x) for x in cds) + "]"
    return params_path.stem


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Apply DP-Cauchy filter with pre-calibrated params"
    )
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--day", required=True, help="Target day YYYY-MM-DD to filter")
    ap.add_argument(
        "--params",
        required=True,
        type=Path,
        help="JSON params file from calibrate_cauchy_params.py",
    )
    ap.add_argument("-o", "--output", default=None, help="Output HTML path")
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
    args = ap.parse_args()

    batch_dir = args.batch_dir.resolve()
    target_day = args.day.strip()
    params_path = args.params.resolve()

    params = json.loads(params_path.read_text(encoding="utf-8"))
    cal_day = _params_source_label(params, params_path)
    print(f"Params source:   {cal_day}", file=sys.stderr)
    print(f"Target day:      {target_day}", file=sys.stderr)
    print(
        f"Default Q_w={params['default']['Q_w']:.2e}  gamma={params['default']['gamma']:.2e}",
        file=sys.stderr,
    )

    ls_theta = None if args.no_level_shift else float(args.level_shift_theta)
    payload = build_payload_for_day(
        batch_dir,
        target_day,
        params,
        level_shift_theta=ls_theta,
        level_shift_n_streak=int(args.level_shift_n),
    )

    param_log = payload.pop("_param_log", [])
    params_info = f"Params: {cal_day} → Applied: {target_day}\n" + "\n".join(param_log)
    print("\nParameter assignments:", file=sys.stderr)
    for line in param_log:
        print(line, file=sys.stderr)

    dq_q = params.get("dq_abs_dev_quantile_pct")
    cal_note = cal_day
    if dq_q is not None and abs(float(dq_q) - 50.0) > 1e-6:
        cal_note = f"{cal_day}, |Δq−med| q={float(dq_q):g}%"
    title = f"q/F DP-Cauchy ({cal_note}) — {target_day}"
    safe = "".join(c if c not in '<>:"/\\|?*' else "_" for c in cal_day)
    default_name = f"q_f_cauchy_{safe}_to_{target_day}.html"
    out = Path(args.output).resolve() if args.output else (batch_dir / default_name)
    out.write_text(render_html(payload, title, params_info), encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)
    print(out.as_uri(), file=sys.stderr)


if __name__ == "__main__":
    main()
