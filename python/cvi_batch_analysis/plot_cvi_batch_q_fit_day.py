#!/usr/bin/env python3
"""
Build a standalone HTML (Plotly CDN) like q_fit_day.html:
  - q(t) and F(t) per expiry (multi-line charts)
  - q histogram + sample Gaussian overlay (all / per expiry)
  - per-expiry stats table
  - per-snapshot fit_summary + CVI_dims table
  - static CVI params block (matches TrDBFitViewer clamped preset)

Usage:
  python plot_cvi_batch_q_fit_day.py <batch_dir> [out.html]
"""

from __future__ import annotations

import base64
import csv
import json
import math
import statistics
import sys
from pathlib import Path


def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = statistics.fmean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _normal_curve_pdf(mu: float, sigma: float, n: int = 200) -> tuple[list[float], list[float]]:
    sig = max(sigma, 1e-12)
    lo, hi = mu - 4.0 * sig, mu + 4.0 * sig
    if lo == hi:
        lo, hi = mu - 1.0, mu + 1.0
    step = (hi - lo) / max(n - 1, 1)
    xs = [lo + i * step for i in range(n)]
    c = 1.0 / (sig * math.sqrt(2.0 * math.pi))
    ys = [c * math.exp(-0.5 * ((x - mu) / sig) ** 2) for x in xs]
    return xs, ys


def _overlay_for_qs(qs: list[float]) -> dict | None:
    if len(qs) < 2:
        return None
    mu = statistics.fmean(qs)
    sigma = _pstdev(qs)
    x, y = _normal_curve_pdf(mu, sigma)
    return {"mu": mu, "sigma": sigma, "n": len(qs), "x": x, "y": y}


def read_fit_summary_row(path: Path) -> dict[str, str | float | int | None]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return {}
    row = rows[0]
    out: dict[str, str | float | int | None] = {}
    mapping = [
        ("fit_num_basis", "num_basis", int),
        ("fit_arb_points", "arb_points", int),
        ("fit_lambda", "lambda", float),
        ("fit_n_options", "n_options", int),
        ("fit_rmse", "rmse", float),
        ("fit_mae", "mae", float),
        ("fit_max_error", "max_error", float),
        ("fit_r_squared", "r_squared", float),
        ("fit_clarabel_obj", "clarabel_obj", float),
        ("fit_clarabel_status", "clarabel_status", int),
        ("fit_bias_vol", "bias_vol", float),
        ("fit_median_abs_vol", "median_abs_vol", float),
        ("fit_p90_abs_vol", "p90_abs_vol", float),
        ("fit_p95_abs_vol", "p95_abs_vol", float),
        ("fit_rmse_vol_vega_wt", "rmse_vol_vega_wt", float),
        ("fit_rmse_vol_wmid_wt", "rmse_vol_wmid_wt", float),
        ("fit_rmse_var", "rmse_var", float),
        ("fit_mae_var", "mae_var", float),
        ("fit_mean_rel_abs_vol", "mean_rel_abs_vol", float),
        ("fit_median_rel_abs_vol", "median_rel_abs_vol", float),
    ]
    for out_k, in_k, typ in mapping:
        v = row.get(in_k)
        if v is None or v == "":
            out[out_k] = None
            continue
        try:
            out[out_k] = typ(v)
        except (TypeError, ValueError):
            out[out_k] = v
    return out


def read_dims_row(path: Path) -> dict[str, str | float | int | None]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        return {}
    row = rows[0]
    mapping = [
        ("dims_m", "m", int),
        ("dims_num_basis", "num_basis", int),
        ("dims_n_v", "n_v", int),
        ("dims_z0", "z0", float),
        ("dims_zn1", "zn1", float),
        ("dims_num_constraint_strikes", "num_constraint_strikes", int),
        ("dims_n_c", "n_c", int),
        ("dims_n_v_orig", "n_v_orig", int),
        ("dims_n_slack_ask", "n_slack_ask", int),
        ("dims_n_slack_bid", "n_slack_bid", int),
        ("dims_n_slack_reg", "n_slack_reg", int),
        ("dims_n_A_rows", "n_A_rows", int),
    ]
    out: dict[str, str | float | int | None] = {}
    for out_k, in_k, typ in mapping:
        v = row.get(in_k)
        if v is None or v == "":
            out[out_k] = None
            continue
        try:
            out[out_k] = typ(v)
        except (TypeError, ValueError):
            out[out_k] = v
    return out


def _calendar_date(timestamp: str) -> str:
    """First token of snapshot string: 2026-04-06 09:30:12-04 -> 2026-04-06."""
    ts = timestamp.strip()
    return ts.split()[0] if ts else ""


def _insert_null_between_calendar_days(
    t: list[str], q: list[float], f: list[float]
) -> tuple[list[str | None], list[float | None], list[float | None]]:
    """Break Plotly lines across overnight: insert null triple when calendar day changes."""
    if len(t) <= 1:
        return list(t), list(q), list(f)
    out_t: list[str | None] = []
    out_q: list[float | None] = []
    out_f: list[float | None] = []
    prev_day: str | None = None
    for i in range(len(t)):
        day = _calendar_date(t[i])
        if prev_day is not None and day != prev_day:
            out_t.append(None)
            out_q.append(None)
            out_f.append(None)
        out_t.append(t[i])
        out_q.append(q[i])
        out_f.append(f[i])
        prev_day = day
    return out_t, out_q, out_f


def default_cvi_params() -> dict:
    return {
        "source": "TrDBFitViewer/CviSingleFit.cpp (writeTestSuiteClampedStyleArtifacts path)",
        "num_basis": 23,
        "lambda": 0.001,
        "num_constraint_strikes": 20,
        "z0": -5,
        "zn1": 5,
        "atm_bunch": 1,
        "z_use_spot": False,
        "use_phi_d2_breakpoints": False,
        "phi_d2_vmid_two_pass": False,
        "weight_mode": "VarianceSpace (enum)",
        "zero_dte_front_two_expiries": False,
        "enable_butterfly_pass2": True,
        "ignore_run_arb_pre_post_check_failure": True,
        "ignore_update_chain_pre_post_check_failure": True,
        "keep_mid_outside": True,
        "keep_ask_outside": True,
        "keep_bid_outside": True,
        "use_test_clamped_even_z_breakpoints": True,
    }


def build_payload(batch_dir: Path) -> dict:
    summary_path = batch_dir / "batch_cvi_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    by_exp: dict[str, dict[str, list]] = {}
    snapshots: list[dict] = []

    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        summary_rows = list(reader)

    summary_rows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))

    for row in summary_rows:
        subfolder = row.get("subfolder", "").strip()
        idx = int(row.get("idx_in_bin", 0) or 0)
        ts = row.get("timestamp", "").strip()
        ok = int(row.get("ok", 0) or 0)
        sub = batch_dir / subfolder
        efq = sub / "expiry_fwd_q.csv"
        has_efq = efq.is_file()

        snap: dict = {
            "subfolder": subfolder,
            "idx_in_bin": idx,
            "timestamp": ts,
            "has_expiry_fwd_q": has_efq,
        }
        snap.update(read_fit_summary_row(sub / "fit_summary.csv"))
        snap.update(read_dims_row(sub / "CVI_dims.csv"))
        snapshots.append(snap)

        if not has_efq:
            continue
        with efq.open(newline="", encoding="utf-8") as ef:
            for erow in csv.DictReader(ef):
                eidx = int(erow["expiry_idx"])
                d = erow.get("expiry_date", "").strip().strip('"')
                lab = f"{eidx}: {d}"
                if lab not in by_exp:
                    by_exp[lab] = {"t": [], "q": [], "F": []}
                by_exp[lab]["t"].append(ts)
                by_exp[lab]["q"].append(float(erow["q"]))
                by_exp[lab]["F"].append(float(erow["F"]))

    exp_labels = sorted(by_exp.keys(), key=lambda s: int(s.split(":", 1)[0]))

    by_exp_plot: dict[str, dict[str, list]] = {}
    for lab in exp_labels:
        raw = by_exp[lab]
        t2, q2, f2 = _insert_null_between_calendar_days(raw["t"], raw["q"], raw["F"])
        by_exp_plot[lab] = {"t": t2, "q": q2, "F": f2}

    hist_by_exp: dict[str, list[float]] = {lab: list(by_exp[lab]["q"]) for lab in exp_labels}
    hist_all: list[float] = [q for lab in exp_labels for q in hist_by_exp[lab]]

    normal_overlay: dict[str, dict | None] = {"__ALL__": _overlay_for_qs(hist_all)}
    for lab in exp_labels:
        normal_overlay[lab] = _overlay_for_qs(hist_by_exp[lab])

    stats: list[dict] = []
    for lab in exp_labels:
        qs = by_exp[lab]["q"]
        Fs = by_exp[lab]["F"]
        sigmas_all: list[float] = []
        vs_all: list[float] = []
        eidx_key = int(lab.split(":", 1)[0])
        for row in summary_rows:
            if int(row.get("ok", 0) or 0) != 1:
                continue
            p = batch_dir / row["subfolder"].strip() / "expiry_fwd_q.csv"
            if not p.is_file():
                continue
            with p.open(newline="", encoding="utf-8") as ef:
                for erow in csv.DictReader(ef):
                    if int(erow["expiry_idx"]) == eidx_key:
                        sigmas_all.append(float(erow["sigma_star"]))
                        vs_all.append(float(erow["v_star"]))
                        break

        def _agg(xs: list[float]) -> tuple[float, float, float, float]:
            if not xs:
                return (float("nan"),) * 4
            return (
                statistics.fmean(xs),
                _pstdev(xs),
                min(xs),
                max(xs),
            )

        smin, smax = (min(qs), max(qs)) if qs else (float("nan"), float("nan"))
        fmean, fstd, fmin, fmax = _agg(Fs)
        stats.append(
            {
                "expiry_idx": eidx_key,
                "expiry_date": lab.split(":", 1)[1].strip(),
                "n_snapshots": len(qs),
                "q_mean": statistics.fmean(qs) if qs else float("nan"),
                "q_std": _pstdev(qs) if len(qs) > 1 else 0.0,
                "q_min": smin,
                "q_max": smax,
                "F_mean": fmean,
                "F_std": fstd,
                "F_min": fmin,
                "F_max": fmax,
                "sigma_star_mean": statistics.fmean(sigmas_all) if sigmas_all else float("nan"),
                "sigma_star_std": _pstdev(sigmas_all) if len(sigmas_all) > 1 else 0.0,
                "sigma_star_min": min(sigmas_all) if sigmas_all else float("nan"),
                "sigma_star_max": max(sigmas_all) if sigmas_all else float("nan"),
                "v_star_mean": statistics.fmean(vs_all) if vs_all else float("nan"),
                "v_star_std": _pstdev(vs_all) if len(vs_all) > 1 else 0.0,
                "v_star_min": min(vs_all) if vs_all else float("nan"),
                "v_star_max": max(vs_all) if vs_all else float("nan"),
            }
        )

    return {
        "byExp": by_exp_plot,
        "expLabels": exp_labels,
        "histAll": hist_all,
        "histByExp": hist_by_exp,
        "normalOverlay": normal_overlay,
        "stats": stats,
        "snapshots": snapshots,
        "cviParams": default_cvi_params(),
    }


def _scrub_non_finite(obj: object) -> object:
    """JSON-safe floats (no NaN/Inf) for strict dumps + browser JSON.parse."""
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
    """Many browsers fail or hang on a single multi‑MB JS string literal; chunk + join."""
    parts = [b64[i : i + chunk_size] for i in range(0, len(b64), chunk_size)]
    lines = ["  " + json.dumps(part, ensure_ascii=True) for part in parts]
    return "[\n" + ",\n".join(lines) + "\n]"


def render_html(payload: dict, title: str) -> str:
    b64 = _payload_b64(payload)
    b64_parts_js = _b64_chunk_array_literal(b64)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; }}
    h1 {{ font-size: 1.2rem; }}
    h2 {{ font-size: 1rem; margin-top: 1.5rem; }}
    pre.params {{ background: #f5f5f5; padding: 12px; border-radius: 6px; overflow: auto; font-size: 0.85rem; }}
    #boot-err {{ color: #b00020; font-weight: 600; white-space: pre-wrap; }}
  </style>
</head>
<body>
<h1>CVI forward q and F — whole batch</h1>
<p><b>Open this file</b> (<code>q_fit_day.html</code>), not only the folder — <code>file:///…/batch/</code> often shows a blank page.</p>
<div id="boot-err"></div>
<p>q(t) and F(t) by expiry; q <b>density</b> histogram with a <b>Gaussian curve</b>
(μ, σ from the sample — for shape comparison only). Tables: per-expiry stats; per-snapshot fit
summary + dims (when present).</p>
<div id="g1" style="width:100%;height:420px;"></div>
<div id="g2" style="width:100%;height:420px;"></div>
<label for="histSel"><b>Histogram q — expiry</b></label>
<select id="histSel" style="margin:8px 0;"></select>
<h2>Per-expiry statistics (across snapshots)</h2>
<div id="tbl1"></div>
<h2>Per-snapshot global fit (fit_summary + CVI_dims)</h2>
<div id="tbl2" style="width:100%;min-height:400px;"></div>
<h2>Fit parameters (TrDBFitViewer — clamped batch preset)</h2>
<pre class="params" id="paramPre"></pre>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>
(function() {{
  const errEl = document.getElementById('boot-err');
  function fail(msg) {{ errEl.textContent = msg; }}

  const _B64_PARTS = {b64_parts_js};

  function decodeP(b64) {{
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return JSON.parse(new TextDecoder('utf-8').decode(bytes));
  }}

  function run() {{
    if (typeof Plotly === 'undefined') {{
      fail('Plotly did not load (check network / corporate block of cdn.plot.ly). Try: cd this folder, run  python -m http.server 8080  then open http://localhost:8080/q_fit_day.html');
      return;
    }}
    let P;
    try {{
      P = decodeP(_B64_PARTS.join(''));
    }} catch (e) {{
      fail('Failed to parse batch data: ' + (e && e.message ? e.message : e));
      return;
    }}
    if (!P.expLabels || !P.expLabels.length) {{
      fail('No expiry series in payload (missing expiry_fwd_q.csv under snapshots?).');
      return;
    }}

    const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
    // Collapse gaps: weekends + outside US regular hours (~9:30–16:00) on snapshot wall clock.
    const TRADING_RB = [
      {{ bounds: ['sat', 'mon'], pattern: 'day of week' }},
      {{ bounds: [16, 9.5], pattern: 'hour' }},
    ];
    function tracesQ() {{
      return P.expLabels.map((lab, i) => ({{
        x: P.byExp[lab].t, y: P.byExp[lab].q, mode: 'lines+markers', name: lab,
        connectgaps: false,
        line: {{ width: 1.5, color: colors[i % colors.length] }},
        marker: {{ size: 4 }}
      }}));
    }}
    function tracesF() {{
      return P.expLabels.map((lab, i) => ({{
        x: P.byExp[lab].t, y: P.byExp[lab].F, mode: 'lines+markers', name: lab,
        connectgaps: false,
        line: {{ width: 1.5, color: colors[i % colors.length] }},
        marker: {{ size: 4 }}
      }}));
    }}
    try {{
      Plotly.newPlot('g1', tracesQ(), {{
        title: 'q (forward yield proxy) vs snapshot time',
        xaxis: {{ title: 'timestamp', type: 'date', rangebreaks: TRADING_RB }}, yaxis: {{ title: 'q' }},
        legend: {{ orientation: 'v', x: 1.02, y: 1 }}, margin: {{ r: 160 }}
      }}, {{ responsive: true }});
      Plotly.newPlot('g2', tracesF(), {{
        title: 'Forward F vs snapshot time',
        xaxis: {{ title: 'timestamp', type: 'date', rangebreaks: TRADING_RB }}, yaxis: {{ title: 'F' }},
        legend: {{ orientation: 'v', x: 1.02, y: 1 }}, margin: {{ r: 160 }}
      }}, {{ responsive: true }});
    }} catch (e) {{
      fail('Plotly chart error: ' + e);
      return;
    }}

    const histHost = document.createElement('div');
    histHost.id = 'ghist';
    histHost.style.width = '100%';
    histHost.style.height = '400px';
    document.getElementById('histSel').after(histHost);
    const sel = document.getElementById('histSel');
    const optAll = document.createElement('option');
    optAll.value = '__ALL__';
    optAll.textContent = 'All expiries';
    sel.appendChild(optAll);
    P.expLabels.forEach((lab) => {{
      const o = document.createElement('option');
      o.value = lab;
      o.textContent = lab;
      sel.appendChild(o);
    }});
    let histReady = false;
    function drawHist(key) {{
      const x = key === '__ALL__' ? P.histAll : P.histByExp[key];
      const ovKey = key === '__ALL__' ? '__ALL__' : key;
      const ov = P.normalOverlay[ovKey];
      const muStr = ov && Number.isFinite(ov.mu) ? ov.mu.toFixed(6) : '';
      const sigStr = ov && Number.isFinite(ov.sigma) ? ov.sigma.toFixed(6) : '';
      const sub = ov && ov.n ? (' — sample Gaussian μ=' + muStr + ', σ=' + sigStr + ', n=' + ov.n) : '';
      const title = (key === '__ALL__' ? 'q density (all expiries)' : ('q density — ' + key)) + sub;
      const data = [{{
        x, type: 'histogram', nbinsx: 40, histnorm: 'probability density',
        name: 'q (density)', marker: {{ color: '#8fbce6' }}, opacity: 0.75
      }}];
      if (ov && ov.x && ov.x.length) {{
        data.push({{
          x: ov.x, y: ov.y, mode: 'lines', name: 'N(μ,σ²) same μ,σ as sample',
          line: {{ color: '#d62728', width: 2.5 }}
        }});
      }}
      const layout = {{ title, xaxis: {{ title: 'q' }}, yaxis: {{ title: 'density' }} }};
      const cfg = {{ responsive: true }};
      if (histReady) Plotly.react('ghist', data, layout, cfg);
      else {{ Plotly.newPlot('ghist', data, layout, cfg); histReady = true; }}
    }}
    sel.addEventListener('change', () => drawHist(sel.value));
    drawHist('__ALL__');

    document.getElementById('paramPre').textContent = JSON.stringify(P.cviParams, null, 2);

    const h1 = ["expiry_idx", "expiry_date", "n_snapshots", "q_mean", "q_std", "q_min", "q_max", "F_mean", "F_std", "F_min", "F_max", "sigma_star_mean", "sigma_star_std", "sigma_star_min", "sigma_star_max", "v_star_mean", "v_star_std", "v_star_min", "v_star_max"];
    const c1 = [
      P.stats.map((r) => r.expiry_idx),
      P.stats.map((r) => r.expiry_date),
      P.stats.map((r) => r.n_snapshots),
      P.stats.map((r) => r.q_mean),
      P.stats.map((r) => r.q_std),
      P.stats.map((r) => r.q_min),
      P.stats.map((r) => r.q_max),
      P.stats.map((r) => r.F_mean),
      P.stats.map((r) => r.F_std),
      P.stats.map((r) => r.F_min),
      P.stats.map((r) => r.F_max),
      P.stats.map((r) => r.sigma_star_mean),
      P.stats.map((r) => r.sigma_star_std),
      P.stats.map((r) => r.sigma_star_min),
      P.stats.map((r) => r.sigma_star_max),
      P.stats.map((r) => r.v_star_mean),
      P.stats.map((r) => r.v_star_std),
      P.stats.map((r) => r.v_star_min),
      P.stats.map((r) => r.v_star_max),
    ];
    Plotly.newPlot('tbl1', [{{
      type: 'table',
      header: {{ values: h1, fill: {{ color: '#e0e0e0' }}, font: {{ size: 11 }} }},
      cells: {{ values: c1, font: {{ size: 10 }} }}
    }}], {{ margin: {{ t: 20 }} }}, {{ responsive: true }});

    const snapKeys = [
      "subfolder", "idx_in_bin", "timestamp", "has_expiry_fwd_q",
      "fit_num_basis", "fit_arb_points", "fit_lambda", "fit_n_options", "fit_rmse", "fit_mae", "fit_max_error", "fit_r_squared",
      "fit_clarabel_obj", "fit_clarabel_status", "fit_bias_vol", "fit_median_abs_vol", "fit_p90_abs_vol", "fit_p95_abs_vol",
      "fit_rmse_vol_vega_wt", "fit_rmse_vol_wmid_wt", "fit_rmse_var", "fit_mae_var", "fit_mean_rel_abs_vol", "fit_median_rel_abs_vol",
      "dims_m", "dims_num_basis", "dims_n_v", "dims_z0", "dims_zn1", "dims_num_constraint_strikes", "dims_n_c", "dims_n_v_orig",
      "dims_n_slack_ask", "dims_n_slack_bid", "dims_n_slack_reg", "dims_n_A_rows"
    ];
    const h2 = snapKeys;
    const c2 = snapKeys.map((k) => P.snapshots.map((s) => s[k] !== undefined && s[k] !== null ? s[k] : ""));
    const nRows = P.snapshots.length;
    const tblH = Math.min(5200, 100 + Math.min(nRows, 240) * 20);
    Plotly.newPlot('tbl2', [{{
      type: 'table',
      header: {{ values: h2, fill: {{ color: '#e8f4ff' }}, font: {{ size: 10 }} }},
      cells: {{ values: c2, font: {{ size: 9 }} }}
    }}], {{ height: tblH, margin: {{ t: 20 }} }}, {{ responsive: true }});
  }}

  try {{
    run();
  }} catch (e) {{
    fail('Run error: ' + (e && e.message ? e.message : e));
  }}
}})();
</script>
</body>
</html>
"""


def write_index_redirect(batch_dir: Path, html_name: str = "q_fit_day.html") -> Path:
    """If user opens file:///.../batch/ in the browser, point them at the real report."""
    idx = batch_dir / "index.html"
    idx.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>CVI batch — open report</title>
</head>
<body style="font-family:system-ui,sans-serif;padding:24px;">
  <p>Browsers usually show <b>nothing useful</b> for a bare <code>file:///…/folder/</code> URL.</p>
  <p>Open the report: <a href="{html_name}"><code>{html_name}</code></a></p>
</body>
</html>
""",
        encoding="utf-8",
    )
    return idx


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_cvi_batch_q_fit_day.py <batch_dir> [out.html]", file=sys.stderr)
        sys.exit(2)
    batch_dir = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else batch_dir / "q_fit_day.html"
    payload = build_payload(batch_dir)
    title = f"CVI q / F — {batch_dir.name}"
    html = render_html(payload, title)
    out.write_text(html, encoding="utf-8")
    idx = write_index_redirect(batch_dir, out.name)
    print(f"Wrote {out}")
    print(f"Wrote {idx}")
    print(f"file:///{str(out).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
