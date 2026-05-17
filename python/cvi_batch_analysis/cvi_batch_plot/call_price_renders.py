"""Call ΔC prediction, ln/SSR Plotly HTML, default q/vol heatmap HTML."""
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
from .batch_payloads import build_ln_scatter_payload
from .snapshots_decouple import _build_snapshot_state, _find_forward_idx, _market_price_at_strike

def build_call_price_prediction_payload(
    batch_dir: Path,
    *,
    day: str,
    expiry_index: int,
    horizons_min: list[float],
    strike_z_offsets: list[float],
    target_source: str = "both",
) -> dict:
    if target_source not in {"market", "cvi", "both"}:
        raise SystemExit("target_source must be one of: market, cvi, both")
    summary = read_summary(batch_dir, day=day)
    if not summary:
        raise SystemExit(f"No successful snapshots found for date {day}.")
    try:
        ln_payload = build_ln_scatter_payload(batch_dir, day=day, expiry_index=expiry_index, sigma_mode="avg3")
        ssr_by_ts = {
            str(r["timestamp"]): {
                "ssr_ols": float(r["ssr_ols"]) if r.get("ssr_ols") is not None else None,
                "ssr_wls": float(r["ssr_wls"]) if r.get("ssr_wls") is not None else None,
                "ssr_huber": float(r["ssr_huber"]) if r.get("ssr_huber") is not None else None,
                "ssr_lad": float(r["ssr_lad"]) if r.get("ssr_lad") is not None else None,
                "ssr_ts": float(r["ssr_ts"]) if r.get("ssr_ts") is not None else None,
            }
            for r in ln_payload.get("pred_series", [])
        }
    except Exception:  # noqa: BLE001
        ssr_by_ts = {}

    snaps: list[dict] = []
    for row in summary:
        st = _build_snapshot_state(batch_dir, row, expiry_index, ssr_by_ts)
        if st is not None and st.get("time") is not None:
            snaps.append(st)
    if len(snaps) < 3:
        raise SystemExit("Not enough snapshots with option data for call-price prediction mode.")

    first = snaps[0]
    selected: list[dict[str, float]] = []
    used_idx: set[int] = set()
    for zt in strike_z_offsets:
        j = min(range(len(first["zs"])), key=lambda ii: abs(float(first["zs"][ii]) - float(zt)))
        if j in used_idx:
            continue
        used_idx.add(j)
        selected.append(
            {
                "strike": float(first["strikes"][j]),
                "z_ref": float(first["zs"][j]),
                "z_target": float(zt),
            }
        )
    if not selected:
        raise SystemExit("Could not select any strikes from requested z offsets.")

    times = [s["time"] for s in snaps]
    target_kinds = ["market", "cvi"] if target_source == "both" else [target_source]
    samples: list[dict[str, object]] = []
    market_source_counter: dict[str, int] = defaultdict(int)
    skipped_missing = 0
    for h in horizons_min:
        for i in range(len(snaps) - 1):
            j = _find_forward_idx(times, i, h)
            if j is None:
                continue
            s0 = snaps[i]
            s1 = snaps[j]
            f0 = float(s0["fwd"])
            f1 = float(s1["fwd"])
            if not (f0 > 0.0 and f1 > 0.0):
                continue
            dlnf = math.log(f1) - math.log(f0)
            dF = f1 - f0
            for sel in selected:
                k = float(sel["strike"])
                sigma0 = interp_linear(s0["strikes"], s0["fitted_vols"], k)
                sigma1 = interp_linear(s1["strikes"], s1["fitted_vols"], k)
                z0 = interp_linear(s0["strikes"], s0["zs"], k)
                if sigma0 is None or sigma1 is None or sigma0 <= 0.0 or sigma1 <= 0.0:
                    skipped_missing += 1
                    continue
                c0_cvi = _lr_call_price_from_fwd(
                    f0,
                    k,
                    sigma0,
                    float(s0["vol_time"]),
                    r=float(s0.get("r", 0.0)),
                    q=float(s0.get("q", 0.0)),
                )
                c1_cvi = _lr_call_price_from_fwd(
                    f1,
                    k,
                    sigma1,
                    float(s1["vol_time"]),
                    r=float(s1.get("r", 0.0)),
                    q=float(s1.get("q", 0.0)),
                )
                if c0_cvi is None or c1_cvi is None:
                    skipped_missing += 1
                    continue
                k_ref = k * f0 / f1
                sigma_ref = interp_linear(s0["strikes"], s0["fitted_vols"], k_ref)
                sigma_ref = sigma_ref if sigma_ref is not None and sigma_ref > 0.0 else sigma0
                c_hat_ss = _lr_call_price_from_fwd(
                    f1,
                    k,
                    sigma0,
                    float(s1["vol_time"]),
                    r=float(s1.get("r", 0.0)),
                    q=float(s1.get("q", 0.0)),
                )
                c_hat_sm = _lr_call_price_from_fwd(
                    f1,
                    k,
                    sigma_ref,
                    float(s1["vol_time"]),
                    r=float(s1.get("r", 0.0)),
                    q=float(s1.get("q", 0.0)),
                )
                greeks = _lr_call_greeks_from_fwd(
                    f0,
                    k,
                    sigma0,
                    float(s0["vol_time"]),
                    r=float(s0.get("r", 0.0)),
                    q=float(s0.get("q", 0.0)),
                )
                c_hat_kl = None
                if greeks is not None:
                    delta = float(greeks["delta"])
                    vega = float(greeks["vega"])
                    gamma = float(greeks["gamma"])
                    vanna = float(greeks["vanna"])
                    volga = float(greeks["volga"])
                    s_atf = _to_float_or_none(s0.get("s_atf_norm"))
                    ssr_wls = _to_float_or_none(s0.get("ssr_wls"))
                    if s_atf is not None and ssr_wls is not None:
                        dln_sigma = ssr_wls * s_atf * dlnf
                        dsigma = sigma0 * (math.exp(dln_sigma) - 1.0)
                        dc_pred = (
                            delta * dF
                            + vega * dsigma
                            + 0.5 * gamma * dF * dF
                            + vanna * dF * dsigma
                            + 0.5 * volga * dsigma * dsigma
                        )
                        c_hat_kl = c0_cvi + dc_pred
                if c_hat_ss is None or c_hat_sm is None:
                    skipped_missing += 1
                    continue
                c1_market, src1 = _market_price_at_strike(s1["rows"], k, f1, float(s1["vol_time"]))
                c0_market, src0 = _market_price_at_strike(s0["rows"], k, f0, float(s0["vol_time"]))
                if src0 != "unavailable":
                    market_source_counter[src0] += 1
                if src1 != "unavailable":
                    market_source_counter[src1] += 1
                pred_dcs = {
                    "sticky_strike": c_hat_ss - c0_cvi,
                    "sticky_moneyness": c_hat_sm - c0_cvi,
                    "klassen_linearized": (c_hat_kl - c0_cvi) if c_hat_kl is not None else None,
                }
                delta_bs_t = None
                if greeks is not None:
                    delta_bs_t = float(greeks["delta"])
                delta_pred_sticky_strike = (pred_dcs["sticky_strike"] / dF) if abs(dF) > 1e-12 else None
                delta_pred_sticky_moneyness = (pred_dcs["sticky_moneyness"] / dF) if abs(dF) > 1e-12 else None
                delta_pred_klassen_linearized = (
                    (pred_dcs["klassen_linearized"] / dF)
                    if pred_dcs["klassen_linearized"] is not None and abs(dF) > 1e-12
                    else None
                )
                target_map = {
                    "cvi": {"c0": c0_cvi, "c1": c1_cvi, "source": "fitted_vol_to_bs_price"},
                    "market": {"c0": c0_market, "c1": c1_market, "source": f"{src0}->{src1}"},
                }
                for tk in target_kinds:
                    trg = target_map[tk]
                    if trg["c0"] is None or trg["c1"] is None:
                        continue
                    dc_real = float(trg["c1"]) - float(trg["c0"])
                    samples.append(
                        {
                            "target": tk,
                            "horizon_min": float(h),
                            "strike": k,
                            "z_ref": float(sel["z_ref"]),
                            "z_t": z0,
                            "from_t": str(s0["timestamp"]),
                            "to_t": str(s1["timestamp"]),
                            "dF": dF,
                            "dlnF": dlnf,
                            "c_t": float(trg["c0"]),
                            "c_realized": float(trg["c1"]),
                            "dc_realized": dc_real,
                            "source": str(trg["source"]),
                            "pred_dc_sticky_strike": pred_dcs["sticky_strike"],
                            "pred_dc_sticky_moneyness": pred_dcs["sticky_moneyness"],
                            "pred_dc_klassen_linearized": pred_dcs["klassen_linearized"],
                            "delta_bs_t": delta_bs_t,
                            "delta_pred_sticky_strike": delta_pred_sticky_strike,
                            "delta_pred_sticky_moneyness": delta_pred_sticky_moneyness,
                            "delta_pred_klassen_linearized": delta_pred_klassen_linearized,
                            "delta_realized": (dc_real / dF) if abs(dF) > 1e-12 else None,
                            "pred_c_sticky_strike": float(trg["c0"]) + float(pred_dcs["sticky_strike"]),
                            "pred_c_sticky_moneyness": float(trg["c0"]) + float(pred_dcs["sticky_moneyness"]),
                            "pred_c_klassen_linearized": (
                                float(trg["c0"]) + float(pred_dcs["klassen_linearized"])
                                if pred_dcs["klassen_linearized"] is not None
                                else None
                            ),
                        }
                    )
    if not samples:
        raise SystemExit("No valid (t, K, h) samples for call-price prediction mode.")

    by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    metrics_h_target: dict[str, dict[str, float | int | None]] = {}
    z_slice: dict[str, dict[str, float]] = {}
    move_slice: dict[str, dict[str, float]] = {}
    for s in samples:
        g = f"h{int(round(float(s['horizon_min'])))}|{s['target']}|K{float(s['strike']):.4f}"
        by_group[g].append(s)
    for g, rows in by_group.items():
        rows.sort(key=lambda r: str(r["to_t"]))
        run = {
            "sticky_strike": {"n": 0, "sse": 0.0},
            "sticky_moneyness": {"n": 0, "sse": 0.0},
            "klassen_linearized": {"n": 0, "sse": 0.0},
        }
        run_delta = {
            "sticky_strike": {"n": 0, "sse": 0.0},
            "sticky_moneyness": {"n": 0, "sse": 0.0},
            "klassen_linearized": {"n": 0, "sse": 0.0},
        }
        for r in rows:
            for m in ("sticky_strike", "sticky_moneyness", "klassen_linearized"):
                pred = _to_float_or_none(r.get(f"pred_dc_{m}"))
                err = None if pred is None else float(r["dc_realized"]) - pred
                r[f"resid_{m}"] = err
                if err is not None:
                    run[m]["n"] += 1
                    run[m]["sse"] += err * err
                    r[f"running_mse_{m}"] = run[m]["sse"] / max(run[m]["n"], 1)
                else:
                    r[f"running_mse_{m}"] = None
                d_real = _to_float_or_none(r.get("delta_realized"))
                d_pred = _to_float_or_none(r.get(f"delta_pred_{m}"))
                d_err = None if d_real is None or d_pred is None else (d_real - d_pred)
                r[f"delta_resid_{m}"] = d_err
                if d_err is not None:
                    run_delta[m]["n"] += 1
                    run_delta[m]["sse"] += d_err * d_err
                    r[f"running_delta_mse_{m}"] = run_delta[m]["sse"] / max(run_delta[m]["n"], 1)
                else:
                    r[f"running_delta_mse_{m}"] = None
                zkey = f"h{int(round(float(r['horizon_min'])))}|{r['target']}|{bucket_z(_to_float_or_none(r.get('z_t')))}"
                mkey = f"h{int(round(float(r['horizon_min'])))}|{r['target']}|{bucket_abs_dlnf(_to_float_or_none(r.get('dlnF')))}"
                update_sse_bucket(z_slice, zkey, m, err)
                update_sse_bucket(move_slice, mkey, m, err)
        h = int(round(float(rows[0]["horizon_min"])))
        tkind = str(rows[0]["target"])
        kkey = f"h{h}|{tkind}"
        mh = metrics_h_target.setdefault(
            kkey,
            {"horizon_min": h, "target": tkind, "n": 0, "sse_sticky_strike": 0.0, "sse_sticky_moneyness": 0.0, "sse_klassen_linearized": 0.0},
        )
        for r in rows:
            mh["n"] = int(mh["n"]) + 1
            for m in ("sticky_strike", "sticky_moneyness", "klassen_linearized"):
                e = _to_float_or_none(r.get(f"resid_{m}"))
                if e is not None:
                    mh[f"sse_{m}"] = float(mh[f"sse_{m}"]) + e * e

    metrics_rows: list[dict[str, float | int | None | str]] = []
    for _k, row in sorted(metrics_h_target.items()):
        n = int(row["n"])
        sse_ss = float(row["sse_sticky_strike"])
        sse_sm = float(row["sse_sticky_moneyness"])
        sse_kl = float(row["sse_klassen_linearized"])
        out = {
            "horizon_min": int(row["horizon_min"]),
            "target": str(row["target"]),
            "n": n,
            "mse_sticky_strike": (sse_ss / n) if n > 0 else None,
            "mse_sticky_moneyness": (sse_sm / n) if n > 0 else None,
            "mse_klassen_linearized": (sse_kl / n) if n > 0 else None,
            "r2_vs_sticky_sticky_moneyness": (1.0 - sse_sm / sse_ss) if sse_ss > 1e-30 else None,
            "r2_vs_sticky_klassen_linearized": (1.0 - sse_kl / sse_ss) if sse_ss > 1e-30 else None,
        }
        metrics_rows.append(out)

    return {
        "date": day,
        "expiry_index": expiry_index,
        "horizons_min": [float(h) for h in horizons_min],
        "target_source": target_source,
        "selected_strikes": selected,
        "groups": {k: v for k, v in by_group.items()},
        "metrics": metrics_rows,
        "z_slice_sse": z_slice,
        "move_slice_sse": move_slice,
        "diagnostics": {
            "n_snapshots_used": len(snaps),
            "n_samples": len(samples),
            "n_skipped_missing": skipped_missing,
            "market_source_counter": dict(market_source_counter),
        },
    }


def render_call_price_prediction_html(payload: dict, title: str) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    .plot {{ width: 100%; height: 520px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; }}
    .row {{ display: flex; gap: 12px; margin: 10px 0; align-items: center; flex-wrap: wrap; }}
  </style>
</head>
<body>
  <h1>ΔC prediction evaluation</h1>
  <div class="row">
    <label>Series <select id="series_pick"></select></label>
  </div>
  <div id="price_ts" class="plot"></div>
  <div id="resid_ts" class="plot"></div>
  <div id="delta_ts" class="plot"></div>
  <div id="delta_resid_ts" class="plot"></div>
  <h3>Metrics (MSE / R² vs sticky-strike)</h3>
  <pre id="metrics"></pre>
  <h3>Diagnostics</h3>
  <pre id="diag"></pre>
  <script>
    const P = {data_json};
    const groups = P.groups || {{}};
    const keys = Object.keys(groups).sort();
    const sel = document.getElementById("series_pick");
    keys.forEach((k, i) => {{
      const o = document.createElement("option");
      o.value = k;
      o.textContent = k;
      if (i === 0) o.selected = true;
      sel.appendChild(o);
    }});
    function redraw() {{
      const k = sel.value;
      const rows = (groups[k] || []).slice();
      const x = rows.map(r => r.to_t);
      Plotly.react("price_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.c_realized), name: "realized C(t+h)" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.pred_c_sticky_strike), name: "pred sticky-strike" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.pred_c_sticky_moneyness), name: "pred sticky-moneyness" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.pred_c_klassen_linearized), name: "pred klassen-linearized" }},
      ], {{
        title: "Realized call price vs one-step-ahead predictions",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "Call price" }},
        margin: {{ l: 70, r: 40, t: 48, b: 100 }},
      }}, {{ responsive: true }});
      Plotly.react("resid_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.resid_sticky_strike), name: "resid sticky-strike" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.resid_sticky_moneyness), name: "resid sticky-moneyness" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.resid_klassen_linearized), name: "resid klassen-linearized" }},
      ], {{
        title: "Residuals ΔC_realized - ΔC_pred",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "Residual ΔC" }},
        margin: {{ l: 70, r: 40, t: 48, b: 100 }},
        shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#888" }} }}]
      }}, {{ responsive: true }});
      Plotly.react("delta_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_realized), name: "realized ΔC/ΔF" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_bs_t), name: "current BS delta(t)" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_pred_sticky_strike), name: "pred delta sticky-strike" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_pred_sticky_moneyness), name: "pred delta sticky-moneyness" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_pred_klassen_linearized), name: "pred delta klassen-linearized" }},
      ], {{
        title: "Current delta prediction vs realized ΔC/ΔF",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "Delta" }},
        margin: {{ l: 70, r: 40, t: 48, b: 100 }},
      }}, {{ responsive: true }});
      Plotly.react("delta_resid_ts", [
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_resid_sticky_strike), name: "delta resid sticky-strike" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_resid_sticky_moneyness), name: "delta resid sticky-moneyness" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.delta_resid_klassen_linearized), name: "delta resid klassen-linearized" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.running_delta_mse_sticky_strike), name: "running delta MSE sticky-strike", yaxis: "y2" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.running_delta_mse_sticky_moneyness), name: "running delta MSE sticky-moneyness", yaxis: "y2" }},
        {{ type: "scatter", mode: "lines+markers", x, y: rows.map(r => r.running_delta_mse_klassen_linearized), name: "running delta MSE klassen-linearized", yaxis: "y2" }},
      ], {{
        title: "Delta residuals and running MSE",
        xaxis: {{ title: "to timestamp", tickangle: -45 }},
        yaxis: {{ title: "Delta residual" }},
        yaxis2: {{ title: "Running delta MSE", overlaying: "y", side: "right", showgrid: false }},
        margin: {{ l: 70, r: 70, t: 48, b: 100 }},
        shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#888" }} }}]
      }}, {{ responsive: true }});
    }}
    if (keys.length > 0) {{
      sel.addEventListener("change", redraw);
      redraw();
    }}
    document.getElementById("metrics").textContent = JSON.stringify(P.metrics || [], null, 2);
    document.getElementById("diag").textContent = JSON.stringify({{
      diagnostics: P.diagnostics || {{}},
      z_slice_sse: P.z_slice_sse || {{}},
      move_slice_sse: P.move_slice_sse || {{}},
      selected_strikes: P.selected_strikes || []
    }}, null, 2);
  </script>
</body>
</html>
"""


def render_ln_scatter_html(payload: dict, title: str) -> str:
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
    .plot {{ width: 100%; height: 620px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>Δln(F) vs Δln(sigma_star) + ATF skew</h1>
  <div id="ln_sc" class="plot"></div>
  <div id="roll_reg" class="plot"></div>
  <div id="roll_ssr" class="plot"></div>
  <div id="pred_dlnsigma" class="plot"></div>
  <div id="pred_sigma" class="plot"></div>
  <div id="pred_parity" class="plot"></div>
  <div id="skew_ts" class="plot"></div>
  <h3>Diagnostics</h3>
  <pre id="diag"></pre>
  <script>
    const P = {data_json};
    const pts = P.points || [];
    const xs = pts.map(p => p.dlnF);
    const ys = pts.map(p => p.dlnSigma);
    const hover = pts.map(p =>
      `${{p.from_t}} → ${{p.to_t}}<br>ΔlnF=${{p.dlnF.toFixed(6)}}<br>Δlnσ*=${{p.dlnSigma.toFixed(6)}}`
    );
    function fitLineWLSOrigin(xv, yv, wv) {{
      if (!xv || xv.length < 2 || yv.length !== xv.length || wv.length !== xv.length) return null;
      const sxx = xv.reduce((a, x, i) => a + wv[i] * x * x, 0);
      const sxy = xv.reduce((a, x, i) => a + wv[i] * x * yv[i], 0);
      if (Math.abs(sxx) <= 1e-24) return null;
      const beta = sxy / sxx;
      if (!Number.isFinite(beta)) return null;
      return {{ beta }};
    }}

    const eps = 1e-12;
    const ws = xs.map((x, i) => 1.0 / (eps + x * x + ys[i] * ys[i]));
    const ols = fitLineWLSOrigin(xs, ys, xs.map(() => 1.0));
    const wls = fitLineWLSOrigin(xs, ys, ws);
    const traces = [{{
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      text: hover,
      marker: {{ size: 9, color: "#1f77b4", opacity: 0.86 }},
      hovertemplate: "%{{text}}<extra></extra>",
      name: "snapshots",
    }}];
    if (ols || wls) {{
      const xmin = Math.min(...xs);
      const xmax = Math.max(...xs);
      const pad = (xmax - xmin) * 0.06 + 1e-12;
      const x0 = xmin - pad;
      const x1 = xmax + pad;
      if (ols) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: [x0, x1],
          y: [ols.beta * x0, ols.beta * x1],
          line: {{ width: 2.5, color: "#2ca02c" }},
          name: `OLS fit (origin): y = ${{ols.beta.toFixed(6)}}x`,
          hovertemplate: "OLS fit<extra></extra>",
        }});
      }}
      if (wls) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: [x0, x1],
          y: [wls.beta * x0, wls.beta * x1],
          line: {{ width: 3, color: "#d62728" }},
          name: `WLS fit (origin): y = ${{wls.beta.toFixed(6)}}x`,
          hovertemplate: "WLS fit<extra></extra>",
        }});
      }}
    }}
    Plotly.newPlot("ln_sc", traces, {{
      title: "Scatter: Δln(F) vs Δln(sigma_star)",
      xaxis: {{ title: "Δln(F)" }},
      yaxis: {{ title: "Δln(sigma_star)" }},
      shapes: [
        {{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#aaa" }} }},
        {{ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: {{ dash: "dot", color: "#aaa" }} }},
      ],
      margin: {{ l: 70, r: 40, t: 48, b: 60 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    const sk = P.skew_series || [];
    const rr = P.rolling || [];
    Plotly.newPlot("roll_reg", [
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.beta_ols ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#2ca02c" }},
        name: "beta OLS (1h rolling)",
        hovertemplate: "%{{x}}<br>beta_ols=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.beta_wls ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#d62728" }},
        name: "beta WLS (1h rolling)",
        hovertemplate: "%{{x}}<br>beta_wls=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.beta_huber ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#ff7f0e" }},
        name: "beta Huber-IRLS (1h rolling)",
        hovertemplate: "%{{x}}<br>beta_huber=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.beta_lad ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#8c564b" }},
        name: "beta LAD-IRLS (1h rolling)",
        hovertemplate: "%{{x}}<br>beta_lad=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.beta_ts ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#17becf" }},
        name: "beta Theil-Sen (pairwise median, 1h rolling)",
        hovertemplate: "%{{x}}<br>beta_ts=%{{y:.6f}}<extra></extra>",
      }},
    ], {{
      title: "Rolling 1-hour regression slope (Δlnσ vs ΔlnF)",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "beta" }},
      margin: {{ l: 70, r: 40, t: 48, b: 90 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    Plotly.newPlot("roll_ssr", [
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.ssr_ols ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#2ca02c" }},
        name: "SSR OLS (1h rolling, beta/avg_skew)",
        hovertemplate: "%{{x}}<br>ssr_ols=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.ssr_wls ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#d62728" }},
        name: "SSR WLS (1h rolling, beta/avg_skew)",
        hovertemplate: "%{{x}}<br>ssr_wls=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.ssr_huber ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#ff7f0e" }},
        name: "SSR Huber-IRLS (1h rolling, beta/avg_skew)",
        hovertemplate: "%{{x}}<br>ssr_huber=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.ssr_lad ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#8c564b" }},
        name: "SSR LAD-IRLS (1h rolling, beta/avg_skew)",
        hovertemplate: "%{{x}}<br>ssr_lad=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.ssr_ts ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2, color: "#17becf" }},
        name: "SSR Theil-Sen (1h rolling, beta/avg_skew)",
        hovertemplate: "%{{x}}<br>ssr_ts=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines",
        x: rr.map(p => p.timestamp),
        y: rr.map(p => p.avg_skew ?? null),
        line: {{ width: 1.5, color: "#1f77b4", dash: "dot" }},
        name: "avg skew (1h window)",
        yaxis: "y2",
        hovertemplate: "%{{x}}<br>avg_skew=%{{y:.6f}}<extra></extra>",
      }},
    ], {{
      title: "Rolling 1-hour SSR at window end",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "SSR" }},
      yaxis2: {{
        title: "avg skew",
        overlaying: "y",
        side: "right",
        showgrid: false
      }},
      margin: {{ l: 70, r: 70, t: 48, b: 90 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    const pr = P.pred_series || [];
    Plotly.newPlot("pred_dlnsigma", [
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_actual ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2.4, color: "#1f77b4" }},
        name: "Δlnσ actual",
        hovertemplate: "%{{x}}<br>dlnSigma_actual=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_naive ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#9467bd" }},
        name: "Δlnσ pred naive (=0)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_naive=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_ols ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#2ca02c" }},
        name: "Δlnσ pred (SSR OLS)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_ols=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_wls ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#d62728" }},
        name: "Δlnσ pred (SSR WLS)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_wls=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_huber ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#ff7f0e" }},
        name: "Δlnσ pred (SSR Huber-IRLS)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_huber=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_lad ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#8c564b" }},
        name: "Δlnσ pred (SSR LAD-IRLS)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_lad=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.dlnSigma_pred_ts ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#17becf" }},
        name: "Δlnσ pred (SSR Theil-Sen)",
        hovertemplate: "%{{x}}<br>dlnSigma_pred_ts=%{{y:.6f}}<extra></extra>",
      }},
    ], {{
      title: "One-step Δlnσ prediction vs actual (no level persistence masking)",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "Δlnσ" }},
      margin: {{ l: 70, r: 40, t: 48, b: 90 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    Plotly.newPlot("pred_sigma", [
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_actual ?? null),
        marker: {{ size: 5 }},
        line: {{ width: 2.4, color: "#1f77b4" }},
        name: "sigma actual",
        hovertemplate: "%{{x}}<br>sigma_actual=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_naive ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#9467bd" }},
        name: "sigma pred naive (sigma_prev)",
        hovertemplate: "%{{x}}<br>sigma_pred_naive=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_ols ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#2ca02c" }},
        name: "sigma pred (SSR OLS)",
        hovertemplate: "%{{x}}<br>sigma_pred_ols=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_wls ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#d62728" }},
        name: "sigma pred (SSR WLS)",
        hovertemplate: "%{{x}}<br>sigma_pred_wls=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_huber ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#ff7f0e" }},
        name: "sigma pred (SSR Huber-IRLS)",
        hovertemplate: "%{{x}}<br>sigma_pred_huber=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_lad ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#8c564b" }},
        name: "sigma pred (SSR LAD-IRLS)",
        hovertemplate: "%{{x}}<br>sigma_pred_lad=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines+markers",
        x: pr.map(p => p.timestamp),
        y: pr.map(p => p.sigma_pred_ts ?? null),
        marker: {{ size: 4 }},
        line: {{ width: 2, color: "#17becf" }},
        name: "sigma pred (SSR Theil-Sen)",
        hovertemplate: "%{{x}}<br>sigma_pred_ts=%{{y:.6f}}<extra></extra>",
      }},
    ], {{
      title: "Predicted sigma series (rolling SSR) vs actual sigma",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "sigma" }},
      margin: {{ l: 70, r: 40, t: 48, b: 90 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    const act = pr.map(p => p.sigma_actual).filter(v => v !== null && v !== undefined && Number.isFinite(v));
    const olsX = pr.map(p => p.sigma_actual ?? null);
    const olsY = pr.map(p => p.sigma_pred_ols ?? null);
    const wlsX = pr.map(p => p.sigma_actual ?? null);
    const wlsY = pr.map(p => p.sigma_pred_wls ?? null);
    const naiX = pr.map(p => p.sigma_actual ?? null);
    const naiY = pr.map(p => p.sigma_pred_naive ?? null);
    const hovPar = pr.map(p =>
      `${{p.timestamp}}<br>actual=${{(p.sigma_actual ?? NaN).toFixed ? p.sigma_actual.toFixed(6) : p.sigma_actual}}`
    );
    const pmin = act.length ? Math.min(...act) : 0.0;
    const pmax = act.length ? Math.max(...act) : 1.0;
    const ppad = (pmax - pmin) * 0.05 + 1e-9;
    const q0 = pmin - ppad;
    const q1 = pmax + ppad;
    Plotly.newPlot("pred_parity", [
      {{
        type: "scatter",
        mode: "markers",
        x: olsX,
        y: olsY,
        marker: {{ size: 6, color: "#2ca02c", opacity: 0.7 }},
        name: "OLS pred vs actual (same timestamp)",
        text: hovPar,
        hovertemplate: "%{{text}}<br>pred_ols=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "markers",
        x: naiX,
        y: naiY,
        marker: {{ size: 6, color: "#9467bd", opacity: 0.6 }},
        name: "Naive pred vs actual",
        text: hovPar,
        hovertemplate: "%{{text}}<br>pred_naive=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "markers",
        x: wlsX,
        y: wlsY,
        marker: {{ size: 6, color: "#d62728", opacity: 0.7 }},
        name: "WLS pred vs actual (same timestamp)",
        text: hovPar,
        hovertemplate: "%{{text}}<br>pred_wls=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "markers",
        x: pr.map(p => p.sigma_actual ?? null),
        y: pr.map(p => p.sigma_pred_ts ?? null),
        marker: {{ size: 6, color: "#17becf", opacity: 0.7 }},
        name: "Theil-Sen SSR pred vs actual",
        text: hovPar,
        hovertemplate: "%{{text}}<br>pred_ts=%{{y:.6f}}<extra></extra>",
      }},
      {{
        type: "scatter",
        mode: "lines",
        x: [q0, q1],
        y: [q0, q1],
        line: {{ width: 2, dash: "dot", color: "#444" }},
        name: "y = x",
        hoverinfo: "skip",
      }},
    ], {{
      title: "Predicted vs Actual sigma at same timestamp",
      xaxis: {{ title: "actual sigma(t)" }},
      yaxis: {{ title: "predicted sigma(t)", scaleanchor: "x", scaleratio: 1 }},
      margin: {{ l: 70, r: 40, t: 48, b: 60 }},
      hovermode: "closest",
    }}, {{ responsive: true }});

    Plotly.newPlot("skew_ts", [{{
      type: "scatter",
      mode: "lines+markers",
      x: sk.map(p => p.timestamp),
      y: sk.map(p => p.s_atf_norm),
      marker: {{ size: 6 }},
      line: {{ width: 2, color: "#d62728" }},
      hovertemplate: "%{{x}}<br>s_atf_norm=%{{y:.6f}}<extra></extra>",
      name: "s_atf_norm",
    }}], {{
      title: "ATF normalized skew over time",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "s_atf_norm" }},
      margin: {{ l: 70, r: 40, t: 48, b: 90 }},
      hovermode: "closest",
    }}, {{ responsive: true }});
    document.getElementById("diag").textContent = JSON.stringify(P.diagnostics || {{}}, null, 2);
  </script>
</body>
</html>
"""


def render_ln_regression_methods_html(payload: dict, title: str) -> str:
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
    .plot {{ width: 100%; height: 620px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>No-Intercept Robust Regression Diagnostics</h1>
  <p style="color:#444;font-size:0.92rem;max-width:920px;line-height:1.45">
    Bottom section: <b>ATM 2h hedge residuals</b> ε = ΔC − δ·ΔF (BS prices, overlapping windows).
    Tighter distributions with mean near zero are generally better for delta-only hedging; overlapping samples are correlated.
  </p>
  <div id="beta_ts" class="plot"></div>
  <div id="ssr_ts" class="plot"></div>
  <div id="hedge_hist" class="plot"></div>
  <div id="hedge_eps_ts" class="plot"></div>
  <h3>Method Summary</h3>
  <pre id="summary"></pre>
  <h3>Diagnostics</h3>
  <pre id="diag"></pre>
  <script>
    const P = {data_json};
    const rr = P.rolling || [];
    const methods = [
      {{ key: "ols", label: "OLS", color: "#2ca02c" }},
      {{ key: "wls", label: "WLS", color: "#d62728" }},
      {{ key: "huber", label: "Huber-IRLS", color: "#ff7f0e" }},
      {{ key: "lad", label: "LAD-IRLS", color: "#8c564b" }},
      {{ key: "ts", label: "Theil-Sen", color: "#17becf" }},
    ];
    Plotly.newPlot("beta_ts", methods.map(m => ({{
      type: "scatter",
      mode: "lines+markers",
      x: rr.map(r => r.timestamp),
      y: rr.map(r => r[`beta_${{m.key}}`] ?? null),
      marker: {{ size: 5 }},
      line: {{ width: 2, color: m.color }},
      name: `beta ${{m.label}}`,
      hovertemplate: "%{{x}}<br>beta=%{{y:.6f}}<extra></extra>",
    }})), {{
      title: "Rolling slope beta (no intercept)",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "beta" }},
      margin: {{ l: 70, r: 40, t: 48, b: 90 }},
    }}, {{ responsive: true }});
    Plotly.newPlot("ssr_ts", methods.map(m => ({{
      type: "scatter",
      mode: "lines+markers",
      x: rr.map(r => r.timestamp),
      y: rr.map(r => r[`ssr_${{m.key}}`] ?? null),
      marker: {{ size: 5 }},
      line: {{ width: 2, color: m.color }},
      name: `SSR ${{m.label}}`,
      hovertemplate: "%{{x}}<br>ssr=%{{y:.6f}}<extra></extra>",
    }})).concat([{{
      type: "scatter",
      mode: "lines",
      x: rr.map(r => r.timestamp),
      y: rr.map(r => r.avg_skew ?? null),
      line: {{ width: 1.5, color: "#1f77b4", dash: "dot" }},
      name: "avg skew (1h window)",
      yaxis: "y2",
      hovertemplate: "%{{x}}<br>avg_skew=%{{y:.6f}}<extra></extra>",
    }}]), {{
      title: "Rolling SSR by method",
      xaxis: {{ title: "timestamp", tickangle: -45 }},
      yaxis: {{ title: "SSR" }},
      yaxis2: {{ title: "avg skew", overlaying: "y", side: "right", showgrid: false }},
      margin: {{ l: 70, r: 70, t: 48, b: 90 }},
    }}, {{ responsive: true }});
    const H = P.hedge_atm_2h || null;
    const hRows = (H && H.rows) ? H.rows : [];
    const epsMethods = [
      {{ col: "eps_naive", label: "Naive δ", color: "#7f7f7f" }},
      {{ col: "eps_hedge_ols", label: "Rolling OLS SSR", color: "#2ca02c" }},
      {{ col: "eps_hedge_wls", label: "Rolling WLS SSR", color: "#d62728" }},
      {{ col: "eps_hedge_huber", label: "Rolling Huber SSR", color: "#ff7f0e" }},
      {{ col: "eps_hedge_lad", label: "Rolling LAD SSR", color: "#8c564b" }},
      {{ col: "eps_hedge_ts", label: "Rolling Theil-Sen SSR", color: "#17becf" }},
    ];
    if (hRows.length > 0) {{
      Plotly.newPlot("hedge_hist", epsMethods.map(em => ({{
        type: "histogram",
        name: em.label,
        x: hRows.map(r => r[em.col]).filter(v => v != null && Number.isFinite(v)),
        opacity: 0.55,
        marker: {{ color: em.color }},
        histnorm: "probability density",
        nbinsx: 45,
      }})), {{
        title: "Distribution of ATM 2h hedge residuals ε",
        xaxis: {{ title: "ε = ΔC − δ·ΔF" }},
        yaxis: {{ title: "density" }},
        barmode: "overlay",
        margin: {{ l: 60, r: 30, t: 48, b: 50 }},
      }}, {{ responsive: true }});
      Plotly.newPlot("hedge_eps_ts", epsMethods.map(em => ({{
        type: "scatter",
        mode: "lines+markers",
        x: hRows.map(r => r.to_t),
        y: hRows.map(r => r[em.col] ?? null),
        name: em.label,
        line: {{ width: 1.4, color: em.color }},
        marker: {{ size: 4 }},
        hovertemplate: "%{{x}}<br>%{{y:.6f}}<extra></extra>",
      }})), {{
        title: "Hedge residual time series (ATM, 2h horizon)",
        xaxis: {{ title: "to_t", tickangle: -45 }},
        yaxis: {{ title: "ε" }},
        margin: {{ l: 60, r: 30, t: 48, b: 100 }},
        shapes: [{{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#888" }} }}],
      }}, {{ responsive: true }});
    }} else {{
      document.getElementById("hedge_hist").innerHTML =
        "<p style='padding:16px;color:#666'>No hedge residual rows (hedge_atm_2h missing).</p>";
      document.getElementById("hedge_eps_ts").innerHTML = "";
    }}

    function median(arr) {{
      if (!arr.length) return null;
      const x = [...arr].sort((a, b) => a - b);
      const m = Math.floor(x.length / 2);
      return (x.length % 2) ? x[m] : 0.5 * (x[m - 1] + x[m]);
    }}
    const summaryRegression = methods.map(m => {{
      const betaVals = rr.map(r => r[`beta_${{m.key}}`]).filter(v => Number.isFinite(v));
      const ssrVals = rr.map(r => r[`ssr_${{m.key}}`]).filter(v => Number.isFinite(v));
      const hedgeEps = hRows.map(r => r[`eps_hedge_${{m.key}}`]).filter(v => v != null && Number.isFinite(v));
      const mseHedge = hedgeEps.length ? hedgeEps.reduce((acc, r) => acc + r * r, 0) / hedgeEps.length : null;
      return {{
        method: m.label,
        n_beta: betaVals.length,
        n_ssr: ssrVals.length,
        n_hedge_eps: hedgeEps.length,
        beta_median: median(betaVals),
        ssr_median: median(ssrVals),
        mse_hedge_eps: mseHedge,
      }};
    }});
    const naiveEps = hRows.map(r => r.eps_naive).filter(v => v != null && Number.isFinite(v));
    const mseNaive = naiveEps.length ? naiveEps.reduce((a, b) => a + b * b, 0) / naiveEps.length : null;
    document.getElementById("summary").textContent = JSON.stringify(
      {{
        regression_by_method: summaryRegression,
        hedge_summary_from_payload: (H && H.summary) ? H.summary : null,
        naive_hedge_mse: mseNaive,
        n_hedge_rows: hRows.length,
      }},
      null,
      2
    );
    document.getElementById("diag").textContent = JSON.stringify(P.diagnostics || {{}}, null, 2);
  </script>
</body>
</html>
"""


def render_ssr_scatter_html(payload: dict, title: str) -> str:
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
    .plot {{ width: 100%; height: 620px; background: #fff; border: 1px solid #ddd; border-radius: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 12px; overflow: auto; }}
    .hint {{ color: #444; font-size: 0.9rem; max-width: 1000px; }}
  </style>
</head>
<body>
  <h1>Quick SSR Scatter</h1>
  <p class="hint">Points are transition-level values with x = ΔlnF and y = Δlnσ*/s_atf_norm(t). The line is origin-constrained SSR fit (β = Σxy/Σx²).</p>
  <div id="ssr_sc" class="plot"></div>
  <h3>Diagnostics</h3>
  <pre id="diag"></pre>
  <script>
    const P = {data_json};
    const pts = P.points || [];
    const d = P.diagnostics || {{}};
    const xs = pts.map(p => p.x);
    const ys = pts.map(p => p.y);
    const hover = pts.map(p =>
      `${{p.from_t}} → ${{p.to_t}}<br>ΔlnF=${{p.x.toFixed(6)}}<br>Δlnσ*/s=${{p.y.toFixed(6)}}<br>s_atf_norm=${{p.s_atf_norm.toFixed(6)}}`
    );
    const traces = [{{
      type: "scatter",
      mode: "markers",
      x: xs,
      y: ys,
      text: hover,
      marker: {{ size: 9, color: "#1f77b4", opacity: 0.85 }},
      hovertemplate: "%{{text}}<extra></extra>",
      name: "transitions",
    }}];
    if (d.ssr_hat_origin !== null && d.ssr_hat_origin !== undefined && isFinite(d.ssr_hat_origin)) {{
      const xmin = Math.min(...xs);
      const xmax = Math.max(...xs);
      traces.push({{
        type: "scatter",
        mode: "lines",
        x: [xmin, xmax],
        y: [d.ssr_hat_origin * xmin, d.ssr_hat_origin * xmax],
        line: {{ width: 3, color: "#d62728" }},
        name: "SSR_hat(origin) = " + d.ssr_hat_origin.toFixed(6),
      }});
    }}
    Plotly.newPlot("ssr_sc", traces, {{
      title: "SSR scatter (Δlnσ*/s_atf_norm vs ΔlnF)",
      xaxis: {{ title: "ΔlnF" }},
      yaxis: {{ title: "Δlnσ* / s_atf_norm(t)" }},
      margin: {{ l: 70, r: 40, t: 48, b: 60 }},
      hovermode: "closest",
      shapes: [
        {{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 0, y1: 0, line: {{ dash: "dot", color: "#aaa" }} }},
        {{ type: "line", xref: "x", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1, line: {{ dash: "dot", color: "#aaa" }} }},
      ],
    }}, {{ responsive: true }});
    document.getElementById("diag").textContent = JSON.stringify(d, null, 2);
  </script>
</body>
</html>
"""


def render_html(payload: dict, title: str) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 16px; background: #fafafa; }}
    h1 {{ font-size: 1.15rem; }}
    h2 {{ font-size: 1rem; margin-top: 24px; }}
    .hint {{ color: #444; font-size: 0.88rem; max-width: 960px; }}
    .plot {{ width: 100%; height: 480px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; }}
    .plot-tall {{ height: 560px; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>CVI batch — q(t) and vol vs spot-proxy changes</h1>
  <p class="hint" id="spotnote"></p>

  <h2>q by expiry (heatmap: time &times; expiry)</h2>
  <div id="q_heat" class="plot plot-tall"></div>

  <h2>q time series — pick expiry</h2>
  <div class="row">
    <label>Expiry <select id="exp_pick"></select></label>
  </div>
  <div id="q_line" class="plot"></div>

  <h2>Fixed strike: WLS of % &Delta;&sigma; on % &Delta;S (one regression line per expiry)</h2>
  <p class="hint">Pick strike K. <b>Markers</b> = observed (%&Delta;S, %&Delta;&sigma;) per transition; <b>line</b> = WLS through those points (weights ∝ 1/(ε+ΔS²+Δσ²), less influence from large joint moves). Axes use <b>1:1 data scaling</b> (y scale-anchored to x) so the line&rsquo;s <b>visual tilt matches &beta;</b> (rise/run in % per %). <b>Color</b>: front expiry = green &rarr; back = red.</p>
  <div class="row">
    <label>Strike K <select id="strike_pick"></select></label>
  </div>
  <div id="sc_strike" class="plot plot-tall"></div>
  <div id="reg_table_wrap" style="overflow:auto;max-width:100%"></div>

  <h2>Per expiry: &beta; across strikes (slope vs moneyness)</h2>
  <p class="hint">Same WLS as above (%&Delta;&sigma; on %&Delta;F(0)), but <b>fixed expiry</b> and varying strike. Table and chart show how &beta; moves as K changes; <b>&Delta;&beta;</b> = change from previous row (sorted by strike).</p>
  <div class="row">
    <label>Expiry <select id="exp_pick_beta"></select></label>
  </div>
  <div id="beta_vs_strike" class="plot"></div>
  <div id="beta_strike_table_wrap" style="overflow:auto;max-width:100%"></div>

  <script>
    const P = {data_json};
    document.getElementById("spotnote").textContent = P.spot_note;

    const timeLabs = P.short_times;

    Plotly.newPlot("q_heat", [{{
      type: "heatmap",
      z: P.q_heatmap,
      x: timeLabs,
      y: P.expiry_labels,
      colorscale: "RdBu",
      zmid: 0,
      hovertemplate: "t=%{{x}}<br>%{{y}}<br>q=%{{z:.5f}}<extra></extra>",
    }}], {{
      title: "Dividend yield q (expiry &times; snapshot time)",
      xaxis: {{ title: "snapshot (local time)", tickangle: -45 }},
      yaxis: {{ title: "expiry" }},
      margin: {{ l: 220, b: 120, t: 40 }},
    }}, {{ responsive: true }});

    const sel = document.getElementById("exp_pick");
    P.expiry_indices.forEach((idx, i) => {{
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = P.expiry_labels[i];
      sel.appendChild(o);
    }});

    function redrawQline() {{
      const ii = parseInt(sel.value, 10);
      const ys = P.q_heatmap[ii];
      Plotly.react("q_line", [{{
        x: timeLabs, y: ys, mode: "lines+markers", type: "scatter",
        line: {{ width: 2 }}, marker: {{ size: 8 }},
        hovertemplate: "t=%{{x}}<br>q=%{{y:.6f}}<extra></extra>",
      }}], {{
        title: "q(t) — " + P.expiry_labels[ii],
        xaxis: {{ title: "snapshot", tickangle: -45 }},
        yaxis: {{ title: "q" }},
        margin: {{ b: 100 }},
      }}, {{ responsive: true }});
    }}
    sel.addEventListener("change", redrawQline);
    redrawQline();

    const selK = document.getElementById("strike_pick");
    P.strikes_list.forEach((ks) => {{
      const o = document.createElement("option");
      o.value = ks;
      o.textContent = ks;
      if (ks === P.default_strike) o.selected = true;
      selK.appendChild(o);
    }});

    function regLineColor(i, nExp) {{
      if (nExp <= 1) return "hsl(120, 78%, 40%)";
      const t = i / (nExp - 1);
      const hue = 120 * (1 - t);
      return `hsl(${{hue}}, 78%, 40%)`;
    }}

    function redrawStrikeScat() {{
      const k = selK.value;
      const inner = P.strike_scatter[k] || {{}};
      const expKeys = Object.keys(inner).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
      const nExp = expKeys.length;
      const traces = [];
      let rows = "";
      for (let i = 0; i < expKeys.length; i++) {{
        const e = expKeys[i];
        const blk = inner[e];
        if (!blk || !blk.line_x) continue;
        const lab = P.expiry_label_by_idx[e] || ("expiry " + e);
        const col = regLineColor(i, nExp);
        const a = blk.alpha;
        const b = blk.beta;
        const nn = blk.n;
        const raw = blk.pts || [];
        traces.push({{
          type: "scatter",
          mode: "markers",
          x: raw.map((p) => p.x),
          y: raw.map((p) => p.y),
          text: raw.map((p) => p.t_label),
          marker: {{ size: 9, color: col, opacity: 0.82, line: {{ width: 1, color: "rgba(255,255,255,0.7)" }} }},
          name: lab + " (obs)",
          showlegend: false,
          hovertemplate: "%{{text}}<br>%ΔS=%{{x:.4f}}%<br>%Δσ=%{{y:.4f}}%<extra></extra>",
        }});
        const hov = lab + "<br>β=" + b.toFixed(6) + "<br>α=" + a.toFixed(6) + "<br>n=" + nn;
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: blk.line_x,
          y: blk.line_y,
          name: lab + "  β=" + b.toFixed(5) + "  α=" + a.toFixed(5),
          line: {{ width: 3, color: col }},
          hovertext: [hov, hov],
          hovertemplate: "%{{hovertext}}<extra></extra>",
        }});
        rows += "<tr><td>" + lab + "</td><td style='font-family:monospace'>" + a.toFixed(6)
          + "</td><td style='font-family:monospace'>" + b.toFixed(6) + "</td><td>" + nn + "</td>"
          + "<td style='background:" + col + ";width:24px'></td></tr>";
      }}
      document.getElementById("reg_table_wrap").innerHTML =
        "<table cellpadding='8' cellspacing='0' style='border-collapse:collapse;font-size:0.9rem'>"
        + "<thead><tr style='background:#eee'><th>Expiry</th><th>α (intercept)</th><th>β (slope)</th><th>n</th><th></th></tr></thead>"
        + "<tbody>" + rows + "</tbody></table>";
      const sh = [
        {{ type: "line", x0: -1e6, x1: 1e6, y0: 0, y1: 0, line: {{ dash: "dot", color: "#aaa" }} }},
        {{ type: "line", x0: 0, x1: 0, y0: -1e6, y1: 1e6, line: {{ dash: "dot", color: "#aaa" }} }},
      ];
      Plotly.react("sc_strike", traces, {{
        title: "K = " + k + " — WLS %Δσ ≈ α + β·(%ΔS) [equal x/y scale: geometric slope = β]",
        xaxis: {{
          title: "% change in S (F at expiry 0 proxy)",
          constrain: "domain",
          zeroline: true,
          zerolinewidth: 1,
        }},
        yaxis: {{
          title: "% change in fitted vol",
          scaleanchor: "x",
          scaleratio: 1,
          constrain: "domain",
          zeroline: true,
          zerolinewidth: 1,
        }},
        legend: {{ orientation: "v", x: 1.02, y: 1, font: {{ size: 10 }} }},
        margin: {{ r: 200, b: 60, t: 48, l: 60 }},
        hovermode: "closest",
        shapes: sh,
      }}, {{ responsive: true }});
    }}
    selK.addEventListener("change", redrawStrikeScat);
    redrawStrikeScat();

    const selBe = document.getElementById("exp_pick_beta");
    const bbe = P.beta_by_expiry || {{}};
    const expBetaKeys = Object.keys(bbe).sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
    expBetaKeys.forEach((ek, j) => {{
      const o = document.createElement("option");
      o.value = ek;
      const lab = (P.expiry_label_by_idx && P.expiry_label_by_idx[ek]) ? P.expiry_label_by_idx[ek] : ("idx " + ek);
      o.textContent = lab;
      if (j === 0) o.selected = true;
      selBe.appendChild(o);
    }});

    function redrawBetaVsStrike() {{
      const ek = selBe.value;
      const rows = bbe[ek] || [];
      if (rows.length === 0) {{
        document.getElementById("beta_strike_table_wrap").innerHTML =
          "<p class='hint'>No &beta; data for this expiry.</p>";
        Plotly.purge("beta_vs_strike");
        return;
      }}
      const xs = rows.map((r) => r.strike);
      const betas = rows.map((r) => r.beta);
      let tableRows = "";
      let prevB = null;
      rows.forEach((r) => {{
        const db = prevB === null ? "" : (r.beta - prevB).toFixed(6);
        prevB = r.beta;
        tableRows += "<tr><td style='font-family:monospace'>" + r.strike_key
          + "</td><td style='font-family:monospace'>" + r.beta.toFixed(6)
          + "</td><td style='font-family:monospace'>" + r.alpha.toFixed(6)
          + "</td><td>" + r.n + "</td><td style='font-family:monospace'>" + db + "</td></tr>";
      }});
      document.getElementById("beta_strike_table_wrap").innerHTML =
        "<table cellpadding='8' cellspacing='0' style='border-collapse:collapse;font-size:0.9rem'>"
        + "<thead><tr style='background:#eee'><th>Strike K</th><th>&beta;</th><th>&alpha;</th><th>n</th><th>&Delta;&beta;</th></tr></thead>"
        + "<tbody>" + tableRows + "</tbody></table>";

      const lab = (P.expiry_label_by_idx && P.expiry_label_by_idx[ek]) ? P.expiry_label_by_idx[ek] : ("idx " + ek);
      Plotly.react("beta_vs_strike", [{{
        type: "scatter",
        mode: "lines+markers",
        x: xs,
        y: betas,
        marker: {{ size: 10 }},
        line: {{ width: 2, color: "#2563eb" }},
        hovertemplate: "K=%{{x}}<br>&beta;=%{{y:.6f}}<extra></extra>",
      }}], {{
        title: "&beta; vs strike — " + lab,
        xaxis: {{ title: "strike K" }},
        yaxis: {{ title: "&beta; (WLS slope, %&Delta;&sigma; per %&Delta;S)" }},
        margin: {{ l: 60, r: 40, t: 48, b: 48 }},
      }}, {{ responsive: true }});
    }}

    if (expBetaKeys.length > 0) {{
      selBe.addEventListener("change", redrawBetaVsStrike);
      redrawBetaVsStrike();
    }} else {{
      document.getElementById("beta_strike_table_wrap").innerHTML =
        "<p class='hint'>No per-expiry strike cross-section (need at least one strike with regressions per expiry).</p>";
    }}
  </script>
</body>
</html>
"""


