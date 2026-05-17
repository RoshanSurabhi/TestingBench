#!/usr/bin/env python3
"""
Build a standalone HTML page (Plotly) that overlays fitted implied vol across time:
  - z-space: fitted_surface.csv (dense z grid)
  - strike-space: option_fit_comparison.csv (strike vs fitted_vol per expiry)

Usage:
  python plot_cvi_batch_vol_overlay.py "C:/path/to/cvi_fits_10_1030_full_artifacts"
  python plot_cvi_batch_vol_overlay.py   # defaults to script's parent/../data/AAPL/cvi_fits_10_1030_full_artifacts if present

Open the generated cvi_batch_vol_overlay.html via file:// in a browser.
No network fetch of CSVs at view time — data is embedded in the HTML.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def read_batch_summary(batch_dir: Path) -> list[dict]:
    path = batch_dir / "batch_cvi_summary.csv"
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ok = int(row.get("ok", "0").strip())
            except ValueError:
                ok = 0
            if not ok:
                continue
            rows.append(
                {
                    "subfolder": row["subfolder"].strip(),
                    "idx_in_bin": int(row["idx_in_bin"]),
                    "timestamp": row["timestamp"].strip(),
                }
            )
    return rows


def read_expiry_labels(expiry_csv: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    with expiry_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["expiry_idx"])
            lab = row.get("expiry_date", "").strip().strip('"')
            out.append((idx, lab))
    out.sort(key=lambda x: x[0])
    return out


def read_fitted_surface(surf_csv: Path) -> dict[int, list[tuple[float, float]]]:
    by_exp: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with surf_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row["expiry_index"])
            z = float(row["z"])
            vol = float(row["vol"])
            by_exp[e].append((z, vol))
    for e in by_exp:
        by_exp[e].sort(key=lambda t: t[0])
    return dict(by_exp)


def read_option_fit_strike_fitted_vol(opt_csv: Path) -> dict[int, list[tuple[float, float]]]:
    """Per expiry: (strike, fitted_vol) sorted by strike."""
    by_exp: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with opt_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            e = int(row["expiry_index"])
            strike = float(row["strike"])
            fv = float(row["fitted_vol"])
            by_exp[e].append((strike, fv))
    for e in by_exp:
        by_exp[e].sort(key=lambda t: t[0])
    return dict(by_exp)


def build_payload(batch_dir: Path) -> dict:
    summary = read_batch_summary(batch_dir)
    if not summary:
        raise SystemExit(f"No ok rows in {batch_dir / 'batch_cvi_summary.csv'}")

    expiries: list[tuple[int, str]] | None = None
    runs_out: list[dict] = []

    for row in summary:
        sub = batch_dir / row["subfolder"]
        surf = sub / "fitted_surface.csv"
        if not surf.is_file():
            continue
        if expiries is None:
            efq = sub / "expiry_fwd_q.csv"
            if efq.is_file():
                expiries = read_expiry_labels(efq)

        by_exp_z = read_fitted_surface(surf)
        by_exp_z_json = {str(k): [[a, b] for a, b in v] for k, v in by_exp_z.items()}

        opt = sub / "option_fit_comparison.csv"
        by_exp_k_json: dict[str, list[list[float]]] = {}
        if opt.is_file():
            by_exp_k = read_option_fit_strike_fitted_vol(opt)
            by_exp_k_json = {str(k): [[a, b] for a, b in v] for k, v in by_exp_k.items()}

        runs_out.append(
            {
                "idx_in_bin": row["idx_in_bin"],
                "timestamp": row["timestamp"],
                "subfolder": row["subfolder"],
                "by_expiry_z": by_exp_z_json,
                "by_expiry_strike": by_exp_k_json,
            }
        )

    if not runs_out:
        raise SystemExit("No fitted_surface.csv found for ok batch rows.")

    if expiries is None:
        keys = sorted(int(k) for k in runs_out[0]["by_expiry_z"])
        expiries = [(k, f"expiry {k}") for k in keys]

    expiries_payload = [{"idx": i, "label": lab} for i, lab in expiries]

    return {"expiries": expiries_payload, "runs": runs_out}


def render_html(payload: dict, title: str) -> str:
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, Segoe UI, Roboto, sans-serif; margin: 16px; background: #fafafa; }}
    h1 {{ font-size: 1.1rem; margin: 0 0 8px 0; }}
    .row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }}
    label {{ font-size: 0.9rem; }}
    select, input {{ font-size: 0.9rem; padding: 4px 8px; }}
    .plot-cell {{ width: 100%; height: 520px; background: #fff; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 16px; }}
    .hint {{ color: #555; font-size: 0.85rem; max-width: 900px; }}
    h2 {{ font-size: 1rem; margin: 12px 0 6px 0; color: #333; }}
  </style>
</head>
<body>
  <h1>CVI fitted vol &mdash; overlays across snapshot times</h1>
  <p class="hint">Top: vol vs <b>z</b> (<code>fitted_surface.csv</code>). Bottom: vol vs <b>strike</b> (<code>option_fit_comparison.csv</code>). Curve color: <b style="color:#1a9f3a">green</b> = earlier in the batch window, <b style="color:#c62828">red</b> = later. Data embedded; <code>file://</code> ok.</p>
  <div class="row">
    <label>Expiry <select id="expiry"></select></label>
    <label>Max curves <input id="maxcurves" type="number" min="1" max="500" value="40"/></label>
    <button type="button" id="redraw">Redraw</button>
  </div>
  <h2>Implied vol vs z</h2>
  <div id="plot_z" class="plot-cell"></div>
  <h2>Implied vol vs strike</h2>
  <div id="plot_k" class="plot-cell"></div>
  <script>
    const DATA = {data_json};

    function spreadIndices(n, maxShow) {{
      if (n <= maxShow) return Array.from({{length: n}}, (_, i) => i);
      if (maxShow <= 1) return [n - 1];
      const out = [];
      for (let j = 0; j < maxShow; j++)
        out.push(Math.round((j * (n - 1)) / (maxShow - 1)));
      return out;
    }}

    function shortTime(ts) {{
      const p = ts.trim().split(/\\s+/);
      if (p.length < 2) return ts;
      let t = p[1];
      if (/^\\d{{2}}:\\d{{2}}:\\d{{2}}-\\d{{2}}$/.test(t)) t = t.slice(0, 8);
      return t;
    }}

    /** ui = index in DATA.runs (chronological). Green (hue 120) early -> red (hue 0) late. */
    function colorForRunIndex(ui, nRuns) {{
      if (nRuns <= 1) return "hsl(120, 78%, 40%)";
      const t = ui / (nRuns - 1);
      const hue = 120 * (1 - t);
      return `hsl(${{hue}}, 78%, 40%)`;
    }}

    const sel = document.getElementById("expiry");
    DATA.expiries.forEach((e) => {{
      const o = document.createElement("option");
      o.value = String(e.idx);
      o.textContent = e.idx + " — " + e.label;
      sel.appendChild(o);
    }});

    function redraw() {{
      const expIdx = sel.value;
      const maxC = Math.max(1, parseInt(document.getElementById("maxcurves").value, 10) || 40);
      const runs = DATA.runs;
      const useIdx = spreadIndices(runs.length, maxC);
      const expLabel = DATA.expiries.find((e) => String(e.idx) === expIdx);
      const sub = expLabel ? expLabel.label : expIdx;

      const tracesZ = [];
      const tracesK = [];
      const nRuns = runs.length;
      for (const ui of useIdx) {{
        const run = runs[ui];
        const col = colorForRunIndex(ui, nRuns);
        const leg = shortTime(run.timestamp) + "  #" + run.idx_in_bin;

        const ptsZ = run.by_expiry_z[expIdx];
        if (ptsZ && ptsZ.length) {{
          tracesZ.push({{
            type: "scatter",
            mode: "lines",
            x: ptsZ.map((p) => p[0]),
            y: ptsZ.map((p) => p[1]),
            name: leg,
            line: {{ width: 1.4, color: col }},
          }});
        }}

        const ptsK = run.by_expiry_strike[expIdx];
        if (ptsK && ptsK.length) {{
          tracesK.push({{
            type: "scatter",
            mode: "lines",
            x: ptsK.map((p) => p[0]),
            y: ptsK.map((p) => p[1]),
            name: leg,
            line: {{ width: 1.4, color: col }},
          }});
        }}
      }}

      const baseTitle = "expiry " + expIdx + " (" + sub + ") — " + useIdx.length + " curves shown / " + runs.length + " snapshots";
      if (typeof Plotly === "undefined") {{
        const msg = "Plotly failed to load (check network / CDN). Charts need the script from cdn.plot.ly.";
        document.getElementById("plot_z").textContent = msg;
        document.getElementById("plot_k").textContent = msg;
        return;
      }}
      const elZ = document.getElementById("plot_z");
      const elK = document.getElementById("plot_k");
      Plotly.react(elZ, tracesZ, {{
        title: "Vol vs z — " + baseTitle,
        xaxis: {{ title: "z" }},
        yaxis: {{ title: "implied vol (fitted)" }},
        legend: {{ orientation: "h", y: -0.22, font: {{ size: 9 }} }},
        margin: {{ t: 44, r: 20, b: 100, l: 52 }},
        hovermode: "closest",
      }}, {{ responsive: true }});

      Plotly.react(elK, tracesK, {{
        title: "Vol vs strike — " + baseTitle + (tracesK.length ? "" : " (no option_fit_comparison data)"),
        xaxis: {{ title: "strike" }},
        yaxis: {{ title: "implied vol (fitted)" }},
        legend: {{ orientation: "h", y: -0.22, font: {{ size: 9 }} }},
        margin: {{ t: 44, r: 20, b: 100, l: 52 }},
        hovermode: "closest",
      }}, {{ responsive: true }});
    }}

    sel.addEventListener("change", redraw);
    document.getElementById("redraw").addEventListener("click", redraw);
    redraw();
  </script>
</body>
</html>
"""


def main() -> None:
    if len(sys.argv) > 1:
        batch_dir = Path(sys.argv[1]).resolve()
    else:
        repo_root = Path(__file__).resolve().parents[2]
        batch_dir = (repo_root / "data" / "AAPL" / "cvi_fits_10_1030_full_artifacts").resolve()

    if not batch_dir.is_dir():
        print("Usage: python plot_cvi_batch_vol_overlay.py <cvi_batch_output_dir>", file=sys.stderr)
        raise SystemExit(2)

    payload = build_payload(batch_dir)
    out_html = batch_dir / "cvi_batch_vol_overlay.html"
    title = f"CVI batch — {batch_dir.name}"
    out_html.write_text(render_html(payload, title), encoding="utf-8")
    print(f"Wrote {out_html}")
    print(f"Open: file:///{str(out_html).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
