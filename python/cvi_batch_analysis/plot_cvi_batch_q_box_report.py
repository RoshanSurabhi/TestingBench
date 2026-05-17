#!/usr/bin/env python3
"""
CVI batch — minimal HTML: plots only (no data tables).

  1) Per-expiry dropdown: q(t) over the full batch time range (ok=1, expiry_fwd_q.csv).
  2) Summary box plots by calendar day: all q values pooled per day (all expiries).

Usage:
  python plot_cvi_batch_q_box_report.py <batch_dir> [out.html]
  Default out: <batch_dir>/q_daily_plots.html
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def day_key(timestamp: str) -> str:
    """Calendar date from snapshot string, e.g. 2026-04-06 09:30:12-04 -> 2026-04-06."""
    ts = timestamp.strip()
    if not ts:
        return ""
    return ts.split()[0]


def build_payload(batch_dir: Path) -> dict:
    summary_path = batch_dir / "batch_cvi_summary.csv"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))

    by_expiry: dict[str, dict[str, list]] = defaultdict(lambda: {"t": [], "q": []})
    by_day: dict[str, list[float]] = defaultdict(list)
    n_snaps = 0

    for row in rows:
        if int(row.get("ok", 0) or 0) != 1:
            continue
        sub = batch_dir / row["subfolder"].strip()
        efq = sub / "expiry_fwd_q.csv"
        if not efq.is_file():
            continue
        ts = row.get("timestamp", "").strip()
        dk = day_key(ts)
        n_row = 0
        with efq.open(newline="", encoding="utf-8") as ef:
            for erow in csv.DictReader(ef):
                qv = float(erow["q"])
                if not math.isfinite(qv):
                    continue
                d = erow.get("expiry_date", "").strip().strip('"')
                lab = f'{int(erow["expiry_idx"])}: {d}'
                by_expiry[lab]["t"].append(ts)
                by_expiry[lab]["q"].append(qv)
                if dk:
                    by_day[dk].append(qv)
                n_row += 1
        if n_row > 0:
            n_snaps += 1

    exp_labels = sorted(by_expiry.keys(), key=lambda s: int(s.split(":", 1)[0]))
    by_expiry_out = {lab: by_expiry[lab] for lab in exp_labels}
    days_sorted = sorted(by_day.keys())

    return {
        "batch_name": batch_dir.name,
        "n_snapshots_used": n_snaps,
        "exp_labels": exp_labels,
        "by_expiry": by_expiry_out,
        "by_day": {d: by_day[d] for d in days_sorted},
    }


def render_html(payload: dict, title: str) -> str:
    # Indented JSON; concatenate so JSON braces are not interpreted as f-string braces
    data_json = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
    bn = payload["batch_name"]
    ns = payload["n_snapshots_used"]
    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 16px; max-width: 1400px; }}
    h1 {{ font-size: 1.1rem; }}
    h2 {{ font-size: 0.95rem; margin-top: 1rem; color: #444; }}
    .plot {{ width: 100%; height: 440px; margin: 10px 0; border: 1px solid #ddd; border-radius: 6px; }}
    #boot-err {{ color: #b00020; font-weight: 600; }}
  </style>
</head>
<body>
<h1>q — {bn} ({ns} snapshots)</h1>
<h2>q vs time — pick expiry</h2>
<p style="font-size:0.9rem;color:#444">Full batch timeline for one expiry at a time.</p>
<div id="boot-err"></div>
<div class="row" style="margin:8px 0">
  <label for="expPick"><b>Expiry</b></label>
  <select id="expPick" style="margin-left:8px;min-width:280px"></select>
</div>
<div id="tsExp" class="plot"></div>
<h2>q by calendar day (all expiries, pooled)</h2>
<div id="boxDay" class="plot"></div>
<script>
const DATA = """
    tail = r""";(function() {
  const err = document.getElementById('boot-err');
  if (typeof Plotly === 'undefined') {
    err.textContent = 'Plotly failed to load. Try: python -m http.server 8080 then http://localhost:8080/this-file.html';
    return;
  }
  const labels = DATA.exp_labels || [];
  const byExp = DATA.by_expiry || {};
  const sel = document.getElementById('expPick');
  labels.forEach((lab, i) => {
    const o = document.createElement('option');
    o.value = lab;
    o.textContent = lab;
    if (i === 0) o.selected = true;
    sel.appendChild(o);
  });
  const xb = [], yb = [];
  for (const day of Object.keys(DATA.by_day).sort()) {
    for (const q of DATA.by_day[day]) { xb.push(day); yb.push(q); }
  }
  const TRADING_RB = [
    { bounds: ['sat', 'mon'], pattern: 'day of week' },
    { bounds: [16, 9.5], pattern: 'hour' },
  ];
  const layoutTs = {
    title: 'q vs snapshot time',
    xaxis: { title: 'Time', type: 'date', rangebreaks: TRADING_RB, tickangle: -45 },
    yaxis: { title: 'q' },
    showlegend: false,
    margin: { b: 120, t: 48, l: 56, r: 24 },
  };
  function traceFor(lab) {
    const d = byExp[lab] || { t: [], q: [] };
    return {
      type: 'scatter',
      mode: 'lines+markers',
      x: d.t,
      y: d.q,
      name: 'q',
      line: { width: 2, color: '#2563eb' },
      marker: { size: 5 },
    };
  }
  function redrawTs() {
    const lab = sel.value;
    layoutTs.title = 'q vs time — ' + lab;
    Plotly.react('tsExp', [traceFor(lab)], layoutTs, { responsive: true });
  }
  try {
    if (!labels.length) {
      err.textContent = 'No expiry series (check expiry_fwd_q.csv).';
    } else {
      Plotly.newPlot('tsExp', [traceFor(labels[0])], Object.assign({}, layoutTs, { title: 'q vs time — ' + labels[0] }), { responsive: true });
      sel.addEventListener('change', redrawTs);
    }

    Plotly.newPlot('boxDay', [{
      type: 'box',
      x: xb,
      y: yb,
      marker: { size: 3 },
      boxpoints: 'suspectedoutliers',
    }], {
      title: 'Distribution of q by day (summary)',
      xaxis: { title: 'Calendar day', categoryorder: 'array', categoryarray: Object.keys(DATA.by_day).sort() },
      yaxis: { title: 'q' },
      showlegend: false,
      margin: { b: 72, t: 36, l: 56, r: 24 },
    }, { responsive: true });
  } catch (e) {
    err.textContent = 'Plot error: ' + (e.message || e);
  }
})();
</script>
</body>
</html>
"""
    return head + data_json + tail


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python plot_cvi_batch_q_box_report.py <batch_dir> [out.html]", file=sys.stderr)
        sys.exit(2)
    batch_dir = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else batch_dir / "q_daily_plots.html"
    payload = build_payload(batch_dir)
    title = f"q daily — {batch_dir.name}"
    out.write_text(render_html(payload, title), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"file:///{str(out).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
