#!/usr/bin/env python3
"""
Decode q_f_cauchy_*.html (chunked base64 JSON from apply_cauchy_filter.py) and
export PNG(s) with only the longest-maturity expiry (max expiry_idx in legend).

Requires: pip install plotly kaleido

Usage:
  python export_qf_cauchy_html_last_expiry_png.py path/to/q_f_cauchy_....html
  python export_qf_cauchy_html_last_expiry_png.py --glob "C:/.../data/*/cvi_fits*/q_f_cauchy_*.html"
  python export_qf_cauchy_html_last_expiry_png.py file.html --separate   # four PNGs + grid
"""

from __future__ import annotations

import argparse
import base64
import glob as globmod
import json
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ExportSkipped(Exception):
    """Payload missing data needed for PNG export (caller may skip file)."""


def _decode_payload_from_html(html: str) -> dict:
    marker = "const _B64_PARTS = "
    i = html.find(marker)
    if i < 0:
        raise SystemExit("Could not find const _B64_PARTS in HTML")
    j = html.find("[", i)
    if j < 0:
        raise SystemExit("Malformed HTML: missing [ after _B64_PARTS")
    arr, _end = json.JSONDecoder().raw_decode(html[j:])
    if not isinstance(arr, list):
        raise SystemExit("_B64_PARTS is not a JSON array")
    b64 = "".join(arr)
    raw = base64.b64decode(b64).decode("utf-8")
    return json.loads(raw)


def _label_sort_key(lab: str) -> int:
    lab = lab.strip()
    if ":" in lab:
        try:
            return int(lab.split(":", 1)[0].strip())
        except ValueError:
            return 0
    return 0


def _pairs(t_list: list, y_list: list) -> tuple[list, list]:
    xs: list = []
    ys: list = []
    for i in range(min(len(t_list), len(y_list))):
        ti = t_list[i]
        yi = y_list[i]
        if ti is None:
            continue
        if yi is None or (isinstance(yi, float) and not math.isfinite(yi)):
            continue
        xs.append(ti)
        ys.append(float(yi))
    return xs, ys


def export_pngs(html_path: Path, *, separate: bool, width: int, height_per: int) -> list[Path]:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    data = _decode_payload_from_html(html)
    labels = list(data.get("exp_labels") or [])
    if not labels:
        raise ExportSkipped("no exp_labels in payload (nothing to plot)")
    labels_sorted = sorted(labels, key=_label_sort_key)
    last_lab = labels_sorted[-1]
    by_exp = data.get("by_exp") or {}
    series = by_exp.get(last_lab)
    if not series:
        raise ExportSkipped(f"no series for label {last_lab!r}")

    t = series.get("t") or []
    q = series.get("q") or []
    F = series.get("F") or []
    qh = series.get("q_cauchy") or []
    fh = series.get("F_cauchy") or []

    out_dir = html_path.parent
    base = html_path.stem
    written: list[Path] = []

    def _fig_single(title: str, y_title: str, y_data: list) -> go.Figure:
        x, y = _pairs(t, y_data)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=last_lab,
                line=dict(width=1.8),
            )
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis_title="timestamp",
            yaxis_title=y_title,
            template="plotly_white",
            legend_title_text="expiry",
            margin=dict(l=56, r=24, t=56, b=48),
            width=width,
            height=height_per,
        )
        return fig

    if separate:
        for suffix, title, ylab, arr in (
            ("q_raw", "q (raw)", "q", q),
            ("F_raw", "Forward F (raw)", "F", F),
            ("q_hat", "q̂ (DP-MAP Cauchy)", "q_hat", qh),
            ("F_hat", "F̂ implied from q̂", "F_hat", fh),
        ):
            out = out_dir / f"{base}_{suffix}_last_expiry.png"
            fig = _fig_single(title, ylab, arr)
            fig.write_image(str(out), scale=2)
            written.append(out)

    # 2x2 grid (default)
    xq, yq = _pairs(t, q)
    xF, yF = _pairs(t, F)
    xqh, yqh = _pairs(t, qh)
    xfh, yfh = _pairs(t, fh)

    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("q (raw)", "Forward F (raw)", "q̂ (DP-MAP Cauchy)", "F̂"),
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )
    fig2.add_trace(
        go.Scatter(x=xq, y=yq, mode="lines", name=last_lab, showlegend=True, legendgroup="a"),
        row=1,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(x=xF, y=yF, mode="lines", name=last_lab, showlegend=False, legendgroup="a"),
        row=1,
        col=2,
    )
    fig2.add_trace(
        go.Scatter(x=xqh, y=yqh, mode="lines", name=last_lab, showlegend=False, legendgroup="a"),
        row=2,
        col=1,
    )
    fig2.add_trace(
        go.Scatter(x=xfh, y=yfh, mode="lines", name=last_lab, showlegend=False, legendgroup="a"),
        row=2,
        col=2,
    )
    fig2.update_xaxes(title_text="timestamp", row=1, col=1)
    fig2.update_xaxes(title_text="timestamp", row=1, col=2)
    fig2.update_xaxes(title_text="timestamp", row=2, col=1)
    fig2.update_xaxes(title_text="timestamp", row=2, col=2)
    fig2.update_yaxes(title_text="q", row=1, col=1)
    fig2.update_yaxes(title_text="F", row=1, col=2)
    fig2.update_yaxes(title_text="q_hat", row=2, col=1)
    fig2.update_yaxes(title_text="F_hat", row=2, col=2)
    fig2.update_layout(
        title_text=f"Last expiry only: {last_lab}",
        template="plotly_white",
        width=int(width * 1.05),
        height=height_per * 2 + 80,
        margin=dict(l=52, r=28, t=72, b=40),
    )
    grid_out = out_dir / f"{base}_last_expiry_2x2.png"
    fig2.write_image(str(grid_out), scale=2)
    written.append(grid_out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("html_paths", nargs="*", type=Path, help="One or more q_f_cauchy HTML files")
    ap.add_argument(
        "--glob",
        default=None,
        help="Glob pattern for HTML files (e.g. .../data/*/cvi_fits*/q_f_cauchy_*.html)",
    )
    ap.add_argument(
        "--separate",
        action="store_true",
        help="Also write four single-panel PNGs (q, F, q_hat, F_hat)",
    )
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=380, help="Height per row (single panels use this)")
    args = ap.parse_args()

    paths: list[Path] = []
    paths.extend(Path(p) for p in args.html_paths)
    if args.glob:
        paths.extend(Path(p) for p in globmod.glob(args.glob))

    # de-dupe, keep order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    paths = uniq

    if not paths:
        ap.error("Provide html_paths and/or --glob")

    all_written: list[Path] = []
    for hp in paths:
        hp = hp.resolve()
        if not hp.is_file():
            print(f"skip missing: {hp}")
            continue
        print(f"export {hp.name} …")
        try:
            outs = export_pngs(
                hp,
                separate=bool(args.separate),
                width=int(args.width),
                height_per=int(args.height),
            )
        except ExportSkipped as e:
            print(f"  skip: {e}")
            continue
        for o in outs:
            print(f"  wrote {o}")
            all_written.append(o)

    print(f"\nTotal PNG files: {len(all_written)}")


if __name__ == "__main__":
    main()
