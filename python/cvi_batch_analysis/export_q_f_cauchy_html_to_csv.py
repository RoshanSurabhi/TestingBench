#!/usr/bin/env python3
"""
Decode Plotly HTML from apply_cauchy_filter.py / plot_cvi_batch_qf_cauchy_compare.py
(chunked base64 JSON) and write a long CSV: one row per (expiry, timestamp).

Skips null timestamp rows inserted for day breaks in plots.

Usage:
  python export_q_f_cauchy_html_to_csv.py path/to/q_f_cauchy_....html [-o out.csv]
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path


def _decode_payload_from_html(html: str) -> dict:
    marker = "const _B64_PARTS = "
    i = html.find(marker)
    if i < 0:
        raise SystemExit("Could not find const _B64_PARTS in HTML (not a chunked payload export?)")
    j = html.find("[", i)
    if j < 0:
        raise SystemExit("Malformed HTML: missing [ after _B64_PARTS")
    arr, _end = json.JSONDecoder().raw_decode(html[j:])
    if not isinstance(arr, list):
        raise SystemExit("_B64_PARTS is not a JSON array")
    b64 = "".join(arr)
    raw = base64.b64decode(b64).decode("utf-8")
    return json.loads(raw)


def _expiry_idx_and_date(label: str) -> tuple[str, str]:
    """Label like '3: 2026-04-17' -> ('3', '2026-04-17')."""
    label = label.strip()
    if ":" in label:
        a, b = label.split(":", 1)
        return a.strip(), b.strip()[:10]
    return "", label[:10]


def export_csv(data: dict, out_path: Path) -> None:
    labels = data.get("exp_labels") or []
    by_exp = data.get("by_exp") or {}
    fieldnames = [
        "expiry_idx",
        "expiry_date",
        "expiry_label",
        "timestamp",
        "q_raw",
        "F_raw",
        "q_hat",
        "F_hat",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for lab in labels:
            s = by_exp.get(lab) or {}
            t_list = s.get("t") or []
            q_list = s.get("q") or []
            F_list = s.get("F") or []
            qh_list = s.get("q_cauchy") or []
            fh_list = s.get("F_cauchy") or []
            n = min(len(t_list), len(q_list), len(F_list), len(qh_list), len(fh_list))
            eidx, edate = _expiry_idx_and_date(lab)
            for k in range(n):
                ts = t_list[k]
                if ts is None:
                    continue
                w.writerow(
                    {
                        "expiry_idx": eidx,
                        "expiry_date": edate,
                        "expiry_label": lab,
                        "timestamp": str(ts),
                        "q_raw": q_list[k],
                        "F_raw": F_list[k],
                        "q_hat": qh_list[k],
                        "F_hat": fh_list[k],
                    }
                )


def main() -> None:
    ap = argparse.ArgumentParser(description="Export q/F Cauchy HTML to CSV")
    ap.add_argument("html_path", type=Path)
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: same stem as .html)",
    )
    args = ap.parse_args()
    html_path = args.html_path.resolve()
    if not html_path.is_file():
        raise SystemExit(f"Not found: {html_path}")

    out = (
        args.output.resolve()
        if args.output
        else html_path.with_suffix(".csv")
    )

    html = html_path.read_text(encoding="utf-8", errors="replace")
    data = _decode_payload_from_html(html)
    export_csv(data, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
