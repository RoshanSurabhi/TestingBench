#!/usr/bin/env python3
"""
Merge every snapshot's expiry_fwd_q.csv into one long CSV (q, F, expiries, timestamps).

Usage:
  python export_batch_q_values_csv.py <batch_dir> [out.csv]
  Default out: <batch_dir>/q_values_long.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python export_batch_q_values_csv.py <batch_dir> [out.csv]",
            file=sys.stderr,
        )
        sys.exit(2)
    batch_dir = Path(sys.argv[1]).resolve()
    out = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else batch_dir / "q_values_long.csv"
    )
    summary = batch_dir / "batch_cvi_summary.csv"
    if not summary.is_file():
        raise SystemExit(f"Missing {summary}")

    fieldnames = [
        "idx_in_bin",
        "subfolder",
        "timestamp",
        "ok",
        "expiry_idx",
        "expiry_date",
        "F",
        "q",
        "volTime",
        "r",
        "sigma_star",
        "v_star",
    ]
    n_rows = 0
    with summary.open(newline="", encoding="utf-8") as sf, out.open(
        "w", newline="", encoding="utf-8"
    ) as of:
        srows = list(csv.DictReader(sf))
        w = csv.DictWriter(of, fieldnames=fieldnames)
        w.writeheader()
        srows.sort(key=lambda r: int(r.get("idx_in_bin", 0) or 0))
        for row in srows:
            sub = row.get("subfolder", "").strip()
            idx = int(row.get("idx_in_bin", 0) or 0)
            ts = row.get("timestamp", "").strip()
            ok = int(row.get("ok", 0) or 0)
            efq = batch_dir / sub / "expiry_fwd_q.csv"
            if not efq.is_file():
                continue
            with efq.open(newline="", encoding="utf-8") as ef:
                for erow in csv.DictReader(ef):
                    w.writerow(
                        {
                            "idx_in_bin": idx,
                            "subfolder": sub,
                            "timestamp": ts,
                            "ok": ok,
                            "expiry_idx": erow.get("expiry_idx", ""),
                            "expiry_date": erow.get("expiry_date", "").strip().strip('"'),
                            "F": erow.get("F", ""),
                            "q": erow.get("q", ""),
                            "volTime": erow.get("volTime", ""),
                            "r": erow.get("r", ""),
                            "sigma_star": erow.get("sigma_star", ""),
                            "v_star": erow.get("v_star", ""),
                        }
                    )
                    n_rows += 1
    print(f"Wrote {n_rows} rows to {out}")


if __name__ == "__main__":
    main()
