"""Per ticker: % of cvi_* dirs with volatility_by_expiry.html; latest calendar date among those."""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data"
FITS = "cvi_fits_json_cvi_fit_params_nb25_z6"
MARKER = "volatility_by_expiry.html"
DATE_IN_NAME = re.compile(r"^cvi_\d+_(\d{4}-\d{2}-\d{2})_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--fits-subdir", default=FITS)
    ap.add_argument("--marker", default=MARKER)
    args = ap.parse_args()
    data: Path = args.data
    fits_name: str = args.fits_subdir
    marker: str = args.marker

    rows: list[tuple[str, int, int, date | None, str]] = []
    for sym in sorted(p for p in data.iterdir() if p.is_dir()):
        fr = sym / fits_name
        if not fr.is_dir():
            rows.append((sym.name, 0, 0, None, "n/a"))
            continue
        total = with_h = 0
        latest: date | None = None
        for e in fr.iterdir():
            if not e.is_dir() or not e.name.startswith("cvi_"):
                continue
            total += 1
            hp = e / marker
            if hp.is_file() and hp.stat().st_size > 0:
                with_h += 1
                m = DATE_IN_NAME.match(e.name)
                if m:
                    d = date.fromisoformat(m.group(1))
                    if latest is None or d > latest:
                        latest = d
        pct = (100.0 * with_h / total) if total else 0.0
        rows.append((sym.name, total, with_h, latest, f"{pct:.2f}%"))

    w = max(6, max(len(r[0]) for r in rows))
    print(f"Data: {data.resolve()}")
    print(f"Marker: <ticker>/{fits_name}/cvi_*/{marker}")
    print()
    hdr = f"{'Ticker':<{w}} {'fits':>6} {'w/html':>7} {'pct':>8}  latest_date_w_html"
    print(hdr)
    print("-" * len(hdr))
    for t, tot, wh, ld, pct in rows:
        ld_s = str(ld) if ld else ("—" if tot else "—")
        print(f"{t:<{w}} {tot:>6} {wh:>7} {pct:>8}  {ld_s}")


if __name__ == "__main__":
    main()
