#!/usr/bin/env python3
"""Shim: implementation in :mod:`cvi_batch_analysis.export_q_f_cauchy_html_to_csv`."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_pkg_root = _root.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from cvi_batch_analysis.export_q_f_cauchy_html_to_csv import main

if __name__ == "__main__":
    main()

