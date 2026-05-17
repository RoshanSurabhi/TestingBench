#!/usr/bin/env python3
"""Shim: :mod:`cvi_batch_analysis.cvi_batch_plot` CLI (scatter, SSR, call-price, ...)."""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_pkg_root = _root.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from cvi_batch_analysis.cvi_batch_plot.cli import main

if __name__ == "__main__":
    main()

