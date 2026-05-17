"""Shared runner for strict C++ decoupling smoke/full checks."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCHBENCH_ROOT = REPO_ROOT / "ResearchBench"
PYTHON_SCRIPTS = RESEARCHBENCH_ROOT / "python" / "scripts"
DECOUPLING_EXE = RESEARCHBENCH_ROOT / "x64" / "Release" / "cvi_decoupling_engine.exe"


def run_decoupling_job(
    *,
    log_path: Path,
    batch_dir: Path,
    expiry_indices: str,
    timeout_seconds: int,
    unbuffered: bool = False,
) -> int:
    os.environ["CVI_DECOUPLING_CPP_STRICT"] = "1"
    os.environ.pop("CVI_DECOUPLING_CPP", None)
    if unbuffered:
        os.environ["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not DECOUPLING_EXE.is_file():
        log_path.write_text(f"ERROR: missing {DECOUPLING_EXE}\n", encoding="utf-8")
        return 2
    if not batch_dir.is_dir():
        log_path.write_text(f"ERROR: missing batch {batch_dir}\n", encoding="utf-8")
        return 2

    cmd = [
        sys.executable,
        "-u" if unbuffered else "",
        str(PYTHON_SCRIPTS / "plot_cvi_batch_q_vol_scatter.py"),
        str(batch_dir),
        "--decoupling-only",
        "--decoupling-dates",
        "2026-04-06",
        "--decoupling-expiry-indices",
        expiry_indices,
        "--decoupling-window-min",
        "5",
        "--decoupling-snapshot-spacing-min",
        "5",
    ]
    cmd = [c for c in cmd if c]

    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"CMD={cmd}\nCWD={PYTHON_SCRIPTS}\nEXE={DECOUPLING_EXE}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(PYTHON_SCRIPTS),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
        logf.write(f"\n=== EXIT={proc.returncode} SEC={time.perf_counter()-t0:.2f} ===\n")
    return int(proc.returncode)
