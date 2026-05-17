"""One-shot decoupling run for subagent: strict C++, all expiries."""
from __future__ import annotations

from pathlib import Path

from decoupling_runner import run_decoupling_job

REPO_ROOT = Path(__file__).resolve().parents[4]
RESEARCHBENCH_ROOT = REPO_ROOT / "ResearchBench"
LOG = RESEARCHBENCH_ROOT / "data" / "runtime" / "decoupling_cpp_strict_run.log"
BATCH = Path(r"C:\Users\RoshanSurabhi\Downloads\data\AAPL\cvi_clamped_z5_basis23_full_batch")


def main() -> int:
    return run_decoupling_job(
        log_path=LOG,
        batch_dir=BATCH,
        expiry_indices="all",
        timeout_seconds=7200,
        unbuffered=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
