"""
Unified CLI for CVI batch / fits folder tools (``cvi_batch_analysis`` package).

Run from ``TrDBClient/tools``::

    python -m cvi_batch_analysis --help
    python -m cvi_batch_analysis q-box --batch "C:/path/to/cvi_..._batch"
    python -m cvi_batch_analysis batch-plot --batch "C:/path/to/batch" -- --ssr-scatter-only

The ``tools`` directory is on ``sys.path`` so imports resolve consistently.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable, Sequence

_TOOLS_ROOT = Path(__file__).resolve().parent.parent


def _ensure_tools_path() -> None:
    s = str(_TOOLS_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


def _run_legacy_main(module: str, argv: Sequence[str]) -> None:
    """Call ``module.main()`` after patching ``sys.argv`` (legacy scripts use sys.argv)."""
    _ensure_tools_path()
    old = sys.argv
    sys.argv = [f"{module}.py", *[str(x) for x in argv]]
    try:
        mod = importlib.import_module(module)
        main_fn: Callable[[], None] = getattr(mod, "main")
        main_fn()
    finally:
        sys.argv = old


def _cmd_q_box(ns: argparse.Namespace) -> None:
    b = ns.batch.resolve()
    argv = [str(b)]
    if ns.out:
        argv.append(str(Path(ns.out).resolve()))
    _run_legacy_main("cvi_batch_analysis.plot_cvi_batch_q_box_report", argv)


def _cmd_q_fit_day(ns: argparse.Namespace) -> None:
    b = ns.batch.resolve()
    argv = [str(b)]
    if ns.out:
        argv.append(str(Path(ns.out).resolve()))
    _run_legacy_main("cvi_batch_analysis.plot_cvi_batch_q_fit_day", argv)


def _cmd_vol_overlay(ns: argparse.Namespace) -> None:
    _run_legacy_main("cvi_batch_analysis.plot_cvi_batch_vol_overlay", [str(ns.batch.resolve())])


def _cmd_qf_cauchy_compare(ns: argparse.Namespace) -> None:
    argv = [str(ns.batch.resolve())]
    if ns.out:
        argv.append(str(Path(ns.out).resolve()))
    if ns.day:
        argv.extend(["--day", ns.day])
    argv.extend(["--gamma-mode", ns.gamma_mode])
    argv.extend(["--k-star", str(ns.k_star)])
    if ns.no_level_shift:
        argv.append("--no-level-shift")
    else:
        argv.extend(["--level-shift-theta", str(ns.level_shift_theta)])
        argv.extend(["--level-shift-n", str(ns.level_shift_n)])
    argv.extend(["--dq-pct-lo", str(ns.dq_pct_lo)])
    argv.extend(["--dq-pct-hi", str(ns.dq_pct_hi)])
    _run_legacy_main("cvi_batch_analysis.plot_cvi_batch_qf_cauchy_compare", argv)


def _cmd_apply_cauchy(ns: argparse.Namespace) -> None:
    argv = [
        str(ns.batch.resolve()),
        "--day",
        ns.day,
        "--params",
        str(Path(ns.params).resolve()),
    ]
    if ns.out:
        argv.extend(["-o", str(Path(ns.out).resolve())])
    if ns.no_level_shift:
        argv.append("--no-level-shift")
    else:
        argv.extend(["--level-shift-theta", str(ns.level_shift_theta)])
        argv.extend(["--level-shift-n", str(ns.level_shift_n)])
    _run_legacy_main("cvi_batch_analysis.apply_cauchy_filter", argv)


def _cmd_calibrate_cauchy(ns: argparse.Namespace) -> None:
    argv: list[str] = []
    if ns.pool_jsons:
        argv.append("--pool-jsons")
        argv.extend(str(Path(p).resolve()) for p in ns.pool_jsons)
    else:
        if not ns.batch:
            raise SystemExit("calibrate-cauchy: need --batch unless using --pool-jsons")
        argv.append(str(ns.batch.resolve()))
    if ns.day:
        argv.extend(["--day", ns.day])
    if ns.days:
        argv.extend(["--days", ns.days])
    if ns.output:
        argv.extend(["-o", str(Path(ns.output).resolve())])
    argv.extend(["--kss-target", str(ns.kss_target)])
    argv.extend(["--kss-range", ns.kss_range])
    argv.extend(["--mad-abs-quantile", str(ns.mad_abs_quantile)])
    _run_legacy_main("cvi_batch_analysis.calibrate_cauchy_params", argv)


def _cmd_report_vol_coverage(ns: argparse.Namespace) -> None:
    argv: list[str] = []
    if ns.data:
        argv.extend(["--data", str(Path(ns.data).resolve())])
    if ns.fits_subdir:
        argv.extend(["--fits-subdir", ns.fits_subdir])
    if ns.marker:
        argv.extend(["--marker", ns.marker])
    _run_legacy_main("cvi_batch_analysis.report_vol_html_coverage", argv)


def _cmd_export_q_csv(ns: argparse.Namespace) -> None:
    b = ns.batch.resolve()
    argv = [str(b)]
    if ns.out:
        argv.append(str(Path(ns.out).resolve()))
    _run_legacy_main("cvi_batch_analysis.export_batch_q_values_csv", argv)


def _cmd_export_qf_html_csv(ns: argparse.Namespace) -> None:
    argv = [str(Path(ns.html).resolve())]
    if ns.out:
        argv.extend(["-o", str(Path(ns.out).resolve())])
    _run_legacy_main("cvi_batch_analysis.export_q_f_cauchy_html_to_csv", argv)


def _cmd_export_qf_png(ns: argparse.Namespace) -> None:
    argv: list[str] = []
    if ns.html:
        argv.append(str(Path(ns.html).resolve()))
    if ns.glob:
        argv.extend(["--glob", ns.glob])
    if not argv:
        raise SystemExit("export-qf-png: pass --html and/or --glob")
    if ns.separate:
        argv.append("--separate")
    _run_legacy_main("cvi_batch_analysis.export_qf_cauchy_html_last_expiry_png", argv)


def _cmd_batch_plot(ns: argparse.Namespace) -> None:
    """Forward to :mod:`cvi_batch_analysis.cvi_batch_plot.cli` (scatter / SSR / call-price)."""
    _ensure_tools_path()
    tail = [str(ns.batch.resolve()), *list(ns.extra or [])]
    old = sys.argv
    sys.argv = ["plot_cvi_batch_q_vol_scatter.py", *tail]
    try:
        from cvi_batch_analysis.cvi_batch_plot.cli import main as batch_main

        batch_main()
    finally:
        sys.argv = old


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="cvi_batch_analysis",
        description="CVI batch / fits folder tools (structured CLI).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("q-box", help="q(t) per expiry + daily q box plots -> q_daily_plots.html")
    p.add_argument("--batch", "-b", type=Path, required=True, help="CVI batch directory")
    p.add_argument("--out", "-o", type=Path, default=None)
    p.set_defaults(_handler=_cmd_q_box)

    p = sub.add_parser("q-fit-day", help="Full q/F report + tables -> q_fit_day.html")
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.add_argument("--out", "-o", type=Path, default=None)
    p.set_defaults(_handler=_cmd_q_fit_day)

    p = sub.add_parser("vol-overlay", help="Fitted vol overlays -> cvi_batch_vol_overlay.html")
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.set_defaults(_handler=_cmd_vol_overlay)

    p = sub.add_parser(
        "qf-cauchy-compare",
        help="Raw vs DP-Cauchy q/F (on-the-fly calibration) -> q_f_cauchy_compare.html",
    )
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.add_argument("--out", "-o", type=Path, default=None)
    p.add_argument("--day", default=None)
    p.add_argument("--gamma-mode", choices=("k_star", "mad_legacy"), default="k_star")
    p.add_argument("--k-star", type=float, default=0.6, dest="k_star")
    p.add_argument("--no-level-shift", action="store_true")
    p.add_argument("--level-shift-theta", type=float, default=10.0)
    p.add_argument("--level-shift-n", type=int, default=3)
    p.add_argument("--dq-pct-lo", type=float, default=25.0)
    p.add_argument("--dq-pct-hi", type=float, default=75.0)
    p.set_defaults(_handler=_cmd_qf_cauchy_compare)

    p = sub.add_parser("apply-cauchy", help="Apply pre-calibrated params -> q_f_cauchy_*.html")
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.add_argument("--day", required=True, help="Target day YYYY-MM-DD")
    p.add_argument("--params", "-p", type=Path, required=True, help="cauchy_params_*.json")
    p.add_argument("--out", "-o", type=Path, default=None)
    p.add_argument("--no-level-shift", action="store_true")
    p.add_argument("--level-shift-theta", type=float, default=10.0)
    p.add_argument("--level-shift-n", type=int, default=3)
    p.set_defaults(_handler=_cmd_apply_cauchy)

    p = sub.add_parser("calibrate-cauchy", help="Estimate cauchy_params_*.json from batch CSVs")
    p.add_argument("--batch", "-b", type=Path, default=None, help="CVI batch dir (not used with --pool-jsons)")
    p.add_argument("--day", default=None)
    p.add_argument("--days", default=None, help="Comma-separated YYYY-MM-DD for pooled calibration")
    p.add_argument("--pool-jsons", nargs="+", default=None, help="Pool from existing JSON files")
    p.add_argument("--output", "-o", type=Path, default=None)
    p.add_argument("--kss-target", type=float, default=2.0)
    p.add_argument("--kss-range", default="1.0,3.0")
    p.add_argument("--mad-abs-quantile", type=float, default=50.0)
    p.set_defaults(_handler=_cmd_calibrate_cauchy)

    p = sub.add_parser(
        "report-vol-coverage",
        help="Ticker x DTE coverage of volatility_by_expiry.html",
    )
    p.add_argument("--data", type=Path, default=None, help="Default: repo ../data")
    p.add_argument("--fits-subdir", default=None)
    p.add_argument("--marker", default=None)
    p.set_defaults(_handler=_cmd_report_vol_coverage)

    p = sub.add_parser("export-q-csv", help="Merge expiry_fwd_q rows -> q_values_long.csv")
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.add_argument("--out", "-o", type=Path, default=None)
    p.set_defaults(_handler=_cmd_export_q_csv)

    p = sub.add_parser("export-qf-html-csv", help="Decode q_f_cauchy HTML -> long CSV")
    p.add_argument("--html", type=Path, required=True)
    p.add_argument("--out", "-o", type=Path, default=None)
    p.set_defaults(_handler=_cmd_export_qf_html_csv)

    p = sub.add_parser("export-qf-png", help="q_f_cauchy HTML -> PNG (needs kaleido)")
    p.add_argument("--html", type=Path, default=None)
    p.add_argument("--glob", default=None, help="Glob string for many HTML files")
    p.add_argument("--separate", action="store_true")
    p.set_defaults(_handler=_cmd_export_qf_png)

    p = sub.add_parser(
        "batch-plot",
        help="SSR / ln-scatter / call-price / ATM hedge (see cvi_batch_plot.cli)",
    )
    p.add_argument("--batch", "-b", type=Path, required=True)
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args for cvi_batch_plot (use `--` before flags, e.g. -- --ssr-scatter-only)",
    )
    p.set_defaults(_handler=_cmd_batch_plot)

    return ap


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_tools_path()
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], None] = args._handler
    handler(args)


if __name__ == "__main__":
    main()
