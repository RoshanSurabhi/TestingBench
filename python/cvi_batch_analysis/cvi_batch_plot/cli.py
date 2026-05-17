"""Argument parsing and dispatch for CVI batch plots."""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from .atm_hedge import (
    atm_hedge_file_prefix,
    compute_atm_hedge_2h_detail,
    write_atm_hedge_2h_artifacts,
    write_atm_hedge_2h_csvs,
    write_atm_hedge_klassen_grid_csv,
    write_call_delta_prediction_csvs,
    write_focused_delta_compare_csvs,
)
from .batch_payloads import (
    build_full_day_ssr_json_document,
    build_ln_scatter_payload,
    build_payload,
    build_ssr_scatter_payload,
    infer_num_expiries_from_batch,
    render_full_day_ssr_html,
    write_ln_scatter_details_csv,
)
from .call_price_renders import (
    build_call_price_prediction_payload,
    render_call_price_prediction_html,
    render_html,
    render_ln_regression_methods_html,
    render_ln_scatter_html,
    render_ssr_scatter_html,
)
from .fundamentals import parse_float_list, parse_int_list
from .snapshots_decouple import (
    build_decoupling_payload,
    render_decoupling_html,
    render_decoupling_mid_wls_html,
    render_decoupling_mid_wls_per_strike_html,
    write_decoupling_detail_csv,
    write_decoupling_mid_wls_betas_csv,
    write_decoupling_mid_wls_betas_per_strike_csv,
    write_decoupling_summary_csv,
)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot CVI q/vol scatter, ln-scatter diagnostics, and call-price ΔC prediction evaluation."
    )
    ap.add_argument("batch_dir", type=Path, help="CVI batch directory")
    ap.add_argument("--date", default=None, help="Filter date YYYY-MM-DD")
    ap.add_argument("--expiry-index", type=int, default=None, help="Expiry index for SSR scatter mode")
    ap.add_argument(
        "--ssr-scatter-only",
        action="store_true",
        help="Generate only quick SSR scatter HTML (x=ΔlnF, y=Δlnσ*/s_atf_norm).",
    )
    ap.add_argument(
        "--ln-scatter-only",
        action="store_true",
        help="Generate only ln(F) vs ln(sigma_star) scatter HTML.",
    )
    ap.add_argument(
        "--min-abs-skew",
        type=float,
        default=1e-4,
        help="Minimum |s_atf_norm| required to keep a point (SSR mode).",
    )
    ap.add_argument(
        "--sigma-mode",
        choices=["avg3", "z0"],
        default="avg3",
        help="Sigma source for ln-scatter mode: avg3 nearest-z or z=0 point.",
    )
    ap.add_argument(
        "--call-price-pred-only",
        action="store_true",
        help="Generate only call-price ΔC prediction evaluation HTML.",
    )
    ap.add_argument(
        "--horizons-min",
        default=None,
        metavar="M1,M2,...",
        help="Comma-separated forecast horizons in minutes for call-price prediction mode. "
        "Default: 1,5,15 (or 1,5,15,30,60,120 with --focused-delta-compare).",
    )
    ap.add_argument(
        "--strike-z-offsets",
        default="0,-1,1",
        help="Comma-separated z targets used to select fixed strikes from first snapshot (e.g., 0,-1,1).",
    )
    ap.add_argument(
        "--target-source",
        choices=["market", "cvi", "both"],
        default="both",
        help="Realized target source in call-price prediction mode.",
    )
    ap.add_argument(
        "--write-delta-csv",
        action="store_true",
        help="With --call-price-pred-only, write per-horizon delta prediction CSVs.",
    )
    ap.add_argument(
        "--focused-delta-compare",
        action="store_true",
        help="Focused compare: CVI-only, horizons 1/5/15, naive vs rolling-regression with hedge/delta losses.",
    )
    ap.add_argument(
        "--atm-hedge-2h-only",
        action="store_true",
        help="CSV-only ATM hedge residual experiment (overlapping windows, BS Greeks). Horizon via --atm-hedge-horizons-min.",
    )
    ap.add_argument(
        "--atm-hedge-ssr-mode",
        choices=["rolling", "prior_day"],
        default="rolling",
        help="SSR for hedge: rolling 1h regression timestamps (default) or one value per method from prior trading day.",
    )
    ap.add_argument(
        "--atm-hedge-prior-fallback",
        choices=["skip", "same_day"],
        default="same_day",
        help="If prior_day mode and no earlier day in batch: error (skip) or fit SSR from same day (labeled same_day_fallback).",
    )
    ap.add_argument(
        "--atm-hedge-prior-ssr-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional full-day SSR JSON (same format as --full-day-ssr-json). When set, prior_day hedge "
        "uses SSR from this file for (ssr_fit_day, expiry) instead of refitting from batch snapshots—"
        "e.g. to match 5-minute full-day regression. Relative paths resolve under the batch directory.",
    )
    ap.add_argument(
        "--atm-hedge-horizons-min",
        default="120",
        metavar="M1,M2,...",
        help="Holding period(s) in minutes for ATM hedge (e.g. 30,60,120). Default: 120.",
    )
    ap.add_argument(
        "--atm-hedge-entry-spacing-min",
        type=float,
        default=0.0,
        metavar="N",
        help="Minimum minutes between hedge entry snapshots (0 = every batch snapshot, e.g. ~1 min). "
        "Use 5 to approximate re-hedging only every 5 minutes.",
    )
    ap.add_argument(
        "--decoupling-only",
        action="store_true",
        help="Compute residualized realized-delta decoupling diagnostic (all strikes, selected dates/expiries).",
    )
    ap.add_argument(
        "--decoupling-dates",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated YYYY-MM-DD for --decoupling-only.",
    )
    ap.add_argument(
        "--decoupling-expiry-indices",
        default=None,
        metavar="I1,I2,...|all",
        help="Expiries for --decoupling-only. Use 'all' to infer from first date.",
    )
    ap.add_argument(
        "--decoupling-window-min",
        type=float,
        default=5.0,
        metavar="M",
        help="Target horizon in minutes for decoupling windows (default 5).",
    )
    ap.add_argument(
        "--decoupling-snapshot-spacing-min",
        type=float,
        default=5.0,
        metavar="M",
        help="Minimum minutes between entry snapshots in decoupling mode (default 5).",
    )
    ap.add_argument(
        "--decoupling-min-abs-df-frac",
        type=float,
        default=0.0005,
        metavar="X",
        help="Skip when |ΔF| < X*F_entry (default 0.0005 = 0.05%%).",
    )
    ap.add_argument(
        "--decoupling-detail-csv",
        type=Path,
        default=None,
        help="Decoupling detail CSV path (default: <batch>/decoupling_details.csv).",
    )
    ap.add_argument(
        "--decoupling-summary-csv",
        type=Path,
        default=None,
        help="Decoupling summary CSV path (default: <batch>/decoupling_summary.csv).",
    )
    ap.add_argument(
        "--decoupling-html",
        type=Path,
        default=None,
        help="Decoupling Plotly HTML path (default: <batch>/decoupling_diagnostic.html).",
    )
    ap.add_argument(
        "--decoupling-mid-wls-csv",
        type=Path,
        default=None,
        help="With --decoupling-only: write pooled WLS betas (mid realized δ ~ α + β·δ_BS) per date×expiry "
        "(default: <batch>/decoupling_mid_wls_betas.csv). Weights ∝ 1/(ε+x²+y²) on scatter coordinates.",
    )
    ap.add_argument(
        "--decoupling-mid-wls-html",
        type=Path,
        default=None,
        help="With --decoupling-only: Plotly page with mid scatter + WLS lines (default: <batch>/decoupling_mid_wls_fit.html).",
    )
    ap.add_argument(
        "--decoupling-mid-wls-per-strike-csv",
        type=Path,
        default=None,
        help="With --decoupling-only: WLS intercept α and slope β per date × expiry × strike on mid-track realized δ "
        "(default: <batch>/decoupling_mid_wls_betas_per_strike.csv). Same weights as --decoupling-mid-wls-csv.",
    )
    ap.add_argument(
        "--decoupling-mid-wls-per-strike-html",
        type=Path,
        default=None,
        help="With --decoupling-only: Plotly page to inspect per-strike mid δ scatter + WLS line "
        "(default: <batch>/decoupling_mid_wls_per_strike_fit.html).",
    )
    ap.add_argument(
        "--full-day-ssr-json",
        type=Path,
        default=None,
        help="Write full-day SSR (OLS/WLS/Huber/LAD/Theil-Sen) for given dates/expiries to this JSON file.",
    )
    ap.add_argument(
        "--full-day-ssr-dates",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated YYYY-MM-DD for --full-day-ssr-json.",
    )
    ap.add_argument(
        "--expiry-indices",
        default=None,
        metavar="I1,I2,...|all",
        help="Comma-separated expiry indices for --full-day-ssr-json, or 'all' (infer m from first date). "
        "Default 0,4,10 if --expiry-index omitted.",
    )
    ap.add_argument(
        "--full-day-ssr-html",
        type=Path,
        default=None,
        help="Plotly HTML for full-day SSR (default: same path as --full-day-ssr-json with .html).",
    )
    ap.add_argument(
        "--full-day-ssr-no-html",
        action="store_true",
        help="With --full-day-ssr-json, do not write the companion HTML chart.",
    )
    ap.add_argument(
        "--full-day-ssr-move-trim-frac",
        type=float,
        default=0.0,
        metavar="P",
        help="Full-day SSR: before regressing, trim the lowest/highest P fraction of |ΔlnF| and of |Δlnσ*| "
        "(independently; keep transitions in both central bands). Example: 0.005 = 0.5%% tails per margin. "
        "0 disables (default).",
    )
    ap.add_argument(
        "--full-day-ssr-snapshot-spacing-min",
        type=float,
        default=0.0,
        metavar="M",
        help="Full-day SSR: minimum minutes between consecutive snapshots used to build Δln transitions "
        "(default 0 = use every OK batch snapshot, typically ~1 min). Example: 5 for 5-minute spacing.",
    )
    ap.add_argument(
        "--atm-hedge-klassen-grid-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write one CSV of Klassen σ-metrics (OLS/WLS/Huber/LAD/Theil–Sen) per date×expiry. "
        "Uses prior_day SSR only. Requires --atm-hedge-klassen-grid-dates. "
        "Uses first value of --atm-hedge-horizons-min (default 120), plus --atm-hedge-entry-spacing-min, "
        "--atm-hedge-prior-fallback, --sigma-mode, optional --atm-hedge-prior-ssr-json.",
    )
    ap.add_argument(
        "--atm-hedge-klassen-grid-dates",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated YYYY-MM-DD for --atm-hedge-klassen-grid-csv.",
    )
    ap.add_argument(
        "--atm-hedge-klassen-grid-expiry-indices",
        default=None,
        metavar="I1,I2,...|all",
        help="Expiries to include (default: all from CVI_dims per day). "
        "Or 'all' using expiry count from the first grid date.",
    )
    ap.add_argument(
        "--atm-hedge-klassen-grid-entry-window",
        choices=["all", "open_2h", "close_2h", "open_and_close"],
        default="all",
        help="Hedge **entry** filter for klassen grid: open_2h = 09:30–11:30 (naive); "
        "close_2h = last ~2h of feasible starts (≥12:00, through t_last−hedge_horizon); "
        "open_and_close = both (entry_window column). Default: all times.",
    )
    args = ap.parse_args()

    batch_dir = args.batch_dir.resolve()
    if not batch_dir.is_dir():
        print("Batch directory not found.", file=sys.stderr)
        sys.exit(2)

    if args.atm_hedge_klassen_grid_csv is not None:
        if not args.atm_hedge_klassen_grid_dates:
            raise SystemExit("--atm-hedge-klassen-grid-dates is required with --atm-hedge-klassen-grid-csv")
        days_kg = [x.strip() for x in args.atm_hedge_klassen_grid_dates.split(",") if x.strip()]
        if not days_kg:
            raise SystemExit("No dates in --atm-hedge-klassen-grid-dates")
        raw_ek = args.atm_hedge_klassen_grid_expiry_indices
        exps_kg: list[int] | None
        if raw_ek is None:
            exps_kg = None
        else:
            rl = raw_ek.strip().lower()
            if rl == "all":
                exps_kg = list(range(infer_num_expiries_from_batch(batch_dir, day=days_kg[0])))
            else:
                exps_kg = parse_int_list(raw_ek)
        h_list = parse_float_list(args.atm_hedge_horizons_min)
        hm_kg = float(h_list[0]) if h_list else 120.0
        outp = args.atm_hedge_klassen_grid_csv.expanduser()
        if not outp.is_absolute():
            outp = (batch_dir / outp).resolve()
        kg_ew = args.atm_hedge_klassen_grid_entry_window
        kg_wins = ["open_2h", "close_2h"] if kg_ew == "open_and_close" else [kg_ew]
        kg_ssr_doc: dict | None = None
        if args.atm_hedge_prior_ssr_json is not None:
            jp = args.atm_hedge_prior_ssr_json.expanduser()
            if not jp.is_absolute():
                jp = (batch_dir / jp).resolve()
            if not jp.is_file():
                raise SystemExit(f"--atm-hedge-prior-ssr-json not found: {jp}")
            kg_ssr_doc = json.loads(jp.read_text(encoding="utf-8"))
        written = write_atm_hedge_klassen_grid_csv(
            batch_dir,
            outp,
            days=days_kg,
            expiry_indices=exps_kg,
            horizon_min=hm_kg,
            sigma_mode=args.sigma_mode,
            prior_day_fallback=args.atm_hedge_prior_fallback,
            min_entry_spacing_min=float(args.atm_hedge_entry_spacing_min),
            entry_time_windows=kg_wins,
            prior_full_day_ssr_json=kg_ssr_doc,
        )
        print(f"Wrote {written}")
        print(f"file:///{str(written).replace(chr(92), '/')}")
        return

    if args.full_day_ssr_json is not None:
        if not args.full_day_ssr_dates:
            raise SystemExit("--full-day-ssr-dates is required with --full-day-ssr-json")
        days_fd = [x.strip() for x in args.full_day_ssr_dates.split(",") if x.strip()]
        if not days_fd:
            raise SystemExit("No dates in --full-day-ssr-dates")
        if args.expiry_indices:
            raw_e = args.expiry_indices.strip().lower()
            if raw_e == "all":
                exps_fd = list(range(infer_num_expiries_from_batch(batch_dir, day=days_fd[0])))
            else:
                exps_fd = parse_int_list(args.expiry_indices)
        elif args.expiry_index is not None:
            exps_fd = [int(args.expiry_index)]
        else:
            exps_fd = [0, 4, 10]
        sigma_fd = "avg3"
        if args.sigma_mode != sigma_fd:
            print(
                "full-day-ssr: σ* uses avg3 only (mean of 3 fitted vols at strikes nearest z=0); "
                f"ignoring --sigma-mode={args.sigma_mode}.",
                file=sys.stderr,
            )
        doc = build_full_day_ssr_json_document(
            batch_dir,
            days=days_fd,
            expiry_indices=exps_fd,
            sigma_mode=sigma_fd,
            move_trim_frac=float(args.full_day_ssr_move_trim_frac),
            snapshot_spacing_min=float(args.full_day_ssr_snapshot_spacing_min),
        )
        out_json = args.full_day_ssr_json.expanduser()
        if not out_json.is_absolute():
            out_json = (batch_dir / out_json).resolve()
        try:
            out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        except PermissionError:
            out_json = out_json.with_name(out_json.stem + "_new" + out_json.suffix)
            out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"Wrote {out_json}")
        print(f"file:///{str(out_json).replace(chr(92), '/')}")
        if not args.full_day_ssr_no_html:
            if args.full_day_ssr_html is not None:
                out_html = args.full_day_ssr_html.expanduser()
                if not out_html.is_absolute():
                    out_html = (batch_dir / out_html).resolve()
            else:
                out_html = out_json.with_suffix(".html")
            tk = doc.get("ticker_guess") or doc.get("batch_folder_name") or batch_dir.name
            title_html = f"Full-day SSR — {tk} — {days_fd[0]} … {days_fd[-1]}"
            html_body = render_full_day_ssr_html(doc, title_html)
            try:
                out_html.write_text(html_body, encoding="utf-8")
            except PermissionError:
                out_html = out_html.with_name(out_html.stem + "_new" + out_html.suffix)
                out_html.write_text(html_body, encoding="utf-8")
            print(f"Wrote {out_html}")
            print(f"file:///{str(out_html).replace(chr(92), '/')}")
        return

    if args.decoupling_only:
        if not args.decoupling_dates:
            raise SystemExit("--decoupling-dates is required with --decoupling-only")
        days_dec = [x.strip() for x in args.decoupling_dates.split(",") if x.strip()]
        if not days_dec:
            raise SystemExit("No dates in --decoupling-dates")
        if not args.decoupling_expiry_indices:
            raise SystemExit("--decoupling-expiry-indices is required with --decoupling-only")
        raw_ed = args.decoupling_expiry_indices.strip().lower()
        if raw_ed == "all":
            exps_dec = list(range(infer_num_expiries_from_batch(batch_dir, day=days_dec[0])))
        else:
            exps_dec = parse_int_list(args.decoupling_expiry_indices)
        if not exps_dec:
            raise SystemExit("No expiry indices resolved for decoupling mode.")
        payload_dec = build_decoupling_payload(
            batch_dir,
            days=days_dec,
            expiry_indices=exps_dec,
            window_min=float(args.decoupling_window_min),
            snapshot_spacing_min=float(args.decoupling_snapshot_spacing_min),
            min_abs_df_frac=float(args.decoupling_min_abs_df_frac),
        )
        out_detail = args.decoupling_detail_csv.expanduser() if args.decoupling_detail_csv is not None else (batch_dir / "decoupling_details.csv")
        if not out_detail.is_absolute():
            out_detail = (batch_dir / out_detail).resolve()
        out_sum = args.decoupling_summary_csv.expanduser() if args.decoupling_summary_csv is not None else (batch_dir / "decoupling_summary.csv")
        if not out_sum.is_absolute():
            out_sum = (batch_dir / out_sum).resolve()
        out_html = args.decoupling_html.expanduser() if args.decoupling_html is not None else (batch_dir / "decoupling_diagnostic.html")
        if not out_html.is_absolute():
            out_html = (batch_dir / out_html).resolve()
        w_detail = write_decoupling_detail_csv(payload_dec, out_detail)
        w_sum = write_decoupling_summary_csv(payload_dec, out_sum)
        title_dec = f"Decoupling diagnostic — {batch_dir.name} — {days_dec[0]} … {days_dec[-1]}"
        try:
            out_html.write_text(render_decoupling_html(payload_dec, title_dec), encoding="utf-8")
            w_html = out_html
        except PermissionError:
            w_html = out_html.with_name(out_html.stem + "_new" + out_html.suffix)
            w_html.write_text(render_decoupling_html(payload_dec, title_dec), encoding="utf-8")
        out_mid_wls = (
            args.decoupling_mid_wls_csv.expanduser()
            if args.decoupling_mid_wls_csv is not None
            else (batch_dir / "decoupling_mid_wls_betas.csv")
        )
        if not out_mid_wls.is_absolute():
            out_mid_wls = (batch_dir / out_mid_wls).resolve()
        w_mid = write_decoupling_mid_wls_betas_csv(payload_dec, out_mid_wls)
        out_mid_html = (
            args.decoupling_mid_wls_html.expanduser()
            if args.decoupling_mid_wls_html is not None
            else (batch_dir / "decoupling_mid_wls_fit.html")
        )
        if not out_mid_html.is_absolute():
            out_mid_html = (batch_dir / out_mid_html).resolve()
        title_mid = f"Mid WLS — {batch_dir.name} — {days_dec[0]} … {days_dec[-1]}"
        try:
            out_mid_html.write_text(render_decoupling_mid_wls_html(payload_dec, title_mid), encoding="utf-8")
            w_mid_html = out_mid_html
        except PermissionError:
            w_mid_html = out_mid_html.with_name(out_mid_html.stem + "_new" + out_mid_html.suffix)
            w_mid_html.write_text(render_decoupling_mid_wls_html(payload_dec, title_mid), encoding="utf-8")
        out_mid_wls_k = (
            args.decoupling_mid_wls_per_strike_csv.expanduser()
            if args.decoupling_mid_wls_per_strike_csv is not None
            else (batch_dir / "decoupling_mid_wls_betas_per_strike.csv")
        )
        if not out_mid_wls_k.is_absolute():
            out_mid_wls_k = (batch_dir / out_mid_wls_k).resolve()
        w_mid_k = write_decoupling_mid_wls_betas_per_strike_csv(payload_dec, out_mid_wls_k)
        out_mid_k_html = (
            args.decoupling_mid_wls_per_strike_html.expanduser()
            if args.decoupling_mid_wls_per_strike_html is not None
            else (batch_dir / "decoupling_mid_wls_per_strike_fit.html")
        )
        if not out_mid_k_html.is_absolute():
            out_mid_k_html = (batch_dir / out_mid_k_html).resolve()
        title_mid_k = f"Mid WLS per strike — {batch_dir.name} — {days_dec[0]} … {days_dec[-1]}"
        try:
            out_mid_k_html.write_text(render_decoupling_mid_wls_per_strike_html(payload_dec, title_mid_k), encoding="utf-8")
            w_mid_k_html = out_mid_k_html
        except PermissionError:
            w_mid_k_html = out_mid_k_html.with_name(out_mid_k_html.stem + "_new" + out_mid_k_html.suffix)
            w_mid_k_html.write_text(render_decoupling_mid_wls_per_strike_html(payload_dec, title_mid_k), encoding="utf-8")
        d = payload_dec.get("diagnostics", {})
        spot_std_msg = ""
        summ = payload_dec.get("summary")
        if isinstance(summ, list) and len(summ) == 1 and isinstance(summ[0], dict):
            spot_std_msg = f" spot_std_dev={summ[0].get('spot_std_dev')} spot_mean={summ[0].get('spot_mean')}"
        print(f"Wrote {w_detail}")
        print(f"Wrote {w_sum}")
        print(f"Wrote {w_mid}")
        print(f"Wrote {w_mid_k}")
        print(f"Wrote {w_mid_k_html}")
        print(f"Wrote {w_mid_html}")
        print(f"Wrote {w_html}")
        print(
            f"n_rows_total={d.get('n_rows_total')} n_rows_valid={d.get('n_rows_valid')} "
            f"n_skipped_small_df={d.get('n_skipped_small_df')} n_skipped_missing_price={d.get('n_skipped_missing_price')}"
            f"{spot_std_msg}"
        )
        print(f"file:///{str(w_html).replace(chr(92), '/')}")
        print(f"file:///{str(w_mid_html).replace(chr(92), '/')}")
        print(f"file:///{str(w_mid_k_html).replace(chr(92), '/')}")
        return

    if args.ssr_scatter_only:
        if not args.date:
            raise SystemExit("--date is required with --ssr-scatter-only")
        if args.expiry_index is None:
            raise SystemExit("--expiry-index is required with --ssr-scatter-only")
        payload = build_ssr_scatter_payload(
            batch_dir,
            day=args.date,
            expiry_index=args.expiry_index,
            min_abs_skew=float(args.min_abs_skew),
        )
        out = batch_dir / f"ssr_scatter_e{args.expiry_index}_{args.date}.html"
        title = f"SSR scatter — {batch_dir.name} — e{args.expiry_index} — {args.date}"
        out.write_text(render_ssr_scatter_html(payload, title), encoding="utf-8")
        diag = payload["diagnostics"]
        print(f"Wrote {out}")
        print(f"n_used={diag['n_points_used']} n_skipped_bad={diag['n_skipped_bad_data']} "
              f"n_skipped_small_skew={diag['n_skipped_small_skew']} ssr_hat={diag['ssr_hat_origin']}")
        print(f"skew_preview={diag['skew_preview']}")
        print(f"file:///{str(out).replace(chr(92), '/')}")
        return

    if args.ln_scatter_only:
        if not args.date:
            raise SystemExit("--date is required with --ln-scatter-only")
        if args.expiry_index is None:
            raise SystemExit("--expiry-index is required with --ln-scatter-only")
        payload = build_ln_scatter_payload(
            batch_dir,
            day=args.date,
            expiry_index=args.expiry_index,
            sigma_mode=args.sigma_mode,
        )
        out = batch_dir / f"dlnF_dlnSigma_scatter_e{args.expiry_index}_{args.date}.html"
        out_reg = batch_dir / f"dlnF_dlnSigma_regression_methods_e{args.expiry_index}_{args.date}.html"
        title = f"Δln(F)-Δln(sigma*) scatter — {batch_dir.name} — e{args.expiry_index} — {args.date}"
        out.write_text(render_ln_scatter_html(payload, title), encoding="utf-8")
        title_reg = f"No-intercept robust regression diagnostics — {batch_dir.name} — e{args.expiry_index} — {args.date}"
        h_detail, h_sum, h_dist, h_klassen = None, None, None, None
        try:
            h_horizons = parse_float_list(args.atm_hedge_horizons_min)
            if not h_horizons:
                h_horizons = [120.0]
            hm0 = float(h_horizons[0])
            h_rows, h_summary, h_diag = compute_atm_hedge_2h_detail(
                batch_dir,
                day=args.date,
                expiry_index=args.expiry_index,
                ln_payload=payload,
                horizon_min=hm0,
                ssr_mode=args.atm_hedge_ssr_mode,
                sigma_mode=args.sigma_mode,
                prior_day_fallback=args.atm_hedge_prior_fallback,
                min_entry_spacing_min=float(args.atm_hedge_entry_spacing_min),
            )
            payload["hedge_atm_2h"] = {"rows": h_rows, "summary": h_summary, "diag": h_diag}
            hpfx = atm_hedge_file_prefix(
                expiry_index=args.expiry_index,
                day=args.date,
                horizon_min=hm0,
                ssr_mode=args.atm_hedge_ssr_mode,
                entry_spacing_min=float(args.atm_hedge_entry_spacing_min),
            )
            h_detail, h_sum, h_dist, h_klassen = write_atm_hedge_2h_artifacts(batch_dir, hpfx, h_rows, h_summary)
        except Exception as exc:
            payload["hedge_atm_2h"] = {"rows": [], "summary": None, "diag": {"error": str(exc)}}
        out_reg.write_text(render_ln_regression_methods_html(payload, title_reg), encoding="utf-8")
        csv_out = batch_dir / f"dlnF_dlnSigma_details_e{args.expiry_index}_{args.date}.csv"
        wrote_csv = csv_out
        try:
            write_ln_scatter_details_csv(payload, csv_out)
        except PermissionError:
            wrote_csv = batch_dir / f"dlnF_dlnSigma_details_e{args.expiry_index}_{args.date}_new.csv"
            write_ln_scatter_details_csv(payload, wrote_csv)
        diag = payload["diagnostics"]
        print(f"Wrote {out}")
        print(f"Wrote {out_reg}")
        print(f"Wrote {wrote_csv}")
        if h_detail is not None:
            print(f"Wrote {h_detail}")
            print(f"Wrote {h_sum}")
            print(f"Wrote {h_dist}")
            if h_klassen is not None:
                print(f"Wrote {h_klassen}")
                print(f"file:///{str(h_klassen).replace(chr(92), '/')}")
            print(f"file:///{str(h_dist).replace(chr(92), '/')}")
            if isinstance(payload.get("hedge_atm_2h"), dict):
                hd = payload["hedge_atm_2h"].get("diag") or {}
                if hd.get("n_skipped_entry_spacing") is not None:
                    print(
                        f"atm_hedge: n_rows={hd.get('n_rows')} n_skipped_entry_spacing="
                        f"{hd.get('n_skipped_entry_spacing')} min_entry_spacing_min={hd.get('min_entry_spacing_min')}"
                    )
        print(
            f"n_used={diag['n_transition_points_used']} n_skew={diag['n_skew_points']} "
            f"n_missing_or_invalid={diag['n_missing_or_invalid']} n_nonpositive={diag['n_nonpositive']}"
        )
        print(f"file:///{str(out).replace(chr(92), '/')}")
        return

    if args.call_price_pred_only:
        if not args.date:
            raise SystemExit("--date is required with --call-price-pred-only")
        if args.expiry_index is None:
            raise SystemExit("--expiry-index is required with --call-price-pred-only")
        if args.horizons_min is None:
            horizons = (
                [1.0, 5.0, 15.0, 30.0, 60.0, 120.0]
                if args.focused_delta_compare
                else [1.0, 5.0, 15.0]
            )
        else:
            horizons = parse_float_list(args.horizons_min)
        z_offsets = parse_float_list(args.strike_z_offsets)
        target_source = args.target_source
        if args.focused_delta_compare:
            target_source = "cvi"
        if not horizons:
            raise SystemExit("--horizons-min must contain at least one value.")
        if not z_offsets:
            raise SystemExit("--strike-z-offsets must contain at least one value.")
        payload = build_call_price_prediction_payload(
            batch_dir,
            day=args.date,
            expiry_index=args.expiry_index,
            horizons_min=horizons,
            strike_z_offsets=z_offsets,
            target_source=target_source,
        )
        out = batch_dir / f"dC_prediction_eval_e{args.expiry_index}_{args.date}.html"
        title = f"ΔC prediction evaluation — {batch_dir.name} — e{args.expiry_index} — {args.date}"
        out.write_text(render_call_price_prediction_html(payload, title), encoding="utf-8")
        wrote_csvs: list[Path] = []
        focused_summary: Path | None = None
        if args.write_delta_csv:
            wrote_csvs = write_call_delta_prediction_csvs(
                payload,
                batch_dir,
                file_prefix=f"delta_prediction_eval_e{args.expiry_index}_{args.date}",
            )
        if args.focused_delta_compare:
            focused_details, focused_summary = write_focused_delta_compare_csvs(
                payload,
                batch_dir,
                file_prefix=f"focused_delta_compare_e{args.expiry_index}_{args.date}",
                delta_floor_abs_df=0.05,
            )
            wrote_csvs.extend(focused_details)
        diag = payload.get("diagnostics", {})
        print(f"Wrote {out}")
        for p in wrote_csvs:
            print(f"Wrote {p}")
        if focused_summary is not None:
            print(f"Wrote {focused_summary}")
        print(
            f"n_samples={diag.get('n_samples')} n_snapshots_used={diag.get('n_snapshots_used')} "
            f"n_skipped_missing={diag.get('n_skipped_missing')}"
        )
        print(f"file:///{str(out).replace(chr(92), '/')}")
        return

    if args.atm_hedge_2h_only:
        if not args.date:
            raise SystemExit("--date is required with --atm-hedge-2h-only")
        if args.expiry_index is None:
            raise SystemExit("--expiry-index is required with --atm-hedge-2h-only")
        horizons_atm = parse_float_list(args.atm_hedge_horizons_min)
        if not horizons_atm:
            horizons_atm = [120.0]
        last_dist: Path | None = None
        last_klassen: Path | None = None
        atm_json_doc: dict | None = None
        if args.atm_hedge_prior_ssr_json is not None and args.atm_hedge_ssr_mode == "prior_day":
            jp = args.atm_hedge_prior_ssr_json.expanduser()
            if not jp.is_absolute():
                jp = (batch_dir / jp).resolve()
            if not jp.is_file():
                raise SystemExit(f"--atm-hedge-prior-ssr-json not found: {jp}")
            atm_json_doc = json.loads(jp.read_text(encoding="utf-8"))
        for hm in horizons_atm:
            fp = atm_hedge_file_prefix(
                expiry_index=args.expiry_index,
                day=args.date,
                horizon_min=float(hm),
                ssr_mode=args.atm_hedge_ssr_mode,
                entry_spacing_min=float(args.atm_hedge_entry_spacing_min),
            )
            detail, summary, dist_html, klassen_html, diag = write_atm_hedge_2h_csvs(
                batch_dir,
                day=args.date,
                expiry_index=args.expiry_index,
                file_prefix=fp,
                horizon_min=float(hm),
                ssr_mode=args.atm_hedge_ssr_mode,
                sigma_mode=args.sigma_mode,
                prior_day_fallback=args.atm_hedge_prior_fallback,
                min_entry_spacing_min=float(args.atm_hedge_entry_spacing_min),
                prior_full_day_ssr_json=atm_json_doc,
            )
            print(f"Wrote {detail}")
            print(f"Wrote {summary}")
            print(f"Wrote {dist_html}")
            print(f"Wrote {klassen_html}")
            print(
                f"horizon_min={diag.get('horizon_min')} ssr_mode={diag.get('ssr_mode')} "
                f"hedge_ssr_source={diag.get('hedge_ssr_source')} "
                f"n_rows={diag.get('n_rows')} n_snapshots_used={diag.get('n_snapshots_used')} "
                f"n_skipped_missing={diag.get('n_skipped_missing')} n_skipped_small_df={diag.get('n_skipped_small_df')} "
                f"n_skipped_entry_spacing={diag.get('n_skipped_entry_spacing')} "
                f"min_entry_spacing_min={diag.get('min_entry_spacing_min')}"
            )
            last_dist = dist_html
            last_klassen = klassen_html
        if last_klassen is not None:
            print(f"file:///{str(last_klassen).replace(chr(92), '/')}")
        if last_dist is not None:
            print(f"file:///{str(last_dist).replace(chr(92), '/')}")
        return

    payload = build_payload(batch_dir)
    out = batch_dir / "cvi_batch_q_vol_analysis.html"
    title = f"CVI q & vol/spot — {batch_dir.name}"
    out.write_text(render_html(payload, title), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"file:///{str(out).replace(chr(92), '/')}")


if __name__ == "__main__":
    main()
