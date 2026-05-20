# ResearchBench

ResearchBench owns analytics modules built on top of TrDB snapshot data.

## Modules

- `fitting/`
  - Snapshot stream adapter + lightweight CVI fit bridge + full `--fit-from-bin` CVI solve pipeline.
- `python/`
  - Batch analysis and q-filtering/cauchy tooling migrated from `TrDBClient/python`.
- `decoupling/`
  - C++ decoupling engine (`cvi_decoupling_engine.exe`) and decoupling helper scripts.
- `cauchy/`
  - Cauchy module target scaffold (module boundary for future migration).
- `tools/mid_delta_wls_cpp/`
  - Standalone C++ tool for batch decoupling/WLS exports and HTML diagnostics.

## Build targets

- `ResearchBench` (`ResearchBench.vcxproj`) - fitting run target.
- `ResearchBenchDecoupling` (`ResearchBenchDecoupling.vcxproj`) - decoupling engine target.
- `ResearchBenchCauchy` (`ResearchBenchCauchy.vcxproj`) - cauchy module target.

## Dependencies

- `TrDBClientFetchCore` from `../TrDBClient/native/TrDBClientFetchCore.vcxproj`
- `UtilLib`
- `CVI` (for fitting ecosystem dependencies)

## Bin fit command

Run full CVI solve directly from TrDB `.bin` snapshots:

`x64/Release/ResearchBench.exe --fit-from-bin="<path-to-bin>" --fit-out-dir="<output-dir>" [--fit-max-snaps=0] [--fit-num-basis=30] [--fit-lambda=0.005] [--fit-arb-points=20]`

Default output contract (minimal, computation-focused):

- `batch_cvi_summary.csv`
- Per-snapshot `expiry_fwd_q.csv`
- Per-snapshot `option_fit_comparison.csv`

Optional extras:

- `--fit-dearb-qp-diagnostics` writes `solve_de_arb_qp_diagnostics.csv` and `de_arb_qp/` Clarabel matrix tree (also included with `--fit-full-artifacts`).
- `--fit-write-price-comparison` adds per-snapshot `price_comparison.csv`
- `--fit-debug-artifacts` adds debug files for fit troubleshooting parity
