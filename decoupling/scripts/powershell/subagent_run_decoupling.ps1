$ErrorActionPreference = 'Stop'
$env:CVI_DECOUPLING_CPP_STRICT = '1'
if ($env:CVI_DECOUPLING_CPP -eq '0') { Remove-Item Env:\CVI_DECOUPLING_CPP -ErrorAction SilentlyContinue }
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
$researchRoot = Join-Path $repoRoot 'ResearchBench'
$pythonScripts = Join-Path $researchRoot 'python\scripts'
$log = Join-Path $researchRoot 'data\runtime\decoupling_cpp_strict_run.log'
$sw = [Diagnostics.Stopwatch]::StartNew()
Set-Location $pythonScripts
$batch = 'C:\Users\RoshanSurabhi\Downloads\data\AAPL\cvi_clamped_z5_basis23_full_batch'
try {
  & python plot_cvi_batch_q_vol_scatter.py $batch `
    --decoupling-only --decoupling-dates 2026-04-06 --decoupling-expiry-indices all `
    --decoupling-window-min 5 --decoupling-snapshot-spacing-min 5 *>&1 | Tee-Object -FilePath $log
  $code = $LASTEXITCODE
} finally {
  $sw.Stop()
  "`n=== EXIT_CODE=$code ELAPSED_SEC=$([math]::Round($sw.Elapsed.TotalSeconds,2)) ===" | Add-Content -Path $log -Encoding UTF8
}
exit $code
