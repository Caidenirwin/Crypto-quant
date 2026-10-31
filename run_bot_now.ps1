# Run ONE full trading cycle right now (V7 + memes + anti-rug + news + ML sleeve),
# on Binance Spot TESTNET (paper money). Safe to use while the autopilot is running:
# if an automatic cycle is in progress, this waits for it to finish first.
#
#   .\run_bot_now.ps1            refresh market data, then trade
#   .\run_bot_now.ps1 -SkipData  trade immediately with the data already on disk

param([switch]$SkipData)

Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (Test-Path "logs\emergency_stop.flag") {
    Write-Host "Emergency stop is ON (logs\emergency_stop.flag) - not trading." -ForegroundColor Red
    Write-Host "Re-enable with .\install_autopilot.ps1 if you meant to resume." -ForegroundColor Yellow
    exit 1
}

if (-not $SkipData) {
    Write-Host "Refreshing market data..." -ForegroundColor Cyan
    python -u update_data.py | Select-String "Done\."
}

Write-Host "Running one trading cycle..." -ForegroundColor Cyan
python -u live_runner.py
$code = $LASTEXITCODE

switch ($code) {
    0 { Write-Host "`nCycle finished. Results: dashboard (http://localhost:8501) or logs\" -ForegroundColor Green }
    2 { Write-Host "`nSAFETY STOP: drawdown limit or exchange problem - review before trading again." -ForegroundColor Red }
    default { Write-Host "`nCycle failed (exit $code) - see the messages above." -ForegroundColor Red }
}
exit $code
