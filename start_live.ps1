$ErrorActionPreference = "Stop"

Write-Host "Starting live paper runner..." -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Optional: load API keys from .env in project root.
$envPath = Join-Path $projectRoot ".env"
if (Test-Path $envPath) {
    Write-Host "Loading environment variables from .env" -ForegroundColor DarkCyan
    Get-Content $envPath | ForEach-Object {
        $line = $_.Trim()
        if (-not $line) { return }
        if ($line.StartsWith("#")) { return }
        $parts = $line.Split("=", 2)
        if ($parts.Count -ne 2) { return }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name) {
            Set-Item -Path ("Env:" + $name) -Value $value
        }
    }
}

if (-not $env:BINANCE_API_KEY -or -not $env:BINANCE_SECRET) {
    Write-Host "Missing BINANCE_API_KEY or BINANCE_SECRET." -ForegroundColor Yellow
    Write-Host "Set them in current session:" -ForegroundColor Yellow
    Write-Host '$env:BINANCE_API_KEY="your_testnet_key"' -ForegroundColor Gray
    Write-Host '$env:BINANCE_SECRET="your_testnet_secret"' -ForegroundColor Gray
    Write-Host "Or create .env in project root with:" -ForegroundColor Yellow
    Write-Host "BINANCE_API_KEY=your_testnet_key" -ForegroundColor Gray
    Write-Host "BINANCE_SECRET=your_testnet_secret" -ForegroundColor Gray
    exit 1
}

Write-Host "Launching live_runner.py..." -ForegroundColor Green
python live_runner.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "live_runner.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Live run completed." -ForegroundColor Green
