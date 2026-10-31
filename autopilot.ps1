# Fully automatic paper trading (Binance Spot TESTNET).
#
# Every 24h cycle:  refresh CSV data -> run live_runner.py hourly for 24h -> repeat.
#   exit 0  = 24h cycle finished        -> refresh data and continue
#   exit 2  = SAFETY STOP (drawdown/API) -> autopilot halts; a human must review
#   other   = crash                      -> wait 5 min and restart
#
# Stop it:   .\stop_autopilot.ps1   (or create logs\emergency_stop.flag)
# Log:       logs\autopilot.log

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$logsDir   = Join-Path $projectRoot "logs"
$logFile   = Join-Path $logsDir "autopilot.log"
$stopFlag  = Join-Path $logsDir "emergency_stop.flag"
$lockFile  = Join-Path $logsDir "autopilot.lock"
New-Item -ItemType Directory -Force $logsDir | Out-Null

function Write-Log([string]$msg) {
    $line = "{0} | AUTOPILOT | {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Host $line
}

# Single instance: two runners would double every trade.
if (Test-Path $lockFile) {
    $otherPid = (Get-Content $lockFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    # Windows reuses PIDs after a reboot, so check the PID is really an autopilot,
    # not just any process (this stale-lock bug kept the bot off Oct 15-22 2026).
    $other = if ($otherPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $otherPid" -ErrorAction SilentlyContinue }
    if ($other -and $other.CommandLine -match "autopilot\.ps1") {
        Write-Log "Another autopilot is already running (PID $otherPid). Exiting."
        exit 0
    }
}
Set-Content -Path $lockFile -Value $PID -Encoding ascii

try {
    Write-Log "Autopilot started (PID $PID)."
    while ($true) {
        if (Test-Path $stopFlag) {
            Write-Log "Emergency stop flag present - not trading. Delete logs\emergency_stop.flag to resume."
            break
        }

        # After a wake/boot Wi-Fi may not be up yet; wait (up to 15 min) instead of
        # burning the runner's error budget on DNS failures.
        for ($i = 0; $i -lt 30; $i++) {
            if (Resolve-DnsName "testnet.binance.vision" -ErrorAction SilentlyContinue) { break }
            if ($i -eq 0) { Write-Log "Network not ready - waiting for testnet.binance.vision..." }
            Start-Sleep -Seconds 30
        }

        Write-Log "Refreshing market data (update_data.py)..."
        cmd /c "python -u update_data.py >> `"$logFile`" 2>&1"
        Write-Log "Data refresh exit code $LASTEXITCODE."

        # Weekly ML research (self-throttles: does nothing if the last run was < 7 days ago):
        # champion/challenger walk-forward, retrain, health gate, report.
        Write-Log "ML research check (ml_autoresearch.py)..."
        cmd /c "python -u ml_autoresearch.py >> `"$logFile`" 2>&1"
        Write-Log "ML research exit code $LASTEXITCODE."

        Write-Log "Starting live_runner.py (hourly, 24h cycle)..."
        cmd /c "python -u live_runner.py --loop --interval-minutes 60 --stop-after-hours 24 >> `"$logFile`" 2>&1"
        $code = $LASTEXITCODE

        if ($code -eq 0) {
            Write-Log "24h cycle complete."
            continue
        }
        if ($code -eq 2) {
            Write-Log "SAFETY STOP (exit 2): drawdown limit or exchange failure. Autopilot halted - review logs before restarting."
            break
        }
        Write-Log "live_runner.py crashed (exit $code). Restarting in 5 minutes."
        Start-Sleep -Seconds 300
    }
}
finally {
    Remove-Item $lockFile -ErrorAction SilentlyContinue
    Write-Log "Autopilot stopped."
}
