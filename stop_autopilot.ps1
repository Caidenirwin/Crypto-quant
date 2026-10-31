# Stop the autopilot and its trading loop immediately.
# Creates logs\emergency_stop.flag so nothing restarts until you delete it
# (install_autopilot.ps1 deletes it for you when you re-enable).

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $projectRoot "logs"
New-Item -ItemType File -Force (Join-Path $logsDir "emergency_stop.flag") | Out-Null

Stop-ScheduledTask -TaskName "CryptoQuantAutopilot" -ErrorAction SilentlyContinue

# Kill the autopilot shell and any runner it started (matched by script name).
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -match "autopilot\.ps1|live_runner\.py|update_data\.py" -and $_.ProcessId -ne $PID } |
    ForEach-Object {
        Write-Host "Stopping PID $($_.ProcessId): $($_.CommandLine)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
Remove-Item (Join-Path $logsDir "autopilot.lock") -ErrorAction SilentlyContinue
Write-Host "Autopilot stopped. Open positions are left as-is on Testnet." -ForegroundColor Yellow
