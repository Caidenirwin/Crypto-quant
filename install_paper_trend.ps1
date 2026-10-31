# Register the PAPER-ONLY BTC trend strategy (paper_trend.py) as its own Windows Scheduled Task.
# Separate from V7's CryptoQuantAutopilot - it sends no orders and only writes logs\paper_trend\.
# Runs hourly + at logon; each run only processes completed daily candles, so extra runs are harmless.
# Remove with:  Unregister-ScheduledTask -TaskName CryptoQuantPaperTrend -Confirm:$false

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path (Split-Path (Get-Command python).Source) "pythonw.exe"   # no console window

$action = New-ScheduledTaskAction -Execute $pythonw -Argument "paper_trend.py" -WorkingDirectory $projectRoot
$trigger = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) -RepetitionInterval (New-TimeSpan -Hours 1))
)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "CryptoQuantPaperTrend" -Action $action -Trigger $trigger -Settings $settings `
    -Description "PAPER-ONLY BTC 200-SMA + 40% vol target + 10pt band (paper_trend.py). No orders." -Force | Out-Null

# Confirm it actually runs (a new line appears in logs\paper_trend\run.log).
$log = Join-Path $projectRoot "logs\paper_trend\run.log"
$before = if (Test-Path $log) { (Get-Content $log).Count } else { 0 }
Start-ScheduledTask -TaskName "CryptoQuantPaperTrend"
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if ((Test-Path $log) -and (Get-Content $log).Count -gt $before) {
        Write-Host "CryptoQuantPaperTrend installed and CONFIRMED running:" -ForegroundColor Green
        Get-Content $log -Tail 1
        exit 0
    }
}
Write-Host "WARNING: task registered but no run confirmed yet - check logs\paper_trend\run.log" -ForegroundColor Red
exit 1
