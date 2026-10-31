# Register autopilot.ps1 as a Windows Scheduled Task that starts at logon and
# restarts if it dies, then start it now. Re-running this is safe (it replaces the task).
# Remove with:  Unregister-ScheduledTask -TaskName CryptoQuantAutopilot -Confirm:$false

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $projectRoot "autopilot.ps1"
Remove-Item (Join-Path $projectRoot "logs\emergency_stop.flag") -ErrorAction SilentlyContinue

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $projectRoot
# At logon, plus an hourly "is it alive?" trigger: after sleep/crash it relaunches within
# an hour. If an autopilot is already running, the new copy sees the lock and exits.
$trigger = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) -RepetitionInterval (New-TimeSpan -Hours 1))
)
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 5) `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName "CryptoQuantAutopilot" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Crypto quant paper trader (Binance Testnet): data refresh + hourly live_runner." -Force | Out-Null
Start-ScheduledTask -TaskName "CryptoQuantAutopilot"
Write-Host "CryptoQuantAutopilot installed and started. Log: logs\autopilot.log" -ForegroundColor Green
