#requires -Version 5.1
#requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$scaleDevices = @(
    Get-PnpDevice -PresentOnly |
        Where-Object { $_.InstanceId -like "USB\VID_1A86&PID_7523*" }
)

if ($scaleDevices.Count -ne 1) {
    throw "Expected one connected CH340 scale adapter, but found $($scaleDevices.Count)."
}

$scaleDevice = $scaleDevices[0]
Write-Host "Restarting $($scaleDevice.FriendlyName)"
& pnputil.exe /restart-device $scaleDevice.InstanceId
if ($LASTEXITCODE -ne 0) {
    throw "Windows could not restart the CH340 device. Unplug and reconnect it, then retry."
}

Start-Sleep -Seconds 2
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found. Run scripts\setup_windows.ps1 first."
}

& $python (Join-Path $PSScriptRoot "test_scale.py")
exit $LASTEXITCODE
