#requires -Version 5.1
#requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$scaleDevices = @(
    Get-PnpDevice -PresentOnly |
        Where-Object { $_.InstanceId -like "USB\VID_1A86&PID_7523*" }
)
if ($scaleDevices.Count -ne 1) {
    throw "Expected one connected CH340 scale adapter, but found $($scaleDevices.Count)."
}

Write-Host "Restarting $($scaleDevices[0].FriendlyName)"
& pnputil.exe /restart-device $scaleDevices[0].InstanceId
if ($LASTEXITCODE -ne 0) {
    throw "Windows could not restart the CH340 device."
}
Write-Host "Scale adapter restarted. Start the backend now; do not open COM12 in another app."
