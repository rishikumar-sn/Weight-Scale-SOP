$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$virtualPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$pythonCommand = if (Test-Path -LiteralPath $virtualPython) { $virtualPython } else { "python" }
$env:PYTHONDONTWRITEBYTECODE = "1"

Push-Location $projectRoot
try {
    if ($env:SKIP_SCALE_RESTART -ne "1") {
        $scaleDevice = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
            Where-Object { $_.InstanceId -like "USB\VID_1A86&PID_7523*" } |
            Select-Object -First 1
        if ($scaleDevice) {
            Write-Host "Resetting the connected scale adapter before automatic port detection..."
            $restartScript = Join-Path $PSScriptRoot "restart_scale_windows.ps1"
            try {
                $restart = Start-Process -FilePath "powershell.exe" -Verb RunAs -Wait -PassThru -ArgumentList @(
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-File", ('"' + $restartScript + '"')
                )
                if ($restart.ExitCode -ne 0) {
                    Write-Warning "Windows did not reset the scale adapter. The backend will still start and detect the scale port automatically. If the scale stays red, unplug and reconnect its USB cable once."
                }
            }
            catch {
                Write-Warning "The scale reset could not be run. The backend will still start and detect the scale port automatically. If the scale stays red, unplug and reconnect its USB cable once."
            }
        }
    }
    # run.py opens the default browser after startup unless a local UI is connected.
    & $pythonCommand run.py
}
finally {
    Pop-Location
}
