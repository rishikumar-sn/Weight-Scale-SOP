param(
    [string]$Output = "",
    [switch]$Notify
)
$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "App Python environment missing: $pythonPath" }
if (-not $Output) { $Output = Join-Path $projectRoot "data\datasets\jewel-thread-crops-20261031" }
New-Item -ItemType Directory -Path $Output -Force | Out-Null
$logPath = Join-Path $Output "run.log"
$previousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $pythonPath -u (Join-Path $PSScriptRoot "build_thread_crop_dataset.py") --output $Output --resume 2>&1 | Tee-Object -FilePath $logPath -Append
    $runExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $previousPreference
}
$message = if ($runExitCode -eq 0) { "Dataset finished. Open $Output\summary.json and review.html." } else { "Dataset stopped or has errors. Check $Output\summary.json and run.log. Run this command again to resume." }
Write-Host $message
if ($Notify) {
    $popup = New-Object -ComObject WScript.Shell
    $popup.Popup($message, 15, "Jewel/thread dataset", 64) | Out-Null
}
exit $runExitCode
