param([string]$TestRoot = "", [string]$PythonExe = "")
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $TestRoot) { $TestRoot = Join-Path $env:TEMP "ws-offline-test" }
$venv = Join-Path $TestRoot ".venv"
if (Test-Path -LiteralPath $venv) { throw "Use a new empty TestRoot to prove a fresh installation: $TestRoot" }
New-Item -ItemType Directory -Force -Path $TestRoot | Out-Null
Start-Transcript -Path (Join-Path $TestRoot "installation.log")
try {
    & (Join-Path $PSScriptRoot "setup_windows_offline.ps1") -VenvPath $venv -PythonExe $PythonExe | Tee-Object -FilePath (Join-Path $TestRoot "dependencies.log")
    & (Join-Path $venv "Scripts\python.exe") (Join-Path $PSScriptRoot "verify_offline_install.py") --report (Join-Path $TestRoot "verification.json") | Tee-Object -FilePath (Join-Path $TestRoot "models.log")
    if ($LASTEXITCODE -ne 0) { throw "Offline smoke test failed." }
    Write-Host "PASS: fresh local-wheel installation and CPU/OCR/model smoke test. See $TestRoot"
} finally { Stop-Transcript }
