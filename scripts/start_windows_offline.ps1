$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Run scripts\setup_windows_offline.ps1 first." }
$env:PYTHONDONTWRITEBYTECODE = "1"
Push-Location $projectRoot
try {
    & $python (Join-Path $PSScriptRoot "offline_runtime.py")
    if ($LASTEXITCODE -ne 0) { throw "Application exited with code $LASTEXITCODE" }
} finally { Pop-Location }
