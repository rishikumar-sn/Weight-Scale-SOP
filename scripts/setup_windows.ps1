$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$virtualPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $virtualPython)) {
    py -3.10 -m venv (Join-Path $projectRoot ".venv")
}

& $virtualPython -m pip install --upgrade pip
& $virtualPython -m pip install -r (Join-Path $projectRoot "requirements.txt")
npm ci --prefix (Join-Path $projectRoot "frontend")
npm run build --prefix (Join-Path $projectRoot "frontend")

Write-Host "Setup complete. Run scripts\start_backend.ps1"
