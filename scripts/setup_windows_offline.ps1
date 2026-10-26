param(
    [string]$VenvPath = "",
    [string]$PythonExe = "",
    [switch]$VerifyModels
)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$bundleRoot = Split-Path $projectRoot -Parent
$packages = Join-Path $bundleRoot "offline\windows-cp310"
if (-not (Test-Path -LiteralPath (Join-Path $packages "requirements-lock.txt"))) {
    throw "Offline Windows packages missing: $packages. Keep the application beside the offline folder."
}
if (-not (Test-Path -LiteralPath (Join-Path $projectRoot "frontend\dist\index.html"))) { throw "Built frontend missing." }
if (-not $VenvPath) { $VenvPath = Join-Path $projectRoot ".venv" }
if (-not $PythonExe) {
    $PythonExe = & py -3.10 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -ne 0) { throw "Install offline\installers\python-3.10.11-amd64.exe first, then reopen PowerShell." }
}
& $PythonExe -c "import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8, 'Requires Python 3.10 x64'"
if ($LASTEXITCODE -ne 0) { throw "Unsupported Python interpreter." }
& $PythonExe (Join-Path $PSScriptRoot "check_windows_offline_paths.py") $VenvPath (Join-Path $packages "wheels")
if ($LASTEXITCODE -ne 0) { throw "Choose a shorter installation path or enable Windows long paths." }
$env:PIP_NO_INDEX = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$env:PIP_CONFIG_FILE = "NUL"
$env:PYTHONDONTWRITEBYTECODE = "1"
if (-not (Test-Path -LiteralPath (Join-Path $VenvPath "Scripts\python.exe"))) {
    if (Test-Path -LiteralPath $VenvPath) { throw "Destination exists but is not a Windows venv: $VenvPath" }
    & $PythonExe -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
}
$python = Join-Path $VenvPath "Scripts\python.exe"
& $python -c "import sys,struct; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8, 'Existing venv must use Python 3.10 x64'"
if ($LASTEXITCODE -ne 0) { throw "Existing venv has an incompatible interpreter." }
& $python -m pip --isolated install --no-index --no-cache-dir --no-compile --only-binary=:all: --find-links (Join-Path $packages "wheels") -r (Join-Path $packages "requirements-lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Offline dependency installation failed. No downloads are allowed." }
& $python -m pip --isolated check
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed." }
& $python (Join-Path $PSScriptRoot "verify_cpu.py")
if ($LASTEXITCODE -ne 0) { throw "CPU verification failed." }
if ($VerifyModels) {
    & $python (Join-Path $PSScriptRoot "verify_offline_install.py")
    if ($LASTEXITCODE -ne 0) { throw "Offline model verification failed." }
}
Write-Host "Offline setup complete: $VenvPath"
Write-Host "Start with scripts\start_windows_offline.ps1"
