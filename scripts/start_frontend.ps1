$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

Push-Location (Join-Path $projectRoot "frontend")
try {
    npm run dev
}
finally {
    Pop-Location
}
