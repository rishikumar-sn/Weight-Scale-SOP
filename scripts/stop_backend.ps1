$ErrorActionPreference = "Stop"

try {
    $response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/shutdown" -TimeoutSec 5
    Write-Host $response.message
}
catch {
    Write-Error "The backend could not be shut down gracefully. Confirm that it is running on port 8000. $($_.Exception.Message)"
}
