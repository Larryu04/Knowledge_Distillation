# Starts CoreCS Global on one machine: the web API/UI and one worker serving both lanes.
# Usage (from the GLOBAL folder):  powershell -ExecutionPolicy Bypass -File scripts\run_platform.ps1
param([int]$Port = 8000, [string]$BindHost = "127.0.0.1")

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Load .env (KEY=VALUE lines; comments and blanks ignored) into this process's environment.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -Encoding utf8) {
        if ($line -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*)\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"').Trim("'"), "Process")
        }
    }
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Create the venv first: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" }
$env:PYTHONIOENCODING = "utf-8"

$worker = Start-Process -FilePath $python -ArgumentList "-m", "corecs_global.worker" -NoNewWindow -PassThru
Write-Host "Worker started (PID $($worker.Id)). Open http://${BindHost}:$Port"
try {
    & $python -m uvicorn corecs_global.api:app --host $BindHost --port $Port
} finally {
    # Ctrl+C: stop the worker too. Interrupted jobs resume from their checkpoint next time.
    if (-not $worker.HasExited) { & taskkill /T /F /PID $worker.Id | Out-Null }
}
