param(
    [switch]$Reload
)

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    throw "Virtual environment not found. Run .\setup_backend.ps1 first."
}

Set-Location $PSScriptRoot

$uvicornArgs = @(
    "-m", "uvicorn",
    "server:app",
    "--host", "0.0.0.0",
    "--port", "5000"
)

if ($Reload) {
    $uvicornArgs += "--reload"
}

& $venvPython @uvicornArgs
