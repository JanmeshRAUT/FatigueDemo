param(
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

$venvPath = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$requirementsPath = Join-Path $PSScriptRoot "requirements.txt"

if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host "Removing existing virtual environment at $venvPath"
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
}

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment creation failed. Expected interpreter not found at $pythonExe"
}

Write-Host "Upgrading pip, setuptools, and wheel"
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "Installing backend dependencies"
& $pythonExe -m pip install -r $requirementsPath

Write-Host "Setup complete. Run backend with: .\run_backend.ps1"
