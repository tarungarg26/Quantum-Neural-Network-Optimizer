# =============================================================================
# qnnbench — Live Demo Script for Teacher Presentation
# Run from: D:\QNNO\qnnbench\
#
# Usage:
#   .\demo.ps1              # Interactive (pauses between modes)
#   .\demo.ps1 -Mode 1      # Quantum analytic only (shots=None)
#   .\demo.ps1 -Mode 2      # Finite-shot quantum only (shots=128)
#   .\demo.ps1 -Mode 3      # Classical baselines only (LogReg + MLP)
#   .\demo.ps1 -All         # Run all three without pausing
# =============================================================================

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet(0, 1, 2, 3)]
    [int]$Mode = 0,

    [switch]$All
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Locate Python
$PyVenv = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (Test-Path $PyVenv) {
    $Py = $PyVenv
} else {
    $Py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $Py) {
        Write-Host "[ERROR] Python environment not found." -ForegroundColor Red
        Write-Host "Please run .\setup_env.ps1 first to create .venv" -ForegroundColor Yellow
        exit 1
    }
}

$DemoPy = Join-Path $ScriptDir "demo.py"
if (-not (Test-Path $DemoPy)) {
    Write-Host "[ERROR] demo.py not found at $DemoPy" -ForegroundColor Red
    exit 1
}

$PyArgs = @($DemoPy)

if ($All) {
    $PyArgs += "--all"
} elseif ($Mode -gt 0) {
    $PyArgs += "--mode"
    $PyArgs += "$Mode"
}

& $Py @PyArgs
exit $LASTEXITCODE
