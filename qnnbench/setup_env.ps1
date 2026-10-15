# =============================================================================
# qnnbench — Environment Setup Script
# Run from: D:\QNNO\qnnbench\
# Usage:  .\setup_env.ps1
#         .\setup_env.ps1 -Recreate    (delete .venv and start fresh)
# =============================================================================

param([switch]$Recreate = $false)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "[OK] $msg"   -ForegroundColor Green }
function Write-FAIL { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

# ─── Paths (defined early so StrictMode never complains) ────────────────────
$PipExe    = ".venv\Scripts\pip.exe"
$PythonExe = ".venv\Scripts\python.exe"
$PytestExe = ".venv\Scripts\pytest.exe"
function Write-INFO { param($msg) Write-Host "  $msg"      -ForegroundColor Gray }

# ─── Check we are in the right directory ────────────────────────────────────
if (-not (Test-Path "pyproject.toml")) {
    Write-FAIL "Run this script from the qnnbench/ directory (where pyproject.toml lives).`nTry:  cd D:\QNNO\qnnbench   then   .\setup_env.ps1"
}

# ─── Check Python ────────────────────────────────────────────────────────────
Write-Step "Checking Python"
try {
    $PyVer = python --version 2>&1
    Write-OK "Python found: $PyVer"
} catch {
    Write-FAIL "Python not found. Install from https://python.org (3.11+)"
}

# ─── Check / install uv ─────────────────────────────────────────────────────
Write-Step "Checking uv"
$uvOk = $false
try {
    $UvVer = uv --version 2>&1
    Write-OK "uv found: $UvVer"
    $uvOk = $true
} catch {
    Write-INFO "uv not found — will use pip instead"
}

# ─── Create virtual environment ─────────────────────────────────────────────
Write-Step "Setting up virtual environment (.venv)"

if ($Recreate -and (Test-Path ".venv")) {
    Write-INFO "Removing existing .venv..."
    Remove-Item -Recurse -Force ".venv"
    Write-OK ".venv removed"
}

if (-not (Test-Path ".venv")) {
    if ($uvOk) {
        Write-INFO "Creating .venv with uv..."
        uv venv
    } else {
        Write-INFO "Creating .venv with python -m venv..."
        python -m venv .venv
    }
    Write-OK ".venv created"
} else {
    Write-OK ".venv already exists (pass -Recreate to rebuild)"
}

# ─── Activate and install ────────────────────────────────────────────────────
Write-Step "Installing dependencies"

if (-not (Test-Path $PythonExe)) {
    Write-FAIL ".venv\Scripts\python.exe not found — venv creation may have failed."
}

if ($uvOk) {
    Write-INFO "Installing via uv pip..."
    uv pip install -e "."
} else {
    Write-INFO "Installing via pip..."
    & $PipExe install -e "."
}

Write-OK "Dependencies installed"

# ─── Verify key packages ─────────────────────────────────────────────────────
Write-Step "Verifying installations"

$packages = @("pennylane", "numpy", "sklearn", "matplotlib", "hypothesis", "pytest")
foreach ($pkg in $packages) {
    $ver = & $PythonExe -c "import $pkg; print($pkg.__version__)" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-OK "${pkg}: $ver"
    } else {
        Write-INFO "WARNING: $pkg not importable — $ver"
    }
}

# verify qnnbench itself
$qnbVer = & $PythonExe -c "import qnnbench; print('imported OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-OK "qnnbench: $qnbVer"
} else {
    Write-FAIL "qnnbench import failed: $qnbVer"
}

# ─── Write ENV_MANIFEST.json ─────────────────────────────────────────────────
Write-Step "Writing docs/ENV_MANIFEST.json"

$envScript = @'
import json, sys, platform, pathlib
try:
    import pennylane as qml
    pl_ver = qml.__version__
except Exception:
    pl_ver = "not installed"
try:
    import numpy as np
    np_ver = np.__version__
except Exception:
    np_ver = "not installed"
try:
    import sklearn
    sk_ver = sklearn.__version__
except Exception:
    sk_ver = "not installed"
try:
    import matplotlib
    mpl_ver = matplotlib.__version__
except Exception:
    mpl_ver = "not installed"

env = {
    "python": sys.version,
    "platform": platform.platform(),
    "pennylane": pl_ver,
    "numpy": np_ver,
    "sklearn": sk_ver,
    "matplotlib": mpl_ver,
}
pathlib.Path("docs").mkdir(exist_ok=True)
with open("docs/ENV_MANIFEST.json", "w") as f:
    json.dump(env, f, indent=2)
print(json.dumps(env, indent=2))
'@

$envScript | & $PythonExe
Write-OK "docs/ENV_MANIFEST.json written"

# ─── Quick smoke test ─────────────────────────────────────────────────────────
Write-Step "Quick smoke test"

$smokeScript = @'
from qnnbench.core.types import Theta, Batch, RunKey, ANGLE_SHAPE, N_GATE_PARAMS, N_TOTAL_PARAMS
from qnnbench.core.seeds import stream, PurposeID
from qnnbench.data.manager import make_task, make_split, fit_scaler
from qnnbench.quantum.circuit_factory import build_qnode, n_gates_report
from qnnbench.budget.ledger import Ledger, BudgetManager
import numpy as np

X, y = make_task()
assert X.shape == (100, 4), f"Expected (100,4), got {X.shape}"
assert set(y) == {-1.0, 1.0}, f"Labels wrong: {set(y)}"

gates = n_gates_report()
assert gates == {"encoding_RY": 4, "trainable": 16, "cnot": 8}, f"Gates wrong: {gates}"

assert N_GATE_PARAMS == 16
assert N_TOTAL_PARAMS == 17

rng = stream(7, 0, PurposeID.INIT)
angles = rng.uniform(0, 2*3.14159, ANGLE_SHAPE)
theta = Theta(angles=angles, bias=0.0)
flat = theta.flat()
assert flat.shape == (17,), f"flat shape wrong: {flat.shape}"

print("ALL CHECKS PASSED")
print(f"  make_task:       100 samples, 2 classes (versicolor/virginica)")
print(f"  n_gates_report:  encoding_RY=4, trainable=16, cnot=8")
print(f"  N_GATE_PARAMS:   {N_GATE_PARAMS}")
print(f"  N_TOTAL_PARAMS:  {N_TOTAL_PARAMS}")
print(f"  Theta.flat():    shape {flat.shape}")
'@

$smokeScript | & $PythonExe
if ($LASTEXITCODE -ne 0) {
    Write-FAIL "Smoke test failed"
}
Write-OK "Smoke test passed"

# ─── Done ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Environment ready!" -ForegroundColor Green
Write-Host ""
Write-Host "  Make sure you are in D:\QNNO\qnnbench\ then run:" -ForegroundColor White
Write-Host "    .\run_tests.ps1                       (fast test suite)" -ForegroundColor White
Write-Host "    .\run_experiments.ps1 -Experiment E1  (verify PSR gradients)" -ForegroundColor White
Write-Host "    .\demo.ps1                            (live demo for teacher)" -ForegroundColor White
Write-Host "    .\run_all.ps1 -Phase P1               (phase 1 unit tests)" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
