# =============================================================================
# qnnbench — Step-by-Step PowerShell Runner
# =============================================================================
# Run this script to execute the entire qnnbench pipeline, phase by phase.
# Each function corresponds to one build phase from FLOW.md (Section 5).
#
# Usage:
#   .\run_all.ps1                     # Run all phases P0 -> P8
#   .\run_all.ps1 -Phase P3           # Run only Phase P3 (and its deps)
#   .\run_all.ps1 -SkipTo P4          # Skip P0-P3, start from P4
#   .\run_all.ps1 -Tiny               # Use --tiny flag for experiments (faster)
#
# Prerequisites:
#   - Python 3.11+ or 3.13 installed
#   - uv installed (https://docs.astral.sh/uv/)
#   - Run from the qnnbench/ directory
# =============================================================================

param(
    [string]$Phase = "all",
    [string]$SkipTo = "",
    [switch]$Tiny = $false,
    [switch]$Help = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─── Colours ─────────────────────────────────────────────────────────────────
function Write-Step   { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Write-OK     { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-FAIL   { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }
function Write-WARN   { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-INFO   { param($msg) Write-Host "  $msg" -ForegroundColor Gray }

# ─── Helpers ─────────────────────────────────────────────────────────────────
function Invoke-Step {
    param([string]$Name, [scriptblock]$Block)
    Write-Step $Name
    try {
        & $Block
        Write-OK "$Name PASSED"
    } catch {
        Write-FAIL "$Name FAILED: $_"
    }
}

function Run-Cmd {
    param([string]$Cmd)
    Write-INFO ">>> $Cmd"
    Invoke-Expression $Cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Command exited with code $LASTEXITCODE"
    }
}

$TinyFlag = if ($Tiny) { "--tiny" } else { "" }
$VenvPy   = ".venv\Scripts\python.exe"
$VenvPip  = ".venv\Scripts\pip.exe"
$Pytest   = ".venv\Scripts\pytest.exe"
$Qnnbench = ".venv\Scripts\qnnbench.exe"

# ─── Show help ────────────────────────────────────────────────────────────────
if ($Help) {
    Write-Host @"
Usage: .\run_all.ps1 [-Phase <P0|P1|...|P8|all>] [-SkipTo <P0|...|P8>] [-Tiny] [-Help]

Phases:
  P0  Scaffold, lint, empty test pass
  P1  Core types, seeds, IO, budget  (unit tests)
  P2  Data, circuit, backend  (golden tests)
  P3  Gradient engines, E1 and E2 verification
  P4  Optimisers, trainer, classical baselines
  P5  Finite-shot path, tuning, shot-noise law
  P6  Allocation policies, E6, E7, E8
  P7  Analysis, visualisation, report, claims lint
  P8  Freeze configs, run 15 replications
  all Run all phases P0-P8 in order
"@
    exit 0
}

# ─── Phase definitions ────────────────────────────────────────────────────────

function Phase-P0 {
    Invoke-Step "P0: Scaffold, lint, fast test pass" {
        # Verify Python
        Run-Cmd "python --version"

        # Create venv with uv
        if (-not (Test-Path ".venv")) {
            Run-Cmd "uv venv --python 3.13"
        }

        # Install project in editable mode with all deps
        Run-Cmd "uv pip install -e ."

        # Lint (ruff)
        Run-Cmd "$VenvPy -m ruff check src/ tests/ --select E,W,F --ignore E501"

        # Fast test pass (unit only, no slow)
        Run-Cmd "$Pytest tests/ -m 'not slow' -q --tb=short --timeout=90 -x 2>&1 | Select-Object -First 50"
    }
}

function Phase-P1 {
    Invoke-Step "P1: Core types, seeds, IO, budget — unit tests" {
        Run-Cmd "$Pytest tests/unit/test_types.py tests/unit/test_seeds.py tests/unit/test_budget.py tests/unit/test_io.py -v --tb=short -x"
    }
}

function Phase-P2 {
    Invoke-Step "P2: Data, circuit, backend — golden tests + leakage" {
        Run-Cmd "$Pytest tests/unit/test_data.py tests/unit/test_circuit.py tests/unit/test_backend.py tests/regression/test_guards.py::test_correction_1_no_leakage -v --tb=short -x"
    }
}

function Phase-P3 {
    Invoke-Step "P3: Gradient engines — E1 (PSR vs autodiff) and E2 (4-term CRot)" {
        # Run E1 and E2 experiments
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e1 $TinyFlag"
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e2 $TinyFlag"

        # Numeric gradient tests
        Run-Cmd "$Pytest tests/numeric/ -v --tb=short -x -m numeric"
    }
}

function Phase-P4 {
    Invoke-Step "P4: Optimisers, trainer, classical baselines + E3 smoke" {
        Run-Cmd "$Pytest tests/unit/test_optimisers.py -v --tb=short -x"

        # E3 smoke: 5 optimisers x 1 seed (tiny)
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e3 --tiny"

        # Classical baselines smoke
        Run-Cmd "$Pytest tests/unit/test_baselines.py -v --tb=short -x"
    }
}

function Phase-P5 {
    Invoke-Step "P5: Finite-shot path, shot-noise law (E5), budget never overshoot" {
        # Property tests (budget)
        Run-Cmd "$Pytest tests/property/ -v --tb=short -x -m property"

        # E4 smoke (finite-shot)
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e4 --tiny"

        # E5 shot noise law
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e5 $TinyFlag"
    }
}

function Phase-P6 {
    Invoke-Step "P6: Allocation policies, E6, E7, E8" {
        Run-Cmd "$Pytest tests/unit/test_allocation.py -v --tb=short -x"

        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e6 $TinyFlag"
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e7 $TinyFlag"
        Run-Cmd "$VenvPy -m qnnbench.experiments.runner e8 $TinyFlag"
    }
}

function Phase-P7 {
    Invoke-Step "P7: Analysis, viz (all 24 figures), claims lint, schema tests" {
        Run-Cmd "$Pytest tests/unit/test_analysis.py -v --tb=short -x"
        Run-Cmd "$Pytest tests/viz/ -v --tb=short -x -m viz"

        # Build report (fixture data)
        Run-Cmd "$VenvPy -m qnnbench.cli report --from-fixtures"
    }
}

function Phase-P8 {
    Invoke-Step "P8: Freeze configs + 15 paired replications" {
        # Tune each optimiser (validation only)
        Run-Cmd "$VenvPy -m qnnbench.cli tune"

        # Freeze configs (hash-lock)
        Run-Cmd "$VenvPy -m qnnbench.cli freeze"

        # Full 15 runs per frozen config
        Run-Cmd "$VenvPy -m qnnbench.cli replicate"

        # Build final report from runs/
        Run-Cmd "$VenvPy -m qnnbench.cli report"

        Write-INFO "Report written to outputs/report/"
    }
}

# ─── Execution order ──────────────────────────────────────────────────────────
$AllPhases = @("P0","P1","P2","P3","P4","P5","P6","P7","P8")
$PhaseFns  = @{
    "P0" = { Phase-P0 }
    "P1" = { Phase-P1 }
    "P2" = { Phase-P2 }
    "P3" = { Phase-P3 }
    "P4" = { Phase-P4 }
    "P5" = { Phase-P5 }
    "P6" = { Phase-P6 }
    "P7" = { Phase-P7 }
    "P8" = { Phase-P8 }
}

$PhasesToRun = if ($Phase -ne "all") {
    @($Phase)
} else {
    $AllPhases
}

if ($SkipTo -ne "") {
    $idx = $AllPhases.IndexOf($SkipTo)
    if ($idx -lt 0) { Write-FAIL "Unknown -SkipTo phase: $SkipTo" }
    $PhasesToRun = $AllPhases[$idx..($AllPhases.Length - 1)]
}

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host " qnnbench pipeline" -ForegroundColor Magenta
Write-Host " Phases: $($PhasesToRun -join ', ')" -ForegroundColor Magenta
Write-Host " Tiny:   $Tiny" -ForegroundColor Magenta
Write-Host "============================================================`n" -ForegroundColor Magenta

foreach ($p in $PhasesToRun) {
    if (-not $PhaseFns.ContainsKey($p)) {
        Write-FAIL "Unknown phase: $p"
    }
    & $PhaseFns[$p]
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " ALL PHASES COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
