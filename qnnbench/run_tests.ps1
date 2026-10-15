# =============================================================================
# qnnbench — Quick Test Suite Runner
# =============================================================================
# Runs different test subsets based on what you need to verify.
#
# Usage:
#   .\run_tests.ps1                      # All tests (excluding slow)
#   .\run_tests.ps1 -Suite unit          # Unit tests only
#   .\run_tests.ps1 -Suite numeric       # Gradient correctness tests
#   .\run_tests.ps1 -Suite property      # Hypothesis property tests
#   .\run_tests.ps1 -Suite regression    # DA1->DA2 correction guards
#   .\run_tests.ps1 -Suite viz           # Figure rendering tests (Agg backend)
#   .\run_tests.ps1 -Suite fast          # All non-slow tests
#   .\run_tests.ps1 -Suite full          # ALL tests including slow
#   .\run_tests.ps1 -File test_seeds.py  # Single file
#   .\run_tests.ps1 -Coverage            # With coverage report
# =============================================================================

param(
    [string]$Suite    = "fast",
    [string]$File     = "",
    [switch]$Coverage = $false,
    [switch]$Verbose  = $false,
    [switch]$FailFast = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Pytest = ".venv\Scripts\pytest.exe"
$PythonVenv = ".venv\Scripts\python.exe"

function Write-Step { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "[OK] $msg"   -ForegroundColor Green }
function Write-FAIL { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }
function Write-INFO { param($msg) Write-Host "  $msg"      -ForegroundColor Gray }

# Verify venv exists
if (-not (Test-Path $Pytest)) {
    Write-FAIL "pytest not found in .venv. Run .\run_all.ps1 -Phase P0 first."
}

$CoverageFlags = if ($Coverage) {
    "--cov=src/qnnbench --cov-report=term-missing --cov-report=html:outputs/coverage"
} else { "" }

$VerboseFlag  = if ($Verbose)  { "-v" } else { "-q" }
$XFlag        = if ($FailFast) { "-x" } else { "" }
$CommonFlags  = "$VerboseFlag $XFlag --tb=short"

# ─── Suite map ────────────────────────────────────────────────────────────────
$Suites = @{
    "unit"       = "tests/unit/       -m unit"
    "integration"= "tests/integration/ -m integration"
    "numeric"    = "tests/numeric/     -m numeric"
    "property"   = "tests/property/    -m property"
    "regression" = "tests/regression/  -m regression"
    "viz"        = "tests/viz/         -m viz --mpl"
    "fast"       = "tests/             -m 'not slow'"
    "full"       = "tests/"
}

# ─── Detailed per-module mapping ──────────────────────────────────────────────
$ModuleTests = @{
    "M01_types"    = "tests/unit/test_types.py"
    "M02_seeds"    = "tests/unit/test_seeds.py"
    "M03_data"     = "tests/unit/test_data.py"
    "M04_circuit"  = "tests/unit/test_circuit.py"
    "M05_backend"  = "tests/unit/test_backend.py"
    "M06_grad"     = "tests/unit/test_optimisers.py tests/numeric/"
    "M07_metric"   = "tests/unit/test_metric.py"
    "M08_budget"   = "tests/unit/test_budget.py tests/property/"
    "M09_optim"    = "tests/unit/test_optimisers.py"
    "M10_alloc"    = "tests/unit/test_allocation.py"
    "M11_trainer"  = "tests/integration/"
    "M13_io"       = "tests/unit/test_io.py"
    "regression"   = "tests/regression/test_guards.py"
    "analysis"     = "tests/unit/test_analysis.py"
}

Write-Step "qnnbench Test Runner"
Write-INFO "Suite: $Suite | Coverage: $Coverage | Verbose: $Verbose | FailFast: $FailFast"

# ─── Run tests ────────────────────────────────────────────────────────────────
if ($File -ne "") {
    # Single file mode — search for it
    $FoundFile = Get-ChildItem -Path tests/ -Recurse -Filter $File | Select-Object -First 1
    if (-not $FoundFile) { Write-FAIL "Test file not found: $File" }
    $TestPath = $FoundFile.FullName
    Write-INFO "Running: $TestPath"
    $Cmd = "$Pytest $TestPath $CommonFlags $CoverageFlags"
} elseif ($ModuleTests.ContainsKey($Suite)) {
    # Module-specific test path
    $TestPath = $ModuleTests[$Suite]
    Write-INFO "Module tests: $TestPath"
    $Cmd = "$Pytest $TestPath $CommonFlags $CoverageFlags"
} elseif ($Suites.ContainsKey($Suite)) {
    $SuiteSpec = $Suites[$Suite]
    Write-INFO "Suite spec: $SuiteSpec"
    $Cmd = "$Pytest $SuiteSpec $CommonFlags $CoverageFlags"
} else {
    Write-FAIL "Unknown suite: '$Suite'. Valid: $($Suites.Keys -join ', '), $($ModuleTests.Keys -join ', '), or use -File <filename>"
}

Write-INFO "Command: $Cmd"
Invoke-Expression $Cmd

if ($LASTEXITCODE -eq 0) {
    Write-OK "Tests PASSED"
    if ($Coverage) {
        Write-INFO "Coverage HTML report: outputs/coverage/index.html"
    }
} else {
    Write-FAIL "Tests FAILED (exit code $LASTEXITCODE)"
}
