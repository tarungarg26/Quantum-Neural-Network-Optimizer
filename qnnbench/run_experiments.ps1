# =============================================================================
# qnnbench — Individual Experiment Runner
# =============================================================================
# Run specific experiments (E1-E8) or the full experiment suite.
#
# Usage:
#   .\run_experiments.ps1                    # Run all E1-E8 (tiny mode)
#   .\run_experiments.ps1 -Experiment E1     # Run only E1
#   .\run_experiments.ps1 -Experiment E3 -Full  # E3 full scale (slow)
#   .\run_experiments.ps1 -Experiment E3 -SplitSeed 7 -InitSeed 0
#
# What each experiment does:
#   E1: PSR 2-term vs autodiff verification (200 random theta/x pairs)
#   E2: PSR 4-term on controlled-rotation circuit verification
#   E3: 5 optimisers, analytic training, 80 updates, 15 paired runs
#   E4: Finite-shot training at 4 budgets x 3 shot levels x 5 optimisers
#   E5: Shot-noise law: Var(f_hat)*N vs 1/2 bound
#   E6: Trainability scan (width/depth/init/observable)
#   E7: QNG geometry ablation (damping, metric inputs, condition numbers)
#   E8: Adaptive vs uniform shot allocation at matched total shots
# =============================================================================

param(
    [string]$Experiment = "all",
    [switch]$Full       = $false,
    [int]$SplitSeed     = -1,
    [int]$InitSeed      = -1,
    [string]$Optimiser  = "",
    [switch]$Help       = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PythonVenv = ".venv\Scripts\python.exe"
$Pytest = ".venv\Scripts\pytest.exe"

function Write-Step { param($msg) Write-Host "`n== $msg ==" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "[OK] $msg"   -ForegroundColor Green }
function Write-FAIL { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }
function Write-INFO { param($msg) Write-Host "  $msg"      -ForegroundColor Gray }

if (-not (Test-Path $PythonVenv)) {
    Write-FAIL "No .venv found. Run .\run_all.ps1 -Phase P0 first."
}

if ($Help) {
    Write-Host @"
Usage: .\run_experiments.ps1 [-Experiment <E1-E8|all>] [-Full] [-SplitSeed <n>] [-InitSeed <n>] [-Optimiser <name>]

Experiments:
  E1  2-term PSR correctness
  E2  4-term PSR on controlled-rotation circuit
  E3  Analytic 5-optimiser comparison (15 paired runs)
  E4  Finite-shot equal-budget comparison
  E5  Shot-noise law (N*Var vs 1/2 bound)
  E6  Trainability scan (width/depth/init/observable)
  E7  QNG geometry ablation
  E8  Adaptive vs uniform allocation
  all Run E1 through E8 in order

Flags:
  -Full         Run at full scale (15 runs, no tiny shortcuts)
  -SplitSeed    Override split seed (for debugging a specific seed)
  -InitSeed     Override init seed
  -Optimiser    Run E3/E4 for a single optimiser only
"@
    exit 0
}

$TinyFlag  = if ($Full) { "" } else { "--tiny" }
$SeedFlags = ""
if ($SplitSeed -ge 0) { $SeedFlags += " --split-seed $SplitSeed" }
if ($InitSeed  -ge 0) { $SeedFlags += " --init-seed $InitSeed" }
if ($Optimiser -ne "")  { $SeedFlags += " --optimiser $Optimiser" }

# ─── Experiment descriptions ──────────────────────────────────────────────────
$ExptDescs = @{
    "E1" = "2-term PSR vs autodiff (RQ1 gradient correctness)"
    "E2" = "4-term PSR on CRot circuit (RQ1 gradient correctness)"
    "E3" = "5-optimiser analytic comparison, 80 updates, 15 paired seeds (RQ2)"
    "E4" = "Equal-shot ranking: budgets x shots x optimisers (RQ2)"
    "E5" = "Shot-noise law: Var(f_hat)*N bound (shot-noise characterisation)"
    "E6" = "Trainability scan: width/depth/init/observable (RQ4)"
    "E7" = "QNG geometry ablation: damping/metric-inputs/cond(G) (RQ3)"
    "E8" = "Adaptive vs uniform shot allocation (RQ5)"
}

function Run-Experiment {
    param([string]$ID)

    $Desc = $ExptDescs[$ID]
    if (-not $Desc) { Write-FAIL "Unknown experiment: $ID" }

    Write-Step "$ID — $Desc"

    $Cmd = "$PythonVenv -m qnnbench.experiments.runner $($ID.ToLower()) $TinyFlag $SeedFlags"
    Write-INFO "Command: $Cmd"
    Invoke-Expression $Cmd

    if ($LASTEXITCODE -ne 0) {
        Write-FAIL "$ID failed (exit $LASTEXITCODE)"
    }
    Write-OK "$ID complete"

    # Show output summary
    $RunsDir = "runs"
    if (Test-Path $RunsDir) {
        $Latest = Get-ChildItem -Path $RunsDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 3
        if ($Latest) {
            Write-INFO "Latest run dirs:"
            $Latest | ForEach-Object { Write-INFO "  $($_.Name)  ($($_.LastWriteTime))" }
        }
    }
}

# ─── Main ─────────────────────────────────────────────────────────────────────
Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host " qnnbench Experiment Runner" -ForegroundColor Magenta
Write-Host " Experiment: $Experiment | Full: $Full" -ForegroundColor Magenta
Write-Host "============================================================`n" -ForegroundColor Magenta

$ExpsToRun = if ($Experiment -eq "all") {
    @("E1","E2","E3","E4","E5","E6","E7","E8")
} else {
    @($Experiment.ToUpper())
}

foreach ($exp in $ExpsToRun) {
    Run-Experiment -ID $exp
}

# ─── Post-experiment: run verifying tests ─────────────────────────────────────
Write-Step "Post-experiment: verify smoke tests"
$SmokeTests = $ExpsToRun | ForEach-Object {
    $id = $_.ToLower()
    "tests/ -k 'smoke_$id or tiny_$id'"
}
# Try to run experiment smoke tests if they exist
$TestRun = "$Pytest tests/ -m 'not slow' -q --tb=short 2>&1 | Select-Object -First 30"
Write-INFO "Running: $TestRun"
Invoke-Expression $TestRun

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host " Experiments complete: $($ExpsToRun -join ', ')" -ForegroundColor Green
Write-Host " Results in: runs/" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
