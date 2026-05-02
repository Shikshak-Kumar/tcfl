# ============================================================================
# train_china_osm.ps1
#
# Trains all three algorithms on the China Urban OSM SUMO map:
#   1. FedDQN-TSC      (Ye et al., Scientific Reports 2023)   [Baseline 1]
#   2. MA2C            (Chu et al., IEEE TITS 2019)            [Baseline 2]
#   3. AdaptFlow-TSC   (our novel algorithm)
#
# Results are stored in:
#   F:\ttraffic\tcfl\results\china_osm\adaptflow\
#   F:\ttraffic\tcfl\results\china_osm\fed_dqn_tsc\
#   F:\ttraffic\tcfl\results\china_osm\multi_agent_ac\
#
# Usage:
#   .\train_china_osm.ps1                    # headless SUMO (real microsim)
#   .\train_china_osm.ps1 -Mock              # mock environment (quick test)
#   .\train_china_osm.ps1 -Rounds 10 -Steps 500
#   .\train_china_osm.ps1 -AlgoOnly adaptflow
# ============================================================================

param(
    [int]    $Rounds     = 10,    # AdaptFlow federated rounds
    [int]    $Steps      = 500,   # Steps per episode (same for all algos)
    [int]    $Nodes      = 6,
    [int]    $Clusters   = 2,
    [int]    $FedRounds  = 10,    # FedDQN rounds (same as AdaptFlow)
    [int]    $FedEps     = 20,    # FedDQN episodes per round
    [int]    $MA2CSteps  = 30000, # MA2C total steps = 6 nodes x 500 x 10 rounds
    [switch] $Mock,
    [switch] $Gui,
    [string] $AlgoOnly   = ""
)

$ErrorActionPreference = "Stop"

# Force UTF-8 for Python output on Windows
$env:PYTHONUTF8 = "1"

# --- locate adaptflow_ac ---
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$AcDir       = Join-Path $ScriptDir "adaptflow_ac"
$ResultsBase = Join-Path $ScriptDir "results\china_osm"

# Ensure results directories exist
New-Item -ItemType Directory -Force -Path "$ResultsBase\adaptflow"       | Out-Null
New-Item -ItemType Directory -Force -Path "$ResultsBase\fed_dqn_tsc"     | Out-Null
New-Item -ItemType Directory -Force -Path "$ResultsBase\multi_agent_ac"  | Out-Null

# --- helpers ---
function Write-Header($msg) {
    Write-Host ""
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 65) -ForegroundColor Cyan
    Write-Host ""
}

function Write-Section($msg) {
    Write-Host ""
    Write-Host "  >> $msg" -ForegroundColor Yellow
}

# --- mode flags ---
if ($Mock) {
    $ModeAdaptFlow = ""
    $ModeFedDQN    = "--mode mock"
    $ModeMA2C      = "--mode mock"
    $ModeLabel     = "MOCK"
} elseif ($Gui) {
    $ModeAdaptFlow = "--gui --sumo-scenario china_osm"
    $ModeFedDQN    = "--mode sumo --sumo-scenario china_osm --gui"
    $ModeMA2C      = "--mode sumo --sumo-scenario china_osm --gui"
    $ModeLabel     = "SUMO-GUI"
} else {
    $ModeAdaptFlow = "--real-sumo --sumo-scenario china_osm"
    $ModeFedDQN    = "--mode sumo --sumo-scenario china_osm --real-sumo"
    $ModeMA2C      = "--mode sumo --sumo-scenario china_osm --real-sumo"
    $ModeLabel     = "SUMO-HEADLESS"
}

Write-Header "China Urban OSM - Multi-Algorithm Training [$ModeLabel]"
Write-Host "  Map      : China Urban (OSM, sumo_configs_china_osm)"
Write-Host "  Rounds   : AdaptFlow=$Rounds  FedDQN=$FedRounds  MA2C=N/A (step-based)"
Write-Host "  Steps    : $Steps / episode"
Write-Host "  Results  : $ResultsBase"
Write-Host ""

Push-Location $AcDir

# --- 1. FedDQN-TSC (Baseline 1) ---
$RunFedDQN = ($AlgoOnly -eq "" -or $AlgoOnly -ieq "fed_dqn_tsc" -or $AlgoOnly -ieq "feddqn")
if ($RunFedDQN) {
    Write-Section "[BASELINE 1/2] Training FedDQN-TSC (Ye et al. 2023)"
    $FedDQNResults = "$ResultsBase\fed_dqn_tsc"

    $cmd = "python train\train_fed_dqn_tsc.py " +
           "--rounds $FedRounds " +
           "--episodes-per-round $FedEps " +
           "--steps $Steps " +
           "--results-dir `"$FedDQNResults`" " +
           $ModeFedDQN

    Write-Host "  CMD: $cmd" -ForegroundColor Gray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] FedDQN-TSC exited with code $LASTEXITCODE" -ForegroundColor Yellow
    } else {
        Write-Host "  FedDQN-TSC complete. Results in $FedDQNResults" -ForegroundColor Green
    }
}

# --- 2. MA2C (Baseline 2) ---
$RunMA2C = ($AlgoOnly -eq "" -or $AlgoOnly -ieq "multi_agent_ac" -or $AlgoOnly -ieq "ma2c")
if ($RunMA2C) {
    Write-Section "[BASELINE 2/2] Training MA2C (Chu et al. 2019)"
    $MA2CResults = "$ResultsBase\multi_agent_ac"

    $cmd = "python train\train_multi_agent_ac.py " +
           "--steps $Steps " +
           "--max-total-steps $MA2CSteps " +
           "--results-dir `"$MA2CResults`" " +
           $ModeMA2C

    Write-Host "  CMD: $cmd" -ForegroundColor Gray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] MA2C exited with code $LASTEXITCODE" -ForegroundColor Yellow
    } else {
        Write-Host "  MA2C complete. Results in $MA2CResults" -ForegroundColor Green
    }
}

# --- 3. AdaptFlow-TSC (Our Algorithm) ---
$RunAdaptFlow = ($AlgoOnly -eq "" -or $AlgoOnly -ieq "adaptflow")
if ($RunAdaptFlow) {
    Write-Section "[OUR ALGO] Training AdaptFlow-TSC"
    $AdaptFlowResults = "$ResultsBase\adaptflow"

    $cmd = "python train\train_adaptflow.py " +
           "--rounds $Rounds " +
           "--nodes $Nodes " +
           "--clusters $Clusters " +
           "--steps $Steps " +
           "--results-dir `"$AdaptFlowResults`" " +
           $ModeAdaptFlow

    Write-Host "  CMD: $cmd" -ForegroundColor Gray
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [WARN] AdaptFlow exited with code $LASTEXITCODE" -ForegroundColor Yellow
    } else {
        Write-Host "  AdaptFlow-TSC complete. Results in $AdaptFlowResults" -ForegroundColor Green

        if (-not $Mock) {
            Write-Section "[DEPLOY EVAL] AdaptFlow - All 6 agents in one multi-TLS sim"
            $evalCmd = "python train\eval_adaptflow_deployed.py " +
                       "--results-dir `"$AdaptFlowResults`" " +
                       "--sumo-scenario china_osm " +
                       "--episodes 3 " +
                       "--steps $Steps"
            if ($Gui) { $evalCmd += " --gui" }
            Write-Host "  CMD: $evalCmd" -ForegroundColor Gray
            Invoke-Expression $evalCmd
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  [WARN] Deployed eval exited with code $LASTEXITCODE" -ForegroundColor Yellow
            } else {
                Write-Host "  Deployed eval complete. deployed_eval.json saved." -ForegroundColor Green
            }
        }
    }
}

Pop-Location

# --- summary ---
Write-Header "ALL TRAINING COMPLETE"
Write-Host "  Results location: $ResultsBase"
Write-Host ""
Write-Host "  Directories:"
Write-Host "    adaptflow\      <- Our AdaptFlow-TSC results"
Write-Host "    fed_dqn_tsc\    <- FedDQN-TSC baseline results"
Write-Host "    multi_agent_ac\ <- MA2C baseline results"
Write-Host ""
Write-Host "  Next: python compare_china_osm.py"
Write-Host ""
