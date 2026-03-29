# India Rural OSM — AdaptFlow training in headless SUMO (no GUI window).
# 3 rounds x 500 steps x 6 nodes. Faster than GUI — no rendering overhead.
#
# Node layout (India rural zones):
#   node_0  Town market / community hub        (begin=0,    jam=12, Tier 1)
#   node_1  Rural school / temple zone         (begin=400,  jam=18, Tier 2)
#   node_2  Agricultural / farming district    (begin=800,  jam=25, Tier 3)
#   node_3  Village residential outskirts      (begin=1200, jam=30, Tier 3)
#   node_4  Industrial zone / highway approach (begin=1600, jam=40, Tier 3)
#   node_5  State highway junction             (begin=2000, jam=22, Tier 3)
#
# Results stored in: results_adaptflow_india_rural_osm/
# GUI version:       .\train_india_rural_osm.ps1
# Requirements:      sumo (not sumo-gui) on PATH or SUMO_HOME set.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$env:SUMO_HEADLESS = "1"

$SC    = "--sumo-scenario", "india_rural_osm"
$R     = "--rounds", "3"
$STEPS = "--steps", "500"

Write-Host ""
Write-Host "===== India Rural OSM — AdaptFlow Training (Headless) =====" -ForegroundColor Yellow
Write-Host "  Nodes  : 6 (market / school / farm / residential / industrial / highway)" -ForegroundColor Yellow
Write-Host "  Rounds : 3  |  Steps/episode: 500" -ForegroundColor Yellow
Write-Host "  Mode   : sumo headless (real TraCI, no GUI window)" -ForegroundColor Yellow
Write-Host "  Results: results_adaptflow_india_rural_osm/" -ForegroundColor Yellow
Write-Host "===========================================================" -ForegroundColor Yellow
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --real-sumo

Write-Host ""
Write-Host "Training complete. Results saved to results_adaptflow_india_rural_osm/" -ForegroundColor Yellow
