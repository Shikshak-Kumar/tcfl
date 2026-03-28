# China OSM — headless SUMO (no GUI windows, faster for long runs).
# 10 rounds x 500 steps x 6 nodes.
#
# Requirements: sumo on PATH or SUMO_HOME set.
# With GUI: .\train_china_osm.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$env:SUMO_HEADLESS = "1"

$SC    = "--sumo-scenario", "china_osm"
$R     = "--rounds", "10"
$STEPS = "--steps", "500"

Write-Host ""
Write-Host "===== China OSM — AdaptFlow Headless =====" -ForegroundColor Cyan
Write-Host "  Nodes  : 6  |  Rounds: 10  |  Steps: 500" -ForegroundColor Cyan
Write-Host "  Mode   : Headless SUMO (SUMO_HEADLESS=1)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --real-sumo

Write-Host ""
Write-Host "AdaptFlow complete. Results in results_adaptflow_china_osm/" -ForegroundColor Green
