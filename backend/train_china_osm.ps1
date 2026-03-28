# China OSM — Full training with sumo-gui (real SUMO, GUI visible).
# 10 rounds x 500 steps x 6 nodes. Each node opens sumo-gui sequentially (one at a time).
#
# Requirements: sumo-gui on PATH or SUMO_HOME set.
# Headless (no windows): .\train_china_osm_headless.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Remove-Item Env:SUMO_HEADLESS -ErrorAction SilentlyContinue

$SC    = "--sumo-scenario", "china_osm"
$R     = "--rounds", "10"
$STEPS = "--steps", "500"

Write-Host ""
Write-Host "===== China OSM — AdaptFlow Training =====" -ForegroundColor Cyan
Write-Host "  Nodes  : 6 (each with unique SUMO config)" -ForegroundColor Cyan
Write-Host "  Rounds : 10  |  Steps/episode: 500" -ForegroundColor Cyan
Write-Host "  Mode   : sumo-gui (real SUMO with GUI)" -ForegroundColor Cyan
Write-Host "  Note   : sumo-gui opens per node sequentially" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --gui

Write-Host ""
Write-Host "AdaptFlow complete. Results in results_adaptflow_china_osm/" -ForegroundColor Green
