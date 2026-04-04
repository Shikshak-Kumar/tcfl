# Jhajjar Rural OSM — AdaptFlow training with sumo-gui.
# 5 rounds x 200 steps x 6 nodes. Each node opens sumo-gui sequentially.
#
# Node layout (Jhajjar, Haryana):
#   node_0  Town market / sabzi mandi       (begin=0,    jam=12, Tier 1)
#   node_1  School / college zone           (begin=600,  jam=18, Tier 2)
#   node_2  Agricultural / farming area     (begin=1200, jam=25, Tier 3)
#   node_3  Residential colony / outskirts  (begin=1800, jam=30, Tier 3)
#   node_4  HSIIDC industrial / factory     (begin=2400, jam=38, Tier 3)
#   node_5  NH-48 highway junction          (begin=3000, jam=22, Tier 3)
#
# Results stored in: results_adaptflow_rural_osm/
# Headless version:  .\train_rural_osm_headless.ps1
# Requirements:      sumo-gui on PATH or SUMO_HOME set.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Remove-Item Env:SUMO_HEADLESS -ErrorAction SilentlyContinue

$SC    = "--sumo-scenario", "rural_osm"
$R     = "--rounds", "5"
$STEPS = "--steps", "200"

Write-Host ""
Write-Host "===== Jhajjar Rural OSM — AdaptFlow Training =====" -ForegroundColor Magenta
Write-Host "  Nodes  : 6 (market / school / farm / residential / industrial / highway)" -ForegroundColor Magenta
Write-Host "  Rounds : 5  |  Steps/episode: 200  (~490 vehicles/episode)" -ForegroundColor Magenta
Write-Host "  Mode   : sumo-gui (real SUMO with GUI)" -ForegroundColor Magenta
Write-Host "  Results: results_adaptflow_rural_osm/" -ForegroundColor Magenta
Write-Host "  Note   : sumo-gui opens per node sequentially (6 windows per round)" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Magenta
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --gui

Write-Host ""
Write-Host "Training complete. Results saved to results_adaptflow_rural_osm/" -ForegroundColor Magenta
