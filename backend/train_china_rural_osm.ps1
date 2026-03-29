# China Rural OSM — AdaptFlow training with sumo-gui.
# 3 rounds x 500 steps x 6 nodes. Each node opens sumo-gui sequentially.
#
# Node layout (rural zones):
#   node_0  Village market / community centre  (begin=0,    jam=15, Tier 1)
#   node_1  Rural school zone                  (begin=600,  jam=20, Tier 2)
#   node_2  Farming district crossroads        (begin=1200, jam=30, Tier 3)
#   node_3  Residential outskirts              (begin=1800, jam=35, Tier 3)
#   node_4  Rural industrial / factory zone    (begin=2400, jam=45, Tier 3)
#   node_5  Highway junction / town entrance   (begin=3000, jam=22, Tier 3)
#
# Results stored in: results_adaptflow_china_rural_osm/
# Headless version:  .\train_china_rural_osm_headless.ps1
# Requirements:      sumo-gui on PATH or SUMO_HOME set.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

Remove-Item Env:SUMO_HEADLESS -ErrorAction SilentlyContinue

$SC    = "--sumo-scenario", "china_rural_osm"
$R     = "--rounds", "3"
$STEPS = "--steps", "500"

Write-Host ""
Write-Host "===== China Rural OSM — AdaptFlow Training =====" -ForegroundColor Green
Write-Host "  Nodes  : 6 (village / school / farm / residential / industrial / highway)" -ForegroundColor Green
Write-Host "  Rounds : 3  |  Steps/episode: 500" -ForegroundColor Green
Write-Host "  Mode   : sumo-gui (real SUMO with GUI)" -ForegroundColor Green
Write-Host "  Results: results_adaptflow_china_rural_osm/" -ForegroundColor Green
Write-Host "  Note   : sumo-gui opens per node sequentially (6 windows per round)" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Green
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --gui

Write-Host ""
Write-Host "Training complete. Results saved to results_adaptflow_china_rural_osm/" -ForegroundColor Green
