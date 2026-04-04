# Jhajjar Rural OSM — AdaptFlow training in headless SUMO (no GUI window).
# 5 rounds x 200 steps x 6 nodes. Faster than GUI — no rendering overhead.
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
# GUI version:       .\train_rural_osm.ps1
# Requirements:      sumo (not sumo-gui) on PATH or SUMO_HOME set.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$env:SUMO_HEADLESS = "1"

$SC    = "--sumo-scenario", "rural_osm"
$R     = "--rounds", "5"
$STEPS = "--steps", "200"

Write-Host ""
Write-Host "===== Jhajjar Rural OSM — AdaptFlow Training (Headless) =====" -ForegroundColor Yellow
Write-Host "  Nodes  : 6 (market / school / farm / residential / industrial / highway)" -ForegroundColor Yellow
Write-Host "  Rounds : 5  |  Steps/episode: 200  (~490 vehicles/episode)" -ForegroundColor Yellow
Write-Host "  Mode   : sumo headless (real TraCI, no GUI window)" -ForegroundColor Yellow
Write-Host "  Results: results_adaptflow_rural_osm/" -ForegroundColor Yellow
Write-Host "=============================================================" -ForegroundColor Yellow
Write-Host ""

python train_adaptflow.py --nodes 6 --clusters 2 @R @STEPS @SC --real-sumo

Write-Host ""
Write-Host "Training complete. Results saved to results_adaptflow_rural_osm/" -ForegroundColor Yellow
