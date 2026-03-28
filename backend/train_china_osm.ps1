# China OSM: real SUMO with sumo-gui by default (no mock).
# Headless: use .\train_china_osm_headless.ps1 or set SUMO_HEADLESS=1 before running.
# Requires sumo-gui / sumo on PATH or SUMO_HOME.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
Remove-Item Env:SUMO_HEADLESS -ErrorAction SilentlyContinue

$SC = "--sumo-scenario", "china_osm"

Write-Host "China OSM: real SUMO + sumo-gui (SUMO_HEADLESS cleared). For headless: .\train_china_osm_headless.ps1" -ForegroundColor Green

python train_adaptflow.py --rounds 3 --nodes 6 @SC
python train_fedflow.py --rounds 3 --nodes 6 @SC
python train_federated.py --mode sim --rounds 15 --clients 2 @SC
python train_fedkd.py --rounds 15 --num-clients 2 @SC
python train_fedcm.py --rounds 2 --num-clients 2 @SC

Write-Host "Done." -ForegroundColor Green
