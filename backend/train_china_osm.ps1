# China OSM map: real SUMO, NO GUI (uses env SUMO_HEADLESS=1 — no --sumo-headless needed on each line).
# Requires SUMO on PATH or SUMO_HOME.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$env:SUMO_HEADLESS = "1"

$SC = "--sumo-scenario", "china_osm"

Write-Host "SUMO_HEADLESS=1 (headless real SUMO, no sumo-gui windows)" -ForegroundColor Green

python train_adaptflow.py --rounds 3 --nodes 6 @SC
python train_fedflow.py --rounds 3 --nodes 6 @SC
python train_federated.py --mode sim --rounds 15 --clients 2 @SC
python train_fedkd.py --rounds 15 --num-clients 2 @SC
python train_fedcm.py --rounds 2 --num-clients 2 @SC

Write-Host "Done." -ForegroundColor Green
