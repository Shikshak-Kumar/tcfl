# Optional: opens sumo-gui (slow, many windows). Default training = train_china_osm.ps1 (no GUI).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
Remove-Item Env:SUMO_HEADLESS -ErrorAction SilentlyContinue

$SC = "--sumo-scenario", "china_osm"
$GUI = "--gui"

Write-Host "WARNING: This script uses --gui (sumo-gui windows). For normal training use .\train_china_osm.ps1" -ForegroundColor Yellow

python train_adaptflow.py --rounds 3 --nodes 6 @SC @GUI
python train_fedflow.py --rounds 3 --nodes 6 @SC @GUI
python train_federated.py --mode sim --rounds 15 --clients 2 @SC @GUI
python train_fedkd.py --rounds 15 --num-clients 2 @SC @GUI
python train_fedcm.py --rounds 2 --num-clients 2 @SC @GUI
