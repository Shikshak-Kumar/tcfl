# Same as train_china_osm.ps1 (headless SUMO only).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $here "train_china_osm.ps1")
