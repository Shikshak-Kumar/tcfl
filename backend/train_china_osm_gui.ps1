# Same as train_china_osm.ps1 — China maps default to sumo-gui.

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $here "train_china_osm.ps1")
