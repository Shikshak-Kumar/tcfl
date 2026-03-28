# Same as train_china_osm.ps1 — GUI is the default for China OSM.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $here "train_china_osm.ps1")
