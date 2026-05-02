# Quick smoke-test using mock environments (no SUMO required).
# Same proportional parameters as real run — just fewer rounds/steps.
# Usage: .\train_dwarka_mor_mock.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\train_dwarka_mor.ps1" `
    -Mock `
    -Rounds 3 `
    -Steps 200 `
    -FedRounds 3 `
    -FedEps 20 `
    -MA2CSteps 3600
