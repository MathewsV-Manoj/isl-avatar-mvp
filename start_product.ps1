$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSCommandPath
$python = Join-Path $root "isl_env\Scripts\python.exe"

Push-Location $root
try {
    & $python production_server.py
}
finally {
    Pop-Location
}
