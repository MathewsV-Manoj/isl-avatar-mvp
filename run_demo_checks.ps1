$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot "isl_env\Scripts\python.exe"
if(-not (Test-Path $python)){throw "Project virtual environment not found: $python"}

Write-Host "[1/7] Validating dictionaries and CISLR shards..."
& $python validate_isl_assets.py

Write-Host "[2/7] Verifying hand orientation correction..."
& $python verify_cislr_orientation_sample.py

Write-Host "[3/7] Filtering supplemental clips with prolonged two-hand loss..."
& $python filter_supplemental_isl.py

Write-Host "[4/7] Checking supplemental multi-clip corpus..."
& $python validate_supplemental_isl.py --dictionary signal_dictionary_supplemental_filtered.json

Write-Host "[5/8] Checking BridgeConn browser shards..."
& $python validate_bridgeconn_browser_shards.py

Write-Host "[6/8] Checking official browser shards..."
& $python validate_official_browser_shards.py

Write-Host "[7/8] Checking meeting vocabulary coverage..."
& $python sign_coverage.py hello thank_you hear repeat slower meeting class name fine understand

Write-Host "[8/8] Verifying unsupported-word fingerspelling fallback..."
& $python verify_fallback.py

Write-Host "Demo checks passed. Open http://127.0.0.1:8080/isl-avatar-prototype.html"
