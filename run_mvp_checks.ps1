$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot 'isl_env\Scripts\python.exe'

& $python -m py_compile production_server.py test_production_server.py mvp_acceptance.py regional_translation.py
& $python test_production_server.py
& $python test_multilingual_pipeline.py
& $python test_dictionary_resolution.py
& $python mvp_acceptance.py

$report = Get-Content reports\mvp_acceptance.json -Raw | ConvertFrom-Json
if ($report.english.passed -ne $report.english.total) { throw 'English MVP acceptance failed.' }
if ($report.reviewed_regional.passed -ne $report.reviewed_regional.total) { throw 'Reviewed regional MVP acceptance failed.' }
Write-Host "MVP acceptance passed: English $($report.english.passed)/$($report.english.total); regional $($report.reviewed_regional.passed)/$($report.reviewed_regional.total)."
