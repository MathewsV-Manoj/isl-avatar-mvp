$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSCommandPath
$python = Join-Path $root "isl_env\Scripts\python.exe"
$checks = @(
    "test_input_normalization.py",
    "test_production_server.py",
    "test_production_api.py",
    "test_product_mode_guard.py",
    "test_deployment_configuration.py",
    "test_static_runtime_export.py",
    "release_readiness_check.py",
    "test_sentence_clip_integration.py",
    "audit_recorded_sentence_geometry.py",
    "mvp_acceptance.py",
    "test_multilingual_pipeline.py",
    "meeting_vocabulary_stress.py",
    "release_interviewer_stress.py"
)

Push-Location $root
try {
    foreach ($check in $checks) {
        Write-Host "Running $check"
        & $python $check
        if ($LASTEXITCODE -ne 0) {
            throw "Preflight failed: $check"
        }
    }
    Write-Host "ISL MVP preflight passed."
}
finally {
    Pop-Location
}
