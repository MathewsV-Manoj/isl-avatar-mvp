$ErrorActionPreference = "Stop"

# This longer-running audit refreshes evidence files for a release review. It
# does not start a server, deploy the application, or change public services.
$python = Join-Path $PSScriptRoot "isl_env\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project Python environment not found: $python"
}

Push-Location $PSScriptRoot
try {
    foreach ($script in @(
        "analyze_product_coverage.py",
        "audit_runtime_coverage.py",
        "audit_runtime_sentence_coverage.py",
        "build_release_candidate_report.py"
    )) {
        Write-Host "Refreshing $script"
        & $python (Join-Path $PSScriptRoot $script)
        if ($LASTEXITCODE -ne 0) {
            throw "$script failed with exit code $LASTEXITCODE"
        }
    }
} finally {
    Pop-Location
}
