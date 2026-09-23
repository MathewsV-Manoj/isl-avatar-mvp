$ErrorActionPreference = 'Stop'
$root = 'E:\ISL_Project_Datasets\isign'
$parts = @(
    'iSign-poses_v1.1_part_ab',
    'iSign-poses_v1.1_part_ac',
    'iSign-poses_v1.1_part_ad'
)
foreach ($part in $parts) {
    $output = Join-Path $root $part
    while (-not (Test-Path $output)) {
        & .\isl_env\Scripts\python.exe .\stream_download_hf_file.py `
            --repo Exploration-Lab/iSign `
            --file $part `
            --output $output
        if (-not (Test-Path $output)) { Start-Sleep -Seconds 5 }
    }
}

# Preserve a machine-readable completion marker for the next processing stage.
$files = Get-ChildItem (Join-Path $root 'iSign-poses_v1.1_part_*') |
    Select-Object Name, Length, LastWriteTime
$files | ConvertTo-Json | Set-Content (Join-Path $root 'pose_parts_complete.json')
Write-Output 'pose_parts_download_complete'
& .\isl_env\Scripts\python.exe .\inspect_isign_split_zip.py --root $root
& .\isl_env\Scripts\python.exe .\extract_isign_word_candidates.py --root $root --max-words 10000
