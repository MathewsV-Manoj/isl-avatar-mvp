$ErrorActionPreference = 'Stop'
$zip = 'E:\ISL_Project_Datasets\isl_csltr\isl-csltr-indian-sign-language-dataset.zip'
$out = 'E:\ISL_Project_Datasets\isl_csltr\unpacked'
$status = 'E:\ISL_Project_Datasets\isl_csltr\postprocess_status.json'
while (-not (Test-Path -LiteralPath $zip)) { Start-Sleep -Seconds 30 }
do {
    $active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -like '*download_isl_csltr.py*' }
    if ($active) { Start-Sleep -Seconds 60 }
} while ($active)
if (-not (Test-Path -LiteralPath $out)) { New-Item -ItemType Directory -Path $out | Out-Null }
if (-not (Test-Path -LiteralPath (Join-Path $out '.extracted'))) {
    Expand-Archive -LiteralPath $zip -DestinationPath $out -Force
    New-Item -ItemType File -Path (Join-Path $out '.extracted') | Out-Null
}
$videos = @(Get-ChildItem -LiteralPath $out -Recurse -File -Include *.mp4,*.avi,*.mov).Count
@{ state = 'complete'; videos = $videos; archive = $zip } | ConvertTo-Json | Set-Content -Encoding utf8 $status
