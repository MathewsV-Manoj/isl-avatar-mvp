param(
  [int]$Epochs = 2,
  [int]$BatchSize = 128
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'isl_env\Scripts\python.exe'
$data = Join-Path $root 'datasets\isign_sentence_pose_v2'
$active = Join-Path $root 'models\isign_text2pose_candidate.pt'
$candidate = Join-Path $root 'models\isign_text2pose_stage_candidate.pt'
$candidateReport = Join-Path $root 'models\isign_text2pose_stage_candidate_report.json'
$activeReport = Join-Path $root 'models\isign_text2pose_candidate_report.json'

& $python (Join-Path $root 'evaluate_isign_pose_shards.py')
if ($LASTEXITCODE -ne 0) { throw 'Pose shard validation failed.' }

& $python (Join-Path $root 'train_isign_text2pose.py') --data $data --epochs $Epochs --batch-size $BatchSize --resume $active --output $candidate --report $candidateReport
if ($LASTEXITCODE -ne 0) { throw 'Native pose refinement failed.' }

$before = (Get-Content $activeReport -Raw | ConvertFrom-Json).best_valid_smooth_l1
$after = (Get-Content $candidateReport -Raw | ConvertFrom-Json).best_valid_smooth_l1
if ([double]$after -lt [double]$before) {
  Copy-Item $active "$active.pre_stage" -Force
  Copy-Item $candidate $active -Force
  Copy-Item $candidateReport $activeReport -Force
  Write-Output ("PROMOTED native pose candidate: {0} -> {1}" -f $before, $after)
} else {
  Write-Output ("REJECTED native pose candidate: {0} -> {1}" -f $before, $after)
}

$listeners = @(Get-NetTCPConnection -LocalPort 8090 -State Listen -ErrorAction SilentlyContinue)
foreach ($listener in $listeners) { Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Process -FilePath $python -ArgumentList (Join-Path $root 'sentence_inference_server.py') -WorkingDirectory $root -WindowStyle Hidden
Start-Sleep -Seconds 1
$health = $null
for ($attempt = 1; $attempt -le 20; $attempt++) {
  try { $health = Invoke-RestMethod 'http://127.0.0.1:8090/health' -TimeoutSec 2; break } catch { Start-Sleep -Seconds 1 }
}
if ($null -eq $health) { throw 'Native pose service did not become healthy after restart.' }
$pose = Invoke-RestMethod -Uri 'http://127.0.0.1:8090/predict_pose' -Method Post -ContentType 'application/json' -Body '{"text":"the doctor will help me"}'
if (-not $health.ok -or ($pose.shape -join 'x') -ne '30x225') { throw 'Native pose service check failed.' }
Write-Output 'Native pose stage completed and service check passed.'
