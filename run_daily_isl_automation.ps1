$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$lockPath = Join-Path $root 'isl_automation.lock'
$statusPath = Join-Path $root 'isl_automation_status.json'

$now = Get-Date
$start = Get-Date -Hour 4 -Minute 10 -Second 0
$end = Get-Date -Hour 21 -Minute 20 -Second 0
$status = [ordered]@{ timestamp = $now.ToString('o'); window = $false; stages = [ordered]@{}; blockers = @() }
if ($now -ge $start -and $now -le $end) { $status.window = $true } else {
  $status.stages['window'] = [ordered]@{ state = 'skipped'; details = 'Outside 04:10-21:20 Asia/Kolkata build window.' }
  $status | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
  exit 0
}
if (Test-Path $lockPath) {
  $age = $now - (Get-Item $lockPath).LastWriteTime
  if ($age.TotalHours -lt 6) {
    $status.stages['lock'] = [ordered]@{ state = 'skipped'; details = "Another run owns the lock ($([int]$age.TotalMinutes) minutes old)." }
    $status | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
    exit 0
  }
  Remove-Item -LiteralPath $lockPath -Force
}
New-Item -ItemType File -Path $lockPath -Force | Out-Null
try {
  $py = Join-Path $root 'isl_env\Scripts\python.exe'
  function Stage($name, $state, $details) { $status.stages[$name] = [ordered]@{ state = $state; details = $details } }

  try {
    & $py verify_dataset_inventory.py *> (Join-Path $logDir 'automation_inventory.log')
    $inventory = Get-Content (Join-Path $root 'dataset_inventory.json') -Raw | ConvertFrom-Json
    Stage 'inventory' ($(if ($inventory.blockers.Count) { 'review' } else { 'passed' })) "labels_batch=$($inventory.islrtc_validated_batch.labels); sparse_sentence_rows=$($inventory.sentence_context.sparse_kept_rows); blockers=$($inventory.blockers.Count)"
  } catch { $status.blockers += "inventory: $($_.Exception.Message)"; Stage 'inventory' 'blocked' $_.Exception.Message }

  try {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root 'run_full_isl_pipeline.ps1') *> (Join-Path $logDir 'automation_pipeline.log')
    if ($LASTEXITCODE -ne 0) { throw 'Full validation returned a non-zero exit code.' }
    Stage 'validation' 'passed' 'Dictionary, orientation, service, vocabulary, and fallback gates passed.'
  } catch { $status.blockers += "validation: $($_.Exception.Message)"; Stage 'validation' 'blocked' $_.Exception.Message }

  try {
    $health = $null
    try { $health = Invoke-RestMethod 'http://127.0.0.1:8090/health' -TimeoutSec 3 } catch {}
    if (-not $health.ok) {
      Start-Process -FilePath $py -ArgumentList '-u','sentence_inference_server.py' -WorkingDirectory $root -RedirectStandardOutput (Join-Path $logDir 'sentence_inference_server.log') -RedirectStandardError (Join-Path $logDir 'sentence_inference_server.err') -WindowStyle Hidden
      Start-Sleep -Seconds 8
      $health = Invoke-RestMethod 'http://127.0.0.1:8090/health' -TimeoutSec 5
    }
    $avatar = (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080/isl-avatar-prototype.html' -TimeoutSec 5).StatusCode
    Stage 'services' 'passed' "inference=$($health.model); avatar_http=$avatar"
  } catch { $status.blockers += "services: $($_.Exception.Message)"; Stage 'services' 'blocked' $_.Exception.Message }

  try {
    & $py evaluate_source_split.py --checkpoint models/bpcc_gloss_tagger.pt *> (Join-Path $logDir 'automation_source_split.log')
    $diagPath = Join-Path $root 'source_split_diagnostic_isltranslate.json'
    if (Test-Path $diagPath) {
      $diag = Get-Content $diagPath -Raw | ConvertFrom-Json
      Stage 'model_gate' 'passed' "bpcc=$($diag.bpcc.token_accuracy); blimp=$($diag.blimp.token_accuracy); diagnostic_only=true"
    } else { Stage 'model_gate' 'review' 'Diagnostic completed without the expected report file.' }
  } catch { $status.blockers += "model_gate: $($_.Exception.Message)"; Stage 'model_gate' 'blocked' $_.Exception.Message }
} finally {
  $status | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
  Remove-Item -LiteralPath $lockPath -Force -ErrorAction SilentlyContinue
}
Get-Content $statusPath
if ($status.blockers.Count -gt 0) { exit 1 }
