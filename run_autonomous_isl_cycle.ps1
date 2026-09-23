$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$status = [ordered]@{ timestamp = (Get-Date).ToString('o'); stages = [ordered]@{}; blockers = @() }

function Stage($name, $state, $details) {
  $status.stages[$name] = [ordered]@{ state = $state; details = $details }
}

try {
  $health = $null
  try { $health = Invoke-RestMethod 'http://127.0.0.1:8090/health' -TimeoutSec 3 } catch {}
  if (-not $health.ok) {
    Start-Process -FilePath "$root\isl_env\Scripts\python.exe" -ArgumentList 'sentence_inference_server.py' -WorkingDirectory $root -RedirectStandardOutput "$logDir\sentence_inference_server.log" -RedirectStandardError "$logDir\sentence_inference_server.err"
    for ($attempt = 0; $attempt -lt 10 -and -not $health.ok; $attempt++) {
      Start-Sleep -Seconds 1
      try { $health = Invoke-RestMethod 'http://127.0.0.1:8090/health' -TimeoutSec 3 } catch {}
    }
    if (-not $health.ok) { throw 'Inference service did not become ready within 10 seconds.' }
  }
  $avatar = $null
  try { $avatar = (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080/isl-avatar-prototype.html' -TimeoutSec 3).StatusCode } catch {}
  if ($avatar -ne 200) {
    Start-Process -FilePath "$root\isl_env\Scripts\python.exe" -ArgumentList '-m','http.server','8080' -WorkingDirectory $root -RedirectStandardOutput "$logDir\avatar_http.log" -RedirectStandardError "$logDir\avatar_http.err"
    for ($attempt = 0; $attempt -lt 10 -and $avatar -ne 200; $attempt++) {
      Start-Sleep -Seconds 1
      try { $avatar = (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080/isl-avatar-prototype.html' -TimeoutSec 3).StatusCode } catch {}
    }
    if ($avatar -ne 200) { throw 'Avatar HTTP server did not become ready within 10 seconds.' }
  }
  Stage 'services' 'passed' "model=$($health.model); avatar_http=$avatar"
} catch { $status.blockers += "services: $($_.Exception.Message)"; Stage 'services' 'blocked' $_.Exception.Message }

try {
  node -e "const fs=require('fs'); const h=fs.readFileSync('isl-avatar-prototype.html','utf8'); for(const part of h.split('<script').slice(1)){const s=part.indexOf('>'),e=part.indexOf('</script>'); if(s>=0&&e>s)new Function(part.slice(s+1,e));}"
  $html = Get-Content "$root\isl-avatar-prototype.html" -Raw
  $micFeatures = @('getUserMedia','SpeechRecognition','speechPermissionPromise','beforeunload') | Where-Object { $html.Contains($_) }
  if ($micFeatures.Count -lt 4) { throw 'Speech/microphone frontend feature set is incomplete.' }
  Stage 'speech_frontend' 'passed' 'Speech recognition, persistent mic stream, permission handling, and cleanup are present.'
} catch { $status.blockers += "speech_frontend: $($_.Exception.Message)"; Stage 'speech_frontend' 'blocked' $_.Exception.Message }

try {
  & "$root\isl_env\Scripts\python.exe" analyze_sentence_coverage.py *> "$logDir\coverage_cycle.log"
  $coverage = Get-Content "$root\sentence_coverage_report.json" -Raw | ConvertFrom-Json
  Stage 'sentence_coverage' 'passed' "exact=$($coverage.exact_coverage); with_inflections=$($coverage.coverage_with_basic_inflections); tokens=$($coverage.tokens)"
} catch { $status.blockers += "coverage: $($_.Exception.Message)"; Stage 'sentence_coverage' 'blocked' $_.Exception.Message }

try {
  & "$root\isl_env\Scripts\python.exe" verify_dataset_inventory.py *> "$logDir\dataset_inventory_cycle.log"
  $inventory = Get-Content "$root\dataset_inventory.json" -Raw | ConvertFrom-Json
  if ($inventory.blockers.Count -gt 0) {
    Stage 'dataset_inventory' 'review' ($inventory.blockers -join '; ')
  } else {
    Stage 'dataset_inventory' 'passed' "local_datasets=$($inventory.local.Count); pose_arrays=$($inventory.pose_arrays.Count)"
  }
} catch { $status.blockers += "dataset_inventory: $($_.Exception.Message)"; Stage 'dataset_inventory' 'blocked' $_.Exception.Message }

$poseFiles = @(Get-ChildItem "$root\datasets" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.npy','.npz','.pt','.pkl' -and $_.Name -match 'pose|landmark|keypoint|skeleton' -and $_.Length -gt 1024 })
if ($poseFiles.Count -gt 0) {
  Stage 'continuous_pose_data' 'available' "Found $($poseFiles.Count) pose-like files; provenance must be checked before training."
} else {
  Stage 'continuous_pose_data' 'blocked' 'No continuous native ISL pose arrays are available locally; isolated pose data remains usable.'
}

try {
  & "$root\isl_env\Scripts\python.exe" "$root\evaluate_source_split.py" *> "$logDir\source_split_cycle.log"
  $split = Get-Content "$root\source_split_diagnostic.json" -Raw | ConvertFrom-Json
  Stage 'source_split_diagnostic' 'passed' "bpcc=$($split.bpcc.token_accuracy); blimp=$($split.blimp.token_accuracy); note=$($split.note)"
} catch { $status.blockers += "source_split: $($_.Exception.Message)"; Stage 'source_split_diagnostic' 'blocked' $_.Exception.Message }

$disjointReport = "$root\models\source_disjoint_bpcc_report.json"
if (Test-Path $disjointReport) {
  $gate = Get-Content $disjointReport -Raw | ConvertFrom-Json
  Stage 'generalization_gate' 'review' "held_out_token_accuracy=$($gate.held_out_token_accuracy); production_promotion=blocked_until_native_pose_data"
  $poseFiles = @(Get-ChildItem "$root\datasets" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.Extension -in '.npy','.npz','.pt','.pkl' -and $_.Name -match 'pose|landmark|keypoint|skeleton' -and $_.Length -gt 1024 })
  $strictReasons = @()
  if ([double]$gate.held_out_token_accuracy -lt 0.90) { $strictReasons += "source-disjoint accuracy $($gate.held_out_token_accuracy) is below 0.90" }
  if ($poseFiles.Count -eq 0) { $strictReasons += 'continuous native ISL pose arrays are missing' }
  if ($strictReasons.Count -gt 0) {
    $status.blockers += "strict_product_gate: $($strictReasons -join '; ')"
    Stage 'strict_product_gate' 'blocked' ($strictReasons -join '; ')
  } else {
    Stage 'strict_product_gate' 'passed' 'All promotion gates passed.'
  }
} else {
  Stage 'generalization_gate' 'pending' 'Source-disjoint experiment report is not available yet.'
  Stage 'strict_product_gate' 'blocked' 'No source-disjoint report is available.'
}

try {
  & powershell -ExecutionPolicy Bypass -File "$root\run_full_isl_pipeline.ps1" *> "$logDir\pipeline_cycle.log"
  if ($LASTEXITCODE -eq 0) { Stage 'validation' 'passed' 'Full pipeline completed.' }
  else { $status.blockers += 'validation: run_full_isl_pipeline.ps1 failed'; Stage 'validation' 'blocked' 'See logs/pipeline_cycle.log.' }
} catch { $status.blockers += "validation: $($_.Exception.Message)"; Stage 'validation' 'blocked' $_.Exception.Message }

$training = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'train_bpcc_gloss_tagger' -and $_.CommandLine -notmatch 'run_autonomous_isl_cycle' }
$reportPath = "$root\models\bpcc_gloss_tagger_report.json"
$reportAge = if (Test-Path $reportPath) { (Get-Date) - (Get-Item $reportPath).LastWriteTime } else { [TimeSpan]::FromDays(999) }
if (-not $training -and $reportAge.TotalHours -ge 24) {
  Start-Process -FilePath "$root\isl_env\Scripts\python.exe" -ArgumentList 'train_bpcc_gloss_tagger.py','--limit','699963','--epochs','2','--batch-size','256','--mask-prob','0.12','--resume','models/bpcc_gloss_tagger.pt','--index','datasets/bpcc_synthetic/index.jsonl','datasets/bpcc_synthetic/blimp_index.jsonl' -WorkingDirectory $root -RedirectStandardOutput "$logDir\autonomous_tune.log" -RedirectStandardError "$logDir\autonomous_tune.err"
  Stage 'controlled_tuning' 'started' 'Checkpoint older than 24 hours; launched one two-epoch robustness pass.'
} elseif ($training) {
  Stage 'controlled_tuning' 'running' 'A training process is already active; no duplicate run started.'
} else {
  Stage 'controlled_tuning' 'skipped' 'Checkpoint is recent; no duplicate training launched.'
}

$status | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 "$root\autonomous_status.json"
Get-Content "$root\autonomous_status.json"
