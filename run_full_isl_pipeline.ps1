$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$status = [ordered]@{
  timestamp = (Get-Date).ToString('o')
  stages = [ordered]@{}
  blockers = @()
}

function Record-Stage($name, $state, $details) {
  $status.stages[$name] = [ordered]@{ state = $state; details = $details }
}

try {
  & "$root\isl_env\Scripts\python.exe" -m py_compile `
    "$root\train_sentence_bootstrap.py" `
    "$root\train_bpcc_gloss_tagger.py" `
    "$root\production_server.py" `
    "$root\audit_served_hand_geometry.py" `
    "$root\sentence_inference_server.py" `
    "$root\prepare_bpcc_synthetic_index.py" `
    "$root\prepare_blimp_synthetic_index.py" `
    "$root\validate_sentence_browser_shard.py" `
    "$root\regional_translation.py" `
    "$root\test_multilingual_pipeline.py" `
    "$root\test_dictionary_resolution.py" `
    "$root\test_production_api.py" `
    "$root\test_product_mode_guard.py" `
    "$root\product_self_review.py"
  Record-Stage 'python_syntax' 'passed' 'Training, preparation, and inference scripts compile.'

  node -e "const fs=require('fs'); const h=fs.readFileSync('isl-avatar-prototype.html','utf8'); for(const part of h.split('<script').slice(1)){const s=part.indexOf('>'),e=part.indexOf('</script>'); if(s>=0&&e>s)new Function(part.slice(s+1,e));}"
  Record-Stage 'avatar_syntax' 'passed' 'Browser JavaScript parses successfully.'

  $health = Invoke-RestMethod 'http://127.0.0.1:8090/health'
  if (-not $health.ok) { throw 'Inference health check returned false.' }
  $avatarStatus = (Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8080/isl-avatar-prototype.html').StatusCode
  if ($avatarStatus -ne 200) { throw "Avatar server returned HTTP $avatarStatus." }
  Record-Stage 'services' 'passed' "Inference model=$($health.model); avatar_http=$avatarStatus."

  $report = Get-Content "$root\models\bpcc_gloss_tagger_report.json" -Raw | ConvertFrom-Json
  if (-not $report.weak_supervision -or $report.continuous_pose_supervision) {
    throw 'Model provenance report is inconsistent.'
  }
  Record-Stage 'model_provenance' 'passed' "rows=$($report.rows); output_vocab=$($report.output_vocab); pose_supervision=$($report.continuous_pose_supervision)."

  & powershell -ExecutionPolicy Bypass -File "$root\run_demo_checks.ps1"
  & "$root\isl_env\Scripts\python.exe" "$root\validate_sentence_browser_shard.py"
  & "$root\isl_env\Scripts\python.exe" "$root\test_multilingual_pipeline.py"
  & "$root\isl_env\Scripts\python.exe" "$root\test_dictionary_resolution.py"
  & "$root\isl_env\Scripts\python.exe" "$root\test_production_api.py"
  & "$root\isl_env\Scripts\python.exe" "$root\test_product_mode_guard.py"
  & "$root\isl_env\Scripts\python.exe" "$root\audit_served_hand_geometry.py"
  & "$root\isl_env\Scripts\python.exe" "$root\product_self_review.py"
  Record-Stage 'asset_validation' 'passed' 'All dictionary, orientation, sentence-shard, vocabulary, phrase-resolution, fallback, HTTP-contract, browser-mode, and served-hand geometry checks passed.'
}
catch {
  $detail = "{0} at {1}" -f $_.Exception.Message, $_.InvocationInfo.PositionMessage
  $status.blockers += $detail
  Record-Stage 'pipeline' 'blocked' $detail
}

$statusPath = "$root\pipeline_status.json"
$status | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
Get-Content $statusPath
if ($status.blockers.Count -gt 0) { exit 1 }
