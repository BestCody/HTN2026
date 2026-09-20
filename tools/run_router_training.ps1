$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot "..\.venv-training\Scripts\python.exe"
$Model = Join-Path $PSScriptRoot "..\checkpoints\specialist-router"
$Output = Join-Path $PSScriptRoot "..\outputs\router_training"

New-Item -ItemType Directory -Force -Path $Output | Out-Null

& $Python (Join-Path $PSScriptRoot "evaluate_physical_router.py") `
  --device cuda `
  --aggregation centroid `
  --samples (Join-Path $PSScriptRoot "..\examples\physical_ai_routing_validation.jsonl") `
  --output (Join-Path $Output "baseline.json")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python (Join-Path $PSScriptRoot "train_specialist_router.py") `
  --train (Join-Path $Output "triplets.jsonl") `
  --output $Model `
  --epochs 2 `
  --learning-rate 0.00001
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$splits = @{
  validation = "physical_ai_routing_validation.jsonl"
  holdout = "physical_ai_routing_holdout.jsonl"
  test = "physical_ai_routing_test.jsonl"
}
foreach ($entry in $splits.GetEnumerator()) {
  & $Python (Join-Path $PSScriptRoot "evaluate_physical_router.py") `
    --device cuda `
    --model $Model `
    --aggregation centroid `
    --samples (Join-Path $PSScriptRoot "..\examples\$($entry.Value)") `
    --output (Join-Path $Output "$($entry.Key).json")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $Python (Join-Path $PSScriptRoot "summarize_router_training.py") `
  --baseline (Join-Path $Output "baseline.json") `
  --validation (Join-Path $Output "validation.json") `
  --holdout (Join-Path $Output "holdout.json") `
  --test (Join-Path $Output "test.json") `
  --checkpoint $Model `
  --output (Join-Path $Output "summary.json")
exit $LASTEXITCODE
