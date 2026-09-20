$ErrorActionPreference = "Stop"

$Workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Source = (Resolve-Path (Join-Path $Workspace "checkpoints\specialist-router")).Path
$Target = Join-Path $Workspace "deploy\baseten_router\model"
$DeployRoot = (Resolve-Path (Join-Path $Workspace "deploy\baseten_router")).Path
$ResolvedParent = (Resolve-Path (Split-Path $Target -Parent)).Path
$SummaryPath = Join-Path $Workspace "outputs\router_training\summary.json"
$ManifestPath = Join-Path $Workspace "outputs\router_training\dataset_manifest.json"

if ($ResolvedParent -ne $DeployRoot) {
  throw "Router model target escaped the expected deployment directory"
}
if (-not (Test-Path (Join-Path $Source "model.safetensors") -PathType Leaf)) {
  throw "Trained router checkpoint is missing model.safetensors"
}
if (-not (Test-Path (Join-Path $Source "moira_training.json") -PathType Leaf)) {
  throw "Trained router checkpoint is missing MoIRA training metadata"
}
if (-not (Test-Path $SummaryPath -PathType Leaf)) {
  throw "Router acceptance summary is missing"
}
if (-not (Test-Path $ManifestPath -PathType Leaf)) {
  throw "Router dataset manifest is missing"
}
$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
if ($Summary.accepted -ne $true) {
  throw "Router checkpoint did not pass the acceptance gate"
}
$CheckpointHash = (Get-FileHash (Join-Path $Source "model.safetensors") -Algorithm SHA256).Hash.ToLower()
if ($CheckpointHash -ne $Summary.checkpoint_model_sha256) {
  throw "Router checkpoint hash does not match the accepted evaluation"
}

New-Item -ItemType Directory -Force -Path $Target | Out-Null
Copy-Item -Path (Join-Path $Source "*") -Destination $Target -Recurse -Force
Copy-Item -LiteralPath $SummaryPath -Destination (Join-Path $Target "moira_evaluation.json") -Force
Copy-Item -LiteralPath $ManifestPath -Destination (Join-Path $Target "moira_dataset_manifest.json") -Force
Write-Output "Staged the accepted router checkpoint at $Target"
