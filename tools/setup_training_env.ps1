[CmdletBinding()]
param(
    [string]$EnvironmentPath = ".venv-training",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu128",
    [switch]$SkipFfmpegInstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $repositoryRoot $EnvironmentPath
$trainingPython = Join-Path $environmentRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $trainingPython)) {
    python -m venv $environmentRoot
}

& $trainingPython -m pip install --upgrade pip wheel "setuptools>=71,<82"
& $trainingPython -m pip install `
    --index-url $TorchIndexUrl `
    "torch==2.11.0" `
    "torchvision==0.26.0"
& $trainingPython -m pip install -r (Join-Path $repositoryRoot "requirements-training.txt")
$editableProject = "$repositoryRoot[prompt,adapters,dev]"
& $trainingPython -m pip install -e $editableProject

function Find-SharedFfmpegBin {
    $packageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    if (-not (Test-Path -LiteralPath $packageRoot)) {
        return $null
    }
    $package = Get-ChildItem -LiteralPath $packageRoot -Directory |
        Where-Object { $_.Name -like "BtbN.FFmpeg.GPL.Shared.8.0*" } |
        Select-Object -First 1
    if ($null -eq $package) {
        return $null
    }
    $executable = Get-ChildItem -LiteralPath $package.FullName -Recurse -Filter "ffmpeg.exe" |
        Select-Object -First 1
    if ($null -eq $executable) {
        return $null
    }
    return $executable.DirectoryName
}

$ffmpegBin = Find-SharedFfmpegBin
if ($null -eq $ffmpegBin -and -not $SkipFfmpegInstall) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget is required to install the FFmpeg 8 shared libraries"
    }
    winget install `
        --id BtbN.FFmpeg.GPL.Shared.8.0 `
        --exact `
        --source winget `
        --silent `
        --accept-source-agreements `
        --accept-package-agreements `
        --disable-interactivity
    $ffmpegBin = Find-SharedFfmpegBin
}
if ($null -eq $ffmpegBin) {
    throw "FFmpeg 8 shared libraries are unavailable; TorchCodec cannot load"
}

# Make the freshly installed DLLs visible without requiring a new terminal.
$env:PATH = "$ffmpegBin;$env:PATH"

& $trainingPython -m pip check
& $trainingPython (Join-Path $repositoryRoot "tools\verify_training_env.py")
