$ErrorActionPreference = "Stop"

$source = Join-Path (Split-Path $PSScriptRoot -Parent) "setup.ps1"
$testDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName())
$oldOS = $env:OS
$oldComposeFile = $env:COMPOSE_FILE
$oldNonInteractive = $env:LVS_SETUP_NONINTERACTIVE
$env:LVS_SETUP_NONINTERACTIVE = "1"
$script:DockerCalls = @()
$script:FailPs = $false
$script:GpuGB = 0

function Get-CimInstance {
    [CmdletBinding()]
    param([string]$ClassName)
    if ($script:GpuGB) { return [pscustomobject]@{ Name = "NVIDIA test GPU" } }
    return @()
}

function nvidia-smi {
    return [string]($script:GpuGB * 1024)
}

function ffmpeg {
    throw "Host FFmpeg must not be used to configure container decoding."
}

function docker {
    $script:DockerCalls += ,@($args)
    $global:LASTEXITCODE = 0
    if ($args[0] -eq "info") { return "nvidia" }
    if ($args[0] -eq "compose" -and $args[1] -eq "config" -and $args[2] -eq "--services") {
        return @("qdrant", "local-vss-api", "local-vss-frontend", "ollama")
    }
    if ($args[0] -eq "compose" -and $args[1] -eq "--profile") {
        if ($script:FailPs) {
            $global:LASTEXITCODE = 1
        } elseif ($args[-1] -eq "vllm-main") {
            return "project-owned-container"
        }
    }
    if ($args[0] -eq "compose" -and $args[1] -eq "pull") {
        $global:LASTEXITCODE = 1
    }
}

function Invoke-RestMethod {
    param([string]$Uri)
    if ($Uri -like "*/status/models") {
        return [pscustomobject]@{ services = @(
            [pscustomobject]@{ ok = $true },
            [pscustomobject]@{ ok = $true },
            [pscustomobject]@{ ok = $true }
        ) }
    }
    throw "Unexpected network request in setup test: $Uri"
}

try {
    New-Item -ItemType Directory -Path $testDir | Out-Null
    $testSetup = Join-Path $testDir "setup.ps1"
    $testEnv = Join-Path $testDir ".env"
    Copy-Item $source $testSetup
    @(
        "HARDWARE_TYPE=nvidia",
        "MODEL_FAMILY=gemma",
        "VIDEO_DIR_1=C:\old-videos",
        "APP_DATA_DIR=C:\old-data",
        "QDRANT_STORAGE_PATH=C:\old-qdrant",
        "MODELS_DIR=./old-models",
        "WHISPER_MODEL=small",
        "CUSTOM_SETTING=preserved"
    ) | Set-Content -Path $testEnv -Encoding UTF8
    $env:OS = "Windows_NT"

    . $testSetup -Force

    $lines = @(Get-Content $testEnv)
    foreach ($expected in @(
        "HARDWARE_TYPE=cpu",
        'COMPOSE_PROFILES=${HARDWARE_TYPE}',
        "MODEL_FAMILY=gemma",
        "VIDEO_DIR_1=C:\old-videos",
        "APP_DATA_DIR=C:\old-data",
        "QDRANT_STORAGE_PATH=C:\old-qdrant",
        "MODELS_DIR=./old-models",
        "WHISPER_MODEL=small",
        "CUSTOM_SETTING=preserved",
        "FFMPEG_HWACCEL=auto"
    )) {
        if ($lines -notcontains $expected) { throw "Missing or incorrect setting: $expected" }
    }
    if (-not (Test-Path -PathType Container (Join-Path $testDir "old-models"))) {
        throw "Setup did not create the model storage folder."
    }
    $removals = @($script:DockerCalls | Where-Object { $_[0] -eq "rm" })
    if ($removals.Count -ne 1 -or $removals[0][-1] -ne "project-owned-container") {
        throw "Setup did not remove only the inactive project-owned container."
    }
    $starts = @($script:DockerCalls | Where-Object { $_[0] -eq "compose" -and $_[1] -eq "up" })
    if ($starts.Count -ne 1 -or $starts[0] -notcontains "--build") {
        throw "Setup did not fall back to building after an image pull failure."
    }

    $lines = @($lines | ForEach-Object {
        if ($_ -eq "FFMPEG_HWACCEL=auto") { "FFMPEG_HWACCEL=off" } else { $_ }
    })
    Set-Content -Path $testEnv -Value $lines -Encoding UTF8

    $script:DockerCalls = @()
    $script:FailPs = $true
    $blocked = $false
    try {
        . $testSetup -Force
    } catch {
        if ($_.Exception.Message -notlike "*refusing to start another GPU backend*") { throw }
        $blocked = $true
    }
    if (-not $blocked -or @($script:DockerCalls | Where-Object { $_[0] -eq "compose" -and $_[1] -eq "up" }).Count) {
        throw "Setup must refuse to start if inactive containers cannot be checked."
    }
    if (@(Get-Content $testEnv) -notcontains "FFMPEG_HWACCEL=off") {
        throw "Setup must preserve an explicitly configured FFmpeg decoder."
    }

    $script:FailPs = $false
    foreach ($case in @(@(24, "low"), @(48, "mid"))) {
        $script:GpuGB = $case[0]
        . $testSetup -Force -Vllm
        $lines = @(Get-Content $testEnv)
        if ($lines -notcontains "HARDWARE_TYPE=nvidia-vllm" -or $lines -notcontains "VRAM_TIER=$($case[1])") {
            throw "Incorrect vLLM tier for $($case[0]) GB."
        }
    }
    Write-Output "PowerShell setup checks passed."
} finally {
    $env:OS = $oldOS
    $env:COMPOSE_FILE = $oldComposeFile
    $env:LVS_SETUP_NONINTERACTIVE = $oldNonInteractive
    Set-Location $PSScriptRoot
    Remove-Item -LiteralPath $testDir -Recurse -Force -ErrorAction SilentlyContinue
}
