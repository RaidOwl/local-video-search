<# 
.SYNOPSIS
    Local Video Search — Smart Setup for Windows
    Auto-detects hardware, configures .env, validates prerequisites, starts stack
.PARAMETER Vllm
    Opt into vLLM on NVIDIA (32 GB+ recommended) instead of the default Ollama backend.
#>

param(
    [switch]$Force,
    [switch]$Vllm,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$script:NonInteractive = $NonInteractive -or ($env:LVS_SETUP_NONINTERACTIVE -eq "1") -or [Console]::IsInputRedirected

# ── Colors ────────────────────────────────────────────────────────────────────
function Write-Log { param($msg) Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Write-Ok  { param($msg) Write-Host "[✓] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[✗] $msg" -ForegroundColor Red }

function Read-WizardValue {
    param([string]$Label, [string]$Default)
    if ($script:NonInteractive) { return $Default }
    $answer = Read-Host "$Label [$Default]"
    if ([string]::IsNullOrWhiteSpace($answer)) { return $Default }
    return $answer.Trim()
}

function Read-WizardChoice {
    param([string]$Label, [int]$Default, [int]$Maximum)
    while ($true) {
        $answer = Read-WizardValue "$Label (1-$Maximum)" "$Default"
        $choice = 0
        if ([int]::TryParse($answer, [ref]$choice) -and $choice -ge 1 -and $choice -le $Maximum) {
            return $choice
        }
        Write-Warn "Choose a number from 1 to $Maximum."
    }
}

function Get-ExistingEnvValue {
    param([string]$Name, [string]$Default)
    $line = @(Get-Content .env -ErrorAction SilentlyContinue | Where-Object { $_ -match "^$([regex]::Escape($Name))=" } | Select-Object -Last 1)
    if ($line.Count -eq 0) { return $Default }
    return (($line[0] -split '=', 2)[1]).Trim().Trim('"').Trim("'")
}

function Set-EnvValue {
    param([string]$Name, [string]$Value)
    $lines = @(Get-Content .env)
    $found = $false
    $escapedName = [regex]::Escape($Name)
    $lines = @($lines | ForEach-Object {
        if ($_ -match "^$escapedName=") {
            $found = $true
            "$Name=$Value"
        } else { $_ }
    })
    if (-not $found) { $lines += "$Name=$Value" }
    Set-Content -Path .env -Value $lines -Encoding UTF8
}

function Get-FreeSpaceGB {
    param([string]$Path)
    try {
        $full = [System.IO.Path]::GetFullPath($Path)
        $drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($full))
        return [math]::Floor($drive.AvailableFreeSpace / 1GB)
    } catch {
        return $null
    }
}

function Read-ModelsDir {
    param([string]$Backend)
    $current = Get-ExistingEnvValue "MODELS_DIR" ""
    $default = if ($current) { $current } else { ".\data\models" }
    switch ($Backend) {
        "nvidia-vllm" { $needed = 30; $note = "vLLM weights, reranker, and speech models" }
        { $_ -in @("nvidia", "cpu") } { $needed = 13; $note = "Ollama models, reranker, and speech models" }
        "native" { $needed = 4; $note = "reranker and speech models; host Ollama keeps its own models in its own storage (OLLAMA_MODELS)" }
        default { $needed = 4; $note = "reranker and speech models" }
    }
    Write-Log "Model storage holds downloaded AI models (~$needed GB: $note)."
    $value = Read-WizardValue "Model storage folder" $default
    if ($value -eq "~" -or $value -like "~\*" -or $value -like "~/*") { $value = $HOME + $value.Substring(1) }
    if ($value -ne $default -and (Test-Path $default) -and @(Get-ChildItem -Force $default -ErrorAction SilentlyContinue).Count) {
        Write-Warn "Models already downloaded in $default are not moved. Move them to $value yourself, or they will download again."
    }
    $free = Get-FreeSpaceGB $value
    if ($null -ne $free -and $free -lt $needed) {
        Write-Warn "Only $free GB is free at $value; about $needed GB is needed."
    }
    return $value
}

function Test-ExternalModel {
    param([string]$Url, [string]$Model, [string]$Name)
    $base = $Url.TrimEnd('/')
    try {
        $response = Invoke-RestMethod -UseBasicParsing -TimeoutSec 10 -Uri "$base/v1/models"
        $ids = @($response.data | ForEach-Object { $_.id })
        if ($ids -notcontains $Model -and $ids -notcontains "${Model}:latest") {
            throw "Model '$Model' was not listed by $base/v1/models."
        }
        Write-Ok "$Name model verified: $Model"
    } catch {
        throw "$Name endpoint/model check failed at $base/v1/models. $($_.Exception.Message)"
    }
}

function Wait-ForReadiness {
    param([int]$TimeoutSeconds = 1800)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    Write-Log "Waiting for the API and all configured model endpoints (up to $([math]::Floor($TimeoutSeconds / 60)) minutes)..."
    while ((Get-Date) -lt $deadline) {
        try {
            $status = Invoke-RestMethod -UseBasicParsing -TimeoutSec 10 -Uri "http://127.0.0.1:$(Get-ExistingEnvValue 'API_PORT' '8000')/status/models"
            $services = @($status.services)
            if ($services.Count -eq 3 -and @($services | Where-Object { -not $_.ok }).Count -eq 0) {
                Write-Ok "API is healthy and all configured model endpoints are responding."
                return
            }
        } catch { }
        Start-Sleep -Seconds 10
    }
    docker compose ps
    docker compose logs --tail 80 local-vss-api ollama-nvidia ollama-amd ollama
    throw "The app did not become ready within $([math]::Floor($TimeoutSeconds / 60)) minutes. Check the service logs above."
}

# ── Detect hardware ───────────────────────────────────────────────────────────
# 64-bit dedicated VRAM of display adapters matching $Pattern, from the registry.
# (Win32_VideoController.AdapterRAM is 32-bit and tops out at 4 GB.)
function Get-AdapterVramGB {
    param([string]$Pattern)
    $best = 0
    $keys = Get-ChildItem 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}' -ErrorAction SilentlyContinue
    foreach ($k in $keys) {
        $p = Get-ItemProperty $k.PSPath -ErrorAction SilentlyContinue
        if (-not $p -or $p.DriverDesc -notmatch $Pattern) { continue }
        $bytes = $p.'HardwareInformation.qwMemorySize'
        if ($bytes) {
            $gb = [int][math]::Round([double]$bytes / 1GB)
            if ($gb -gt $best) { $best = $gb }
        }
    }
    return $best
}

function Test-HostOllama {
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 http://localhost:11434/api/tags | Out-Null
        return $true
    } catch { return $false }
}

function Get-HardwareConfig {
    $hwType = ""
    $composeFiles = "docker-compose.yml"
    $vramTier = "mid"
    $modelFamily = "qwen"
    $extraEnv = @()
    $memoryGB = 0
    $hardwareLabel = "CPU"

    # $IsWindows only exists in PowerShell 6+; Windows PowerShell 5.1 sets $env:OS.
    $onWindows = $IsWindows -or ($env:OS -eq 'Windows_NT')

    if ($onWindows) {
        $adapters = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue)
        $computer = Get-CimInstance -ClassName Win32_ComputerSystem -ErrorAction SilentlyContinue
        if ($computer.TotalPhysicalMemory) { $memoryGB = [int][math]::Round([double]$computer.TotalPhysicalMemory / 1GB) }

        # NVIDIA GPU: Ollama (or opt-in vLLM) in Docker; Docker Desktop's WSL2 backend passes the GPU through.
        $nvidia = $adapters | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
        if ($nvidia) {
            $hardwareLabel = $nvidia.Name
            $vramGB = 0
            $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if ($smi) {
                $mb = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
                if ("$mb".Trim() -match '^\d+$') { $vramGB = [int][math]::Round([int]"$mb".Trim() / 1024) }
            }
            if (-not $vramGB) { $vramGB = Get-AdapterVramGB 'NVIDIA' }
            $memoryGB = $vramGB
            Write-Log "Detected NVIDIA GPU: $($nvidia.Name) (${vramGB}GB VRAM)"

            $dockerInfo = docker info 2>$null | Out-String
            if ($dockerInfo -notmatch 'nvidia') {
                Write-Warn "Docker doesn't report NVIDIA GPU support. Use Docker Desktop with the WSL2 backend"
                Write-Warn "and a current NVIDIA driver (GPU support is built in), then re-run setup."
                $extraEnv += "REQUIRE_NVIDIA_TOOLKIT=1"
            }

            # Same conservative thresholds as setup.sh pick_ollama_tier / pick_vllm_tier.
            # Only low/mid are served locally; larger models are external-endpoint only.
            if ($Vllm) {
                $hwType = "nvidia-vllm"
                if ($vramGB -ge 48) { $vramTier = "mid" } else { $vramTier = "low" }
                if ($vramGB -lt 32) {
                    Write-Warn "${vramGB}GB VRAM is below the conservative 32GB recommendation for vLLM low; re-run without -Vllm to use Ollama."
                }
            } else {
                $hwType = "nvidia"
                if ($vramGB -ge 12) { $vramTier = "mid" } else { $vramTier = "low" }
                if ($vramGB -lt 8) {
                    Write-Warn "${vramGB}GB VRAM is below the ~9GB the low tier needs; Ollama will run part of the model on the CPU (slower)."
                }
            }
        }
        else {
            # AMD / Intel GPUs: Docker Desktop has no ROCm passthrough, so run Ollama natively
            # on Windows (it uses the GPU directly) and point the containers at it.
            $gpu = $adapters | Where-Object { $_.Name -match 'AMD|Radeon|Intel\(R\) Arc' } | Select-Object -First 1
            if ($gpu) {
                $hardwareLabel = $gpu.Name
                $vramGB = Get-AdapterVramGB 'AMD|Radeon|Arc'
                $memoryGB = $vramGB
                Write-Log "Detected GPU: $($gpu.Name) (${vramGB}GB VRAM) — using native Ollama on the host"
                $hwType = "native"
                if ($vramGB -ge 12) { $vramTier = "mid" } else { $vramTier = "low" }
                if (-not (Test-HostOllama)) {
                    Write-Warn "Ollama isn't running. Install it from https://ollama.com/download, then re-run setup."
                    $extraEnv += "REQUIRE_OLLAMA=1"
                }
            }
        }

        if (-not $hwType) {
            Write-Log "No supported GPU detected — using CPU mode (Ollama in Docker; analysis will be slow)"
            $hwType = "cpu"
            $vramTier = "low"
        }
    }
    elseif ($IsMacOS) {
        Write-Warn "On macOS, use ./setup.sh instead"
        exit 1
    }
    elseif ($IsLinux) {
        Write-Warn "On Linux, use ./setup.sh instead"
        exit 1
    }

    # FFmpeg runs in the API container, so its GPU access must be checked there.
    $ffmpegHwaccel = "auto"
    return @($hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel, $memoryGB, $hardwareLabel)
}

# ── Write .env ────────────────────────────────────────────────────────────────
function Write-EnvFile {
    param($hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel, $wizardEnv)

    $content = @"
# ── Auto-generated by setup.ps1 on $(Get-Date) ──────────────────────────────────────
# Edit this file to customize, then re-run setup.ps1 or docker compose up -d

HARDWARE_TYPE=$hwType
MODEL_FAMILY=$modelFamily
VRAM_TIER=$vramTier
COMPOSE_FILE=$composeFiles
COMPOSE_PROFILES=`${HARDWARE_TYPE}

# ── STORAGE — SET YOUR VIDEO DIRECTORIES ──────────────────────────────────────
VIDEO_DIR_1=C:\path\to\your\videos
# VIDEO_DIR_2=
# VIDEO_DIR_3=

# ── DATA PATHS (override to relocate) ──────────────────────────────────────────
# MODELS_DIR=.\data\models
# APP_DATA_DIR=.\data
# QDRANT_STORAGE_PATH=.\data\qdrant

# ── PROCESSING ─────────────────────────────────────────────────────────────────
FRAME_EXTRACTION_INTERVAL=5
EMBED_BATCH_SIZE=16
FFMPEG_HWACCEL=$ffmpegHwaccel

# ── AUTOMATION (defaults; change in Settings UI after first run) ──────────────
# AUTO_IMPORT_INTERVAL=0
# AUTO_ANALYZE=off

# ── PORTS (uncomment to override) ──────────────────────────────────────────────
# FRONTEND_PORT=3000
# API_PORT=8000
# QDRANT_PORT=6333
"@

    foreach ($e in $extraEnv) {
        if ($e) { $content += "`n# $e" }
    }
    foreach ($setting in $wizardEnv) {
        if ($setting) { $content += "`n$setting" }
    }

    Set-Content -Path .env -Value $content -Encoding UTF8
    Write-Ok "Written .env (HARDWARE_TYPE=$hwType, VRAM_TIER=$vramTier, MODEL_FAMILY=$modelFamily, FFMPEG_HWACCEL=$ffmpegHwaccel)"
}

function Restore-EnvSettings {
    param([string[]]$PreviousLines)
    $derived = @("HARDWARE_TYPE", "VRAM_TIER", "COMPOSE_FILE", "COMPOSE_PROFILES", "VLLM_ROCM_IMAGE")
    $lines = @(Get-Content .env)
    foreach ($oldLine in $PreviousLines) {
        if ($oldLine -notmatch '^\s*([A-Za-z_][A-Za-z0-9_]*)=') { continue }
        $key = $Matches[1]
        if ($derived -contains $key -or $oldLine -eq "VLLM_QUANTIZATION=") { continue }
        $replaced = $false
        $lines = @($lines | ForEach-Object {
            if ($_ -match "^$key=") { $replaced = $true; $oldLine } else { $_ }
        })
        if (-not $replaced) { $lines += $oldLine }
    }
    Set-Content -Path .env -Value $lines -Encoding UTF8
}

# ── Check video dirs ──────────────────────────────────────────────────────────
function Check-VideoDirs {
    $found = $false
    for ($i=1; $i -le 5; $i++) {
        $var = "VIDEO_DIR_$i"
        $dir = Get-Content .env | Where-Object { $_ -like "$var=*" } | ForEach-Object { ($_ -split '=',2)[1] }
        if ($dir -and $dir -ne "C:\path\to\your\videos" -and $dir -ne "/path/to/your\videos") {
            if (Test-Path $dir) {
                Write-Ok "${var}: $dir (exists)"
                $found = $true
            } else {
                Write-Warn "${var}: $dir (NOT FOUND — create it or update .env)"
            }
        }
    }
    if (-not $found) { Write-Warn "No video directories configured. Edit .env and set VIDEO_DIR_1." }
}

# ── Main ──────────────────────────────────────────────────────────────────────
Write-Host @"
╔═══════════════════════════════════════════════════════════════════╗
║        Local Video Search — Smart Setup (Windows)               ║
╚═══════════════════════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan

Set-Location $PSScriptRoot

$hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel, $memoryGB, $hardwareLabel = Get-HardwareConfig

if (-not $hwType) {
    Write-Err "Could not detect hardware. Set HARDWARE_TYPE manually in .env."
    exit 1
}

$detectedHwType = $hwType
if ($detectedHwType -eq "nvidia-vllm") { $detectedHwType = "nvidia" }
$previousEnv = @(Get-Content .env -ErrorAction SilentlyContinue)
$configuredHwaccel = Get-ExistingEnvValue "FFMPEG_HWACCEL" $ffmpegHwaccel
if ($configuredHwaccel -in @("off", "auto", "cuda", "vaapi", "drm", "vulkan")) {
    $ffmpegHwaccel = $configuredHwaccel
} elseif (-not [string]::IsNullOrWhiteSpace($configuredHwaccel)) {
    Write-Warn "Ignoring unsupported FFMPEG_HWACCEL value '$configuredHwaccel'; using $ffmpegHwaccel."
}
$modelFamily = Get-ExistingEnvValue "MODEL_FAMILY" $modelFamily
$modelFamilyDefault = if ($modelFamily -eq "gemma") { 2 } else { 1 }
$videoDefault = Get-ExistingEnvValue "VIDEO_DIR_1" ""
$memoryDefault = [math]::Max(1, $memoryGB)
Write-Log "Detected system: $hardwareLabel ($detectedHwType; $memoryGB GB total memory). The budget below is the memory to reserve for AI models."
$systemChoice = Read-WizardChoice "System type: 1) Keep detected 2) CPU only 3) Enter a platform type" 1 3
if ($systemChoice -eq 2) {
    $detectedHwType = "cpu"
    $hardwareLabel = "CPU"
} elseif ($systemChoice -eq 3) {
    $platform = Read-WizardValue "Platform (nvidia/native/cpu)" $detectedHwType
    if ($platform -notin @("nvidia", "native", "cpu")) {
        throw "Windows setup supports nvidia, native (host Ollama), or cpu."
    }
    $detectedHwType = $platform
    $hardwareLabel = $platform
}
$memoryInput = Read-WizardValue "Memory budget for the AI models, in GB" "$memoryDefault"
$memoryBudget = 0
if (-not [int]::TryParse($memoryInput, [ref]$memoryBudget) -or $memoryBudget -lt 1) {
    throw "Memory budget must be a positive whole number."
}

$backendDefault = if ($Vllm -and $detectedHwType -eq "nvidia") { 2 } else { 1 }
if ($detectedHwType -eq "nvidia") {
    Write-Host "AI backend: 1) Ollama (recommended Q4)  2) vLLM (advanced)  3) External OpenAI-compatible servers"
    $backendChoice = Read-WizardChoice "AI backend" $backendDefault 3
} else {
    Write-Host "AI backend: 1) Ollama (recommended Q4)  2) External OpenAI-compatible servers"
    $backendChoice = Read-WizardChoice "AI backend" 1 2
    if ($backendChoice -eq 2) { $backendChoice = 3 }
}

$wizardEnv = @()
if ($backendChoice -eq 2) {
    if ($detectedHwType -ne "nvidia") {
        Write-Warn "Bundled vLLM is supported only on NVIDIA in the Windows setup; using Ollama."
        $backendChoice = 1
    } else {
        $hwType = "nvidia-vllm"
        $composeFiles = "docker-compose.yml"
        $vramTier = if ($memoryBudget -ge 48) { "mid" } else { "low" }
        if ($memoryBudget -lt 32) { Write-Warn "The detected budget is below vLLM's conservative 32 GB low-tier recommendation." }
        Write-Host "Model family: 1) Qwen3-VL (recommended)  2) Gemma 3"
        $familyChoice = Read-WizardChoice "Model family" $modelFamilyDefault 2
        $modelFamily = if ($familyChoice -eq 2) { "gemma" } else { "qwen" }
    }
}
if ($backendChoice -eq 3) {
    $hwType = "custom"
    $composeFiles = "docker-compose.yml"
    $embedUrl = Read-WizardValue "Embedding endpoint base URL (unauthenticated; no API-key support)" (Get-ExistingEnvValue "EMBED_URL" "http://localhost:8001")
    $embedModel = Read-WizardValue "Embedding model ID" (Get-ExistingEnvValue "EMBED_MODEL" "Qwen/Qwen3-Embedding-4B")
    $vlmUrl = Read-WizardValue "Vision endpoint base URL" (Get-ExistingEnvValue "VLM_URL" "http://localhost:8002")
    $vlmModel = Read-WizardValue "Vision model ID" (Get-ExistingEnvValue "VLM_MODEL" "llava-hf/llava-v1.6-vicuna-7b-hf")
    $llmUrl = Read-WizardValue "Summarization endpoint base URL" (Get-ExistingEnvValue "LLM_URL" $vlmUrl)
    $llmModel = Read-WizardValue "Summarization model ID" (Get-ExistingEnvValue "LLM_MODEL" $vlmModel)
    $wizardEnv += @(
        "EMBED_URL=$embedUrl", "EMBED_MODEL=$embedModel",
        "VLM_URL=$vlmUrl", "VLM_MODEL=$vlmModel",
        "LLM_URL=$llmUrl", "LLM_MODEL=$llmModel"
    )
    Write-Warn "External endpoints must be reachable from the API container and do not support API-key authentication."
    Test-ExternalModel $embedUrl $embedModel "Embedding"
    Test-ExternalModel $vlmUrl $vlmModel "Vision"
    Test-ExternalModel $llmUrl $llmModel "Summarization"
} elseif ($backendChoice -eq 1) {
    $hwType = $detectedHwType
    if ($hwType -eq "nvidia" -or $hwType -eq "amd") {
        $vramTier = if ($memoryBudget -ge 12) { "mid" } else { "low" }
    } else {
        $vramTier = if ($memoryBudget -ge 16) { "mid" } else { "low" }
    }
    Write-Host "Model family: 1) Qwen3-VL (recommended)  2) Gemma 3"
    $familyChoice = Read-WizardChoice "Model family" $modelFamilyDefault 2
    $modelFamily = if ($familyChoice -eq 2) { "gemma" } else { "qwen" }
}

if ($hwType -ne "custom") {
    $tierDefault = if ($vramTier -eq "mid") { "mid" } else { "low" }
    $vramTier = Read-WizardValue "Local model tier (low or mid)" $tierDefault
    if ($vramTier -notin @("low", "mid")) { throw "Model tier must be low or mid." }
} else {
    $vramTier = Get-ExistingEnvValue "VRAM_TIER" "mid"
    if ($vramTier -notin @("low", "mid")) { $vramTier = "mid" }
}
$videoDir = Read-WizardValue "Video library path (blank to add it later in Settings)" $videoDefault
if (-not [string]::IsNullOrWhiteSpace($videoDir)) { $wizardEnv += "VIDEO_DIR_1=$videoDir" }
$modelsDir = Read-ModelsDir $hwType
if ((Get-ExistingEnvValue "MODELS_DIR" "") -or $modelsDir -ne ".\data\models") { $wizardEnv += "MODELS_DIR=$modelsDir" }

if ($hwType -eq "amd" -or $hwType -eq "amd-vllm") {
    $composeFiles = "docker-compose.yml:docker-compose.rocm.yml"
} elseif ($hwType -eq "custom" -or $hwType -eq "nvidia-vllm") {
    $composeFiles = "docker-compose.yml"
}
Write-Log "Review: system=$hardwareLabel, backend=$hwType, models=$modelFamily/$vramTier, memory budget=${memoryBudget}GB, model storage=$modelsDir"
if ($hwType -eq "custom") {
    Write-Log "External models: embedding=$embedModel at $embedUrl; vision=$vlmModel at $vlmUrl; summarization=$llmModel at $llmUrl"
}
if (-not $script:NonInteractive) {
    $confirmation = Read-WizardValue "Write configuration and start the app? (Y/n)" "Y"
    if ($confirmation -match '^[Nn]') {
        Write-Log "Setup cancelled; existing configuration was not changed."
        return
    }
}

if (Test-Path .env) {
    if (-not $Force) {
        Write-Warn ".env exists — backing up to .env.bak"
        Copy-Item .env .env.bak -Force
    }
}

Write-EnvFile $hwType $composeFiles $vramTier $modelFamily $extraEnv $ffmpegHwaccel $wizardEnv
if ($previousEnv.Count -gt 0) {
    Restore-EnvSettings $previousEnv
}
Set-EnvValue "HARDWARE_TYPE" $hwType
Set-EnvValue "MODEL_FAMILY" $modelFamily
Set-EnvValue "VRAM_TIER" $vramTier
Set-EnvValue "COMPOSE_FILE" $composeFiles
Set-EnvValue "COMPOSE_PROFILES" '${HARDWARE_TYPE}'
Set-EnvValue "FFMPEG_HWACCEL" $ffmpegHwaccel
foreach ($setting in $wizardEnv) {
    $parts = $setting -split '=', 2
    Set-EnvValue $parts[0] $parts[1]
}
Check-VideoDirs
try {
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    Write-Ok "Model storage: $modelsDir"
} catch {
    Write-Warn "Could not create $modelsDir; Docker will try to create it when the stack starts."
}

# Ollama on the host: pull the embedding + vision models for the selected tier.
# Names must match src/local_models.py TIER_MODELS["ollama"]. The bare qwen3-vl
# tags are thinking models that return empty captions; use -instruct.
if ($hwType -eq "native" -and (Get-Command ollama -ErrorAction SilentlyContinue) -and (Test-HostOllama)) {
    $mainModel = @{ low = "qwen3-vl:4b-instruct"; mid = "qwen3-vl:8b-instruct" }[$vramTier]
    if ($modelFamily -eq "gemma") {
        $mainModel = @{ low = "gemma3:4b"; mid = "gemma3:12b" }[$vramTier]
    }
    Write-Log "Pulling Ollama models (qwen3-embedding:4b, $mainModel)..."
    & ollama pull qwen3-embedding:4b
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull the embedding model." }
    & ollama pull $mainModel
    if ($LASTEXITCODE -ne 0) { throw "Failed to pull the vision model." }
    Write-Ok "Models ready"
}

Write-Log "Validating Docker Compose config..."
$env:COMPOSE_FILE = $composeFiles
docker compose config -q | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker Compose config is invalid." }
Write-Ok "Compose config valid"

# Never leave the other backend running: two backends on one GPU run it out of
# memory. Disabled-profile services aren't stopped by "up".
$active = @(docker compose config --services 2>$null)
foreach ($svc in @("vllm-embed", "vllm-main", "ollama", "ollama-nvidia", "ollama-amd")) {
    if ($active -notcontains $svc) {
        $owned = @(docker compose --profile '*' ps --all --quiet $svc)
        if ($LASTEXITCODE -ne 0) { throw "Could not check this project's $svc containers; refusing to start another GPU backend." }
        foreach ($containerId in $owned) {
            if (-not $containerId) { continue }
            Write-Log "Removing this project's $svc container (not part of HARDWARE_TYPE=$hwType)"
            docker rm -f $containerId | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "Could not remove the inactive $svc container." }
        }
    }
}

Write-Log "Pulling Docker images (this may take a few minutes on first run)..."
docker compose pull
$buildFlag = @()
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Pull failed — building the app images from this checkout instead."
    $buildFlag = @("--build")
}
Write-Log "Starting stack..."
docker compose up -d --remove-orphans @buildFlag
if ($LASTEXITCODE -ne 0) { throw "Docker Compose failed to start the stack." }

Write-Host ""
Write-Ok "Containers started."
$frontendPort = Get-ExistingEnvValue "FRONTEND_PORT" "3000"
$apiPort = Get-ExistingEnvValue "API_PORT" "8000"
Write-Host "Frontend:  http://localhost:$frontendPort" -ForegroundColor Cyan
Write-Host "API docs:  http://localhost:$apiPort/docs" -ForegroundColor Cyan
if ($hwType -eq "nvidia") {
    Write-Host "First start downloads the AI models (~9 GB). Follow progress with: docker compose logs -f ollama-nvidia" -ForegroundColor Cyan
} elseif ($hwType -eq "cpu") {
    Write-Host "First start downloads the AI models (~9 GB). Follow progress with: docker compose logs -f ollama" -ForegroundColor Cyan
} elseif ($hwType -eq "nvidia-vllm") {
    Write-Host "First start downloads the AI models (~20 GB). Follow progress with: docker compose logs -f vllm-main" -ForegroundColor Cyan
}
Write-Host "Settings:  http://localhost:$frontendPort/settings" -ForegroundColor Cyan
Wait-ForReadiness
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:$frontendPort"
Write-Host "  2. Go to Settings → verify AI backend shows '$hwType'"
Write-Host "  3. Videos → Import → select videos → Analyze"
Write-Host "  4. Search!"
