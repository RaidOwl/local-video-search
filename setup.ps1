<# 
.SYNOPSIS
    Local Video Search — Smart Setup for Windows
    Auto-detects hardware, configures .env, validates prerequisites, starts stack
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ── Colors ────────────────────────────────────────────────────────────────────
function Write-Log { param($msg) Write-Host "[setup] $msg" -ForegroundColor Cyan }
function Write-Ok  { param($msg) Write-Host "[✓] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[!] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[✗] $msg" -ForegroundColor Red }

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

    # $IsWindows only exists in PowerShell 6+; Windows PowerShell 5.1 sets $env:OS.
    $onWindows = $IsWindows -or ($env:OS -eq 'Windows_NT')

    if ($onWindows) {
        $adapters = @(Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue)

        # NVIDIA GPU: vLLM in Docker (Docker Desktop's WSL2 backend passes the GPU through).
        $nvidia = $adapters | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
        if ($nvidia) {
            $vramGB = 0
            $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if ($smi) {
                $mb = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
                if ("$mb".Trim() -match '^\d+$') { $vramGB = [int][math]::Round([int]"$mb".Trim() / 1024) }
            }
            if (-not $vramGB) { $vramGB = Get-AdapterVramGB 'NVIDIA' }
            Write-Log "Detected NVIDIA GPU: $($nvidia.Name) (${vramGB}GB VRAM)"

            $dockerInfo = docker info 2>$null | Out-String
            if ($dockerInfo -notmatch 'nvidia') {
                Write-Warn "Docker doesn't report NVIDIA GPU support. Use Docker Desktop with the WSL2 backend"
                Write-Warn "and a current NVIDIA driver (GPU support is built in), then re-run setup."
                $extraEnv += "REQUIRE_NVIDIA_TOOLKIT=1"
            }

            # Same thresholds as setup.sh pick_vllm_tier (FP8 on RTX 30-series and newer).
            if ($vramGB -ge 51) { $vramTier = "high" }
            elseif ($vramGB -ge 24) { $vramTier = "mid" }
            else { $vramTier = "low" }
            if ($vramGB -lt 12) {
                Write-Warn "${vramGB}GB VRAM is tight for vLLM. Native Ollama is lighter: install it from"
                Write-Warn "https://ollama.com/download and set HARDWARE_TYPE=native in .env."
            }
            $hwType = "nvidia"
        }
        else {
            # AMD / Intel GPUs: Docker Desktop has no ROCm passthrough, so run Ollama natively
            # on Windows (it uses the GPU directly) and point the containers at it.
            $gpu = $adapters | Where-Object { $_.Name -match 'AMD|Radeon|Intel\(R\) Arc' } | Select-Object -First 1
            if ($gpu) {
                $vramGB = Get-AdapterVramGB 'AMD|Radeon|Arc'
                Write-Log "Detected GPU: $($gpu.Name) (${vramGB}GB VRAM) — using native Ollama on the host"
                $hwType = "native"
                if ($vramGB -ge 24) { $vramTier = "high" }
                elseif ($vramGB -ge 10) { $vramTier = "mid" }
                else { $vramTier = "low" }
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

    # ── FFmpeg HW acceleration detection ──────────────────────────────────────
    $ffmpegHwaccel = "off"
    try {
        $raw = & ffmpeg -hide_banner -hwaccels 2>&1 | Select-Object -Skip 1
        $hwaccels = $raw | ForEach-Object { $_.Trim() }
        foreach ($candidate in @('cuda', 'vaapi', 'drm', 'vulkan', 'vdpau')) {
            if ($hwaccels -contains $candidate) {
                $ffmpegHwaccel = "auto"
                break
            }
        }
    } catch { }

    return @($hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel)
}

# ── Write .env ────────────────────────────────────────────────────────────────
function Write-EnvFile {
    param($hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel)

    $content = @"
# ── Auto-generated by setup.ps1 on $(Get-Date) ──────────────────────────────────────
# Edit this file to customize, then re-run setup.ps1 or docker compose up -d

HARDWARE_TYPE=$hwType
MODEL_FAMILY=$modelFamily
VRAM_TIER=$vramTier
COMPOSE_FILE=$composeFiles
COMPOSE_PROFILES=\${HARDWARE_TYPE}

# ── STORAGE — SET YOUR VIDEO DIRECTORIES ──────────────────────────────────────
VIDEO_DIR_1=C:\path\to\your\videos
# VIDEO_DIR_2=
# VIDEO_DIR_3=

# ── DATA PATHS (override to relocate) ──────────────────────────────────────────
# MODELS_DIR=.\data\models
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

    Set-Content -Path .env -Value $content -Encoding UTF8
    Write-Ok "Written .env (HARDWARE_TYPE=$hwType, VRAM_TIER=$vramTier, MODEL_FAMILY=$modelFamily, FFMPEG_HWACCEL=$ffmpegHwaccel)"
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

$hwType, $composeFiles, $vramTier, $modelFamily, $extraEnv, $ffmpegHwaccel = Get-HardwareConfig

if (-not $hwType) {
    Write-Err "Could not detect hardware. Set HARDWARE_TYPE manually in .env."
    exit 1
}

if ($ffmpegHwaccel -eq "auto") {
    Write-Ok "FFmpeg GPU decode available — configured as 'auto'"
} else {
    Write-Log "No FFmpeg GPU decode detected — using software decode"
}

if ((Test-Path .env) -and -not $Force) {
    Write-Warn ".env exists — backing up to .env.bak"
    Copy-Item .env .env.bak -Force
}

Write-EnvFile $hwType $composeFiles $vramTier $modelFamily $extraEnv $ffmpegHwaccel
Check-VideoDirs

# Ollama on the host: pull the embedding + vision models for the selected tier.
if ($hwType -eq "native" -and (Get-Command ollama -ErrorAction SilentlyContinue) -and (Test-HostOllama)) {
    $mainModel = @{ low = "qwen3-vl:4b"; mid = "qwen3-vl:8b"; high = "qwen3-vl:32b"; ultra = "qwen3-vl:235b" }[$vramTier]
    if ($modelFamily -eq "gemma") {
        $mainModel = @{ low = "gemma3:4b"; mid = "gemma3:12b"; high = "gemma3:27b"; ultra = "gemma3:27b" }[$vramTier]
    }
    Write-Log "Pulling Ollama models (bge-m3, $mainModel)..."
    & ollama pull bge-m3
    & ollama pull $mainModel
    Write-Ok "Models ready"
}

Write-Log "Validating Docker Compose config..."
$env:COMPOSE_FILE = $composeFiles
docker compose config -q | Out-Null
Write-Ok "Compose config valid"

Write-Log "Starting stack (this may take a few minutes on first run)..."
docker compose up -d

Write-Host ""
Write-Ok "Stack started!"
Write-Host "Frontend:  http://localhost:3000" -ForegroundColor Cyan
Write-Host "API docs:  http://localhost:8000/docs" -ForegroundColor Cyan
if ($hwType -eq "nvidia") {
    Write-Host "First start downloads the AI models (~20 GB). Follow progress with: docker compose logs -f vllm-main" -ForegroundColor Cyan
}
Write-Host "Settings:  http://localhost:3000/settings" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:3000"
Write-Host "  2. Go to Settings → verify AI backend shows '$hwType'"
Write-Host "  3. Videos → Import → select videos → Analyze"
Write-Host "  4. Search!"
