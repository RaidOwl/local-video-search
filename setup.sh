#!/usr/bin/env bash
# Local Video Search — Smart Setup
# Auto-detects hardware, configures .env, validates prerequisites, starts stack

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()   { echo -e "${BLUE}[setup]${NC} $*" >&2; }
ok()    { echo -e "${GREEN}[✓]${NC} $*" >&2; }
warn()  { echo -e "${YELLOW}[!]${NC} $*" >&2; }
err()   { echo -e "${RED}[✗]${NC} $*" >&2; }
info()  { echo -e "${CYAN}[i]${NC} $*" >&2; }

# ── Detect hardware ───────────────────────────────────────────────────────────
detect_hardware() {
    local hw_type=""
    local compose_files="docker-compose.yml"
    local vram_tier="mid"
    local model_family="qwen"
    local extra_env=()

    # macOS / Apple Silicon
    if [[ "$(uname -s)" == "Darwin" ]]; then
        hw_type="apple"
        local chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
        local mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        info "Detected macOS: $chip (${mem_gb}GB unified memory)"

        if [[ $mem_gb -ge 96 ]]; then vram_tier="ultra"
        elif [[ $mem_gb -ge 48 ]]; then vram_tier="high"
        elif [[ $mem_gb -ge 16 ]]; then vram_tier="mid"
        else vram_tier="low"; fi

        # Check Ollama
        if ! command -v ollama &>/dev/null; then
            warn "Ollama not installed. Run: brew install ollama && ollama serve"
            extra_env+=("REQUIRE_OLLAMA=1")
        elif ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
            warn "Ollama not running. Start with: ollama serve"
            extra_env+=("REQUIRE_OLLAMA=1")
        else
            # Check/advise keep_alive — during analysis the VLM is called hundreds
            # of times per video, so the default 5m keep_alive causes constant reloads.
            # Recommend indefinite (-1) to match vLLM behavior.
            if pgrep -f ollama >/dev/null 2>&1; then
                # /proc/.../environ is Linux-only (not available on macOS)
                env_line=""
                if [[ "$(uname -s)" == "Linux" ]]; then
                    pid=$(pgrep -f 'ollama serve' | head -1)
                    env_line=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep OLLAMA_KEEP_ALIVE || true)
                fi
                if [ -z "$env_line" ]; then
                    info "Ollama keep_alive not set — models unload after ~5m idle. For best analysis performance, set OLLAMA_KEEP_ALIVE=-1 (indefinite): edit the systemd override or run OLLAMA_KEEP_ALIVE=-1 ollama serve"
                fi
            fi
        fi

    # Linux
    elif [[ "$(uname -s)" == "Linux" ]]; then
        # NVIDIA GPU?
        if command -v nvidia-smi &>/dev/null; then
            local vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
            local vram_gb=$(( vram_mb / 1024 ))
            info "Detected NVIDIA GPU: ${vram_gb}GB VRAM"

            if ! command -v nvidia-container-toolkit &>/dev/null && ! dpkg -l | grep -q nvidia-container-toolkit; then
                warn "nvidia-container-toolkit not installed. Run: curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && sudo apt update && sudo apt install -y nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
                extra_env+=("REQUIRE_NVIDIA_TOOLKIT=1")
            fi

            hw_type="nvidia"
            compose_files="docker-compose.yml"
            if [[ $vram_gb -ge 80 ]]; then vram_tier="ultra"
            elif [[ $vram_gb -ge 40 ]]; then vram_tier="high"
            elif [[ $vram_gb -ge 16 ]]; then vram_tier="mid"
            else vram_tier="low"; fi

        # AMD GPU (ROCm)?
        elif [[ -e /dev/kfd ]] || [[ -e /dev/dri/renderD128 ]]; then
            info "Detected AMD GPU (ROCm)"
            hw_type="amd"
            compose_files="docker-compose.yml:docker-compose.rocm.yml"

            # Check Docker Compose version for !reset support
            local compose_ver=$(docker compose version --short 2>/dev/null | sed 's/v//')
            local major=$(echo "$compose_ver" | cut -d. -f1)
            local minor=$(echo "$compose_ver" | cut -d. -f2)
            if [[ $major -lt 2 ]] || [[ $major -eq 2 && $minor -lt 24 ]]; then
                warn "Docker Compose $compose_ver < 2.24.0 — ROCm override needs >=2.24.0 for device passthrough"
                extra_env+=("REQUIRE_COMPOSE_UPGRADE=1")
            fi

            # Estimate VRAM from system memory (unified on APUs)
            local mem_gb=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
            if [[ $mem_gb -ge 128 ]]; then vram_tier="ultra"
            elif [[ $mem_gb -ge 64 ]]; then vram_tier="high"
            elif [[ $mem_gb -ge 32 ]]; then vram_tier="mid"
            else vram_tier="low"; fi

        # CPU only
        else
            info "No GPU detected — using CPU mode"
            hw_type="cpu"
            vram_tier="low"
        fi
    fi

    echo "$hw_type|$compose_files|$vram_tier|$model_family|${extra_env[*]-}"
}

# ── Detect FFmpeg HW acceleration ────────────────────────────────────────────
detect_ffmpeg_hwaccel() {
    if ! command -v ffmpeg &>/dev/null; then
        echo "off"
        return
    fi
    local hwaccels
    hwaccels=$(ffmpeg -hide_banner -hwaccels 2>/dev/null | tail -n +2 | tr -d ' ')
    for candidate in cuda vaapi drm vulkan vdpau; do
        if echo "$hwaccels" | grep -qi "^$candidate\$"; then
            echo "$candidate"
            return
        fi
    done
    echo "off"
}

# ── Write .env ────────────────────────────────────────────────────────────────
write_env() {
    local hw_type="$1" compose_files="$2" vram_tier="$3" model_family="$4" ffmpeg_hwaccel="$5"
    shift 5
    local extra_env=("$@")

    cat > .env <<ENVEOF
# ── Auto-generated by setup.sh on $(date) ──────────────────────────────────────
# Edit this file to customize, then re-run setup.sh or docker compose up -d

HARDWARE_TYPE=$hw_type
MODEL_FAMILY=$model_family
VRAM_TIER=$vram_tier
COMPOSE_FILE=$compose_files
COMPOSE_PROFILES=\${HARDWARE_TYPE}

# ── STORAGE — SET YOUR VIDEO DIRECTORIES ──────────────────────────────────────
VIDEO_DIR_1=/path/to/your/videos
# VIDEO_DIR_2=
# VIDEO_DIR_3=

# ── DATA PATHS (override to relocate) ──────────────────────────────────────────
# MODELS_DIR=./data/models
# QDRANT_STORAGE_PATH=./data/qdrant

# ── PROCESSING ─────────────────────────────────────────────────────────────────
FRAME_EXTRACTION_INTERVAL=5
EMBED_BATCH_SIZE=16
FFMPEG_HWACCEL=$ffmpeg_hwaccel

# ── AUTOMATION (defaults; change in Settings UI after first run) ──────────────
# AUTO_IMPORT_INTERVAL=0
# AUTO_ANALYZE=off

# ── PORTS (uncomment to override) ──────────────────────────────────────────────
# FRONTEND_PORT=3000
# API_PORT=8000
# QDRANT_PORT=6333
ENVEOF

    # Append any extra requirements
    for e in "${extra_env[@]}"; do
        [[ -n "$e" ]] && echo "# $e" >> .env
    done

    local hw_label=""
    case "$ffmpeg_hwaccel" in
        cuda)   hw_label=" (cuda)" ;;
        vaapi)  hw_label=" (vaapi)" ;;
        drm)    hw_label=" (drm)" ;;
        vulkan) hw_label=" (vulkan)" ;;
        *)      hw_label="" ;;
    esac
    ok "Written .env (HARDWARE_TYPE=$hw_type, VRAM_TIER=$vram_tier, MODEL_FAMILY=$model_family, FFMPEG_HWACCEL=$ffmpeg_hwaccel$hw_label)"
}

# ── Validate video dirs ────────────────────────────────────────────────────────
check_video_dirs() {
    local found=0
    for i in {1..5}; do
        local var="VIDEO_DIR_$i"
        local dir="${!var:-}"
        if [[ -n "$dir" && "$dir" != "/path/to/your/videos" ]]; then
            if [[ -d "$dir" ]]; then
                ok "VIDEO_DIR_$i: $dir (exists)"
                found=1
            else
                warn "VIDEO_DIR_$i: $dir (NOT FOUND — create it or update .env)"
            fi
        fi
    done
    [[ $found -eq 0 ]] && warn "No video directories configured. Edit .env and set VIDEO_DIR_1."
}

# ── Pull models for Apple ──────────────────────────────────────────────────────
pull_apple_models() {
    local tier="$1" family="$2"
    local embed_model="bge-m3"
    local main_model=""

    if [[ "$family" == "gemma" ]]; then
        case "$tier" in
            low)   main_model="gemma3:4b" ;;
            mid)   main_model="gemma3:12b" ;;
            high)  main_model="gemma3:27b" ;;
            ultra) main_model="gemma3:27b" ;;
        esac
    else
        case "$tier" in
            low)   main_model="qwen3-vl:4b" ;;
            mid)   main_model="qwen3-vl:8b" ;;
            high)  main_model="qwen3-vl:32b" ;;
            ultra) main_model="qwen3-vl:235b" ;;
        esac
    fi

    log "Pulling Ollama models for Apple Silicon ($family / $tier)..."
    ollama pull "$embed_model"
    ollama pull "$main_model"
    ok "Models ready: $embed_model, $main_model"
}

# ── Main ───────────────────────────────────────────────────────────────────────
main() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║        Local Video Search — Smart Setup                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    cd "$(dirname "$0")"

    # 1. Detect hardware
    log "Detecting hardware..."
    IFS='|' read -r hw_type compose_files vram_tier model_family extra_env <<< "$(detect_hardware)"

    if [[ -z "$hw_type" ]]; then
        err "Could not detect hardware. Set HARDWARE_TYPE manually in .env."
        exit 1
    fi

    # 2. Detect FFmpeg hardware acceleration backend
    log "Checking FFmpeg hardware acceleration..."
    local ffmpeg_hwaccel
    ffmpeg_hwaccel=$(detect_ffmpeg_hwaccel)
    if [[ "$ffmpeg_hwaccel" != "off" ]]; then
        ok "FFmpeg GPU decode available: $ffmpeg_hwaccel"
        # Use auto so the container probes at runtime (ffmpeg -hwaccels inside
        # the container may differ from the host, e.g. Docker Desktop on macOS).
        ffmpeg_hwaccel="auto"
    else
        info "No FFmpeg GPU decode detected — using software decode"
    fi

    # 3. Write .env
    if [[ -f .env ]]; then
        warn ".env exists — backing up to .env.bak"
        cp .env .env.bak
    fi
    # shellcheck disable=SC2086
    write_env "$hw_type" "$compose_files" "$vram_tier" "$model_family" "$ffmpeg_hwaccel" ${extra_env[@]+"${extra_env[@]}"}

    # 3. Check video dirs
    check_video_dirs

    # 4. Apple Silicon: pull models
    if [[ "$hw_type" == "apple" ]]; then
        if [[ "${extra_env[*]-}" =~ REQUIRE_OLLAMA ]]; then
            warn "Install/start Ollama first, then re-run setup.sh"
        else
            pull_apple_models "$vram_tier" "$model_family"
        fi
    fi

    # 5. Validate Docker Compose
    log "Validating Docker Compose config..."
    COMPOSE_FILE="$compose_files" docker compose config -q && ok "Compose config valid" || {
        err "Compose config invalid. Check .env and docker-compose*.yml"
        exit 1
    }

    # 6. Start
    log "Starting stack (this may take a few minutes on first run)..."
    COMPOSE_FILE="$compose_files" docker compose up -d

    echo
    ok "Stack started!"
    info "Frontend:  http://localhost:3000"
    info "API docs:  http://localhost:8000/docs"
    info "Settings:  http://localhost:3000/settings"
    echo
    info "Next steps:"
    echo "  1. Open http://localhost:3000"
    echo "  2. Go to Settings → verify AI backend shows '$hw_type'"
    echo "  3. Videos → Import → select videos → Analyze"
    echo "  4. Search!"
    echo
    if [[ "$hw_type" == "apple" ]]; then
        info "Note: Ollama runs on host (not in Docker). Keep 'ollama serve' running."
    fi
}

main "$@"
