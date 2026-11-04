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

# ── Pick the largest model tier that fits ────────────────────────────────────
# Mirrors the memory math in docker/vllm/serve.py: main model weights + ~8 GB KV
# cache, plus ~3.5 GB for the embedder and headroom. Unified-memory systems keep
# ~20% for the OS and stay on "mid" by default: their memory bandwidth makes the
# 32B model several times slower per frame. Set VRAM_TIER=high in .env to opt in.
# "ultra" (235B MoE) needs multiple GPUs and is never picked automatically.
pick_vllm_tier() {
    local mem_gb="$1" unified="$2" fp8="$3"
    local usable=$mem_gb
    (( unified )) && usable=$(( mem_gb * 8 / 10 ))
    local need_high need_mid need_low
    if (( fp8 )); then need_high=51; need_mid=24; need_low=12
    else need_high=81; need_mid=32; need_low=16; fi
    if (( ! unified && usable >= need_high )); then echo high
    elif (( usable >= need_mid )); then echo mid
    else
        (( usable < need_low )) && warn "${mem_gb}GB GPU memory is below the ~${need_low}GB the smallest vLLM tier needs; consider HARDWARE_TYPE=cpu (Ollama) or custom."
        echo low
    fi
}

# ── Host Ollama (HARDWARE_TYPE=native) ───────────────────────────────────────
# For GPUs vLLM can't use (Intel Arc, older AMD...), Ollama on the host drives the
# GPU itself (Metal/CUDA/ROCm/Vulkan) while the Dockerized CPU Ollama can't.
host_ollama_up() {
    curl -sf -m 3 http://localhost:11434/api/version &>/dev/null
}

# Containers reach host Ollama through the Docker bridge, so on Linux it must
# listen on more than 127.0.0.1 (Docker Desktop on macOS/Windows forwards loopback).
warn_if_ollama_loopback_only() {
    command -v ss &>/dev/null || return 0
    local listen
    listen=$(ss -ltnH 'sport = :11434' 2>/dev/null || true)
    if [[ -n "$listen" ]] && ! echo "$listen" | grep -q -E '(\*|0\.0\.0\.0|\[::\]):11434'; then
        warn "Ollama only listens on 127.0.0.1, so the containers can't reach it. Make it listen on all interfaces:"
        warn "  sudo mkdir -p /etc/systemd/system/ollama.service.d"
        warn "  printf '[Service]\\nEnvironment=\"OLLAMA_HOST=0.0.0.0:11434\"\\n' | sudo tee /etc/systemd/system/ollama.service.d/host.conf"
        warn "  sudo systemctl daemon-reload && sudo systemctl restart ollama"
        extra_env+=("REQUIRE_OLLAMA_HOST=1")
    fi
    return 0
}

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

        # qwen3-vl:235b is ~143 GB at Q4, so "ultra" needs a 192 GB+ machine.
        if [[ $mem_gb -ge 192 ]]; then vram_tier="ultra"
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
        local sys_mem_gb=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) / 1024 / 1024 ))

        # NVIDIA GPU?
        if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
            local gpu_name vram_mb cc unified=0
            gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
            local vram_gb
            if [[ "$vram_mb" =~ ^[0-9]+$ ]]; then
                # Round, don't floor: an RTX 4090 reports 24564 MiB, which is a 24 GB card.
                vram_gb=$(( (vram_mb + 512) / 1024 ))
            else
                # Unified-memory GPUs (DGX Spark GB10, Jetson) report [N/A]: the GPU shares system RAM.
                unified=1
                vram_gb=$sys_mem_gb
            fi
            info "Detected NVIDIA GPU: $gpu_name, ${vram_gb}GB $( ((unified)) && echo unified || echo VRAM ), compute capability ${cc:-unknown}"

            if ! command -v nvidia-container-toolkit &>/dev/null && ! command -v nvidia-ctk &>/dev/null \
                && ! (command -v dpkg &>/dev/null && dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit); then
                warn "nvidia-container-toolkit not installed. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
                extra_env+=("REQUIRE_NVIDIA_TOOLKIT=1")
            fi

            hw_type="nvidia"
            compose_files="docker-compose.yml"
            # vLLM runs FP8 on Ampere and newer (native on Ada/Hopper/Blackwell, weight-only before).
            local fp8=0
            [[ "${cc%%.*}" =~ ^[0-9]+$ ]] && (( ${cc%%.*} >= 8 )) && fp8=1
            vram_tier=$(pick_vllm_tier "$vram_gb" "$unified" "$fp8")

        # AMD GPU (ROCm)?
        elif [[ -e /dev/kfd ]]; then
            hw_type="amd"
            compose_files="docker-compose.yml:docker-compose.rocm.yml"

            # Check Docker Compose version for !reset support
            local compose_ver=$(docker compose version --short 2>/dev/null | sed 's/v//')
            local major=$(echo "$compose_ver" | cut -d. -f1)
            local minor=$(echo "$compose_ver" | cut -d. -f2)
            if [[ -z "$major" ]] || [[ $major -lt 2 ]] || [[ $major -eq 2 && $minor -lt 24 ]]; then
                warn "Docker Compose ${compose_ver:-unknown} < 2.24.0 — ROCm override needs >=2.24.0 for device passthrough"
                extra_env+=("REQUIRE_COMPOSE_UPGRADE=1")
            fi

            # GPU family from the KFD topology (gfx_target_version 110501 -> gfx1151).
            local gfx_ver
            gfx_ver=$(cat /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null \
                | awk '$1=="gfx_target_version" && $2>0 {print $2; exit}')
            local gfx_major=$(( ${gfx_ver:-0} / 10000 ))
            local gfx_name
            gfx_name=$(printf 'gfx%d%d%x' "$gfx_major" $(( ${gfx_ver:-0} / 100 % 100 )) $(( ${gfx_ver:-0} % 100 )))

            # VRAM of the largest AMD GPU. APUs (Strix Halo / Point) have a small BIOS
            # carve-out and map the rest of system RAM as GTT, so use GTT for those.
            local vram_b=0 gtt_b=0 f v
            for f in /sys/class/drm/card*/device/mem_info_vram_total; do
                [[ -r "$f" ]] || continue
                v=$(cat "$f")
                if (( v > vram_b )); then
                    vram_b=$v
                    gtt_b=$(cat "${f%vram_total}gtt_total" 2>/dev/null || echo 0)
                fi
            done
            local vram_gb=$(( (vram_b + 536870912) / 1073741824 )) unified=0
            if (( vram_gb < 8 )) && (( gtt_b > vram_b )); then
                unified=1
                vram_gb=$(( gtt_b / 1024 / 1024 / 1024 ))
            fi
            info "Detected AMD GPU: $gfx_name, ${vram_gb}GB $( ((unified)) && echo "unified (APU)" || echo VRAM )"

            local fp8=0
            case "$gfx_name" in
                gfx94*|gfx95*)
                    fp8=1
                    extra_env+=("VLLM_ROCM_IMAGE=rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0") ;;
                gfx11*|gfx12*)
                    extra_env+=("VLLM_ROCM_IMAGE=rocm/vllm:rocm7.14.1_rdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0") ;;
                *)
                    warn "AMD $gfx_name is not supported by the ROCm vLLM images (need RDNA 3+ or MI300+)."
                    hw_type=""
                    compose_files="docker-compose.yml" ;;
            esac
            if [[ "$hw_type" == "amd" ]]; then
                vram_tier=$(pick_vllm_tier "$vram_gb" "$unified" "$fp8")
            fi
        fi

        # No GPU vLLM can use: prefer Ollama on the host (it can still use the GPU).
        if [[ -z "$hw_type" ]]; then
            local intel_gpu=0
            command -v lspci &>/dev/null && lspci 2>/dev/null | grep -i -E 'vga|3d|display' | grep -q -i 'intel.*(arc|battlemage|alchemist|dg2)' && intel_gpu=1
            if host_ollama_up; then
                hw_type="native"
                compose_files="docker-compose.yml"
                vram_tier="mid"
                info "Using Ollama on the host (HARDWARE_TYPE=native). VRAM_TIER=mid (qwen3-vl:8b, ~6 GB); set low for GPUs under 8 GB."
                warn_if_ollama_loopback_only
            elif (( intel_gpu )); then
                warn "Intel Arc GPU detected. vLLM in Docker can't use it, but Ollama can (Vulkan):"
                warn "  curl -fsSL https://ollama.com/install.sh | sh   (then set OLLAMA_VULKAN=1 and OLLAMA_HOST=0.0.0.0 for the service)"
                warn "and re-run setup to use it. Continuing in CPU mode for now."
            fi
        fi

        if [[ -z "$hw_type" ]]; then

            info "No supported GPU detected — using CPU mode (Ollama in Docker; analysis will be slow)"
            hw_type="cpu"
            compose_files="docker-compose.yml"
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
# Uncomment and set at least one video directory. If none are set (all commented
# out), an empty placeholder mounts and you configure paths via the UI + import.
# VIDEO_DIR_1=/path/to/your/videos
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

    # Append detected settings (KEY=value) and unmet requirements (REQUIRE_*, as comments)
    for e in "${extra_env[@]}"; do
        [[ -z "$e" ]] && continue
        if [[ "$e" == REQUIRE_* ]]; then echo "# $e" >> .env; else echo "$e" >> .env; fi
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

# ── Carry user settings over from the previous .env ──────────────────────────
# Re-running setup re-detects hardware but must not drop the user's own settings
# (video dirs, data paths, ports, custom URLs...). Keys setup derives from the
# hardware are rewritten; every other KEY=value from the backup is appended.
carry_over_env() {
    local old="$1"
    [[ -f "$old" ]] || return 0
    local derived=" HARDWARE_TYPE VRAM_TIER COMPOSE_FILE COMPOSE_PROFILES FFMPEG_HWACCEL VLLM_ROCM_IMAGE "
    local line key kept=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)= ]] || continue
        key="${BASH_REMATCH[1]}"
        [[ "$derived" == *" $key "* ]] && continue
        # Older READMEs told AMD users to blank this; serve.py now picks it per GPU.
        [[ "$line" == "VLLM_QUANTIZATION=" ]] && continue
        if grep -q "^${key}=" .env; then
            # Replace the default setup just wrote with the user's value.
            local tmp; tmp=$(mktemp)
            awk -v k="$key" -v l="$line" 'index($0, k"=")==1 {print l; next} {print}' .env > "$tmp" && cat "$tmp" > .env
            rm -f "$tmp"
        else
            echo "$line" >> .env
        fi
        kept=$((kept + 1))
    done < "$old"
    (( kept )) && ok "Kept $kept setting(s) from your previous .env (backup: $old)"
    return 0
}

# ── Validate video dirs ────────────────────────────────────────────────────────
check_video_dirs() {
    local found=0
    for i in {1..5}; do
        local var="VIDEO_DIR_$i"
        local dir
        dir=$(grep -E "^${var}=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//' || true)
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
    return 0
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

    log "Pulling Ollama models into host Ollama ($family / $tier)..."
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
        ok "FFmpeg GPU decode available on host: $ffmpeg_hwaccel"
    fi
    # Always "auto": decoding runs in the API container, which probes its own
    # ffmpeg at runtime (the host may not even have ffmpeg installed).
    ffmpeg_hwaccel="auto"

    # 3. Write .env
    if [[ -f .env ]]; then
        warn ".env exists — backing up to .env.bak"
        cp .env .env.bak
    fi
    # shellcheck disable=SC2086
    write_env "$hw_type" "$compose_files" "$vram_tier" "$model_family" "$ffmpeg_hwaccel" ${extra_env[@]+"${extra_env[@]}"}
    carry_over_env .env.bak

    # 3. Check video dirs
    check_video_dirs

    # 4. Ollama on the host (Apple Silicon / native): pull models
    if [[ "$hw_type" == "apple" || "$hw_type" == "native" ]]; then
        if [[ " ${extra_env[*]-} " == *" REQUIRE_OLLAMA=1 "* ]]; then
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

    # 6. Pull images (shows progress) then start
    log "Pulling Docker images (this may take a few minutes on first run)..."
    COMPOSE_FILE="$compose_files" docker compose pull 2>&1 || warn "Pull failed — trying to start anyway (may need a retry)"
    log "Starting stack..."
    COMPOSE_FILE="$compose_files" docker compose up -d

    local fe_port api_port
    fe_port=$(grep -E '^FRONTEND_PORT=' .env | tail -1 | cut -d= -f2 || true)
    api_port=$(grep -E '^API_PORT=' .env | tail -1 | cut -d= -f2 || true)
    fe_port=${fe_port:-3000}; api_port=${api_port:-8000}

    echo
    ok "Stack started!"
    info "Frontend:  http://localhost:$fe_port"
    info "API docs:  http://localhost:$api_port/docs"
    info "Settings:  http://localhost:$fe_port/settings"
    if [[ "$hw_type" == "nvidia" || "$hw_type" == "amd" ]]; then
        info "First start downloads the AI models (~20 GB for the mid tier). Follow progress with:"
        echo "       docker compose logs -f vllm-main"
    fi
    echo
    info "Next steps:"
    local n=1
    if ! grep -qE '^VIDEO_DIR_[1-5]=.+' .env; then
        echo "  $n. Add your video directories to .env:"; n=$((n + 1))
        echo "       echo 'VIDEO_DIR_1=/path/to/your/videos' >> .env"
        echo "       docker compose up -d   # remounts without restarting"
    fi
    echo "  $n. Open http://localhost:$fe_port"; n=$((n + 1))
    echo "  $n. Go to Settings → verify the AI backend and that all models show as reachable"; n=$((n + 1))
    echo "  $n. Videos → Import → select videos → Analyze"; n=$((n + 1))
    echo "  $n. Search!"
    echo
    if [[ "$hw_type" == "apple" || "$hw_type" == "native" ]]; then
        info "Note: Ollama runs on the host (not in Docker). Keep it running."
    fi
}

main "$@"
