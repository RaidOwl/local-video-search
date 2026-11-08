#!/usr/bin/env bash
# Local Video Search — Smart Setup
# Detects hardware, guides configuration, validates prerequisites, starts stack
#
# Usage: ./setup.sh           Ollama backend (default; runs on any GPU, 8 GB+)
#        ./setup.sh --vllm    opt into vLLM on NVIDIA/AMD (32 GB+ recommended)
#        ./setup.sh --non-interactive   accept detected recommendations

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

USE_VLLM=0
NON_INTERACTIVE=0

# ── Pick the largest model tier that fits ────────────────────────────────────
# Only "low" and "mid" are served locally — larger models (Qwen3-VL-32B,
# 235B-A22B) are external-endpoint only (HARDWARE_TYPE=custom).
#
# Ollama (Q4, 8k context), both models loaded: low ~9 GB, mid ~11 GB, measured.
# On smaller GPUs Ollama still runs, spilling layers to the CPU (slower).
# Must match TIER_VRAM["ollama"] in src/local_models.py.
pick_ollama_tier() {
    local mem_gb="$1"
    if (( mem_gb >= 12 )); then echo mid
    else
        (( mem_gb < 8 )) && warn "${mem_gb}GB GPU memory is below the ~9GB the low tier needs; Ollama will run part of the model on the CPU (slower)."
        echo low
    fi
}

# vLLM (opt-in): conservative BF16 budgets for both model families, including
# the embedding server, main-model KV cache and headroom (see serve.py).
pick_vllm_tier() {
    local mem_gb="$1" unified="$2"
    local usable=$mem_gb
    (( unified )) && usable=$(( mem_gb * 8 / 10 ))
    if (( usable >= 48 )); then echo mid
    else
        (( usable < 32 )) && warn "${mem_gb}GB GPU memory is below the conservative 32GB recommendation for vLLM low; re-run without --vllm to use Ollama."
        echo low
    fi
}

# ── Host Ollama (HARDWARE_TYPE=native) ───────────────────────────────────────
# For GPUs Docker can't pass through (Intel Arc...), Ollama on the host drives the
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
    local memory_gb=0 hardware_label="CPU"
    local extra_env=()

    # macOS / Apple Silicon
    if [[ "$(uname -s)" == "Darwin" ]]; then
        hw_type="apple"
        local chip=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
        local mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
        memory_gb=$mem_gb
        hardware_label="$chip"
        info "Detected macOS: $chip (${mem_gb}GB unified memory)"

        # Only "low" and "mid" are served locally; "high"/"ultra" are
        # external-endpoint only (HARDWARE_TYPE=custom).
        if [[ $mem_gb -ge 16 ]]; then vram_tier="mid"
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
        memory_gb=$sys_mem_gb

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
            hardware_label="$gpu_name"
            memory_gb=$vram_gb

            if ! command -v nvidia-container-toolkit &>/dev/null && ! command -v nvidia-ctk &>/dev/null \
                && ! (command -v dpkg &>/dev/null && dpkg -l 2>/dev/null | grep -q nvidia-container-toolkit); then
                warn "nvidia-container-toolkit not installed. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
                extra_env+=("REQUIRE_NVIDIA_TOOLKIT=1")
            fi

            compose_files="docker-compose.yml"
            if (( USE_VLLM )); then
                hw_type="nvidia-vllm"
                vram_tier=$(pick_vllm_tier "$vram_gb" "$unified")
            else
                hw_type="nvidia"
                vram_tier=$(pick_ollama_tier "$vram_gb")
            fi

        # AMD GPU (ROCm)?
        elif [[ -e /dev/kfd ]]; then
            hw_type="amd"
            # The ROCm overlay adds GPU video decode (VAAPI) to the API and the
            # ROCm vLLM image; Ollama's ROCm image is in docker-compose.yml.
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
            hardware_label="$gfx_name"
            memory_gb=$vram_gb

            if (( USE_VLLM )); then
                hw_type="amd-vllm"
                case "$gfx_name" in
                    gfx94*|gfx95*)
                        extra_env+=("VLLM_ROCM_IMAGE=rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0") ;;
                    gfx11*|gfx12*)
                        extra_env+=("VLLM_ROCM_IMAGE=rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0") ;;
                    *)
                        warn "AMD $gfx_name is not supported by the ROCm vLLM images (need RDNA 3+ or MI300+). Using Ollama instead."
                        hw_type="amd" ;;
                esac
                [[ "$hw_type" == "amd-vllm" ]] && vram_tier=$(pick_vllm_tier "$vram_gb" "$unified")
            fi
            if [[ "$hw_type" == "amd" ]]; then
                # Ollama's ROCm build covers RDNA 2+ and Instinct; on other GPUs it falls back to the CPU.
                vram_tier=$(pick_ollama_tier "$vram_gb")
            fi
            if [[ "$(uname -m)" != "x86_64" ]]; then
                warn "The bundled AMD ROCm images require x86-64. Using host Ollama instead."
                hw_type="native"
                compose_files="docker-compose.yml"
                vram_tier=$(pick_ollama_tier "$vram_gb")
                if host_ollama_up; then
                    warn_if_ollama_loopback_only
                else
                    warn "Install and start Ollama on the host, then re-run setup."
                    extra_env+=("REQUIRE_OLLAMA=1")
                fi
            fi
        fi

        # No GPU Docker can use: prefer Ollama on the host (it can still use the GPU).
        if [[ -z "$hw_type" ]]; then
            local intel_gpu=0
            command -v lspci &>/dev/null && lspci 2>/dev/null | grep -i -E 'vga|3d|display' | grep -q -i 'intel.*(arc|battlemage|alchemist|dg2)' && intel_gpu=1
            if host_ollama_up; then
                hw_type="native"
                compose_files="docker-compose.yml"
                vram_tier="mid"
                info "Using Ollama on the host (HARDWARE_TYPE=native). VRAM_TIER=mid (~11 GB); set low (~9 GB) for smaller GPUs."
                warn_if_ollama_loopback_only
            elif (( intel_gpu )); then
                warn "Intel Arc GPU detected. Docker can't use it, but Ollama on the host can (Vulkan):"
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

    echo "$hw_type|$compose_files|$vram_tier|$model_family|${extra_env[*]-}|$memory_gb|$hardware_label"
}

prompt_value() {
    local label="$1" default="$2" value
    if [[ "${LVS_SETUP_TEST_INPUT:-}" == 1 ]]; then
        read -r value
        printf '%s' "${value:-$default}"
        return
    fi
    if (( NON_INTERACTIVE )) || [[ ! -t 0 ]]; then
        printf '%s' "$default"
        return
    fi
    read -r -p "$label [$default]: " value </dev/tty
    printf '%s' "${value:-$default}"
}

prompt_choice() {
    local label="$1" default="$2" maximum="$3" choice
    while true; do
        choice=$(prompt_value "$label (1-$maximum)" "$default")
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= maximum )); then
            printf '%s' "$choice"
            return
        fi
        warn "Choose a number from 1 to $maximum."
    done
}

apply_env_value() {
    local key="$1" value="$2" tmp
    tmp=$(mktemp)
    if awk -v key="$key" -v value="$value" '
        index($0, key "=") == 1 { print key "=" value; found=1; next }
        { print }
        END { if (!found) print key "=" value }
    ' .env > "$tmp"; then
        mv "$tmp" .env
    else
        rm -f "$tmp"
        return 1
    fi
}

existing_env_value() {
    local key="$1"
    [[ -f .env ]] || return 0
    sed -n "s/^${key}=//p" .env | tail -1 | sed -e 's/^["'\'']//' -e 's/["'\'']$//'
}

configured_ffmpeg_hwaccel() {
    local configured
    configured=$(existing_env_value FFMPEG_HWACCEL)
    case "$configured" in
        off|auto|cuda|vaapi|drm|vulkan) printf '%s' "$configured" ;;
        "") printf 'auto' ;;
        *) warn "Ignoring unsupported FFMPEG_HWACCEL value '$configured'; using auto."; printf 'auto' ;;
    esac
}

probe_external_model() {
    local url="${1%/}" model="$2" name="$3" response
    command -v python3 >/dev/null || {
        warn "python3 is required to validate external model endpoints."
        return 1
    }
    response=$(curl -fsS --max-time 10 "$url/v1/models") || {
        warn "$name endpoint is unreachable: $url/v1/models"
        return 1
    }
    if ! printf '%s' "$response" | python3 -c '
import json, sys
try:
    models = [item.get("id") for item in json.load(sys.stdin).get("data", [])]
except (ValueError, AttributeError, TypeError):
    raise SystemExit(1)
raise SystemExit(0 if sys.argv[1] in models or sys.argv[1] + ":latest" in models else 1)
' "$model"; then
        warn "$name model '$model' was not listed by $url/v1/models."
        return 1
    fi
    ok "$name model verified: $model"
}

free_space_gb() {
    local dir="$1"
    while [[ ! -d "$dir" && "$dir" != "/" && "$dir" != "." ]]; do
        dir=$(dirname "$dir")
    done
    df -Pk "$dir" 2>/dev/null | awk 'NR == 2 { print int($4 / 1048576) }'
}

prompt_models_dir() {
    local backend="$1" current default value needed free note
    current=$(existing_env_value MODELS_DIR)
    default="${current:-./data/models}"
    case "$backend" in
        nvidia-vllm|amd-vllm) needed=30 note="vLLM weights, reranker, and speech models" ;;
        nvidia|amd|cpu) needed=13 note="Ollama models, reranker, and speech models" ;;
        apple|native) needed=4 note="reranker and speech models; host Ollama keeps its own models in its own storage (OLLAMA_MODELS)" ;;
        *) needed=4 note="reranker and speech models" ;;
    esac
    info "Model storage holds downloaded AI models (~${needed} GB: $note)."
    value=$(prompt_value "Model storage folder" "$default")
    value="${value:-$default}"
    [[ "$value" == "~" || "$value" == "~/"* ]] && value="$HOME${value#\~}"
    if [[ "$value" == *:* ]]; then
        err "Model storage folder cannot contain ':' (Docker uses it to separate mount paths)."
        return 1
    fi
    if [[ "$value" != "$default" && -d "$default" ]] && [[ -n "$(ls -A "$default" 2>/dev/null)" ]]; then
        warn "Models already downloaded in $default are not moved. Move them to $value yourself, or they will download again."
    fi
    free=$(free_space_gb "$value")
    if [[ -n "$free" ]] && (( free < needed )); then
        warn "Only ${free} GB is free at $value; about ${needed} GB is needed."
    fi
    WIZARD_MODELS_DIR="$value"
    if [[ -n "$current" || "$value" != "./data/models" ]]; then
        WIZARD_ENV+=("MODELS_DIR=$value")
    fi
}

wait_for_readiness() {
    local timeout="${LVS_SETUP_READY_TIMEOUT:-1800}" start=$SECONDS
    local backend_service=""
    case "$hw_type" in
        nvidia) backend_service=ollama-nvidia ;;
        amd) backend_service=ollama-amd ;;
        cpu) backend_service=ollama ;;
    esac
    log "Waiting for the API and all configured model endpoints to become ready (up to $((timeout / 60)) minutes)..."
    while (( SECONDS - start < timeout )); do
        if COMPOSE_FILE="$compose_files" docker compose exec -T local-vss-api python3 -c '
import json, urllib.request
data=json.load(urllib.request.urlopen("http://127.0.0.1:8000/status/models", timeout=8))
services=data.get("services", [])
raise SystemExit(0 if len(services) == 3 and all(service.get("ok") for service in services) else 1)
' >/dev/null 2>&1; then
            ok "API is healthy and all configured model endpoints are responding."
            return 0
        fi
        sleep 10
    done
    err "The app did not become ready within $((timeout / 60)) minutes."
    COMPOSE_FILE="$compose_files" docker compose ps >&2 || true
    COMPOSE_FILE="$compose_files" docker compose logs --tail 80 local-vss-api $backend_service >&2 || true
    return 1
}

configure_wizard() {
    local detected_hw="$1" hardware_label="$2" memory_gb="$3" family="$4"
    local tier_default backend_default backend_choice memory_choice
    local video_dir embed_url embed_model vlm_url vlm_model llm_url llm_model
    local system_choice system_override family_choice
    WIZARD_ENV=()
    case "$detected_hw" in
        nvidia-vllm) detected_hw=nvidia ;;
        amd-vllm) detected_hw=amd ;;
    esac
    if [[ -t 0 ]]; then
        printf 'System type: 1) Keep detected %s (%s)  2) CPU only  3) Enter a platform type\n' "$hardware_label" "$detected_hw" >/dev/tty
    fi
    system_choice=$(prompt_choice "System type" 1 3)
    if [[ "$system_choice" == 2 ]]; then
        detected_hw=cpu
        hardware_label="CPU"
    elif [[ "$system_choice" == 3 ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
            system_override=$(prompt_value "Platform (apple/native/cpu)" "$detected_hw")
            [[ "$system_override" == "apple" || "$system_override" == "native" || "$system_override" == "cpu" ]] || {
                err "On macOS choose apple, native, or cpu."
                return 1
            }
        else
            system_override=$(prompt_value "Platform (nvidia/amd/native/cpu)" "$detected_hw")
            [[ "$system_override" == "nvidia" || "$system_override" == "amd" || "$system_override" == "native" || "$system_override" == "cpu" ]] || {
                err "On Linux choose nvidia, amd, native, or cpu."
                return 1
            }
        fi
        detected_hw="$system_override"
        hardware_label="$system_override"
    fi
    if [[ "$detected_hw" == "amd" ]]; then
        compose_files="docker-compose.yml:docker-compose.rocm.yml"
    else
        compose_files=docker-compose.yml
    fi
    tier_default=low
    (( memory_gb >= 12 )) && tier_default=mid

    log "Detected system: $hardware_label ($detected_hw; $memory_gb GB total memory). The budget below is the memory to reserve for AI models."
    memory_choice=$(prompt_value "Memory budget for the AI models, in GB" "$memory_gb")
    if [[ ! "$memory_choice" =~ ^[0-9]+$ ]] || (( memory_choice < 1 )); then
        err "Memory budget must be a positive whole number."
        return 1
    fi
    memory_gb=$memory_choice
    tier_default=low
    (( memory_gb >= 12 )) && tier_default=mid

    backend_default=1
    if (( USE_VLLM )) && [[ "$detected_hw" == "nvidia" || "$detected_hw" == "amd" ]]; then
        backend_default=2
    fi
    if [[ "$detected_hw" == "nvidia" || "$detected_hw" == "amd" ]]; then
        if [[ -t 0 ]]; then printf 'AI backend: 1) Ollama (recommended Q4)  2) vLLM (advanced)  3) External OpenAI-compatible servers\n' >/dev/tty; fi
        backend_choice=$(prompt_choice "AI backend" "$backend_default" 3)
    else
        if [[ -t 0 ]]; then printf 'AI backend: 1) Ollama (recommended Q4)  2) External OpenAI-compatible servers\n' >/dev/tty; fi
        backend_choice=$(prompt_choice "AI backend" 1 2)
        [[ "$backend_choice" == 2 ]] && backend_choice=3
    fi

    if [[ "$backend_choice" == 2 && "$detected_hw" != "nvidia" && "$detected_hw" != "amd" ]]; then
        warn "Bundled vLLM is supported only on Linux NVIDIA and AMD systems; using Ollama."
        backend_choice=1
    fi
    if [[ "$backend_choice" == 2 ]]; then
        WIZARD_HW_TYPE="${detected_hw}-vllm"
        (( memory_gb >= 48 )) && tier_default=mid || tier_default=low
        (( memory_gb < 32 )) && warn "The detected budget is below vLLM's conservative 32 GB low-tier recommendation."
        if [[ -t 0 ]]; then printf 'Model family: 1) Qwen3-VL (recommended)  2) Gemma 3\n' >/dev/tty; fi
        family_choice=$(prompt_choice "Model family" 1 2)
        [[ "$family_choice" == 2 ]] && family=gemma || family=qwen
    elif [[ "$backend_choice" == 3 ]]; then
        WIZARD_HW_TYPE=custom
        compose_files=docker-compose.yml
        embed_url=$(prompt_value "Embedding endpoint base URL (unauthenticated; no API-key support)" "http://localhost:8001")
        embed_model=$(prompt_value "Embedding model ID" "Qwen/Qwen3-Embedding-4B")
        vlm_url=$(prompt_value "Vision endpoint base URL" "http://localhost:8002")
        vlm_model=$(prompt_value "Vision model ID" "llava-hf/llava-v1.6-vicuna-7b-hf")
        llm_url=$(prompt_value "Summarization endpoint base URL" "$vlm_url")
        llm_model=$(prompt_value "Summarization model ID" "$vlm_model")
        WIZARD_ENV+=("EMBED_URL=$embed_url" "EMBED_MODEL=$embed_model"
            "VLM_URL=$vlm_url" "VLM_MODEL=$vlm_model"
            "LLM_URL=$llm_url" "LLM_MODEL=$llm_model")
    else
        WIZARD_HW_TYPE="$detected_hw"
        if [[ "$WIZARD_HW_TYPE" == "nvidia" || "$WIZARD_HW_TYPE" == "amd" ]]; then
            if (( memory_gb >= 12 )); then tier_default=mid; else tier_default=low; fi
        fi
        if [[ -t 0 ]]; then printf 'Model family: 1) Qwen3-VL (recommended)  2) Gemma 3\n' >/dev/tty; fi
        family_choice=$(prompt_choice "Model family" 1 2)
        [[ "$family_choice" == 2 ]] && family=gemma || family=qwen
    fi

    if [[ "$WIZARD_HW_TYPE" != custom ]]; then
        WIZARD_TIER=$(prompt_value "Local model tier (low or mid)" "$tier_default")
        if [[ "$WIZARD_TIER" != low && "$WIZARD_TIER" != mid ]]; then
            err "Model tier must be low or mid."
            return 1
        fi
    else
        WIZARD_TIER=$(existing_env_value VRAM_TIER)
        [[ "$WIZARD_TIER" == low || "$WIZARD_TIER" == mid ]] || WIZARD_TIER=mid
    fi
    WIZARD_FAMILY="$family"
    video_dir=$(prompt_value "Video library path (blank to add it later in Settings)" "$(existing_env_value VIDEO_DIR_1)")
    [[ -n "$video_dir" ]] && WIZARD_ENV+=("VIDEO_DIR_1=$video_dir")
    prompt_models_dir "$WIZARD_HW_TYPE" || return 1

    echo
    info "Review: system=$detected_hw, backend=$WIZARD_HW_TYPE, models=$family/$WIZARD_TIER, memory budget=${memory_gb}GB, model storage=$WIZARD_MODELS_DIR"
    if [[ "$WIZARD_HW_TYPE" == "custom" ]]; then
        info "External models: embedding=$embed_model at $embed_url; vision=$vlm_model at $vlm_url; summarization=$llm_model at $llm_url"
        warn "External endpoints must be reachable from the API container and do not support API-key authentication."
        probe_external_model "$embed_url" "$embed_model" "Embedding" || return 1
        probe_external_model "$vlm_url" "$vlm_model" "Vision" || return 1
        probe_external_model "$llm_url" "$llm_model" "Summarization" || return 1
    fi
    if (( ! NON_INTERACTIVE )) && { [[ -t 0 ]] || [[ "${LVS_SETUP_TEST_INPUT:-}" == 1 ]]; }; then
        local confirmation
        confirmation=$(prompt_value "Write configuration and start the app? (Y/n)" "Y")
        if [[ "$confirmation" =~ ^[Nn] ]]; then
            info "Setup cancelled; existing configuration was not changed."
            return 2
        fi
    fi
    return 0
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
# APP_DATA_DIR=./data
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
    local derived=" HARDWARE_TYPE VRAM_TIER COMPOSE_FILE COMPOSE_PROFILES VLLM_ROCM_IMAGE "
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

# ── Pull models into host Ollama (apple / native) ───────────────────────────────
# Names must match src/local_models.py TIER_MODELS["ollama"]. The bare qwen3-vl
# tags are thinking models that return empty captions; use -instruct.
pull_apple_models() {
    local tier="$1" family="$2"
    local embed_model="qwen3-embedding:4b"
    local main_model=""

    if [[ "$family" == "gemma" ]]; then
        case "$tier" in
            low)   main_model="gemma3:4b" ;;
            *)     main_model="gemma3:12b" ;;
        esac
    else
        case "$tier" in
            low)   main_model="qwen3-vl:4b-instruct" ;;
            *)     main_model="qwen3-vl:8b-instruct" ;;
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

    local arg
    for arg in "$@"; do
        case "$arg" in
            --vllm) USE_VLLM=1 ;;
            --non-interactive) NON_INTERACTIVE=1 ;;
            -h|--help) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
            *) err "Unknown option: $arg (see --help)"; exit 1 ;;
        esac
    done

    # 1. Detect hardware
    log "Detecting hardware..."
    local detected_memory hardware_label
    IFS='|' read -r hw_type compose_files vram_tier model_family extra_env detected_memory hardware_label <<< "$(detect_hardware)"

    if [[ -z "$hw_type" ]]; then
        err "Could not detect hardware. Set HARDWARE_TYPE manually in .env."
        exit 1
    fi
    local wizard_status=0
    configure_wizard "$hw_type" "$hardware_label" "$detected_memory" "$model_family" || wizard_status=$?
    if (( wizard_status == 2 )); then exit 0; fi
    (( wizard_status == 0 )) || exit "$wizard_status"
    hw_type="$WIZARD_HW_TYPE"
    vram_tier="$WIZARD_TIER"
    model_family="$WIZARD_FAMILY"
    if [[ "$hw_type" == "amd" || "$hw_type" == "amd-vllm" ]]; then
        compose_files="docker-compose.yml:docker-compose.rocm.yml"
    elif [[ "$hw_type" == "custom" || "$hw_type" == "nvidia-vllm" ]]; then
        compose_files="docker-compose.yml"
    fi

    # 2. FFmpeg and its GPU access are checked inside the API container.
    log "FFmpeg will check for a usable GPU decoder inside the API container."
    local ffmpeg_hwaccel
    ffmpeg_hwaccel=$(configured_ffmpeg_hwaccel)

    # 3. Write .env
    local previous_env=""
    if [[ -f .env ]]; then
        warn ".env exists — backing up to .env.bak"
        cp .env .env.bak
        previous_env=".env.bak"
    fi
    # shellcheck disable=SC2086
    write_env "$hw_type" "$compose_files" "$vram_tier" "$model_family" "$ffmpeg_hwaccel" ${extra_env[@]+"${extra_env[@]}"} ${WIZARD_ENV[@]+"${WIZARD_ENV[@]}"}
    [[ -n "$previous_env" ]] && carry_over_env "$previous_env"
    apply_env_value HARDWARE_TYPE "$hw_type"
    apply_env_value MODEL_FAMILY "$model_family"
    apply_env_value VRAM_TIER "$vram_tier"
    apply_env_value COMPOSE_FILE "$compose_files"
    apply_env_value COMPOSE_PROFILES "\${HARDWARE_TYPE}"
    apply_env_value FFMPEG_HWACCEL "$ffmpeg_hwaccel"
    local wizard_setting
    for wizard_setting in "${WIZARD_ENV[@]+"${WIZARD_ENV[@]}"}"; do
        apply_env_value "${wizard_setting%%=*}" "${wizard_setting#*=}"
    done
    model_family=$(grep -E '^MODEL_FAMILY=' .env | tail -1 | cut -d= -f2- | tr -d "\"'")

    # 3. Check video dirs and prepare model storage
    check_video_dirs
    if mkdir -p "$WIZARD_MODELS_DIR" 2>/dev/null; then
        ok "Model storage: $WIZARD_MODELS_DIR"
    else
        warn "Could not create $WIZARD_MODELS_DIR; Docker will try to create it when the stack starts."
    fi

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

    # 6. Never leave the other backend running: two backends on one GPU run it
    # out of memory. Disabled-profile services aren't stopped by "up".
    local active svc container_id inactive_ids
    active=" $(COMPOSE_FILE="$compose_files" docker compose config --services 2>/dev/null | tr '\n' ' ') "
    for svc in vllm-embed vllm-main ollama ollama-nvidia ollama-amd; do
        if [[ "$active" != *" $svc "* ]]; then
            inactive_ids=$(COMPOSE_FILE="$compose_files" docker compose --profile '*' ps --all --quiet "$svc") || {
                err "Could not check the existing $svc containers; refusing to start another GPU backend."
                exit 1
            }
            while IFS= read -r container_id; do
                [[ -n "$container_id" ]] || continue
                log "Removing this project's $svc container (not part of HARDWARE_TYPE=$hw_type)"
                docker rm -f "$container_id" >/dev/null
            done <<< "$inactive_ids"
        fi
    done

    # 7. Pull images (shows progress) then start
    log "Pulling Docker images (this may take a few minutes on first run)..."
    local build_flag=()
    COMPOSE_FILE="$compose_files" docker compose pull 2>&1 || {
        warn "Pull failed — building the app images from this checkout instead."
        build_flag=(--build)
    }
    log "Starting stack..."
    COMPOSE_FILE="$compose_files" docker compose up -d --remove-orphans ${build_flag[@]+"${build_flag[@]}"}

    local fe_port api_port
    fe_port=$(grep -E '^FRONTEND_PORT=' .env | tail -1 | cut -d= -f2 || true)
    api_port=$(grep -E '^API_PORT=' .env | tail -1 | cut -d= -f2 || true)
    fe_port=${fe_port:-3000}; api_port=${api_port:-8000}

    echo
    ok "Containers started."
    info "Frontend:  http://localhost:$fe_port"
    info "API docs:  http://localhost:$api_port/docs"
    info "Settings:  http://localhost:$fe_port/settings"
    wait_for_readiness
    case "$hw_type" in
        nvidia|amd|cpu)
            local osvc="ollama"; [[ "$hw_type" != cpu ]] && osvc="ollama-$hw_type"
            info "First start downloads the AI models (~9 GB for the mid tier). Follow progress with:"
            echo "       docker compose logs -f $osvc" ;;
        nvidia-vllm|amd-vllm)
            info "First start downloads the AI models (~20 GB for the mid tier). Follow progress with:"
            echo "       docker compose logs -f vllm-main" ;;
    esac
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

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
