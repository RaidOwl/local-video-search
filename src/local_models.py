"""
Local model tier registry.

Used when HARDWARE_TYPE is not custom. nvidia, amd and cpu run Ollama in Docker,
apple and native use Ollama on the host, and nvidia-vllm / amd-vllm opt into vLLM.
Maps (backend_family, model_family, vram_tier) → model names + service URLs.

In local mode a single multimodal model handles both vision captioning (VLM)
and text summarization (LLM), so vlm_model == llm_model.

Only low and mid are served locally. Larger models (Qwen3-VL-32B, 235B-A22B...)
are external-endpoint only: point HARDWARE_TYPE=custom at your own server.
"""

# ---------------------------------------------------------------------------
# Model names per family × tier
# ollama and ollama-native share the same model names (differ only in URL)
# ---------------------------------------------------------------------------
TIER_MODELS: dict = {
    "ollama": {
        # Ollama's bare qwen3-vl tags are the *thinking* variants: they spend the
        # whole token budget reasoning and return empty captions. Always use the
        # -instruct tags. Both are Q4_K_M.
        "qwen": {
            "low":   {"model": "qwen3-vl:4b-instruct", "embed": "qwen3-embedding:4b"},
            "mid":   {"model": "qwen3-vl:8b-instruct", "embed": "qwen3-embedding:4b"},
        },
        "gemma": {
            "low":   {"model": "gemma3:4b",  "embed": "qwen3-embedding:4b"},
            "mid":   {"model": "gemma3:12b", "embed": "qwen3-embedding:4b"},
        },
    },
    "vllm": {
        # FP8 quantizes only the LLM decoder; vision encoder stays BF16.
        # Practical VRAM needed (with --gpu-memory-utilization 0.75):
        #   4B        → ~4 GB    low GPU
        #   8B        → ~9 GB    mid GPU
        "qwen": {
            "low":   {"model": "Qwen/Qwen3-VL-4B-Instruct", "embed": "Qwen/Qwen3-Embedding-4B"},
            "mid":   {"model": "Qwen/Qwen3-VL-8B-Instruct", "embed": "Qwen/Qwen3-Embedding-4B"},
        },
        "gemma": {
            "low":   {"model": "google/gemma-3-4b-it",  "embed": "Qwen/Qwen3-Embedding-4B"},
            "mid":   {"model": "google/gemma-3-12b-it", "embed": "Qwen/Qwen3-Embedding-4B"},
        },
    },
}

# ---------------------------------------------------------------------------
# Service URLs per backend
# ---------------------------------------------------------------------------
BACKEND_URLS: dict = {
    # Ollama running inside Docker (GPU on nvidia/amd, CPU on cpu)
    "ollama": {
        "model_url": "http://ollama:11434",
        "embed_url": "http://ollama:11434",
    },
    # Ollama running natively on the host (best for Apple Silicon / Metal)
    "ollama-native": {
        "model_url": "http://host.docker.internal:11434",
        "embed_url": "http://host.docker.internal:11434",
    },
    # vLLM: two separate containers (embed + combined VLM/LLM)
    "vllm": {
        "model_url": "http://vllm-main:8002",
        "embed_url": "http://vllm-embed:8001",
    },
}

# ---------------------------------------------------------------------------
# Tier memory guidance (for UI display only): both models loaded, 8k context.
# Ollama (Q4) measured on Strix Halo. vLLM uses the conservative BF16 budgets
# used by setup.sh for both families (see docker/vllm/serve.py).
# ---------------------------------------------------------------------------
TIER_VRAM: dict = {
    "ollama": {"low": "~9 GB",  "mid": "~11 GB"},
    "vllm":   {"low": "32 GB+", "mid": "48 GB+"},
}


def resolve_local_config(backend: str, family: str, tier: str) -> dict:
    """
    Return the base config dict for a local backend mode.
    backend: ollama | ollama-native | vllm
    family:  qwen | gemma
    tier:    low | mid (anything else falls back to mid)

    URLs and model names are fully determined by the tier registry. The
    EMBED_URL / VLM_URL / LLM_URL (and *_MODEL) env vars are only honoured in
    custom mode — local backends always auto-configure from the registry.
    """
    backend_key = "ollama" if backend in ("ollama", "ollama-native") else "vllm"
    family = family if family in TIER_MODELS[backend_key] else "qwen"
    tier = tier if tier in TIER_MODELS[backend_key][family] else "mid"

    models = TIER_MODELS[backend_key][family][tier]
    urls = BACKEND_URLS[backend]

    embed_url = urls["embed_url"]
    vlm_url   = urls["model_url"]
    llm_url   = urls["model_url"]
    embed_model = models["embed"]
    vlm_model   = models["model"]
    llm_model   = models["model"]

    return {
        "embed_url":   embed_url,
        "vlm_url":     vlm_url,
        "llm_url":     llm_url,
        "embed_model": embed_model,
        "vlm_model":   vlm_model,
        "llm_model":   llm_model,
        # resolved metadata (not used by services, but surfaced in /settings)
        "_resolved_backend": backend,
        "_resolved_family":  family,
        "_resolved_tier":    tier,
        "_resolved_model":   llm_model,
        "_resolved_embed":   embed_model,
    }
