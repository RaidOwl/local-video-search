"""
Local model tier registry.

Used when AI_BACKEND is ollama, ollama-native, or vllm.
Maps (backend_family, model_family, vram_tier) → model names + service URLs.

In local mode a single multimodal model handles both vision captioning (VLM)
and text summarization (LLM), so vllm_vlm_model == vllm_model.
"""

# ---------------------------------------------------------------------------
# Model names per family × tier
# ollama and ollama-native share the same model names (differ only in URL)
# ---------------------------------------------------------------------------
TIER_MODELS: dict = {
    "ollama": {
        "qwen": {
            "low":   {"model": "qwen2.5vl:3b",  "embed": "nomic-embed-text"},
            "mid":   {"model": "qwen2.5vl:7b",  "embed": "nomic-embed-text"},
            "high":  {"model": "qwen2.5vl:32b", "embed": "nomic-embed-text"},
            "ultra": {"model": "qwen2.5vl:72b", "embed": "nomic-embed-text"},
        },
        "gemma": {
            "low":   {"model": "gemma3:4b",  "embed": "nomic-embed-text"},
            "mid":   {"model": "gemma3:12b", "embed": "nomic-embed-text"},
            "high":  {"model": "gemma3:27b", "embed": "nomic-embed-text"},
            "ultra": {"model": "gemma3:27b", "embed": "nomic-embed-text"},
        },
    },
    "vllm": {
        "qwen": {
            "low":   {"model": "Qwen/Qwen2.5-VL-3B-Instruct",  "embed": "BAAI/bge-m3"},
            "mid":   {"model": "Qwen/Qwen2.5-VL-7B-Instruct",  "embed": "BAAI/bge-m3"},
            "high":  {"model": "Qwen/Qwen2.5-VL-32B-Instruct", "embed": "BAAI/bge-m3"},
            "ultra": {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "embed": "BAAI/bge-m3"},
        },
        "gemma": {
            "low":   {"model": "google/gemma-3-4b-it",  "embed": "BAAI/bge-m3"},
            "mid":   {"model": "google/gemma-3-12b-it", "embed": "BAAI/bge-m3"},
            "high":  {"model": "google/gemma-3-27b-it", "embed": "BAAI/bge-m3"},
            "ultra": {"model": "google/gemma-3-27b-it", "embed": "BAAI/bge-m3"},
        },
    },
}

# ---------------------------------------------------------------------------
# Service URLs per backend
# ---------------------------------------------------------------------------
BACKEND_URLS: dict = {
    # Ollama running inside Docker
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
# Tier VRAM guidance (for UI display only)
# ---------------------------------------------------------------------------
TIER_VRAM: dict = {
    "low":   "≤8 GB",
    "mid":   "≤16 GB",
    "high":  "≤24 GB",
    "ultra": "24 GB+",
}


def resolve_local_config(backend: str, family: str, tier: str, settings=None) -> dict:
    """
    Return the base config dict for a local backend mode.
    backend: ollama | ollama-native | vllm
    family:  qwen | gemma
    tier:    low | mid | high | ultra

    If `settings` is provided, any explicitly set URL env vars override the
    auto-resolved defaults. This lets users point vllm/ollama mode at existing
    servers without switching to custom mode.
    """
    backend_key = "ollama" if backend in ("ollama", "ollama-native") else "vllm"
    family = family if family in TIER_MODELS[backend_key] else "qwen"
    tier = tier if tier in TIER_MODELS[backend_key][family] else "mid"

    models = TIER_MODELS[backend_key][family][tier]
    urls = BACKEND_URLS[backend]

    # URL defaults from backend registry — overridable via env vars
    default_embed_url = urls["embed_url"]
    default_model_url = urls["model_url"]

    # If the user explicitly set URL env vars, honour them over the defaults.
    # We detect "explicitly set" by comparing against the pydantic default values,
    # which are localhost fallbacks that would never match a real service URL.
    LOCALHOST_DEFAULTS = {"http://localhost:8001", "http://localhost:8002", "http://localhost:8000"}
    if settings:
        if getattr(settings, "vllm_embed_url", None) not in LOCALHOST_DEFAULTS:
            default_embed_url = settings.vllm_embed_url
        if getattr(settings, "vllm_url", None) not in LOCALHOST_DEFAULTS:
            default_model_url = settings.vllm_url

    return {
        "vllm_embed_url":   default_embed_url,
        "vllm_vlm_url":     default_model_url,
        "vllm_url":         default_model_url,
        "vllm_embed_model": models["embed"],
        "vllm_vlm_model":   models["model"],
        "vllm_model":       models["model"],
        # resolved metadata (not used by services, but surfaced in /settings)
        "_resolved_backend": backend,
        "_resolved_family":  family,
        "_resolved_tier":    tier,
        "_resolved_model":   models["model"],
        "_resolved_embed":   models["embed"],
    }
