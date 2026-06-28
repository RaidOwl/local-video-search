"""
Local model tier registry.

Used when AI_BACKEND is ollama, ollama-native, or vllm.
Maps (backend_family, model_family, vram_tier) → model names + service URLs.

In local mode a single multimodal model handles both vision captioning (VLM)
and text summarization (LLM), so vlm_model == llm_model.
"""

# ---------------------------------------------------------------------------
# Model names per family × tier
# ollama and ollama-native share the same model names (differ only in URL)
# ---------------------------------------------------------------------------
TIER_MODELS: dict = {
    "ollama": {
        "qwen": {
            "low":   {"model": "qwen2.5vl:3b",  "embed": "bge-m3"},
            "mid":   {"model": "qwen2.5vl:7b",  "embed": "bge-m3"},
            "high":  {"model": "qwen2.5vl:32b", "embed": "bge-m3"},
            "ultra": {"model": "qwen2.5vl:72b", "embed": "bge-m3"},
        },
        "gemma": {
            "low":   {"model": "gemma3:4b",  "embed": "bge-m3"},
            "mid":   {"model": "gemma3:12b", "embed": "bge-m3"},
            "high":  {"model": "gemma3:27b", "embed": "bge-m3"},
            "ultra": {"model": "gemma3:27b", "embed": "bge-m3"},
        },
    },
    "vllm": {
        # FP8 quantizes only the LLM decoder; vision encoder stays BF16.
        # Practical VRAM needed (with --gpu-memory-utilization 0.75):
        #   3B  → ~3 GB    low GPU
        #   7B  → ~7 GB    mid GPU
        #   32B → ~35 GB   high tier (≥40 GB)
        #   72B → ~80 GB   ultra tier (≥128 GB, e.g. DGX Spark GB10)
        "qwen": {
            "low":   {"model": "Qwen/Qwen2.5-VL-3B-Instruct",  "embed": "BAAI/bge-m3"},
            "mid":   {"model": "Qwen/Qwen2.5-VL-7B-Instruct",  "embed": "BAAI/bge-m3"},
            "high":  {"model": "Qwen/Qwen2.5-VL-32B-Instruct", "embed": "BAAI/bge-m3"},
            "ultra": {"model": "Qwen/Qwen2.5-VL-72B-Instruct", "embed": "BAAI/bge-m3"},
        },
        "gemma": {
            "low":   {"model": "google/gemma-3-4b-it",  "embed": "BAAI/bge-m3"},
            "mid":   {"model": "google/gemma-3-12b-it", "embed": "BAAI/bge-m3"},
            "high":  {"model": "google/gemma-3-12b-it", "embed": "BAAI/bge-m3"},
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
# vLLM tiers assume FP8 quantization (default). Ollama uses Q4 automatically.
# ---------------------------------------------------------------------------
TIER_VRAM: dict = {
    "low":   "≤8 GB",
    "mid":   "≤16 GB",
    "high":  "≤48 GB",
    "ultra": "≥80 GB",
}


def resolve_local_config(backend: str, family: str, tier: str) -> dict:
    """
    Return the base config dict for a local backend mode.
    backend: ollama | ollama-native | vllm
    family:  qwen | gemma
    tier:    low | mid | high | ultra

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
