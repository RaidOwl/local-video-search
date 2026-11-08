"""
Local model tier registry.

Used when HARDWARE_TYPE is nvidia, amd, apple, or cpu (i.e. not custom).
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
            "low":   {"model": "qwen3-vl:4b",   "embed": "qwen3-embedding:4b"},
            "mid":   {"model": "qwen3-vl:8b",   "embed": "qwen3-embedding:4b"},
            "high":  {"model": "qwen3-vl:32b",  "embed": "qwen3-embedding:4b"},
            "ultra": {"model": "qwen3-vl:235b", "embed": "qwen3-embedding:4b"},
        },
        "gemma": {
            "low":   {"model": "gemma3:4b",  "embed": "qwen3-embedding:4b"},
            "mid":   {"model": "gemma3:12b", "embed": "qwen3-embedding:4b"},
            "high":  {"model": "gemma3:27b", "embed": "qwen3-embedding:4b"},
            "ultra": {"model": "gemma3:27b", "embed": "qwen3-embedding:4b"},
        },
    },
    "vllm": {
        # FP8 quantizes only the LLM decoder; vision encoder stays BF16.
        # Practical VRAM needed (with --gpu-memory-utilization 0.75):
        #   4B        → ~4 GB    low GPU
        #   8B        → ~9 GB    mid GPU
        # Only low/mid are served locally. high (32B dense, ~67 GB) and ultra
        # (235B MoE, ~471 GB) need a multi-GPU host, so they are external-endpoint
        # only — use HARDWARE_TYPE=custom to point at your own server.
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
# vLLM tiers assume FP8 (native FP8 GPUs) or BF16 (everything else) — see
# docker/vllm/serve.py. Ollama uses Q4 automatically.
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
