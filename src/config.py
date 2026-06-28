from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Deployment mode
    ai_backend: str = "custom"   # custom | ollama | ollama-native | vllm
    vram_tier: str = "mid"       # low | mid | high | ultra
    model_family: str = "qwen"   # qwen | gemma

    # AI Service Endpoints (used in custom mode or as overrides)
    # Backend-agnostic: these may point at vLLM, Ollama, MLX, or a cloud API.
    embed_url: str = "http://localhost:8001"
    vlm_url: str = "http://localhost:8002"
    llm_url: str = "http://localhost:8000"
    qdrant_url: str = "http://localhost:6333"

    # Model names (used in custom mode or as overrides)
    embed_model: str = "BAAI/bge-m3"
    vlm_model: str = "llava-hf/llava-v1.6-vicuna-7b-hf"
    llm_model: str = "Qwen/Qwen3-8B"
    whisper_model: str = "base"

    # Storage
    database_url: str = "sqlite+aiosqlite:///./data/metadata.db"

    # Processing
    frame_extraction_interval: int = 5
    embed_batch_size: int = 16

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ('settings_',)


@lru_cache()
def get_settings() -> Settings:
    return Settings()