from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Hardware platform — determines which AI backend is used
    # nvidia → vllm | amd → vllm | apple → ollama-native | cpu → ollama | custom → custom
    hardware_type: str = "custom"
    vram_tier: str = "mid"       # low | mid | high | ultra
    model_family: str = "qwen"   # qwen | gemma

    @property
    def ai_backend(self) -> str:
        """Derive the AI backend from hardware_type."""
        mapping = {
            "nvidia": "vllm",
            "amd": "vllm",
            "apple": "ollama-native",
            "cpu": "ollama",
            "custom": "custom",
        }
        return mapping.get(self.hardware_type, "custom")

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
    whisper_beam_size: int = 1                    # 1 = greedy (fast); higher is slower for marginal gain
    whisper_condition_on_previous: bool = False   # False avoids hallucination loops on silence/room noise

    # Storage
    database_url: str = "sqlite+aiosqlite:///./data/metadata.db"

    # Processing
    frame_extraction_interval: int = 5
    embed_batch_size: int = 16
    ffmpeg_threads: int = 0       # max CPU threads ffmpeg may use; 0 = auto (half cores)

    # Automation (defaults; the Settings UI override file takes precedence at runtime)
    auto_import_interval: int = 0      # minutes between library scans; 0 = scan on startup only
    auto_analyze: str = "off"          # off | audio | video | both
    import_blacklist: str = ""         # comma-separated glob patterns matched against file paths
    default_context: str = ""          # context applied to auto-analyzed videos

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ('settings_',)


@lru_cache()
def get_settings() -> Settings:
    return Settings()