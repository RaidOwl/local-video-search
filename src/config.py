from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # AI Service Endpoints
    vllm_embed_url: str = "http://localhost:8001"
    vllm_vlm_url: str = "http://localhost:8002"
    vllm_url: str = "http://localhost:8000"
    qdrant_url: str = "http://localhost:6333"

    # Storage
    video_dir: str = "/data/videos"
    database_url: str = "sqlite+aiosqlite:///./data/metadata.db"

    # Processing
    frame_extraction_interval: int = 5
    embed_batch_size: int = 16

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()