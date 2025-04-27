from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    model_dir: str = "models"
    n_topics: int = 20
    embedder_name: str = "intfloat/e5-base-v2"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    return Settings()
