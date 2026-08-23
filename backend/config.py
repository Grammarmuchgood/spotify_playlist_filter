from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str

    anthropic_api_key: str
    embedding_api_key: str
    genius_api_key: str

    database_url: str = "sqlite:///./backend/data/vibe_filter.db"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
