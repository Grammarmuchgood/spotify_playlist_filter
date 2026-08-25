from functools import lru_cache

from pydantic_settings import BaseSettings


# BaseSettings (not a plain BaseModel) auto-populates each field from an
# environment variable of the same name, case-insensitive - e.g.
# spotify_client_id reads SPOTIFY_CLIENT_ID with no extra code.
class Settings(BaseSettings):
    # No default value = required. If the matching env var is missing,
    # creating a Settings() instance raises a validation error immediately
    # at startup, rather than silently returning None deep in some module.
    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str

    anthropic_api_key: str
    embedding_api_key: str
    genius_api_key: str

    # Has a default, so it's optional - falls back to local SQLite if
    # DATABASE_URL isn't set in the environment.
    database_url: str = "sqlite:///./backend/data/vibe_filter.db"

    class Config:
        # Also read from a .env file (via python-dotenv under the hood),
        # not just real OS environment variables - works the same in local
        # dev and in a real deployment with no code change.
        env_file = ".env"


# lru_cache with no arguments makes this a de facto singleton: the first
# call builds and validates Settings() once; every later call anywhere in
# the app returns that same cached instance instantly instead of
# re-reading and re-validating the environment every time.
@lru_cache
def get_settings() -> Settings:
    return Settings()
