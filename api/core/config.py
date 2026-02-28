from pathlib import Path

from pydantic_settings import BaseSettings

# Always resolve .env relative to this file → api/.env
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    google_client_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    cloudflare_account_id: str
    d1_database_id: str
    d1_api_token: str

    class Config:
        env_file = str(_ENV_FILE)


settings = Settings()
