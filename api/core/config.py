from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_client_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    cloudflare_account_id: str
    d1_database_id: str
    d1_api_token: str

    class Config:
        env_file = ".env"


settings = Settings()
