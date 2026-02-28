from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from api.core.config import settings


def verify_google_token(token: str) -> dict | None:
    """Validate a Google ID token and return the payload or None on failure."""
    try:
        payload = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            settings.google_client_id,
        )
        return payload
    except Exception:
        return None
