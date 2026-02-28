from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.core.security import verify_google_token

_security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """Extract and validate the Bearer token; return user payload or raise 401."""
    payload = verify_google_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return {
        "sub": payload["sub"],
        "email": payload["email"],
        "name": payload.get("name", ""),
    }
