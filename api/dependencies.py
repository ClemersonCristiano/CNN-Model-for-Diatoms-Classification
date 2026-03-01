from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.core.security import verify_google_token

# auto_error=False prevents HTTPBearer from raising 403 when the header is absent;
# we raise 401 ourselves so all auth errors are consistent.
_security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict:
    """Extract and validate the Bearer token; return user payload or raise 401."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = verify_google_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return {
        "sub": payload["sub"],
        "email": payload["email"],
        "name": payload.get("name", ""),
    }
