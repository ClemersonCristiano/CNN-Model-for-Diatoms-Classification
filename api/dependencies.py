from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.core.security import verify_google_token
from api.services import d1_service

_security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_security),
) -> dict:
    """Extract and validate the Bearer token; return persisted user payload or raise 401."""
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = verify_google_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    sub: str = payload["sub"]
    email: str = payload["email"]
    name: str = payload.get("name", "")

    user = await d1_service.get_or_create_user(sub, email, name)
    return {
        "sub": user["id"],
        "email": user["email"],
        "name": user.get("name", ""),
    }
