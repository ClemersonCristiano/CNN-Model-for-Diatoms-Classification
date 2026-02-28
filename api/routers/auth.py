"""Auth router — /api/auth/*"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from api.schemas.auth import LoginRequest, UserOut
from api.core.security import verify_google_token
from api.services import d1_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=UserOut)
async def login(body: LoginRequest) -> UserOut:
    """Validate a Google ID token and return (or create) the user."""
    payload = verify_google_token(body.token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Google token")

    sub: str = payload["sub"]
    email: str = payload["email"]
    name: str = payload.get("name", "")

    user = await d1_service.get_or_create_user(sub, email, name)
    return UserOut(**user)


@router.get("/me", response_model=UserOut)
async def me(current_user: dict = Depends(get_current_user)) -> UserOut:
    """Return the currently authenticated user's data."""
    result = await d1_service.execute(
        "SELECT id, email, name, created_at FROM users WHERE id = ?",
        [current_user["sub"]],
    )
    rows = result["result"][0]["results"]
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return UserOut(**rows[0])
