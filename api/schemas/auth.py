from datetime import datetime

from pydantic import BaseModel


class LoginRequest(BaseModel):
    token: str


class UserOut(BaseModel):
    id: str
    email: str
    name: str | None
    created_at: datetime
