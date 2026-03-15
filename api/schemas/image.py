from datetime import datetime

from pydantic import BaseModel


class ImageOut(BaseModel):
    id: str
    user_id: str
    r2_key: str
    original_name: str | None
    created_at: datetime
    url: str  # presigned URL


class TreatResponse(BaseModel):
    image_id: str
    url: str  # presigned URL of the treated image


class DeleteImageResponse(BaseModel):
    success: bool
    image_id: str
    message: str
