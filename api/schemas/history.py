from datetime import datetime

from pydantic import BaseModel


class ClassificationOut(BaseModel):
    id: str
    user_id: str
    image_id: str
    model_used: str
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    created_at: datetime
    image_url: str  # presigned URL


class HistoryList(BaseModel):
    items: list[ClassificationOut]
    total: int
