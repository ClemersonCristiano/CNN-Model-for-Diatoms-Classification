from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class ClassificationOut(BaseModel):
    id: str
    user_id: str
    image_id: str
    model_used: Literal["model_7k", "model_10k", "model_22k"]
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    created_at: datetime
    image_url: str  # presigned URL


class HistoryList(BaseModel):
    items: list[ClassificationOut]
    total: int
