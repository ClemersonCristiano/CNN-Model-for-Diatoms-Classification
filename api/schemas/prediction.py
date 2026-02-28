from typing import Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_id: str
    model: Literal["7k", "10k", "22k"]


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    id: str
    description: str
