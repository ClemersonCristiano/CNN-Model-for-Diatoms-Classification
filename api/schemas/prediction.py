from typing import Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_id: str
    model: Literal["model_7k", "model_10k", "model_22k"]


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    id: str
    description: str
