"""Predict router — /api/predict, /api/models"""

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from api.schemas.prediction import ModelInfo, PredictRequest, PredictResponse
from api.services import d1_service, model_service, r2_service

router = APIRouter(tags=["predict"])

_MODEL_DESCRIPTIONS = {
    "model_7k":  "trained with ~7k curated images and dynamic augmentation (best model)",
    "model_10k": "trained with ~10k pure images and dynamic augmentation",
    "model_22k": "trained with ~22k curated images and 3x augmentations (raw augmentation, images from model_7k x3)",
}

_401 = {"description": "Token inválido ou expirado",        "content": {"application/json": {"example": {"code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}}}}
_404 = {"description": "Imagem não encontrada",             "content": {"application/json": {"example": {"code": 404, "error": "Image not found", "detail": "Image not found"}}}}
_503 = {"description": "Modelo solicitado não carregado",   "content": {"application/json": {"example": {"code": 503, "error": "Model not loaded", "detail": "Model not loaded"}}}}


@router.get(
    "/models",
    response_model=list[ModelInfo],
    status_code=status.HTTP_200_OK,
)
async def list_models() -> list[ModelInfo]:
    """Return the list of available models."""
    return [ModelInfo(id=k, description=v) for k, v in _MODEL_DESCRIPTIONS.items()]


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    responses={
        401: _401,
        404: _404,
        503: _503,
    },
)
async def predict(
    body: PredictRequest,
    current_user: dict = Depends(get_current_user),
) -> PredictResponse:
    """Run inference on a saved image and persist the classification to D1."""
    user_id = current_user["sub"]

    try:
        image_row = await d1_service.get_image(body.image_id, user_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    if body.model not in model_service.MODELS:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    # Download image bytes from R2 (blocking — run in thread pool)
    image_bytes = await asyncio.to_thread(r2_service.get_image_bytes, image_row["r2_key"])

    # Model inference (blocking TensorFlow call — run in thread pool)
    result = await asyncio.to_thread(model_service.predict, image_bytes, body.model)

    classification_id = str(uuid.uuid4())
    await d1_service.create_classification(
        id=classification_id,
        user_id=user_id,
        image_id=body.image_id,
        model_used=body.model,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=json.dumps(result["probabilities"]),
    )

    return PredictResponse(**result)
