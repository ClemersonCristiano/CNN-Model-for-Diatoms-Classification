"""Process router — /api/process/*

The crop is performed client-side (react-image-crop).
This router only handles the server-side treatment step.
"""

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status

from api.dependencies import get_current_user
from api.schemas.image import TreatResponse
from api.services import image_service, r2_service, d1_service

router = APIRouter(prefix="/process", tags=["process"])

_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/tiff"}
_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/treat", response_model=TreatResponse)
async def treat(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
) -> TreatResponse:
    """Receive an image, apply rembg + CLAHE + resize, save to R2 + D1."""
    if file.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type: {file.content_type}",
        )

    raw_bytes = await file.read()
    if len(raw_bytes) > _MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 10 MB limit",
        )

    try:
        # image_service uses cv2/rembg — blocking; run in thread pool
        treated_bytes = await asyncio.to_thread(image_service.treat_image, raw_bytes)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    image_id = str(uuid.uuid4())
    user_id = current_user["sub"]

    # r2_service uses boto3 (sync) — run in thread pool
    r2_key = await asyncio.to_thread(r2_service.upload_image, user_id, image_id, treated_bytes)
    await d1_service.create_image(image_id, user_id, r2_key, file.filename)
    url = await asyncio.to_thread(r2_service.get_presigned_url, r2_key)

    return TreatResponse(image_id=image_id, url=url)
