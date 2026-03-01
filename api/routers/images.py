"""Images router — /api/images/*"""

import asyncio
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from api.dependencies import get_current_user
from api.schemas.image import ImageOut
from api.services import d1_service, r2_service

router = APIRouter(prefix="/images", tags=["images"])

_ALLOWED_TYPES = {"image/png", "image/jpeg", "image/bmp", "image/tiff"}
_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

_401 = {"description": "Token inválido ou expirado",       "content": {"application/json": {"example": {"code": 401, "error": "Invalid or expired token", "detail": "Invalid or expired token"}}}}
_404 = {"description": "Imagem não encontrada",            "content": {"application/json": {"example": {"code": 404, "error": "Image not found", "detail": "Image not found"}}}}
_413 = {"description": "Arquivo maior que 10 MB",          "content": {"application/json": {"example": {"code": 413, "error": "File exceeds the 10 MB limit", "detail": "File exceeds the 10 MB limit"}}}}
_422 = {"description": "Formato de arquivo não suportado", "content": {"application/json": {"example": {"code": 422, "error": "Unsupported file type: image/gif", "detail": "Unsupported file type: image/gif"}}}}


@router.post(
    "",
    response_model=ImageOut,
    status_code=status.HTTP_200_OK,
    responses={
        401: _401,
        413: _413,
        422: _422,
    },
)
async def upload_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
) -> ImageOut:
    """Save a pre-treated image to R2 + D1 and return its metadata."""
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

    user_id = current_user["sub"]
    image_id = str(uuid.uuid4())

    r2_key = await asyncio.to_thread(r2_service.upload_image, user_id, image_id, raw_bytes)
    row = await d1_service.create_image(image_id, user_id, r2_key, file.filename)
    url = await asyncio.to_thread(r2_service.get_presigned_url, r2_key)

    return ImageOut(**row, url=url)


@router.get(
    "",
    response_model=list[ImageOut],
    status_code=status.HTTP_200_OK,
    responses={
        401: _401,
    },
)
async def list_images(current_user: dict = Depends(get_current_user)) -> list[ImageOut]:
    """List all images saved by the current user."""
    user_id = current_user["sub"]
    rows = await d1_service.list_images(user_id)
    result = []
    for row in rows:
        url = await asyncio.to_thread(r2_service.get_presigned_url, row["r2_key"])
        result.append(ImageOut(**row, url=url))
    return result


@router.delete(
    "/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: _401,
        404: _404,
    },
)
async def delete_image(
    image_id: str,
    current_user: dict = Depends(get_current_user),
) -> None:
    """Delete an image from R2 and D1."""
    user_id = current_user["sub"]

    try:
        row = await d1_service.get_image(image_id, user_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    await asyncio.to_thread(r2_service.delete_image, row["r2_key"])
    await d1_service.delete_image(image_id, user_id)
