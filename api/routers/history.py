"""History router — /api/history/*"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_current_user
from api.schemas.history import ClassificationOut, HistoryList
from api.services import d1_service, r2_service

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryList)
async def list_history(current_user: dict = Depends(get_current_user)) -> HistoryList:
    """Return all classifications for the current user, most recent first."""
    user_id = current_user["sub"]
    # list_classifications already JOINs with images — no N+1
    rows = await d1_service.list_classifications(user_id)
    items = [await _row_to_out(row) for row in rows]
    return HistoryList(items=items, total=len(items))


@router.get("/{classification_id}", response_model=ClassificationOut)
async def get_classification(
    classification_id: str,
    current_user: dict = Depends(get_current_user),
) -> ClassificationOut:
    """Return a single classification with the image's presigned URL."""
    user_id = current_user["sub"]

    try:
        row = await d1_service.get_classification(classification_id, user_id)
    except LookupError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Classification not found")

    return await _row_to_out(row)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _row_to_out(row: dict) -> ClassificationOut:
    """Build a ClassificationOut from a JOIN row (r2_key already included)."""
    r2_key: str | None = row.get("image_r2_key")
    if r2_key:
        image_url = await asyncio.to_thread(r2_service.get_presigned_url, r2_key)
    else:
        image_url = ""

    probabilities = row["probabilities"]
    if isinstance(probabilities, str):
        probabilities = json.loads(probabilities)

    return ClassificationOut(
        id=row["id"],
        user_id=row["user_id"],
        image_id=row["image_id"],
        model_used=row["model_used"],
        predicted_class=row["predicted_class"],
        confidence=row["confidence"],
        probabilities=probabilities,
        created_at=row["created_at"],
        image_url=image_url,
    )
