"""Cloudflare D1 service.

D1 has no official Python SDK — communication is done via the Cloudflare
REST API using httpx (async HTTP client).

Reference:
  POST https://api.cloudflare.com/client/v4/accounts/{account_id}/d1/database/{database_id}/query
"""

import httpx

from api.core.config import settings

_BASE = "https://api.cloudflare.com/client/v4"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.d1_api_token}",
        "Content-Type": "application/json",
    }


def _url() -> str:
    return (
        f"{_BASE}/accounts/{settings.cloudflare_account_id}"
        f"/d1/database/{settings.d1_database_id}/query"
    )


async def execute(sql: str, params: list | None = None) -> dict:
    """Send a SQL query to D1 and return the parsed JSON response."""
    payload = {"sql": sql, "params": params or []}
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(_url(), headers=_headers(), json=payload)
    response.raise_for_status()
    data = response.json()
    if not data.get("success"):
        raise RuntimeError(f"D1 query failed: {data.get('errors')}")
    return data


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------

async def get_or_create_user(sub: str, email: str, name: str) -> dict:
    """Return existing user or create a new one. Uses Google sub as primary key."""
    result = await execute(
        "SELECT id, email, name, created_at FROM users WHERE id = ?",
        [sub],
    )
    rows = result["result"][0]["results"]
    if rows:
        return rows[0]

    await execute(
        "INSERT INTO users (id, email, name) VALUES (?, ?, ?)",
        [sub, email, name],
    )
    result = await execute(
        "SELECT id, email, name, created_at FROM users WHERE id = ?",
        [sub],
    )
    return result["result"][0]["results"][0]


# ---------------------------------------------------------------------------
# Image operations
# ---------------------------------------------------------------------------

async def create_image(id: str, user_id: str, r2_key: str, original_name: str | None) -> dict:
    await execute(
        "INSERT INTO images (id, user_id, r2_key, original_name) VALUES (?, ?, ?, ?)",
        [id, user_id, r2_key, original_name],
    )
    return await get_image(id, user_id)


async def get_image(id: str, user_id: str) -> dict:
    result = await execute(
        "SELECT id, user_id, r2_key, original_name, created_at FROM images WHERE id = ? AND user_id = ?",
        [id, user_id],
    )
    rows = result["result"][0]["results"]
    if not rows:
        raise LookupError(f"Image {id} not found for user {user_id}")
    return rows[0]


async def list_images(user_id: str) -> list[dict]:
    result = await execute(
        "SELECT id, user_id, r2_key, original_name, created_at FROM images "
        "WHERE user_id = ? ORDER BY created_at DESC",
        [user_id],
    )
    return result["result"][0]["results"]


async def delete_image(id: str, user_id: str) -> None:
    await execute(
        "DELETE FROM images WHERE id = ? AND user_id = ?",
        [id, user_id],
    )


# ---------------------------------------------------------------------------
# Classification operations
# ---------------------------------------------------------------------------

async def create_classification(
    id: str,
    user_id: str,
    image_id: str,
    model_used: str,
    predicted_class: str,
    confidence: float,
    probabilities: str,  # JSON string
) -> dict:
    await execute(
        """INSERT INTO classifications
           (id, user_id, image_id, model_used, predicted_class, confidence, probabilities)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [id, user_id, image_id, model_used, predicted_class, confidence, probabilities],
    )
    return await get_classification(id, user_id)


async def list_classifications(user_id: str) -> list[dict]:
    """Return classifications joined with their image's r2_key (avoids N+1 queries)."""
    result = await execute(
        """SELECT c.id, c.user_id, c.image_id, c.model_used,
                  c.predicted_class, c.confidence, c.probabilities, c.created_at,
                  i.r2_key AS image_r2_key
           FROM classifications c
           LEFT JOIN images i ON c.image_id = i.id
           WHERE c.user_id = ?
           ORDER BY c.created_at DESC""",
        [user_id],
    )
    return result["result"][0]["results"]


async def get_classification(id: str, user_id: str) -> dict:
    result = await execute(
        """SELECT c.id, c.user_id, c.image_id, c.model_used,
                  c.predicted_class, c.confidence, c.probabilities, c.created_at,
                  i.r2_key AS image_r2_key
           FROM classifications c
           LEFT JOIN images i ON c.image_id = i.id
           WHERE c.id = ? AND c.user_id = ?""",
        [id, user_id],
    )
    rows = result["result"][0]["results"]
    if not rows:
        raise LookupError(f"Classification {id} not found for user {user_id}")
    return rows[0]
