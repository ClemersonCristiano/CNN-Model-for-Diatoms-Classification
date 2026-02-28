"""Cloudflare R2 service (S3-compatible via boto3)."""

import boto3
from botocore.config import Config

from api.core.config import settings

_BUCKET = "diatoms"
_EXPIRES_DEFAULT = 3600


def _client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.cloudflare_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def upload_image(user_id: str, image_id: str, image_bytes: bytes) -> str:
    """Upload a PNG image to R2 and return the r2_key.

    Key format: users/{user_id}/images/{image_id}.png
    """
    key = f"users/{user_id}/images/{image_id}.png"
    _client().put_object(
        Bucket=_BUCKET,
        Key=key,
        Body=image_bytes,
        ContentType="image/png",
    )
    return key


def get_presigned_url(r2_key: str, expires_in: int = _EXPIRES_DEFAULT) -> str:
    """Generate a temporary download URL for the given R2 key."""
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _BUCKET, "Key": r2_key},
        ExpiresIn=expires_in,
    )


def get_image_bytes(r2_key: str) -> bytes:
    """Download an object from R2 and return its raw bytes."""
    obj = _client().get_object(Bucket=_BUCKET, Key=r2_key)
    return obj["Body"].read()


def delete_image(r2_key: str) -> None:
    """Delete an object from R2."""
    _client().delete_object(Bucket=_BUCKET, Key=r2_key)
