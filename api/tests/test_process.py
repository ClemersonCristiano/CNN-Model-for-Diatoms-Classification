"""Tests for the image processing pipeline (image_service)."""

import io

import numpy as np
import pytest
from PIL import Image


def _make_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a simple RGB PNG in memory for testing."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[10:90, 10:90] = 200  # bright square on black background
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_grayscale_png_bytes(width: int = 100, height: int = 100) -> bytes:
    """Create a grayscale PNG in memory for testing."""
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[10:90, 10:90] = 180
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_treat_image_returns_bytes():
    from api.services.image_service import treat_image

    result = treat_image(_make_png_bytes())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_treat_image_output_is_400x400():
    from api.services.image_service import treat_image

    result = treat_image(_make_png_bytes())
    img = Image.open(io.BytesIO(result))
    assert img.size == (400, 400)


def test_treat_image_accepts_grayscale_input():
    from api.services.image_service import treat_image

    result = treat_image(_make_grayscale_png_bytes())
    img = Image.open(io.BytesIO(result))
    assert img.size == (400, 400)


def test_treat_image_rejects_invalid_bytes():
    from api.services.image_service import treat_image

    with pytest.raises(ValueError):
        treat_image(b"not an image")
