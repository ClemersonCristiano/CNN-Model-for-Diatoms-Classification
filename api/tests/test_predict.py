"""Tests for the model inference pipeline (model_service)."""

import io

import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch


def _make_grayscale_400_bytes() -> bytes:
    """Create a 400×400 grayscale PNG in memory."""
    arr = np.zeros((400, 400), dtype=np.uint8)
    arr[100:300, 100:300] = 180
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_returns_expected_keys():
    from api.services import model_service

    # Mock model that returns uniform probabilities across 5 classes
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
    model_service.MODELS["model_22k"] = mock_model

    result = model_service.predict(_make_grayscale_400_bytes(), "model_22k")

    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert len(result["probabilities"]) == 5


def test_predict_probabilities_sum_to_one():
    from api.services import model_service

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.6, 0.1, 0.1, 0.1, 0.1]])
    model_service.MODELS["model_7k"] = mock_model

    result = model_service.predict(_make_grayscale_400_bytes(), "model_7k")
    total = sum(result["probabilities"].values())
    assert abs(total - 1.0) < 1e-5


def test_predict_raises_on_unknown_model():
    from api.services import model_service

    with pytest.raises(KeyError):
        model_service.predict(_make_grayscale_400_bytes(), "unknown")
