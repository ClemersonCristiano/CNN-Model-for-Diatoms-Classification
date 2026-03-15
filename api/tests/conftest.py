"""Shared pytest fixtures for the API test suite."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

import api.services.model_service as model_service_module
from api.main import app


def _make_mock_model(probs: list[float] | None = None) -> MagicMock:
    if probs is None:
        probs = [0.6, 0.1, 0.1, 0.1, 0.1]
    mock = MagicMock()
    mock.predict.return_value = np.array([probs])
    return mock


@pytest.fixture(scope="session", autouse=True)
def patch_models():
    """Patch MODELS with mocks so tests never need .keras files on disk."""
    model_service_module.MODELS = {
        "model_7k":  _make_mock_model(),
        "model_10k": _make_mock_model(),
        "model_22k": _make_mock_model([0.9, 0.025, 0.025, 0.025, 0.025]),
    }


@pytest.fixture(scope="session")
def client(patch_models):
    """Return a TestClient with models already patched."""
    with TestClient(app) as c:
        yield c
