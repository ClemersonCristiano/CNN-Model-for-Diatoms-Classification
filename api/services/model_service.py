"""Model inference service.

Loads the three fine-tuned Keras models once at startup and exposes a
predict() function that works entirely in memory.

Image preprocessing mirrors CNN/predict.py but uses bytes instead of file paths:
  PNG bytes (grayscale 400×400)
  → decode to numpy
  → grayscale → RGB (ResNet expects 3-channel)
  → resize+pad to 400×400 (already done by image_service, kept for safety)
  → resnet_v2.preprocess_input
  → model.predict()
  → argmax + probabilities dict
"""

import io
from typing import Any

import numpy as np
from PIL import Image

# Lazy imports — TensorFlow is heavy; imported only once at module level.
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input


_IMAGE_SIZE = 400
_CLASSES = ["Encyonema", "Eunotia", "Gomphonema", "Navicula", "Pinnularia"]

_MODEL_PATHS: dict[str, str] = {
    "model_7k":  "CNN/models/modelo_7k/fineTuned_model_7k/Diatom_Classifier_FineTuned_Model_7k.keras",
    "model_10k": "CNN/models/modelo_10k/fineTuned_model_10k/Diatom_Classifier_FineTuned_Model_10k.keras",
    "model_22k": "CNN/models/modelo_22k/fineTuned_model_22k/Diatom_Classifier_FineTuned_Model_22k.keras",
}

# Global model registry — populated by load_models() at startup.
MODELS: dict[str, Any] = {}


def load_models() -> None:
    """Load all three fine-tuned models into memory.

    Must be called once during the FastAPI lifespan startup event.
    TensorFlow is not fork-safe — use --workers 1 with uvicorn.
    """
    for model_id, path in _MODEL_PATHS.items():
        MODELS[model_id] = tf.keras.models.load_model(path)


def predict(image_bytes: bytes, model_id: str) -> dict:
    """Run inference on a pre-processed image.

    Args:
        image_bytes: PNG bytes of the 400×400 grayscale image produced by
                     image_service.treat_image().
        model_id: One of "model_7k", "model_10k", "model_22k".

    Returns:
        {
            "predicted_class": str,
            "confidence": float,
            "probabilities": {"Encyonema": float, ...}
        }

    Raises:
        KeyError: If model_id is not loaded.
        ValueError: If image cannot be decoded.
    """
    model = MODELS[model_id]
    arr = _preprocess(image_bytes)
    preds: np.ndarray = model.predict(arr, verbose=0)[0]

    idx = int(np.argmax(preds))
    probabilities = {cls: float(preds[i]) for i, cls in enumerate(_CLASSES)}

    return {
        "predicted_class": _CLASSES[idx],
        "confidence": float(preds[idx]),
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _preprocess(image_bytes: bytes) -> np.ndarray:
    """Convert grayscale PNG bytes to a batch array ready for ResNet50V2."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")  # ensure grayscale
    pil_rgb = pil_img.convert("RGB")  # ResNet expects 3-channel input
    pil_resized = pil_rgb.resize((_IMAGE_SIZE, _IMAGE_SIZE), Image.LANCZOS)

    arr = np.array(pil_resized, dtype=np.float32)  # (400, 400, 3)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)  # (1, 400, 400, 3)
