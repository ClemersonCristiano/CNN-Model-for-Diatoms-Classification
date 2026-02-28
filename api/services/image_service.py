"""Image processing service.

Adapts the CLI pipeline (segmentacao_com_ia + pipeline_tratamento) to work
entirely in memory (bytes in/bytes out) without touching the filesystem.

Pipeline:
  raw bytes (any format)
  → BGR numpy array (cv2.imdecode)
  → BGRA numpy array (rembg background removal)
  → grayscale + CLAHE (contrast enhancement)
  → resize + pad to 400×400
  → PNG bytes (cv2.imencode)
"""

import io

import cv2
import numpy as np
from PIL import Image
from rembg import remove


_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_TARGET_SIZE = 400


def treat_image(image_bytes: bytes) -> bytes:
    """Apply background removal, CLAHE and resize to 400×400.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, BMP or TIFF).

    Returns:
        PNG bytes of the processed 400×400 grayscale image.
    """
    bgr = _decode_to_bgr(image_bytes)
    bgra = _remove_background(bgr)
    gray = _apply_clahe(bgra)
    padded = _resize_and_pad(gray, _TARGET_SIZE)
    return _encode_to_png(padded)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _decode_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file")
    return bgr


def _remove_background(bgr: np.ndarray) -> np.ndarray:
    """Run rembg on a BGR image and return a BGRA result."""
    # rembg expects RGBA input (PIL/Pillow); convert BGR → RGB → PIL → rembg → BGRA
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_in = Image.fromarray(rgb)
    pil_out: Image.Image = remove(pil_in)  # returns RGBA PIL image
    rgba = np.array(pil_out)
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bgra


def _apply_clahe(bgra: np.ndarray) -> np.ndarray:
    """Convert BGRA to grayscale and apply CLAHE contrast enhancement."""
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE.apply(gray)
    return enhanced


def _resize_and_pad(gray: np.ndarray, target: int) -> np.ndarray:
    """Proportional resize to fit inside target×target, then pad with zeros."""
    h, w = gray.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target, target), dtype=np.uint8)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def _encode_to_png(gray: np.ndarray) -> bytes:
    success, buf = cv2.imencode(".png", gray)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")
    return buf.tobytes()
