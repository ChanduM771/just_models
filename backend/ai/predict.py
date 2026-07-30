"""
Shamba AI - Prediction Module
Smart router: ONNX → Gemini Vision → Fallback
Returns same format as original: {class_name, confidence, display, crop, emoji}
"""

import logging

from .model import get_model
from .gemini_vision import predict_gemini_vision

logger = logging.getLogger(__name__)


def predict_disease(
    image_bytes: bytes, crop_type: str = None, mime_type: str = "image/jpeg"
) -> dict:
    """
    Run disease prediction on an image.

    Priority:
    1. Gemini Vision API — independent, open-ended image analysis grounded in
       actual observed symptoms (not constrained to a fixed coarse label set)
    2. Local ONNX model — fast offline fallback if Gemini is unavailable
       (note: several ONNX classes are only approximate matches to our labels)
    3. Fallback — return unknown

    Args:
        image_bytes: Raw image bytes
        crop_type: Crop type from user selection (e.g., "Tomato", "Kale")

    Returns:
        Dict with class_name, confidence, disease_name, display, crop, emoji
    """
    # ==========================================
    # PATH 1: Try Gemini Vision API (independent diagnosis)
    # ==========================================
    try:
        result = predict_gemini_vision(image_bytes, crop_type, mime_type=mime_type)
        if result and result.get("confidence", 0) > 20:
            logger.info(
                f"[AI] Gemini result: {result['display']} ({result['confidence']}%)"
            )
            return result
    except Exception as e:
        logger.warning(f"[AI] Gemini API error: {e}")

    # ==========================================
    # PATH 2: Try local ONNX model
    # ==========================================
    try:
        model = get_model()
        result = model.predict(image_bytes, crop_type)
        if result and result.get("confidence", 0) > 40:
            logger.info(
                f"[AI] ONNX result: {result['display']} ({result['confidence']}%)"
            )
            return result
    except Exception as e:
        logger.warning(f"[AI] ONNX model error: {e}")

    # ==========================================
    # PATH 3: Fallback — return unknown
    # ==========================================
    logger.warning("[AI] All prediction methods failed")
    return {
        "class_name": "Unknown",
        "confidence": 0.0,
        "disease_name": "Unknown Disease",
        "display": "Could not identify disease",
        "crop": crop_type or "Unknown",
        "emoji": "❓",
    }
