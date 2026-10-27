"""Crop-routed AgroVision prediction with Gemini explanation."""

import logging

from .agrovision import get_agrovision_registry
from .gemini_vision import (
    explain_classifier_result,
    predict_gemini_vision,
    verify_selected_crop,
)

logger = logging.getLogger(__name__)


def predict_disease(
    image_bytes: bytes, crop_type: str = None, mime_type: str = "image/jpeg"
) -> dict:
    """Run the selected crop's AgroVision model and explain its result."""
    registry = get_agrovision_registry()
    normalized_crop = (crop_type or "").strip().lower()
    if normalized_crop not in registry.specs:
        return {
            "unsupported": True,
            "message": "crop not supported.",
            "crop": crop_type or "Unknown",
        }

    try:
        crop_matches = verify_selected_crop(image_bytes, crop_type, mime_type=mime_type)
        if crop_matches is False:
            return {
                "crop_mismatch": True,
                "message": f"The image does not appear to be a {crop_type}. Please select the correct crop.",
                "crop": crop_type,
            }
    except Exception as exc:
        logger.warning("[AI] Gemini crop verification error: %s", exc)

    try:
        result = registry.predict(image_bytes, crop_type)
        if result:
            explained = explain_classifier_result(result)
            logger.info(
                "[AI] AgroVision result: %s (%s%%)",
                explained["display"],
                explained["confidence"],
            )
            return explained
    except Exception as exc:
        logger.warning("[AI] AgroVision prediction error: %s", exc)

    try:
        fallback = predict_gemini_vision(image_bytes, crop_type, mime_type=mime_type)
        if fallback:
            return fallback
    except Exception as exc:
        logger.warning("[AI] Gemini fallback error: %s", exc)

    return {
        "class_name": "Unknown",
        "confidence": 0.0,
        "disease_name": "Unknown Disease",
        "display": "Could not identify disease",
        "crop": crop_type or "Unknown",
        "emoji": "❓",
        "source": "agrovision",
    }
