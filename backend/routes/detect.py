"""
AgroVision AI - Detection Routes
POST /api/detect - Accept image + crop type, return disease prediction
"""

import os
import json
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Header
from typing import Optional

from config import UPLOAD_DIR
from db.database import get_connection
from ai.predict import predict_disease
from ai.advice import get_advice
from utils.image_utils import validate_image, read_image_file

router = APIRouter(prefix="/api", tags=["Detection"])


def get_optional_user(authorization: str = None):
    """Get user ID from token if provided, otherwise return None (anonymous)."""
    if not authorization:
        return None
    from utils.auth_utils import get_active_user
    return get_active_user(authorization)["user_id"]


@router.post("/detect")
async def detect_disease(
    image: UploadFile = File(...),
    crop_type: str = Form(...),
    authorization: Optional[str] = Header(None),
):
    """
    Detect crop disease from uploaded image.

    - Accepts multipart form with image file and crop_type
    - Validates image (size, format)
    - Runs AI prediction (simulated)
    - Looks up treatment advice
    - Saves detection to database
    - Returns comprehensive result
    """
    # Validate image
    validation = validate_image(image)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["error"])

    # Read image bytes
    image_bytes = read_image_file(image)
    mime_type = image.content_type or "image/jpeg"

    # Run prediction
    prediction = predict_disease(image_bytes, crop_type, mime_type=mime_type)
    if prediction.get("unsupported"):
        raise HTTPException(status_code=400, detail=prediction["message"])
    if prediction.get("crop_mismatch"):
        raise HTTPException(status_code=400, detail=prediction["message"])
    if prediction.get("verification_unavailable"):
        raise HTTPException(status_code=503, detail=prediction["message"])

    # Get advice for the predicted disease
    gemini_meta = (
        {
            "severity": prediction.get("severity"),
            "cause_type": prediction.get("cause_type"),
            "display": prediction.get("display"),
            "brief_explanation": prediction.get("brief_explanation"),
        }
        if prediction.get("source") == "gemini_vision"
        else None
    )
    advice = get_advice(prediction["class_name"], gemini_meta=gemini_meta)

    # Get optional user
    user_id = get_optional_user(authorization)

    # Save to uploads directory if it exists
    image_path = None
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        extension = os.path.splitext(image.filename or "")[1].lower() or ".jpg"
        filename = f"{uuid.uuid4().hex}{extension}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)
        image_path = filepath
    except Exception:
        pass  # Non-critical — don't fail if we can't save the file

    # Save detection to database
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO detections 
               (user_id, image_path, crop_type, disease_name, confidence, severity, symptoms, treatment_steps, products)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                image_path,
                crop_type,
                prediction["class_name"],
                prediction["confidence"],
                advice["severity"],
                json.dumps(advice["symptoms"]),
                json.dumps(advice["treatment_steps"]),
                json.dumps(advice["products"]),
            ),
        )
        conn.commit()
        detection_id = cursor.lastrowid
        conn.close()
    except Exception as e:
        detection_id = None
        print(f"[Detect] Warning: Could not save detection to DB: {e}")

    return {
        "id": detection_id,
        "disease": prediction["display"],
        "disease_name": prediction["class_name"],
        "confidence": prediction["confidence"],
        "source": prediction.get("source", "agrovision"),
        "model": prediction.get("model"),
        "brief_explanation": prediction.get("brief_explanation", ""),
        "advice_summary": prediction.get("advice_summary", ""),
        "crop": prediction["crop"],
        "emoji": prediction["emoji"],
        "severity": advice["severity"],
        "symptoms": advice["symptoms"],
        "treatment_steps": advice["treatment_steps"],
        "products": advice["products"],
        "prevention": advice["prevention"],
    }
