"""
Shamba AI - Image Utilities
Image validation, resize, base64 handling
"""
import os
import io
import base64
from pathlib import Path

from PIL import Image

from config import MAX_IMAGE_SIZE, ALLOWED_EXTENSIONS


def validate_image(file) -> dict:
    """
    Validate uploaded image file.
    Returns dict with 'valid' boolean and 'error' message if invalid.
    """
    if not file:
        return {"valid": False, "error": "No image file provided"}

    # Check file extension
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {"valid": False, "error": f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Reset to beginning

    if size > MAX_IMAGE_SIZE:
        max_mb = MAX_IMAGE_SIZE / (1024 * 1024)
        return {"valid": False, "error": f"File too large. Maximum size: {max_mb}MB"}

    return {"valid": True, "error": None}


def resize_image(image_bytes: bytes, size: tuple = (224, 224)) -> bytes:
    """
    Resize image to the specified size using PIL.
    Returns JPEG bytes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 encoded string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def read_image_file(file) -> bytes:
    """Read all bytes from an uploaded file object."""
    return file.file.read()
