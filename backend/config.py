"""
 AgroVision AI - Configuration Settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'shamba.db'}")

# JWT Authentication
JWT_SECRET = os.getenv("JWT_SECRET", "local-development-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
LABELS_PATH = os.path.join(MODEL_PATH, "labels.json")
ADVICE_PATH = os.path.join(MODEL_PATH, "disease_advice.json")
AGROVISION_ROOT = os.path.join(
	os.path.dirname(os.path.dirname(__file__)),
	"AgroVision_Models",
)
AGROVISION_MODELS_PATH = os.path.join(AGROVISION_ROOT, "models")
AGROVISION_MAPPINGS_PATH = os.path.join(AGROVISION_ROOT, "class_mappings")

# Upload settings
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# App settings
APP_NAME = "AgroVision AI"
APP_VERSION = "1.0.0"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@shamba.ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# ===== Gemini Vision API =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ===== ONNX Model =====
ONNX_MODEL_PATH = os.path.join(MODEL_PATH, "plantvillage_mobilenetv2.onnx")
ONNX_LABELS_PATH = os.path.join(MODEL_PATH, "plantvillage_labels.json")

# Crops that the local ONNX model can handle (PlantVillage dataset)
LOCAL_MODEL_CROPS = {"tomato", "maize", "corn", "potato", "pepper", "apple"}