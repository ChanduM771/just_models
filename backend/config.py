"""
Shamba AI - Configuration Settings
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root (gitignored, never committed)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shamba.db")

# JWT Authentication
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
LABELS_PATH = os.path.join(MODEL_PATH, "labels.json")
ADVICE_PATH = os.path.join(MODEL_PATH, "disease_advice.json")

# Upload settings
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# App settings
APP_NAME = "Shamba AI"
APP_VERSION = "1.0.0"

# ===== Gemini Vision API =====
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ===== ONNX Model =====
ONNX_MODEL_PATH = os.path.join(MODEL_PATH, "plantvillage_mobilenetv2.onnx")
ONNX_LABELS_PATH = os.path.join(MODEL_PATH, "plantvillage_labels.json")

# Crops that the local ONNX model can handle (PlantVillage dataset)
LOCAL_MODEL_CROPS = {"tomato", "maize", "corn", "potato", "pepper", "apple"}