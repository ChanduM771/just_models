"""
Shamba AI - Crop Disease Model
Real ONNX inference (MobileNetV2) + Gemini Vision API fallback
"""

import json
import os
import logging

from config import LABELS_PATH, ONNX_MODEL_PATH, ONNX_LABELS_PATH

logger = logging.getLogger(__name__)


class CropDiseaseModel:
    """ONNX-based plant disease classifier using MobileNetV2"""

    def __init__(self):
        self.labels = []  # App labels (12 diseases)
        self.onnx_labels = []  # PlantVillage labels (38 classes)
        self.session = None
        self.input_name = None
        self._load_labels()
        self._load_onnx()

    def _load_labels(self):
        """Load app disease labels from labels.json."""
        try:
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = json.load(f)
            logger.info(f"[AI] Loaded {len(self.labels)} disease labels.")
        except FileNotFoundError:
            logger.warning("[AI] Warning: labels.json not found.")
            self.labels = []

    def _load_onnx(self):
        """Load ONNX model and PlantVillage labels if available."""
        try:
            if os.path.exists(ONNX_LABELS_PATH):
                with open(ONNX_LABELS_PATH, "r", encoding="utf-8") as f:
                    self.onnx_labels = json.load(f)
                logger.info(f"[AI] Loaded {len(self.onnx_labels)} ONNX labels.")
        except Exception as e:
            logger.warning(f"[AI] Could not load ONNX labels: {e}")

        try:
            if os.path.exists(ONNX_MODEL_PATH):
                import onnxruntime as ort

                self.session = ort.InferenceSession(ONNX_MODEL_PATH)
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"[AI] ONNX model loaded: {ONNX_MODEL_PATH}")
            else:
                logger.warning(
                    f"[AI] ONNX model not found: {ONNX_MODEL_PATH}. "
                    f"Gemini API will be used for all predictions."
                )
        except ImportError:
            logger.warning("[AI] onnxruntime not installed. pip install onnxruntime")
        except Exception as e:
            logger.error(f"[AI] Failed to load ONNX model: {e}")

    @property
    def is_loaded(self) -> bool:
        """Check if the ONNX model is ready for inference."""
        return self.session is not None and len(self.onnx_labels) > 0

    def preprocess_image(self, image_bytes: bytes) -> bytes:
        """Preprocess image — resize to 224x224 JPEG."""
        from utils.image_utils import resize_image

        return resize_image(image_bytes, size=(224, 224))

    def _onnx_predict(self, image_bytes: bytes, crop_type: str = None) -> dict:
        """Run ONNX inference and map result to app label format."""
        import numpy as np
        from PIL import Image
        import io

        if not self.is_loaded:
            return None

        try:
            # Preprocess for MobileNetV2
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std

            # HWC -> CHW, add batch dimension
            img_array = np.transpose(img_array, (2, 0, 1))
            img_array = np.expand_dims(img_array, axis=0)

            # Run inference
            outputs = self.session.run(None, {self.input_name: img_array})
            probabilities = outputs[0][0]

            # Softmax
            exp_probs = np.exp(probabilities - np.max(probabilities))
            probabilities = exp_probs / exp_probs.sum()

            # Get top prediction
            top_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[top_idx])

            if top_idx < len(self.onnx_labels):
                onnx_label = self.onnx_labels[top_idx]
            else:
                onnx_label = f"Class_{top_idx}"

            # Map PlantVillage label to our app label
            return self._map_onnx_to_app_label(onnx_label, round(confidence * 100, 1))

        except Exception as e:
            logger.error(f"[AI] ONNX prediction failed: {e}")
            return None

    def _map_onnx_to_app_label(self, onnx_label: str, confidence: float) -> dict:
        """Map a PlantVillage ONNX label to our app's label format."""
        # Build a mapping from ONNX label keywords to our app labels
        keyword_map = {
            "Tomato_Early_blight": "Tomato_Early_Blight",
            "Tomato_Late_blight": "Tomato_Late_Blight",
            "Tomato_Bacterial_spot": "Tomato_Bacterial_Spot",
            "Tomato_Yellow_Leaf_Curl": "Tomato_Yellow_Leaf_Curl",
            "Tomato_Leaf_Mold": "Tomato_Early_Blight",  # Approximate
            "Tomato_Septoria_leaf_spot": "Tomato_Early_Blight",  # Approximate
            "Tomato_Spider_mites": "Tomato_Early_Blight",  # Approximate
            "Tomato_Target_Spot": "Tomato_Early_Blight",  # Approximate
            "Tomato_mosaic_virus": "Tomato_Yellow_Leaf_Curl",  # Approximate
            "Tomato_healthy": None,  # Healthy = no disease
            "Corn_Cercospora": "Corn_(maize)_Gray_leaf_spot",
            "Corn_Common_rust": "Maize_Rust",
            "Corn_Northern_Leaf_Blight": "Corn_(maize)_Northern_Leaf_Blight",
            "Corn_healthy": None,
            "Potato_Early_blight": "Potato_Late_Blight",  # Map to our potato disease
            "Potato_Late_blight": "Potato_Late_Blight",
            "Potato_healthy": None,
        }

        # Find matching app label
        app_class_name = None
        for onnx_key, app_name in keyword_map.items():
            if onnx_key.lower().replace("_", "").replace(
                " ", ""
            ) in onnx_label.lower().replace("_", "").replace(" ", ""):
                app_class_name = app_name
                break

        if app_class_name is None:
            # It's a healthy plant or unknown
            if "healthy" in onnx_label.lower():
                # Find healthy entry in our labels
                healthy = [
                    l for l in self.labels if "healthy" in l.get("name", "").lower()
                ]
                if healthy:
                    h = healthy[0]
                    return {
                        "class_name": h["name"],
                        "confidence": confidence,
                        "disease_name": h["name"],
                        "display": h["display"],
                        "crop": h["crop"],
                        "emoji": h["emoji"],
                        "source": "local_onnx",
                    }
            return None  # Can't map — let Gemini handle it

        # Look up in our labels
        for label in self.labels:
            if label["name"] == app_class_name:
                return {
                    "class_name": label["name"],
                    "confidence": confidence,
                    "disease_name": label["name"],
                    "display": label["display"],
                    "crop": label["crop"],
                    "emoji": label["emoji"],
                    "source": "local_onnx",
                }

        # Fallback — couldn't find in our labels
        return None

    def predict(self, image_bytes: bytes, crop_type: str = None) -> dict:
        """
        Predict disease from leaf image.
        Try ONNX first, fall back to None (caller will try Gemini).

        Returns dict with: class_name, confidence, disease_name, display, crop, emoji
        """
        # Try ONNX if loaded
        if self.is_loaded:
            result = self._onnx_predict(image_bytes, crop_type)
            if result:
                return result

        # If ONNX not available or couldn't map, return None
        # The caller (predict.py) will try Gemini instead
        return None


# Singleton instance
_model_instance = None


def get_model() -> CropDiseaseModel:
    """Get or create the singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = CropDiseaseModel()
    return _model_instance
