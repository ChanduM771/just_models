"""Crop-specific AgroVision Keras model registry and inference."""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from config import AGROVISION_MAPPINGS_PATH, AGROVISION_MODELS_PATH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    crop: str
    model_path: str
    mapping_path: str


def _canonical_crop(crop: str) -> str:
    normalized = (crop or "").strip().lower()
    return {"corn": "corn", "tomato": "tomato", "apple": "apple", "potato": "potato"}.get(normalized, normalized)


def _crop_from_model_name(filename: str) -> str:
    return _canonical_crop(filename.removesuffix("_EfficientNetB0.keras"))


def discover_models() -> dict[str, ModelSpec]:
    """Discover crop models and their matching class mappings from disk."""
    registry = {}
    if not os.path.isdir(AGROVISION_MODELS_PATH):
        return registry

    for filename in sorted(os.listdir(AGROVISION_MODELS_PATH)):
        if not filename.endswith("_EfficientNetB0.keras"):
            continue
        crop = _crop_from_model_name(filename)
        model_crop_name = filename.removesuffix("_EfficientNetB0.keras").lower()
        mapping_candidates = [
            os.path.join(AGROVISION_MAPPINGS_PATH, f"{model_crop_name}_class_mapping.json"),
            os.path.join(AGROVISION_MAPPINGS_PATH, f"{crop}_class_mapping.json"),
        ]
        mapping_path = next((path for path in mapping_candidates if os.path.isfile(path)), None)
        if mapping_path:
            registry[crop] = ModelSpec(
                crop=crop,
                model_path=os.path.join(AGROVISION_MODELS_PATH, filename),
                mapping_path=mapping_path,
            )
        else:
            logger.warning("Skipping %s: mapping file not found", filename)
    return registry


class AgroVisionRegistry:
    """Lazy-load and cache the crop-specific Keras models."""

    def __init__(self):
        self.specs = discover_models()
        self._models = {}
        self._mappings = {}

    @property
    def supported_crops(self) -> list[str]:
        return sorted(self.specs)

    def _load_model(self, crop: str):
        if crop not in self._models:
            try:
                import keras
            except ImportError as exc:
                raise RuntimeError("TensorFlow is required for AgroVision models") from exc
            self._models[crop] = keras.models.load_model(self.specs[crop].model_path, compile=False)
        return self._models[crop]

    def _load_mapping(self, crop: str) -> dict[int, str]:
        if crop not in self._mappings:
            with open(self.specs[crop].mapping_path, encoding="utf-8") as file:
                raw_mapping = json.load(file)
            self._mappings[crop] = {int(label_index): label for label, label_index in raw_mapping.items()}
        return self._mappings[crop]

    def predict(self, image_bytes: bytes, crop_type: str) -> dict:
        crop = _canonical_crop(crop_type)
        if crop not in self.specs:
            return None

        image = Image.open(__import__("io").BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        probabilities = self._load_model(crop).predict(batch, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])
        class_name = self._load_mapping(crop).get(class_index, "Unknown")

        return {
            "class_name": class_name,
            "confidence": round(confidence * 100, 1),
            "disease_name": class_name,
            "display": class_name.replace("___", " - ").replace("__", " ").replace("_", " "),
            "crop": "Corn" if crop == "corn" else crop.title(),
            "emoji": {"corn": "🌽", "tomato": "🍅", "apple": "🍎", "potato": "🥔"}[crop],
            "source": "agrovision",
            "model": os.path.basename(self.specs[crop].model_path),
        }

    def status(self) -> dict:
        return {
            crop: {"model": os.path.basename(spec.model_path), "loaded": crop in self._models}
            for crop, spec in self.specs.items()
        }


_registry = AgroVisionRegistry()


def get_agrovision_registry() -> AgroVisionRegistry:
    return _registry