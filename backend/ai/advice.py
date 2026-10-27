"""
AgroVision AI - Disease Advice Module
Looks up disease_advice.json and returns treatment steps, products, etc.
"""

import json

from config import ADVICE_PATH

# Cache for advice data
_advice_cache = None


def _load_advice() -> dict:
    """Load and cache disease advice data."""
    global _advice_cache
    if _advice_cache is None:
        try:
            with open(ADVICE_PATH, encoding="utf-8", mode="r") as f:
                _advice_cache = json.load(f)
        except FileNotFoundError:
            print("[AI] Warning: disease_advice.json not found.")
            _advice_cache = {}
    return _advice_cache


def _normalize(name: str) -> str:
    """Loose-match key: lowercase, strip underscores/parens/spaces."""
    return (
        name.lower().replace("_", "").replace("(", "").replace(")", "").replace(" ", "")
    )


def get_advice(disease_name: str, gemini_meta: dict = None) -> dict:
    """
    Look up advice for a given disease.

    Args:
        disease_name: The internal disease name (e.g., "Maize_Lethal_Necrosis")
        gemini_meta: Optional dict from gemini_vision result containing display,
                     severity, cause_type, brief_explanation for unknown diseases.

    Returns:
        Dict with symptoms, treatment_steps, products, severity, prevention
    """
    advice_data = _load_advice()
    advice = advice_data.get(disease_name)

    # Exact key missed — try a normalized match before giving up on the curated entry.
    if not advice:
        target = _normalize(disease_name)
        for key, val in advice_data.items():
            if _normalize(key) == target:
                advice = val
                break

    if advice:
        return {
            "symptoms": advice.get("symptoms", ["Symptoms data not available"]),
            "treatment_steps": advice.get(
                "treatment_steps",
                [{"step": 1, "text": "Consult local agricultural officer"}],
            ),
            "products": advice.get(
                "products",
                [
                    {
                        "name": "Consult agrovet",
                        "type": "N/A",
                        "availability": "Local agrovet",
                    }
                ],
            ),
            "severity": advice.get("severity", "Medium"),
            "prevention": advice.get("prevention", ["Practice good field hygiene"]),
        }

    # Disease not in database — build generic advice from Gemini metadata
    meta = gemini_meta or {}
    cause_type = meta.get("cause_type", "unknown")
    display = meta.get("display", disease_name.replace("_", " "))
    explanation = meta.get("brief_explanation", "")
    severity = meta.get("severity", "Medium")

    cause_advice = {
        "fungal": {
            "treatment_steps": [
                {"step": 1, "text": "Remove and destroy affected plant parts"},
                {
                    "step": 2,
                    "text": "Apply a broad-spectrum fungicide (e.g. Mancozeb or Copper-based)",
                },
                {"step": 3, "text": "Improve air circulation around plants"},
                {"step": 4, "text": "Avoid overhead watering"},
            ],
            "prevention": [
                "Use disease-resistant varieties",
                "Rotate crops each season",
                "Apply preventive fungicide sprays during wet weather",
            ],
            "products": [
                {
                    "name": "Mancozeb 80WP",
                    "type": "Fungicide",
                    "availability": "Local agrovet",
                },
                {
                    "name": "Copper Oxychloride",
                    "type": "Contact Fungicide",
                    "availability": "Local agrovet",
                },
            ],
        },
        "bacterial": {
            "treatment_steps": [
                {"step": 1, "text": "Remove and destroy infected plant material"},
                {"step": 2, "text": "Apply copper-based bactericide"},
                {"step": 3, "text": "Avoid working with plants when wet"},
                {"step": 4, "text": "Disinfect tools between plants"},
            ],
            "prevention": [
                "Use certified disease-free seeds",
                "Avoid overhead irrigation",
                "Practice crop rotation",
            ],
            "products": [
                {
                    "name": "Copper Oxychloride",
                    "type": "Bactericide",
                    "availability": "Local agrovet",
                },
                {
                    "name": "Kocide (Copper Hydroxide)",
                    "type": "Bactericide",
                    "availability": "Certified agrovet",
                },
            ],
        },
        "physiological": {
            "treatment_steps": [
                {
                    "step": 1,
                    "text": "Identify and correct the underlying stress factor (water, nutrients, temperature)",
                },
                {
                    "step": 2,
                    "text": "Ensure consistent irrigation — avoid wet/dry cycles",
                },
                {
                    "step": 3,
                    "text": "Apply foliar calcium spray if calcium deficiency is suspected",
                },
                {"step": 4, "text": "Mulch around plants to retain soil moisture"},
            ],
            "prevention": [
                "Maintain consistent watering schedule",
                "Test soil and correct nutrient deficiencies before planting",
                "Avoid excessive nitrogen fertilisation",
            ],
            "products": [
                {
                    "name": "Calcium Nitrate",
                    "type": "Foliar Feed",
                    "availability": "Agrovet",
                },
                {
                    "name": "Calmax",
                    "type": "Calcium Supplement",
                    "availability": "Local agrovet",
                },
            ],
        },
        "pest": {
            "treatment_steps": [
                {"step": 1, "text": "Scout field and identify pest species"},
                {"step": 2, "text": "Apply appropriate insecticide or pesticide"},
                {"step": 3, "text": "Remove heavily infested plant parts"},
                {"step": 4, "text": "Monitor regularly for re-infestation"},
            ],
            "prevention": [
                "Use integrated pest management (IPM)",
                "Introduce natural predators where possible",
                "Use pheromone traps for monitoring",
            ],
            "products": [
                {
                    "name": "Consult agrovet for pest ID",
                    "type": "Insecticide",
                    "availability": "Local agrovet",
                }
            ],
        },
    }

    defaults = cause_advice.get(
        cause_type,
        {
            "treatment_steps": [
                {
                    "step": 1,
                    "text": "Consult your local agricultural extension officer for diagnosis confirmation",
                }
            ],
            "prevention": ["Practice good field hygiene", "Monitor crops regularly"],
            "products": [
                {
                    "name": "Consult agrovet",
                    "type": "N/A",
                    "availability": "Local agrovet",
                }
            ],
        },
    )

    symptoms = (
        [explanation]
        if explanation
        else [f"{display} detected — specific symptom data not yet in database"]
    )
    symptoms.append("Consult a local agronomist to confirm diagnosis")

    return {
        "symptoms": symptoms,
        "treatment_steps": defaults["treatment_steps"],
        "products": defaults["products"],
        "severity": severity,
        "prevention": defaults["prevention"],
    }


def get_all_advice() -> dict:
    """Return all advice data."""
    return _load_advice()
