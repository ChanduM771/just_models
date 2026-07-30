"""
Gemini Vision API — open-ended plant disease diagnosis.
Diagnoses freely, then maps to known DB entries where possible.
Falls back to returning the real diagnosis with generic advice if unrecognised.
"""

import requests
import json
import base64
import logging

from config import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

# All disease keys currently in disease_advice.json
KNOWN_DISEASES = {
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_healthy",
    "Corn_(maize)_Gray_leaf_spot",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Maize_Lethal_Necrosis",
    "Kale_Black_Rot",
    "Kale_Downy_Mildew",
    "Kale_Aphids",
    "Kale_Cabbage_Worm",
    "Fall_Armyworm",
    "Potato_Late_blight",
    "Onion_Purple_Blotch",
    "Maize_Rust",
}

DISEASE_PROMPT = """You are an expert plant pathologist specializing in East African agriculture (Kenya, Nakuru region).

Analyze this plant image carefully and independently — base your diagnosis ONLY on what you actually observe (lesion shape, color, margin, distribution, location on plant), not on assumptions from the crop label alone. Diagnose ANY disease, disorder, or pest damage you observe — do NOT restrict yourself to a fixed list. Physiological disorders (e.g. Blossom End Rot, sunscald, nutrient deficiency), fungal, bacterial, viral, and pest damage are all valid diagnoses.

Crops of interest: Maize, Tomato, Kale (Sukuma Wiki), Potato, Onion

If what you see matches one of these known conditions, use its EXACT class_name string (copy it verbatim, including underscores/parentheses/case):
Tomato_Early_blight, Tomato_Late_blight, Tomato_Bacterial_spot, Tomato_Yellow_Leaf_Curl_Virus, Tomato_healthy,
Corn_(maize)_Gray_leaf_spot, Corn_(maize)_Northern_Leaf_Blight, Maize_Lethal_Necrosis, Maize_Rust,
Kale_Black_Rot, Kale_Downy_Mildew, Kale_Aphids, Kale_Cabbage_Worm, Fall_Armyworm, Potato_Late_blight, Onion_Purple_Blotch.

If what you see does NOT match any of those, do not force-fit it — give your own accurate Title_Case class_name reflecting your actual observation, prefixed with the crop (e.g. Tomato_Blossom_End_Rot).

IMPORTANT: Return ONLY valid JSON with this EXACT structure (no markdown, no extra text):
{
  "class_name": "Short_Underscored_Name",
  "display": "Human-readable diagnosis name",
  "crop": "Crop name (Maize/Tomato/Kale/Potato/Onion/Unknown)",
  "confidence": 0.0-1.0,
  "is_healthy": true/false,
  "severity": "Low/Medium/High/Critical",
  "cause_type": "fungal/bacterial/viral/pest/physiological/unknown",
  "brief_explanation": "One sentence explaining what you see and why"
}

Rules:
- Diagnose what you actually SEE in the image, not what fits a predefined list, and not what the user's selected crop_type primes you to expect — if the image looks like a different crop or condition, say so.
- Set confidence based on how unambiguous the visual evidence is. Reserve 0.85+ for textbook-clear presentations with no plausible alternative; use 0.4-0.7 when lesion shape/location is atypical or multiple causes are plausible.
- If the plant is healthy use class_name "Healthy", is_healthy true, confidence 0.9, severity "Low".
- If the image is not a plant use class_name "Unknown", crop "Unknown", confidence 0.0.
- class_name must use underscores and match the crop prefix."""


def predict_gemini_vision(
    image_bytes: bytes, crop_type: str = None, mime_type: str = "image/jpeg"
) -> dict:
    """
    Call Gemini Vision API for open-ended plant disease detection.
    Returns dict in same format as model.predict(): {class_name, confidence, display, crop, emoji}
    If the diagnosis is not in KNOWN_DISEASES, sets in_database=False so the caller
    can show the real diagnosis with generic advice instead of forcing a wrong match.
    """
    key = GEMINI_API_KEY
    if not key:
        logger.error("No Gemini API key configured")
        return None

    try:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={key}"
        )

        prompt = DISEASE_PROMPT
        if crop_type:
            prompt += f"\n\nThe user selected crop type: {crop_type}. Use this as context but still diagnose what you actually see."

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 400,
            },
        }

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # Strip markdown fences if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            parsed = json.loads(text.strip())

            emoji_map = {
                "Maize": "🌽",
                "Tomato": "🍅",
                "Kale": "🥬",
                "Potato": "🥔",
                "Onion": "🧅",
                "Unknown": "❓",
            }

            crop = parsed.get("crop", "Unknown")
            class_name = parsed.get("class_name", "Unknown")
            confidence_val = parsed.get("confidence", 0.5)
            in_database = class_name in KNOWN_DISEASES

            return {
                "class_name": class_name,
                "confidence": round(confidence_val * 100, 1),
                "disease_name": class_name,
                "display": parsed.get("display", class_name.replace("_", " ")),
                "crop": crop,
                "emoji": emoji_map.get(crop, "❓"),
                "source": "gemini_vision",
                "severity": parsed.get("severity", "Medium"),
                "cause_type": parsed.get("cause_type", "unknown"),
                "brief_explanation": parsed.get("brief_explanation", ""),
                "in_database": in_database,
            }

        elif response.status_code == 429:
            logger.warning("Gemini rate limited (429)")
            return None
        elif response.status_code == 503:
            logger.warning("Gemini overloaded (503)")
            return None
        else:
            logger.warning(
                f"Gemini error: {response.status_code} — {response.text[:200]}"
            )
            return None

    except Exception as e:
        logger.error(f"Gemini Vision failed: {e}")
        return None
