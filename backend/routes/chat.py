"""
AgroVision AI - /api/chat endpoint
Sends farmer's question + diagnosis context to Gemini and returns a reply.
"""

from fastapi import APIRouter, Header
from pydantic import BaseModel
from typing import List, Optional
import httpx
import re
import json

from config import GEMINI_API_KEY, GEMINI_MODEL
from db.database import get_connection
from utils.auth_utils import get_active_user

router = APIRouter()


# ---------------------------------------------------------------------------
# Image search helpers
# ---------------------------------------------------------------------------
_IMG_PATTERNS = re.compile(
    r"\b(show me|image of|picture of|photo of|what does .+ look like|"
    r"how does .+ look|appearance of|identify this|search for image|search images|"
    r"search the web|search web|see the web|looks like|"
    r"picha ya|nionyeshe picha|onyesha picha)\b",
    re.IGNORECASE,
)

_SUBJECT_STRIP = re.compile(
    r"(?:show me(?: an? image of)?|image of|picture of|photo of|search images? of|"
    r"search the web for|see the web for|search web for|"
    r"picha ya|nionyeshe picha ya|onyesha picha ya)\s+",
    re.IGNORECASE,
)

_IMG_SYSTEM_NOTE = (
    "SYSTEM RULE (override all other instructions): "
    "When the user asks for images, pictures, or photos of ANY product, disease, or item, "
    "you MUST respond with ONLY this exact sentence and nothing else: "
    "'Sure, here are some images I found for you.' "
    "Do NOT describe the product. Do NOT apologize. Do NOT add extra text. "
    "The backend will automatically append a Google Images search link after your reply.\n\n"
)


def _wants_image(text: str) -> bool:
    return bool(_IMG_PATTERNS.search(text))


def _build_search_query(message: str, context: str) -> str:
    import re as _re

    # Try stripping common image request trigger phrases first
    cleaned = _SUBJECT_STRIP.sub("", message).strip(" ?.")

    # Generic/vague single-word terms
    _vague = _re.compile(
        r"^(medicine|medicines|dawa|fungicide|fungicides|product|products|bidhaa|"
        r"image|images|picha|photo|photos|it|them|these|hizo|hii)$",
        _re.IGNORECASE,
    )

    # If what's left is short/vague OR is still a long conversational sentence
    # (>4 words means we didn't successfully isolate a product/subject name),
    # use the diagnosis context instead.
    is_sentence = len(cleaned.split()) > 4
    is_vague = len(cleaned) < 4 or bool(_vague.match(cleaned))

    if is_vague or is_sentence:
        disease_m = _re.search(r"Disease:([^,]+)", context)
        products_m = _re.search(r"Products:([^.;]+)", context)
        disease = disease_m.group(1).strip() if disease_m else ""
        products = products_m.group(1).strip() if products_m else ""
        if products:
            first_product = products.split(",")[0].strip()
            return f"{first_product} {disease} fungicide Kenya".strip()
        if disease:
            return f"{disease} treatment fungicide Kenya"
        return message  # last resort: raw message

    return cleaned


def _ddg_images(query: str, max_results: int = 3) -> list[dict]:
    import time

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("[DDG images] neither ddgs nor duckduckgo_search installed")
            return []
    for attempt in range(3):
        try:
            results = DDGS().images(query, max_results=max_results)
            return [
                {
                    "title": r.get("title", ""),
                    "image": r.get("image", ""),
                    "source": r.get("url", ""),
                }
                for r in results
                if r.get("image")
            ]
        except Exception as e:
            msg = str(e)
            if "403" in msg or "Ratelimit" in msg:
                time.sleep(2**attempt)
                continue
            print(f"[DDG images] {e}")
            return []
    print("[DDG images] rate limited after 3 retries")
    return []


def _build_google_img_block(query: str, lang: str) -> dict:
    """
    Fetch up to 3 preview image URLs from DDG.
    Returns dict: { 'text': markdown_link, 'preview_urls': [url, ...] }
    """
    import urllib.parse

    query_encoded = urllib.parse.quote_plus(query)
    google_link = f"https://www.google.com/search?tbm=isch&q={query_encoded}"
    label = "Google ಚಿತ್ರಗಳನ್ನು ಹುಡುಕಿ" if lang == "kn" else "Search Google Images"

    preview_urls: list[str] = []
    try:
        imgs = _ddg_images(query, max_results=3)
        preview_urls = [img["image"] for img in imgs if img.get("image")][:3]
    except Exception:
        pass

    return {
        "text": f"\n\n[{label}]({google_link})",
        "preview_urls": preview_urls,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ChatMsg(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    context: str
    history: Optional[List[ChatMsg]] = []
    detection_id: Optional[int] = None
    lang: Optional[str] = "en"  # "en" | "sw"


_ERR_QUOTA = {
    "en": "Sorry, our AI is currently receiving too many requests. Please try again in a few minutes.",
    "sw": "Samahani, AI yetu inapokea maombi mengi sana sasa hivi. Tafadhali jaribu tena baadaye.",
    "kn": "ಕ್ಷಮಿಸಿ, ನಮ್ಮ AI ಗೆ ಈಗ ತುಂಬಾ ವಿನಂತಿಗಳು ಬರುತ್ತಿವೆ. ದಯವಿಟ್ಟು ಕೆಲವು ನಿಮಿಷಗಳ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
}
_ERR_NETWORK = {
    "en": "Sorry, I couldn't reach the AI right now. Please try again shortly.",
    "sw": "Samahani, sikuweza kufikia AI sasa hivi. Tafadhali jaribu tena baadaye.",
    "kn": "ಕ್ಷಮಿಸಿ, ಈಗ AI ಅನ್ನು ಸಂಪರ್ಕಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಸಮಯದ ನಂತರ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
}


def get_optional_user(authorization: str = None):
    if not authorization:
        return None
    return get_active_user(authorization)["user_id"]


@router.post("/api/chat")
async def chat(req: ChatRequest, authorization: Optional[str] = Header(None)):
    user_id = get_optional_user(authorization)
    lang = req.lang or "en"
    wants_img = _wants_image(req.message)

    if user_id and req.detection_id:
        conn = get_connection()
        owned = conn.execute(
            "SELECT 1 FROM detections WHERE id = ? AND user_id = ?",
            (req.detection_id, user_id),
        ).fetchone()
        conn.close()
        if not owned:
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Detection does not belong to this user")

    # Save user message to DB
    if user_id and req.detection_id:
        try:
            conn = get_connection()
            conn.execute(
                "INSERT INTO chat_messages (user_id, detection_id, role, content) VALUES (?, ?, ?, ?)",
                (user_id, req.detection_id, "user", req.message),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Chat] Could not save user message: {e}")

    # Build Gemini contents array
    # Inject image-system note into context when user wants images
    language_name = {"kn": "Kannada"}.get(lang, "English")
    lang_instruction = f" CRITICAL INSTRUCTION: You MUST reply entirely in {language_name}. Ignore the language of the diagnosis context if it differs."
    context_text = (
        (_IMG_SYSTEM_NOTE if wants_img else "") + req.context + lang_instruction
    )

    contents = []
    contents.append({"role": "user", "parts": [{"text": context_text}]})
    contents.append(
        {
            "role": "model",
            "parts": [
                {
                    "text": "Understood. I'm ready to answer questions about this diagnosis."
                }
            ],
        }
    )

    for m in req.history or []:
        role = "model" if m.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": m.content}]})

    contents.append({"role": "user", "parts": [{"text": req.message}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    try:
        payload = {"contents": contents}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 429:
                return {"reply": _ERR_QUOTA.get(lang, _ERR_QUOTA["en"])}
            resp.raise_for_status()
            data = resp.json()

            reply = ""
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    reply = part["text"]
                    break
            if not reply:
                reply = _ERR_NETWORK.get(lang, _ERR_NETWORK["en"])

            # Google Image Search link + DDG preview thumbnails
            has_images = False
            preview_urls: list[str] = []
            if wants_img:
                query = _build_search_query(req.message, req.context)
                img_block = _build_google_img_block(query, lang)
                reply = reply + img_block["text"]
                preview_urls = img_block["preview_urls"]
                has_images = True

            # Save assistant reply to DB
            if user_id and req.detection_id:
                try:
                    conn = get_connection()
                    conn.execute(
                        "INSERT INTO chat_messages (user_id, detection_id, role, content, preview_urls) VALUES (?, ?, ?, ?, ?)",
                        (
                            user_id,
                            req.detection_id,
                            "assistant",
                            reply,
                            json.dumps(preview_urls) if preview_urls else None,
                        ),
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print(f"[Chat] Could not save assistant message: {e}")

            return {
                "reply": reply,
                "has_images": has_images,
                "preview_urls": preview_urls,
            }
    except Exception as e:
        return {"reply": _ERR_NETWORK.get(lang, _ERR_NETWORK["en"]), "error": str(e)}
