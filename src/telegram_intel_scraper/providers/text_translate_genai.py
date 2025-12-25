from __future__ import annotations

import os
from typing import Tuple

from telegram_intel_scraper.utils.text import normalize_whitespace


def _get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def _get_genai_client(api_key: str):
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed") from exc
    return genai.Client(api_key=api_key)


def detect_and_translate_to_english(
    text: str,
    model: str = "gemini-3-flash-preview",
) -> Tuple[str, str]:
    """
    Returns:
      (language_code, english_text)

    If text is already English, english_text == original text.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "unknown", cleaned

    prompt = (
        "Detect the language of the following text.\n"
        "If it is NOT English, translate it to English.\n\n"
        "Return ONLY a valid JSON object with this schema:\n"
        '{ "language": "<ISO 639-1 code>", "english_text": "<text>" }\n\n'
        f"Text:\n{cleaned[:3000]}"
    )

    client = _get_genai_client(api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    raw = (getattr(response, "text", "") or "").strip()

    try:
        import json
        data = json.loads(raw)
        language = data.get("language", "unknown")
        english_text = data.get("english_text", cleaned)
        return language, normalize_whitespace(english_text)
    except Exception:
        # Fail-safe: assume English
        return "unknown", cleaned
