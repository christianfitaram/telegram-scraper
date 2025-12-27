from __future__ import annotations

import os
from typing import Optional

from telegram_intel_scraper.utils.text import normalize_whitespace


def _get_api_key() -> str:
    # Prefer the official env var name expected by google-genai.
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""


def _get_genai_client(api_key: str):
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("google-genai is not installed. Run: poetry add google-genai") from exc
    return genai.Client(api_key=api_key)


def generate_title_genai(
    text: str,
    model: str = "gemini-2.0-flash",
    timeout: int = 60,
) -> str:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment/.env")

    cleaned = normalize_whitespace(text)
    if not cleaned:
        return "Telegram message"

    prompt = (
        "Create a descriptive but short title (max 12 words) for the following Telegram message.\n"
        "Return ONLY the title, no quotes, no punctuation at the end unless necessary.\n\n"
        f"Message:\n{cleaned[:2500]}"
    )

    client = _get_genai_client(api_key)

    # google-genai does not always expose a universal request timeout argument on generate_content;
    # keep 'timeout' param for interface symmetry; network timeouts are handled by underlying transport.
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    title = (getattr(response, "text", "") or "").strip()
    return title[:120] if title else "Telegram message"
