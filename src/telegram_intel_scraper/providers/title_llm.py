from __future__ import annotations
import requests


def generate_title_ollama(text: str, ollama_url: str, ollama_model: str) -> str:
    prompt = (
        "Generate a short news-style title (max 12 words) for this Telegram message. "
        "Return ONLY the title.\n\n"
        f"Message:\n{text[:2500]}"
    )

    r = requests.post(
        ollama_url,
        json={"model": ollama_model, "prompt": prompt, "stream": False},
        timeout=60,
    )
    r.raise_for_status()
    title = (r.json().get("response") or "").strip()
    return title[:120] if title else "Telegram message"

