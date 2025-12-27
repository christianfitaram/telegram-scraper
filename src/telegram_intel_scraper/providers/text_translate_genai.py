from __future__ import annotations
import os
import json
from typing import NamedTuple
from google import genai
from google.genai import types

class ScraperResult(NamedTuple):
    language: str
    english_text: str
    title: str

def _get_api_key() -> str:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

def detect_translate_and_title(
    text: str,
    model: str = "gemini-2.0-flash",
) -> ScraperResult:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")

    cleaned = (text or "").strip()
    if not cleaned:
        return ScraperResult("unknown", "", "Empty Content")

    client = genai.Client(api_key=api_key)

    # Moving instructions to system_instruction for better adherence
    system_instruction = (
        "You are a translation and summarization assistant. "
        "Your task is to detect the language of the input, translate it to English, "
        "and provide a concise, professional title. "
        "You must output valid JSON only."
    )

    # Define the expected schema for the model
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "language": {"type": "STRING"},
            "english_text": {"type": "STRING"},
            "title": {"type": "STRING"}
        },
        "required": ["language", "english_text", "title"]
    }

    try:
        response = client.models.generate_content(
            model=model,
            contents=f"Process this text:\n{cleaned[:4000]}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema
            ),
        )

        # Gemini returns a direct JSON string when response_mime_type is set
        data = json.loads(response.text)
        
        return ScraperResult(
            language=data.get("language", "unknown"),
            english_text=data.get("english_text", cleaned),
            title=data.get("title", "Untitled")
        )

    except Exception as e:
        # Log the error here if you have a logger
        print(f"Error processing text: {e}")
        return ScraperResult("unknown", cleaned, "Untitled")