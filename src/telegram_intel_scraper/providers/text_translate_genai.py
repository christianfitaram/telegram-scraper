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
    """Retrieves the API key from environment variables."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

def detect_translate_and_title(
    text: str,
    model: str = "gemini-2.0-flash",
) -> ScraperResult:
    """
    Uses Gemini 2.0 Flash to detect language, translate text to English,
    and generate a professional title using structured JSON output.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")

    # Clean and validate input
    cleaned = (text or "").strip()
    if not cleaned:
        return ScraperResult("unknown", "", "Empty Content")

    client = genai.Client(api_key=api_key)

    # 1. System Instruction: Defines the 'Persona' and 'Rules'
    # Separating this from the content improves instruction following.
    system_instruction = (
        "You are a specialized translation and metadata extraction assistant. "
        "Your goal is to process the provided text to: \n"
        "1. Detect the original language.\n"
        "2. Translate the full text into clear, fluent English.\n"
        "3. Create a concise, professional title for the content.\n"
        "You must respond ONLY with valid JSON matching the provided schema."
    )

    # 2. Response Schema: Forces the model to return structured data
    response_schema = types.Schema(
        type="OBJECT",
        properties={
            "language": types.Schema(type="STRING"),
            "english_text": types.Schema(type="STRING"),
            "title": types.Schema(type="STRING")
        },
        required=["language", "english_text", "title"]
    )

    try:
        # 3. API Call with Gemini 2.0 Flash
        response = client.models.generate_content(
            model=model,
            contents=f"Process this text:\n\n{cleaned[:12000]}", # Flash handles large chunks easily
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )

        # 4. Parse response
        if not response.text:
            return ScraperResult("unknown", cleaned, "Content Filtered by Safety Policy")

        data = json.loads(response.text)
        
        return ScraperResult(
            language=data.get("language", "unknown"),
            english_text=data.get("english_text", cleaned),
            title=data.get("title", "Untitled")
        )

    except Exception as e:
        # Log error and return fallback
        print(f"Error processing text: {str(e)}")
        return ScraperResult("unknown", cleaned, "Processing Error")

# --- Example Usage ---
if __name__ == "__main__":
    sample_text = "El rápido desarrollo de la inteligencia artificial está cambiando el mundo."
    result = detect_translate_and_title(sample_text)
    
    print(f"Detected: {result.language}")
    print(f"Title: {result.title}")
    print(f"English: {result.english_text}")