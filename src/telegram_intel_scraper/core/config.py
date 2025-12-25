from __future__ import annotations

from dataclasses import dataclass
import os
from typing import List
from typing import Optional
from datetime import datetime

DEFAULT_CHANNEL_URLS: List[str] = [
    "https://t.me/anna_news",
    "https://t.me/The_Wrong_Side",
    "https://t.me/rnintel",
    "https://t.me/CIG_telegram",
    "https://t.me/BellumActaNews",
    "https://t.me/israelportalnoti",
    "https://t.me/unhispano",
    "https://t.me/Latam24hrs",
    "https://t.me/nayaforiraq",
    "https://t.me/me_observer_TG",
    "https://t.me/DDGeopolitics",
    "https://t.me/entre_guerras",
    "https://t.me/Suriyak_map",
]


def _split_lines(value: str) -> List[str]:
    # Supports newline or comma separated
    raw = [v.strip() for v in value.replace(",", "\n").splitlines()]
    return [v for v in raw if v]

def _parse_utc_iso(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    # Accept ...Z
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Settings:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session: str

    out_jsonl: str
    state_file: str

    enable_llm_titles: bool
    ollama_url: str
    ollama_model: str

    channels: List[str]
    include_empty_text: bool

    mongo_uri: Optional[str]
    mongo_db: str
    mongo_collection: str

    title_provider: str
    genai_model: str

       
    scrape_since: Optional[datetime]
    scrape_until: Optional[datetime]
    translate_to_en: bool


    @staticmethod
    def from_env() -> "Settings":
        api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
        api_hash = os.getenv("TELEGRAM_API_HASH", "")
        session = os.getenv("TELEGRAM_SESSION", "telegram_scraper_session")

        if not api_id or not api_hash:
            raise RuntimeError("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH")

        channels_env = os.getenv("TELEGRAM_CHANNELS", "").strip()
        channels = _split_lines(channels_env) if channels_env else DEFAULT_CHANNEL_URLS

        return Settings(
            telegram_api_id=api_id,
            telegram_api_hash=api_hash,
            telegram_session=session,
            out_jsonl=os.getenv("OUT_JSONL", "output.jsonl"),
            state_file=os.getenv("STATE_FILE", "telegram_state.json"),
            enable_llm_titles=os.getenv("ENABLE_LLM_TITLES", "0").lower() in ("1", "true", "yes"),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            channels=channels,
            include_empty_text=os.getenv("INCLUDE_EMPTY_TEXT", "0").lower() in ("1", "true", "yes"),
            mongo_uri=os.getenv("MONGO_URI"),
            mongo_db=os.getenv("MONGO_DB", "agents"),
            mongo_collection=os.getenv("MONGO_COLLECTION", "articles"),
            title_provider=os.getenv("TITLE_PROVIDER", "heuristic").strip().lower(),  # heuristic | ollama | genai
            genai_model=os.getenv("GENAI_MODEL", "gemini-3-flash-preview"),
            scrape_since=_parse_utc_iso(os.getenv("SCRAPE_SINCE")),
            scrape_until=_parse_utc_iso(os.getenv("SCRAPE_UNTIL")),
            translate_to_en=os.getenv("TRANSLATE_TO_EN", "0").lower() in ("1", "true", "yes"),

        )
