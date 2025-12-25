from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from telethon import TelegramClient

from telegram_intel_scraper.core.config import Settings
from telegram_intel_scraper.core.mongo import get_articles_collection
from telegram_intel_scraper.core.state import load_state, save_state
from telegram_intel_scraper.core.writer import write_jsonl
from telegram_intel_scraper.repositories.articles_repository import ArticlesRepository

from telegram_intel_scraper.providers.telegram import parse_username, iter_channel_messages
from telegram_intel_scraper.providers.text_translate_genai import detect_and_translate_to_english
from telegram_intel_scraper.providers.title_genai import generate_title_genai
from telegram_intel_scraper.providers.title_llm import generate_title_ollama
from telegram_intel_scraper.utils.text import normalize_whitespace, title_heuristic
from telethon.errors import UsernameInvalidError, UsernameNotOccupiedError


def _resolve_title(settings: Settings, text: str) -> str:
    """
    Single source of truth for how titles are generated.
    Priority:
      1) settings.title_provider if set to 'genai' | 'ollama' | 'heuristic'
      2) legacy fallback: enable_llm_titles => use 'ollama'
      3) default: heuristic
    """
    provider = (getattr(settings, "title_provider", "") or "").strip().lower()

    # Backward compatible behavior if title_provider isn't set
    if not provider:
        provider = "ollama" if getattr(settings, "enable_llm_titles", False) else "heuristic"

    if not text:
        return "Telegram message"

    if provider == "genai":
        try:
            return generate_title_genai(text, model=getattr(settings, "genai_model", "gemini-3-flash-preview"))
        except Exception:
            return title_heuristic(text)

    if provider == "ollama":
        try:
            return generate_title_ollama(text, settings.ollama_url, settings.ollama_model)
        except Exception:
            return title_heuristic(text)

    # heuristic (default)
    return title_heuristic(text)


async def run_scrape(settings: Settings) -> None:
    state = load_state(settings.state_file)

    repo: ArticlesRepository | None = None
    if settings.mongo_uri:
        collection = get_articles_collection(
            settings.mongo_uri,
            settings.mongo_db,
            settings.mongo_collection,
        )
        repo = ArticlesRepository(collection)

    async with TelegramClient(
        settings.telegram_session,
        settings.telegram_api_id,
        settings.telegram_api_hash,
    ) as client:
        for url in settings.channels:
            username = parse_username(url)
            last_id = int(state.get(username, {}).get("last_id", 0))
            print(f"[{username}] resume after last_id={last_id}")
            try:
                async for msg in iter_channel_messages(
                    client,
                    username=username,
                    min_id_exclusive=last_id,
                    since=settings.scrape_since,
                    until=settings.scrape_until,
                ):
                    raw_text = (msg.message or "").strip()

                    if not raw_text and not settings.include_empty_text:
                        # Skip purely media posts without captions, etc.
                        continue

                    original_text = normalize_whitespace(raw_text)

                    language = "unknown"
                    text_en = original_text

                    if settings.translate_to_en and original_text:
                        try:
                            language, text_en = detect_and_translate_to_english(
                                original_text,
                                model=settings.genai_model,
                            )
                        except Exception:
                            text_en = original_text

                    title = _resolve_title(settings, text_en)

                    record: Dict[str, Any] = {
                        "title": title,
                        "url": url,
                        "text": text_en,  # canonical text = English
                        "source": username,
                        "scraped_at": msg.date,
                    }

                    if repo is not None:
                        repo.upsert_article(
                            {
                                **record,
                                "text_original": original_text,
                                "text_en": text_en,
                                "language": language,
                                "external_id": msg.id,
                                "telegram_date": msg.date,
                                "telegram_channel": username,
                                "telegram_url": f"https://t.me/{username}/{msg.id}",
                            }
                        )
                    else:
                        # Optional JSONL fallback / audit log
                        write_jsonl(settings.out_jsonl, record)

                    # checkpoint
                    state[username] = {"last_id": msg.id}
                    save_state(settings.state_file, state)
            except (UsernameInvalidError, UsernameNotOccupiedError) as e:
                print(f"[{username}] SKIP: invalid/unknown username ({e.__class__.__name__})")
                continue

            print(f"[{username}] done")
