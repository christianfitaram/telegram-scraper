from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from telegram_intel_scraper.core.config import Settings
from telegram_intel_scraper.core.scrape import run_scrape


def _load_env() -> None:
    # Always load .env from repo root (two levels up from this file: src/telegram_intel_scraper/main.py)
    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env")


_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)


def _parse_since_until(value: Optional[str]) -> Optional[datetime]:
    """
    Accepts:
      - ISO 8601 UTC: 2025-12-25T00:00:00Z (or with +00:00)
      - Durations: 24h, 7d, 30m, 15s, 2w
    Returns timezone-aware UTC datetime.
    """
    if not value:
        return None

    v = value.strip()

    # Duration form
    m = _DURATION_RE.match(v)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()

        seconds = 0
        if unit == "s":
            seconds = n
        elif unit == "m":
            seconds = n * 60
        elif unit == "h":
            seconds = n * 60 * 60
        elif unit == "d":
            seconds = n * 60 * 60 * 24
        elif unit == "w":
            seconds = n * 60 * 60 * 24 * 7

        return datetime.now(timezone.utc) - timedelta(seconds=seconds)

    # ISO form
    try:
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Treat naive input as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception as exc:
        raise SystemExit(
            f"Invalid datetime/duration '{value}'. Use ISO like 2025-12-25T00:00:00Z or duration like 24h/7d/30m."
        ) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Telegram intel scraper")
    p.add_argument("--since", type=str, default=None, help="Start time (ISO) or duration like 24h, 7d, 30m")
    p.add_argument("--until", type=str, default=None, help="End time (ISO) or duration like 12h, or ISO datetime")
    return p


def main() -> None:
    _load_env()
    settings = Settings.from_env()

    args = _build_arg_parser().parse_args()

    since = _parse_since_until(args.since) if args.since else settings.scrape_since
    until = _parse_since_until(args.until) if args.until else settings.scrape_until

    # Sanity: if both are set, enforce since <= until
    if since and until and since > until:
        raise SystemExit("--since must be <= --until")

    settings = replace(settings, scrape_since=since, scrape_until=until)

    asyncio.run(run_scrape(settings))


if __name__ == "__main__":
    main()
