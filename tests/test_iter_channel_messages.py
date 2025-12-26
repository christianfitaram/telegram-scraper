from __future__ import annotations

from dataclasses import dataclass
import unittest
from datetime import datetime, timedelta, timezone

from telegram_intel_scraper.providers.telegram import iter_channel_messages


@dataclass
class FakeMessage:
    id: int
    date: datetime


class FakeClient:
    def __init__(self, pages: list[list[FakeMessage]]):
        self.pages = pages
        self.calls = 0

    async def get_entity(self, username: str) -> str:
        return username

    async def get_messages(self, entity: str, limit: int, offset_id: int) -> list[FakeMessage]:
        if self.calls >= len(self.pages):
            return []
        page = self.pages[self.calls]
        self.calls += 1
        return page


class IterChannelMessagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_since_filter_stops_after_older_message(self) -> None:
        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=10)

        # Page returns newest->oldest to mimic Telethon behavior.
        page = [
            FakeMessage(id=3, date=now - timedelta(minutes=1)),
            FakeMessage(id=2, date=now - timedelta(minutes=5)),
            FakeMessage(id=1, date=now - timedelta(minutes=20)),
        ]

        client = FakeClient(pages=[page, page])

        seen_ids: list[int] = []
        async for message in iter_channel_messages(
            client,
            username="test_channel",
            min_id_exclusive=0,
            since=since,
            until=None,
        ):
            seen_ids.append(message.id)

        self.assertEqual(seen_ids, [2, 3])
        self.assertEqual(client.calls, 1)
