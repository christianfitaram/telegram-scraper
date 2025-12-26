from __future__ import annotations

from datetime import datetime
from typing import AsyncIterator, Optional, List

from telethon import TelegramClient
from telethon.tl.types import Message


def parse_username(tme_url: str) -> str:
    return tme_url.rstrip("/").split("/")[-1]


async def iter_channel_messages(
    client: TelegramClient,
    username: str,
    min_id_exclusive: int = 0,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> AsyncIterator[Message]:
    """
    Iterates messages in ascending order (old -> new) but paginates from newest -> oldest.
    Filters:
      - min_id_exclusive: resume checkpoint
      - since: only messages with msg.date >= since (UTC aware)
      - until: only messages with msg.date <= until (UTC aware)
    """
    entity = await client.get_entity(username)

    fetched = 0
    offset_id = 0
    page_size = 200

    print(f"[{username}] iter_channel_messages since={since} until={until} min_id_exclusive={min_id_exclusive}")
    while True:
        msgs: List[Message] = await client.get_messages(entity, limit=page_size, offset_id=offset_id)
        print(f"[{username}] fetched page {offset_id} count={len(msgs)}")
        if not msgs:
            break

        # msgs is newest->oldest within this page
        offset_id = msgs[-1].id

        # Keep only messages beyond last_id checkpoint
        new_msgs = [m for m in msgs if m and m.id and m.id > min_id_exclusive]

        # If we paged into older-than-last_id, we can stop after processing new ones
        stop_after = msgs[-1].id <= min_id_exclusive

        stop_after_since = False
        # Yield in ascending order for deterministic downstream processing
        print(f"[{username}] new_msgs ids={[m.id for m in new_msgs]}")
        for m in sorted(new_msgs, key=lambda x: x.id):
            # Telethon msg.date is typically timezone-aware UTC datetime
            if not m.date:
                continue

            # If message is newer than "until", skip it (but keep going; older ones may match)
            if until is not None and m.date > until:
                print(f"[{username}] skip {m.id} newer-than-until ({m.date})")
                continue

            # If message is older than "since", we can stop entirely because pagination goes older from here
            if since is not None and m.date < since:
                print(f"[{username}] stop: {m.id} older-than-since ({m.date})")
                stop_after_since = True
                continue

            yield m
            fetched += 1
            if limit is not None and fetched >= limit:
                return

        if stop_after_since:
            return

        if stop_after:
            break
