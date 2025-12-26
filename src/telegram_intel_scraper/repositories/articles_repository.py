from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError


class ArticlesRepository:
    def __init__(self, collection: Collection):
        self._collection = collection
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # Deduplication index: same telegram channel + telegram message id
        self._collection.create_index(
            [("telegram_channel", ASCENDING), ("external_id", ASCENDING)],
            unique=True,
            name="uniq_channel_external_id",
        )

        # Legacy index on source + external_id (still useful if source diverges from telegram_channel)
        self._collection.create_index(
            [("source", ASCENDING), ("external_id", ASCENDING)],
            unique=True,
            name="uniq_source_external_id",
        )

        self._collection.create_index("scraped_at", name="scraped_at_idx")

    def upsert_article(self, doc: Dict[str, Any]) -> bool:
        """
        Insert the document if the telegram_channel + external_id combo is new.
        Returns True when a new document was written; False if skipped.
        """
        now = datetime.utcnow()
        identifier = {
            "telegram_channel": doc["telegram_channel"],
            "external_id": doc["external_id"],
        }

        if self._collection.find_one(identifier, {"_id": 1}):
            return False

        insert_doc = {
            **doc,
            "created_at": now,
            "updated_at": now,
        }

        try:
            self._collection.insert_one(insert_doc)
            return True
        except DuplicateKeyError:
            return False
