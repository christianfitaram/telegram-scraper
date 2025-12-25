from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection


class ArticlesRepository:
    def __init__(self, collection: Collection):
        self._collection = collection
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        # Deduplication index: same source + telegram message id
        self._collection.create_index(
            [("source", ASCENDING), ("external_id", ASCENDING)],
            unique=True,
            name="uniq_source_external_id",
        )

        self._collection.create_index("scraped_at", name="scraped_at_idx")

    def upsert_article(self, doc: Dict[str, Any]) -> None:
        """
        Idempotent insert/update.
        """
        now = datetime.utcnow()

        self._collection.update_one(
            {
                "source": doc["source"],
                "external_id": doc["external_id"],
            },
            {
                "$set": {
                    **doc,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )
