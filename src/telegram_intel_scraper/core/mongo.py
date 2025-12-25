from __future__ import annotations

from pymongo import MongoClient
from pymongo.collection import Collection


def get_articles_collection(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
) -> Collection:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]
