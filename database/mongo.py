import logging
import os
from datetime import datetime, timezone

from agent_sdk.database.mongo import BaseMongoDatabase

logger = logging.getLogger("agent_financials.mongo")

_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_financials")


class MongoDB(BaseMongoDatabase):
    @classmethod
    def db_name(cls) -> str:
        return _DB_NAME

    @classmethod
    def watchlist_collection(cls):
        return cls.get_client()[cls.db_name()]["watchlists"]

    @classmethod
    async def create_watchlist(cls, user_id: str, name: str, tickers: list[str]) -> str:
        doc = {
            "user_id": user_id,
            "name": name,
            "tickers": tickers,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = await cls.watchlist_collection().insert_one(doc)
        return str(result.inserted_id)

    @classmethod
    async def get_watchlists(cls, user_id: str) -> list[dict]:
        cursor = cls.watchlist_collection().find({"user_id": user_id})
        docs = await cursor.to_list(length=100)
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs

    @classmethod
    async def get_watchlist(cls, user_id: str, watchlist_id: str) -> dict | None:
        from bson import ObjectId
        try:
            val = ObjectId(watchlist_id)
        except Exception:
            return None
        doc = await cls.watchlist_collection().find_one({"_id": val, "user_id": user_id})
        if doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    @classmethod
    async def update_watchlist(cls, user_id: str, watchlist_id: str, name: str | None = None, tickers: list[str] | None = None) -> bool:
        from bson import ObjectId
        updates = {"updated_at": datetime.now(timezone.utc)}
        if name is not None:
            updates["name"] = name
        if tickers is not None:
            updates["tickers"] = tickers
        try:
            val = ObjectId(watchlist_id)
        except Exception:
            return False
        result = await cls.watchlist_collection().update_one(
            {"_id": val, "user_id": user_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    @classmethod
    async def delete_watchlist(cls, user_id: str, watchlist_id: str) -> bool:
        from bson import ObjectId
        try:
            val = ObjectId(watchlist_id)
        except Exception:
            return False
        result = await cls.watchlist_collection().delete_one({"_id": val, "user_id": user_id})
        return result.deleted_count > 0

    @classmethod
    async def ensure_indexes(cls) -> None:
        await super().ensure_indexes()
