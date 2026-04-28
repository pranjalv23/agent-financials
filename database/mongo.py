import logging
import os
from datetime import datetime, timezone

from bson import ObjectId
from agent_sdk.database.mongo import BaseMongoDatabase
from agent_sdk.database.gridfs_mixin import GridFSMixin

logger = logging.getLogger("agent_financials.mongo")

_DB_NAME = os.getenv("MONGO_DB_NAME", "agent_financials")


class MongoDB(GridFSMixin, BaseMongoDatabase):

    @classmethod
    def db_name(cls) -> str:
        return _DB_NAME

    @classmethod
    def _db(cls):
        return cls.get_client()[cls.db_name()]

    @classmethod
    def _files(cls):
        return cls._db()["files"]

    @classmethod
    def watchlist_collection(cls):
        return cls._db()["watchlists"]

    @classmethod
    def profile_collection(cls):
        return cls._db()["profiles"]

    @classmethod
    def holdings_collection(cls):
        return cls._db()["holdings"]

    @classmethod
    async def get_profile(cls, user_id: str) -> dict | None:
        return await cls.profile_collection().find_one({"user_id": user_id}, {"_id": 0})

    @classmethod
    async def upsert_profile(cls, user_id: str, data: dict) -> None:
        await cls.profile_collection().update_one(
            {"user_id": user_id},
            {
                "$set": {**data, "updated_at": datetime.now(timezone.utc)},
                "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
            },
            upsert=True,
        )

    @staticmethod
    def _normalize_tickers(raw_tickers: list) -> list[dict]:
        """Accept both legacy list[str] and new list[dict] ticker formats."""
        result = []
        for t in raw_tickers:
            if isinstance(t, str):
                result.append({"symbol": t, "entry_price": None, "added_at": None})
            elif isinstance(t, dict):
                result.append({
                    "symbol": t.get("symbol", ""),
                    "entry_price": t.get("entry_price"),
                    "added_at": t.get("added_at"),
                })
        return result

    @classmethod
    async def create_watchlist(cls, user_id: str, name: str, tickers: list) -> str:
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
        try:
            val = ObjectId(watchlist_id)
        except Exception:
            return None
        doc = await cls.watchlist_collection().find_one({"_id": val, "user_id": user_id})
        if doc:
            doc["id"] = str(doc.pop("_id"))
        return doc

    @classmethod
    async def update_watchlist(cls, user_id: str, watchlist_id: str, name: str | None = None, tickers: list | None = None) -> bool:
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
        try:
            val = ObjectId(watchlist_id)
        except Exception:
            return False
        result = await cls.watchlist_collection().delete_one({"_id": val, "user_id": user_id})
        return result.deleted_count > 0

    # ── Holdings ──

    @classmethod
    async def create_holding(cls, user_id: str, ticker: str, quantity: float, avg_buy_price: float) -> str:
        now = datetime.now(timezone.utc)
        doc = {
            "user_id": user_id,
            "ticker": ticker.upper(),
            "quantity": quantity,
            "avg_buy_price": avg_buy_price,
            "created_at": now,
            "updated_at": now,
        }
        result = await cls.holdings_collection().insert_one(doc)
        return str(result.inserted_id)

    @classmethod
    async def get_holdings(cls, user_id: str) -> list[dict]:
        cursor = cls.holdings_collection().find({"user_id": user_id})
        docs = await cursor.to_list(length=1000)
        for doc in docs:
            doc["id"] = str(doc.pop("_id"))
        return docs

    @classmethod
    async def update_holding(cls, user_id: str, holding_id: str, quantity: float | None = None, avg_buy_price: float | None = None) -> bool:
        updates = {"updated_at": datetime.now(timezone.utc)}
        if quantity is not None:
            updates["quantity"] = quantity
        if avg_buy_price is not None:
            updates["avg_buy_price"] = avg_buy_price
        try:
            val = ObjectId(holding_id)
        except Exception:
            return False
        result = await cls.holdings_collection().update_one(
            {"_id": val, "user_id": user_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    @classmethod
    async def delete_holding(cls, user_id: str, holding_id: str) -> bool:
        try:
            val = ObjectId(holding_id)
        except Exception:
            return False
        result = await cls.holdings_collection().delete_one({"_id": val, "user_id": user_id})
        return result.deleted_count > 0

    @classmethod
    async def ensure_indexes(cls) -> None:
        await super().ensure_indexes()
        await cls.profile_collection().create_index("user_id", unique=True)
        await cls._files().create_index("created_at", expireAfterSeconds=2_592_000)
        await cls._db()["fs.files"].create_index("uploadDate", expireAfterSeconds=2_592_000)
        await cls.watchlist_collection().create_index("user_id")
        await cls.holdings_collection().create_index([("user_id", 1), ("ticker", 1)], unique=True)
