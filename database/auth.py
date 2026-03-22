import logging
import os
import uuid
from datetime import datetime, timezone

import bcrypt
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("agent_financials.auth")

_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
_AUTH_DB = "agent_auth"  # shared across all agents


class AuthDB:
    _client: AsyncIOMotorClient | None = None

    @classmethod
    def _collection(cls):
        if cls._client is None:
            cls._client = AsyncIOMotorClient(_MONGO_URI)
        return cls._client[_AUTH_DB]["users"]

    @classmethod
    async def ensure_index(cls):
        await cls._collection().create_index("email", unique=True)

    @classmethod
    async def create_user(cls, email: str, password: str) -> dict:
        user = {
            "user_id": uuid.uuid4().hex,
            "email": email.lower().strip(),
            "password_hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
            "created_at": datetime.now(timezone.utc),
        }
        await cls._collection().insert_one(user)
        logger.info("Created user email='%s'", user["email"])
        return {"user_id": user["user_id"], "email": user["email"]}

    @classmethod
    async def get_user_by_email(cls, email: str) -> dict | None:
        return await cls._collection().find_one(
            {"email": email.lower().strip()}, {"_id": 0}
        )

    @classmethod
    def verify_password(cls, plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
