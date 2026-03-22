import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

_SECRET = os.getenv("AUTH_JWT_SECRET", "change-me-in-production")
_ALGORITHM = "HS256"
_EXPIRE_DAYS = 30

security = HTTPBearer(auto_error=False)


def create_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=_EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "exp": expire}, _SECRET, algorithm=_ALGORITHM)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """FastAPI dependency — returns user_id if a valid Bearer token is present, else None."""
    if credentials is None:
        return None
    return decode_token(credentials.credentials)


def require_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """FastAPI dependency — raises 401 if no valid Bearer token is present."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return decode_token(credentials.credentials)
