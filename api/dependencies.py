"""FastAPI dependencies - shared resources and auth."""

import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
from jose import JWTError, jwt

from src.cache_manager import SemanticCacheManager
from src.history_manager import HistoryManager
from src.session_manager import SessionManager

load_dotenv()

_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")
_ALGORITHM = "HS256"

security = HTTPBearer(auto_error=False)

# ── Shared instances ───────────────────────────────

_cache_manager: Optional[SemanticCacheManager] = None
_session_manager: Optional[SessionManager] = None
_history_manager: Optional[HistoryManager] = None


def get_cache_manager() -> SemanticCacheManager:
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = SemanticCacheManager()
    return _cache_manager


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_history_manager() -> HistoryManager:
    global _history_manager
    if _history_manager is None:
        _history_manager = HistoryManager()
    return _history_manager


# ── Auth ───────────────────────────────────────────

def create_access_token(password: str) -> str:
    data = {"sub": "admin", "pwd": password}
    return jwt.encode(data, _SECRET_KEY, algorithm=_ALGORITHM)


def verify_token(token: str) -> bool:
    try:
        payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
        return payload.get("pwd") == _ADMIN_PASSWORD
    except JWTError:
        return False


def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> bool:
    if credentials and verify_token(credentials.credentials):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )
