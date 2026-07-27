"""FastAPI dependencies - shared resources and auth."""

import os
from datetime import datetime, timedelta, timezone
from hmac import compare_digest
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from dotenv import load_dotenv
from jose import JWTError, jwt

from src.cache_manager import SemanticCacheManager
from src.history_manager import HistoryManager
from src.session_manager import SessionManager
from src.application_service import ApplicationService
from src.document_service import DocumentService

load_dotenv()

_ALGORITHM = "HS256"
_DEFAULT_TOKEN_LIFETIME = timedelta(minutes=30)

security = HTTPBearer(auto_error=False)

# ── Shared instances ───────────────────────────────

_cache_manager: Optional[SemanticCacheManager] = None
_session_manager: Optional[SessionManager] = None
_history_manager: Optional[HistoryManager] = None
_application_service: Optional[ApplicationService] = None
_document_service: Optional[DocumentService] = None


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


def get_application_service() -> ApplicationService:
    global _application_service
    if _application_service is None:
        _application_service = ApplicationService()
    return _application_service


def get_document_service() -> DocumentService:
    global _document_service
    if _document_service is None:
        _document_service = DocumentService()
    return _document_service


# ── Auth ───────────────────────────────────────────

def get_auth_config() -> tuple[str, str]:
    """Return required authentication secrets, failing closed when absent."""
    secret_key = os.getenv("JWT_SECRET_KEY", "").strip()
    admin_password = os.getenv("ADMIN_PASSWORD", "")
    if not secret_key or not admin_password:
        raise RuntimeError("Authentication is not configured")
    return secret_key, admin_password


def password_matches(candidate: str, expected: str) -> bool:
    """Compare credentials without leaking the matching prefix through timing."""
    return compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def create_access_token(
    *,
    issued_at: Optional[datetime] = None,
    expires_delta: timedelta = _DEFAULT_TOKEN_LIFETIME,
) -> str:
    secret_key, _ = get_auth_config()
    issued_at = issued_at or datetime.now(timezone.utc)
    data = {
        "sub": "admin",
        "iat": issued_at,
        "exp": issued_at + expires_delta,
    }
    return jwt.encode(data, secret_key, algorithm=_ALGORITHM)


def verify_token(token: str) -> bool:
    try:
        secret_key, _ = get_auth_config()
        payload = jwt.decode(token, secret_key, algorithms=[_ALGORITHM])
        return compare_digest(str(payload.get("sub", "")), "admin")
    except (JWTError, RuntimeError):
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
