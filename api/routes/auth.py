"""Authentication routes."""

from fastapi import APIRouter, HTTPException, status

from api.schemas import LoginRequest, TokenResponse
from api.dependencies import (
    create_access_token,
    get_auth_config,
    password_matches,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    try:
        _, admin_password = get_auth_config()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    if not password_matches(body.password, admin_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )
    token = create_access_token()
    return TokenResponse(access_token=token)
