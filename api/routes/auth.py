"""Authentication routes."""

import os

from fastapi import APIRouter, HTTPException, status
from dotenv import load_dotenv

from api.schemas import LoginRequest, TokenResponse
from api.dependencies import create_access_token

load_dotenv()

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    admin_password = os.getenv("ADMIN_PASSWORD", "admin")
    if body.password != admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )
    token = create_access_token(body.password)
    return TokenResponse(access_token=token)
