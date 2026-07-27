"""Security contract tests for the control API authentication boundary."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from api.dependencies import create_access_token, verify_token
from api.main import create_app


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-only-secret")
    monkeypatch.setenv("ADMIN_PASSWORD", "correct horse battery staple")
    monkeypatch.setenv("API_CORS_ORIGINS", "http://localhost:8501,https://demo.example")


def test_login_fails_closed_when_authentication_is_not_configured(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    response = TestClient(create_app()).post(
        "/api/auth/login", json={"password": "admin"}
    )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Authentication is not configured"


def test_login_token_contains_identity_and_lifetime_but_not_password(auth_env):
    response = TestClient(create_app()).post(
        "/api/auth/login", json={"password": "correct horse battery staple"}
    )

    assert response.status_code == 200
    claims = jwt.get_unverified_claims(response.json()["access_token"])
    assert claims["sub"] == "admin"
    assert isinstance(claims["iat"], int)
    assert claims["exp"] > claims["iat"]
    assert "pwd" not in claims
    assert "password" not in claims


def test_invalid_password_is_a_plain_text_user_error(auth_env):
    response = TestClient(create_app()).post(
        "/api/auth/login", json={"password": "wrong"}
    )

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Invalid password"


def test_expired_token_is_rejected(auth_env):
    expired = create_access_token(
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        expires_delta=timedelta(minutes=1),
    )

    assert verify_token(expired) is False


def test_cors_only_allows_explicit_configured_origins(auth_env):
    client = TestClient(create_app())

    allowed = client.options(
        "/api/auth/login",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )
    rejected = client.options(
        "/api/auth/login",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in rejected.headers


def test_cors_rejects_wildcard_origin_with_credentials(monkeypatch):
    monkeypatch.setenv("API_CORS_ORIGINS", "*")

    with pytest.raises(ValueError, match="explicit origins"):
        create_app()
