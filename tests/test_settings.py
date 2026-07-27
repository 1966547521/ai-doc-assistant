import pytest

from src.settings import Settings


REQUIRED_SETTINGS = {
    "LLM_API_KEY": "chat-key",
    "LLM_BASE_URL": "https://chat.example/v1",
    "LLM_MODEL": "chat-model",
    "EMBEDDING_API_KEY": "embedding-key",
    "EMBEDDING_BASE_URL": "https://embedding.example/v1",
    "EMBEDDING_MODEL": "embedding-model",
}


def set_required_settings(monkeypatch):
    for key, value in REQUIRED_SETTINGS.items():
        monkeypatch.setenv(key, value)


def test_settings_requires_separate_embedding_configuration(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "chat-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://chat.example/v1")
    monkeypatch.setenv("LLM_MODEL", "chat-model")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        Settings.from_environment()


def test_settings_exposes_explicit_timeout_and_retry_values(monkeypatch):
    set_required_settings(monkeypatch)
    monkeypatch.setenv("LLM_TIMEOUT", "75.5")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("EMBEDDING_TIMEOUT", "40")
    monkeypatch.setenv("EMBEDDING_MAX_RETRIES", "6")

    settings = Settings.from_environment()

    assert settings.llm_timeout == 75.5
    assert settings.llm_max_retries == 5
    assert settings.llm_max_tokens == 8192
    assert settings.embedding_timeout == 40
    assert settings.embedding_max_retries == 6


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LLM_TIMEOUT", "0"),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_MAX_TOKENS", "0"),
        ("EMBEDDING_TIMEOUT", "not-a-number"),
        ("EMBEDDING_MAX_RETRIES", "1.5"),
    ],
)
def test_settings_rejects_invalid_timeout_and_retry_values(monkeypatch, key, value):
    set_required_settings(monkeypatch)
    monkeypatch.setenv(key, value)

    with pytest.raises(RuntimeError, match=key):
        Settings.from_environment()
