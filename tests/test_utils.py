"""Tests for construction of real OpenAI-compatible API clients."""

import pytest

from src import utils
from src.jina_embeddings import JinaEmbeddings


@pytest.fixture
def api_environment(monkeypatch):
    values = {
        "LLM_API_KEY": "chat-secret",
        "LLM_BASE_URL": "https://chat.example/v1",
        "LLM_MODEL": "chat-model",
        "LLM_TIMEOUT": "45",
        "LLM_MAX_RETRIES": "3",
        "LLM_MAX_TOKENS": "8192",
        "EMBEDDING_API_KEY": "embedding-secret",
        "EMBEDDING_BASE_URL": "https://embedding.example/v1",
        "EMBEDDING_MODEL": "embedding-model",
        "EMBEDDING_TIMEOUT": "30",
        "EMBEDDING_MAX_RETRIES": "4",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return values


def test_get_llm_uses_only_explicit_chat_configuration(api_environment):
    llm = utils.get_llm(temperature=0.25)

    assert llm.model_name == "chat-model"
    assert str(llm.openai_api_base).rstrip("/") == "https://chat.example/v1"
    assert llm.openai_api_key.get_secret_value() == "chat-secret"
    assert llm.request_timeout == 45
    assert llm.max_retries == 3
    assert llm.max_tokens == 8192
    assert llm.temperature == 0.25


def test_get_embeddings_uses_separate_embedding_configuration(api_environment):
    embeddings = utils.get_embeddings()

    assert embeddings.model == "embedding-model"
    assert str(embeddings.openai_api_base).rstrip("/") == "https://embedding.example/v1"
    assert embeddings.openai_api_key.get_secret_value() == "embedding-secret"
    assert embeddings.request_timeout == 30
    assert embeddings.max_retries == 4


def test_get_embeddings_uses_jina_adapter_for_explicit_jina_endpoint(api_environment):
    api_environment["EMBEDDING_BASE_URL"] = "https://api.jina.ai/v1"
    api_environment["EMBEDDING_MODEL"] = "jina-embeddings-v5-text-small"

    import os
    os.environ.update(api_environment)

    embeddings = utils.get_embeddings()

    assert isinstance(embeddings, JinaEmbeddings)
    assert embeddings.model == "jina-embeddings-v5-text-small"


def test_get_llm_does_not_fall_back_to_legacy_provider_keys(monkeypatch):
    for key in (
        "LLM_API_KEY",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "legacy-key")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")

    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        utils.get_llm()


def test_get_last_llm_provider_reports_configured_model(api_environment):
    utils.get_llm()

    assert utils.get_last_llm_provider() == "OpenAI Compatible (chat-model)"
