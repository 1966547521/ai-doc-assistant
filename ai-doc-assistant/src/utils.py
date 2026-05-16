"""Shared utilities for the AI Document Assistant."""

import logging
import os

import requests
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_env_values = dotenv_values(".env")


def _get_env(key: str, default=None):
    """Get config value: .env file first, then system env."""
    if key in _env_values:
        val = _env_values[key]
        if val:
            return val
        return default if default else ""
    return os.getenv(key) or default


def _get_api_key(key: str) -> str:
    """Get API key from system environment only (never from .env)."""
    return os.getenv(key) or ""


def is_ollama_running(url="http://localhost:11434"):
    """Check if Ollama service is running."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_available_ollama_models(url="http://localhost:11434"):
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except Exception:
        pass
    return []


def get_default_ollama_embedding_model():
    """Get the best available embedding model from Ollama."""
    available = get_available_ollama_models()
    preferred = ["nomic-embed-text", "all-minilm", "mxbai-embed-large", "gte-large"]

    for model in preferred:
        if model in available:
            return model

    if available:
        return available[0]
    return "nomic-embed-text"


def get_default_ollama_llm_model():
    """Get the best available LLM model from Ollama."""
    available = get_available_ollama_models()

    qwen_models = [m for m in available if "qwen" in m.lower()]
    if qwen_models:
        return qwen_models[0]

    preferred = ["llama3.2", "llama3", "mistral", "phi3"]
    for model in preferred:
        if model in available:
            return model

    if available:
        return available[0]
    return "llama3.2"


def _is_deepseek():
    """Check if using DeepSeek API."""
    key = _get_api_key("LLM_API_KEY") or _get_api_key("DEEPSEEK_API_KEY") or ""
    url = _get_env("LLM_BASE_URL") or ""
    return bool(key) and ("deepseek" in url.lower())


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Get an LLM instance.

    Priority:
    1. DeepSeek API (when LLM_API_KEY is configured)
    2. DashScope API (when DASHSCOPE_API_KEY is configured, no LLM_API_KEY)
    3. Local Ollama (fallback)
    """
    api_key = (
        _get_api_key("LLM_API_KEY")
        or _get_api_key("DEEPSEEK_API_KEY")
        or _get_api_key("DASHSCOPE_API_KEY")
        or _get_api_key("OPENAI_API_KEY")
        or ""
    )
    base_url = (
        _get_env("LLM_BASE_URL")
        or _get_env("DASHSCOPE_BASE_URL")
        or "https://api.deepseek.com/v1"
    )

    placeholder_keys = ["your-api-key-here", "your-deepseek-api-key-here", ""]
    has_valid_api_key = api_key and api_key not in placeholder_keys

    if not has_valid_api_key:
        if not is_ollama_running():
            raise RuntimeError("No valid API key found and Ollama is not running!")

        base_url = _get_env("LOCAL_LLM_URL", "http://localhost:11434/v1")
        model = _get_env("LOCAL_LLM_MODEL") or get_default_ollama_llm_model()

        return ChatOpenAI(
            model=model, api_key="ollama", base_url=base_url, temperature=temperature  # type: ignore
        )

    if (
        _get_api_key("DASHSCOPE_API_KEY")
        and not _get_api_key("LLM_API_KEY")
        and not _get_api_key("DEEPSEEK_API_KEY")
    ):
        base_url = (
            _get_env("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        model = _get_env("DASHSCOPE_MODEL") or "qwen-plus"
    else:
        model = "deepseek-v4-flash"

    try:
        return ChatOpenAI(
            model=model, api_key=api_key, base_url=base_url, temperature=temperature  # type: ignore
        )
    except Exception as e:
        if is_ollama_running():
            logger.warning("API call failed (%s), falling back to Ollama", e)
            base_url = _get_env("LOCAL_LLM_URL", "http://localhost:11434/v1")
            model = _get_env("LOCAL_LLM_MODEL") or get_default_ollama_llm_model()
            return ChatOpenAI(
                model=model,
                api_key="ollama",  # type: ignore
                base_url=base_url,
                temperature=temperature,
            )
        raise


class FallbackEmbeddings:
    """Wrapper for embeddings with fallback to local Ollama on API errors."""

    def __init__(self, primary_embeddings, fallback_embeddings):
        self.primary = primary_embeddings
        self.fallback = fallback_embeddings
        self._use_fallback = False

    def embed_documents(self, texts):
        """Embed documents with fallback."""
        if self._use_fallback:
            return self.fallback.embed_documents(texts)

        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            logger.warning("API embeddings failed: %s", e)
            if is_ollama_running():
                logger.info("Falling back to local Ollama embeddings...")
                self._use_fallback = True
                return self.fallback.embed_documents(texts)
            raise

    def embed_query(self, text):
        """Embed a query with fallback."""
        if self._use_fallback:
            return self.fallback.embed_query(text)

        try:
            return self.primary.embed_query(text)
        except Exception as e:
            logger.warning("API embeddings failed: %s", e)
            if is_ollama_running():
                logger.info("Falling back to local Ollama embeddings...")
                self._use_fallback = True
                return self.fallback.embed_query(text)
            raise


def get_embeddings():
    """Get embeddings instance.

    Priority:
    1. DashScope API (when DASHSCOPE_API_KEY is in env)
    2. OpenAI compatible API (when other API keys are available)
    3. Local Ollama (fallback)
    """
    dashscope_key = _get_api_key("DASHSCOPE_API_KEY")
    api_key = (
        _get_api_key("LLM_API_KEY")
        or _get_api_key("DEEPSEEK_API_KEY")
        or dashscope_key
        or _get_api_key("OPENAI_API_KEY")
        or ""
    )

    placeholder_keys = ["your-api-key-here", "your-deepseek-api-key-here", ""]
    has_valid_api_key = api_key and api_key not in placeholder_keys

    # Prepare Ollama fallback if available
    ollama_available = is_ollama_running()
    fallback_embeddings = None
    if ollama_available:
        from langchain_ollama import OllamaEmbeddings

        model = (
            _get_env("LOCAL_EMBEDDING_MODEL") or get_default_ollama_embedding_model()
        )
        ollama_url = _get_env("LOCAL_LLM_URL", "http://localhost:11434").rstrip("/v1")
        fallback_embeddings = OllamaEmbeddings(model=model, base_url=ollama_url)

    # Try DashScope API first (supports embeddings)
    if dashscope_key and dashscope_key not in placeholder_keys:
        from langchain_community.embeddings import DashScopeEmbeddings

        primary = DashScopeEmbeddings(
            model="text-embedding-v3", dashscope_api_key=dashscope_key
        )
        if fallback_embeddings:
            return FallbackEmbeddings(primary, fallback_embeddings)
        return primary

    # Try other API-based embeddings
    if has_valid_api_key:
        from langchain_openai import OpenAIEmbeddings

        base_url = _get_env("LLM_BASE_URL") or "https://api.deepseek.com/v1"
        primary = OpenAIEmbeddings(
            model=_get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=api_key,
            base_url=base_url,
        )
        if fallback_embeddings:
            return FallbackEmbeddings(primary, fallback_embeddings)
        return primary

    # Fallback to Ollama
    if fallback_embeddings:
        return fallback_embeddings

    raise RuntimeError(
        "No API key configured and Ollama is not running!\n"
        "Please set DASHSCOPE_API_KEY in system env or start Ollama."
    )
