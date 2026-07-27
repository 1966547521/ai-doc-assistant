"""Factories for the application's real OpenAI-compatible API clients."""

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.jina_embeddings import JinaEmbeddings
from src.settings import Settings


_last_llm_provider: str | None = None


def get_last_llm_provider() -> str | None:
    """Return the configured provider label for display in the UI."""
    return _last_llm_provider


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Build the explicitly configured OpenAI-compatible chat client."""
    global _last_llm_provider

    settings = Settings.from_environment()
    client = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        max_tokens=settings.llm_max_tokens,
    )
    _last_llm_provider = f"OpenAI Compatible ({settings.llm_model})"
    return client


def get_embeddings() -> Embeddings:
    """Build the explicitly configured embedding client."""
    settings = Settings.from_environment()
    if "api.jina.ai" in settings.embedding_base_url:
        return JinaEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            timeout=settings.embedding_timeout,
            max_retries=settings.embedding_max_retries,
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.embedding_timeout,
        max_retries=settings.embedding_max_retries,
    )
