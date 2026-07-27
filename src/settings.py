"""Validated configuration for the real OpenAI-compatible API clients."""

import os
from dataclasses import dataclass


_REQUIRED_KEYS = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
)


def _positive_float(key: str, default: str) -> float:
    raw_value = os.getenv(key, default).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{key} 必须是正数") from exc
    if value <= 0:
        raise RuntimeError(f"{key} 必须是正数")
    return value


def _non_negative_int(key: str, default: str) -> int:
    raw_value = os.getenv(key, default).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{key} 必须是非负整数") from exc
    if value < 0:
        raise RuntimeError(f"{key} 必须是非负整数")
    return value


def _positive_int(key: str, default: str) -> int:
    value = _non_negative_int(key, default)
    if value == 0:
        raise RuntimeError(f"{key} 必须是正整数")
    return value


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout: float
    llm_max_retries: int
    llm_max_tokens: int
    embedding_api_key: str
    embedding_base_url: str
    embedding_model: str
    embedding_timeout: float
    embedding_max_retries: int

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read and validate the application's explicit API configuration."""
        values = {key: os.getenv(key, "").strip() for key in _REQUIRED_KEYS}
        missing = [key for key, value in values.items() if not value]
        if missing:
            raise RuntimeError("缺少真实 API 配置: " + ", ".join(missing))

        return cls(
            llm_api_key=values["LLM_API_KEY"],
            llm_base_url=values["LLM_BASE_URL"],
            llm_model=values["LLM_MODEL"],
            llm_timeout=_positive_float("LLM_TIMEOUT", "60"),
            llm_max_retries=_non_negative_int("LLM_MAX_RETRIES", "2"),
            llm_max_tokens=_positive_int("LLM_MAX_TOKENS", "4096"),
            embedding_api_key=values["EMBEDDING_API_KEY"],
            embedding_base_url=values["EMBEDDING_BASE_URL"],
            embedding_model=values["EMBEDDING_MODEL"],
            embedding_timeout=_positive_float("EMBEDDING_TIMEOUT", "60"),
            embedding_max_retries=_non_negative_int(
                "EMBEDDING_MAX_RETRIES", "2"
            ),
        )
