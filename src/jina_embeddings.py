"""Jina embedding adapter with retrieval-specific task types."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.embeddings import Embeddings


class JinaEmbeddings(Embeddings):
    """Call Jina's embedding endpoint with raw text rather than OpenAI token IDs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        max_retries: int,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "task": task,
            "auto_truncate": True,
            "normalized": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = httpx.post(
            f"{self.base_url}/embeddings",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("Jina Embedding API returned an incomplete response")
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed document chunks for asymmetric retrieval."""
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> list[float]:
        """Embed a user query for asymmetric retrieval."""
        return self._embed([text], task="retrieval.query")[0]
