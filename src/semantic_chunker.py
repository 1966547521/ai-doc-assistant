"""Chinese semantic chunking backed by a local multilingual embedding model."""

import math
import os
import re
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MODEL_NAME = "minishlab/potion-multilingual-128M"
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "potion-multilingual-128M"


class SemanticChunker:
    def __init__(
        self,
        *,
        encoder=None,
        model_name: str = _DEFAULT_MODEL_NAME,
        model_path: str | Path | None = None,
        similarity_threshold: float = 0.55,
        max_chars: int = 1000,
    ) -> None:
        self._encoder = encoder
        self.model_name = model_name
        self.model_path = Path(
            model_path
            or os.getenv("SEMANTIC_MODEL_PATH", str(_DEFAULT_MODEL_PATH))
        )
        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars

    @property
    def encoder(self):
        if self._encoder is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:
                raise RuntimeError(
                    "语义分块模型未安装，请安装 model2vec"
                ) from exc
            if not self.model_path.exists():
                from huggingface_hub import snapshot_download

                snapshot_download(repo_id=self.model_name, local_dir=self.model_path)
            self._encoder = StaticModel.from_pretrained(
                self.model_path, force_download=False
            )
        return self._encoder

    @staticmethod
    def _cosine(left, right) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(
            sum(b * b for b in right)
        )
        return dot / norm if norm else 0.0

    def split(self, text: str) -> list[str]:
        sentences = [part for part in re.split(r"(?<=[。！？!?；;])", text) if part]
        if len(sentences) <= 1:
            return sentences
        try:
            vectors = self.encoder.encode(sentences, normalize_embeddings=True)
        except TypeError:
            vectors = self.encoder.encode(sentences)
        chunks = [sentences[0]]
        for index, sentence in enumerate(sentences[1:], start=1):
            related = self._cosine(vectors[index - 1], vectors[index]) >= self.similarity_threshold
            if related and len(chunks[-1]) + len(sentence) <= self.max_chars:
                chunks[-1] += sentence
            else:
                chunks.append(sentence)
        return chunks
