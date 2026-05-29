from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import TCMemConfig


class SemanticScorer(Protocol):
    def embed_query(self, query: str) -> np.ndarray:
        ...

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        ...

    def score(self, query: str, text: str) -> float:
        ...

    def rank(self, query: str, candidates: list[tuple[str, str]], limit: int) -> list[tuple[str, float]]:
        ...


@dataclass(slots=True)
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str


class EmbeddingClient:
    def __init__(self, config: TCMemConfig) -> None:
        self.config = config
        self._model = None
        self._device = ""
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    @property
    def model_name(self) -> str:
        return self.config.embedding_model

    @property
    def device(self) -> str:
        if not self._device:
            self._device = self._resolve_device()
        return self._device

    def embed_texts(self, texts: list[str], *, kind: str = "document") -> EmbeddingResponse:
        embeddings = self._encode(texts, kind=kind)
        return EmbeddingResponse(embeddings=[vector.tolist() for vector in embeddings], model=self.model_name)

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode([query], kind="query")[0]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._encode(texts, kind="document")

    def score(self, query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        return self._cosine(self.embed_query(query), self.embed_documents([text])[0])

    def rank(self, query: str, candidates: list[tuple[str, str]], limit: int) -> list[tuple[str, float]]:
        if not query or not candidates or limit <= 0:
            return []
        query_vector = self.embed_query(query)
        document_vectors = self.embed_documents([text for _item_id, text in candidates])
        ranked = [
            (item_id, self._cosine(query_vector, vector))
            for (item_id, _text), vector in zip(candidates, document_vectors)
        ]
        ranked = [item for item in ranked if item[1] > 0.0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("TCMem embedding requires sentence-transformers.") from exc
        self._model = SentenceTransformer(self.config.embedding_model, device=self.device)
        return self._model

    def _encode(self, texts: list[str], *, kind: str) -> list[np.ndarray]:
        if not texts:
            return []
        output: list[np.ndarray | None] = []
        missing: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            normalized = str(text or "").strip()
            cache_key = (kind, normalized)
            cached = self._cache.get(cache_key)
            output.append(cached)
            if cached is None:
                missing.append((index, normalized))

        if missing:
            model = self._load_model()
            for start in range(0, len(missing), max(1, self.config.embedding_batch_size)):
                batch = missing[start : start + max(1, self.config.embedding_batch_size)]
                vectors = model.encode(
                    [text for _index, text in batch],
                    batch_size=max(1, self.config.embedding_batch_size),
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.embedding_normalize,
                    show_progress_bar=False,
                ).astype("float32")
                for (index, normalized), vector in zip(batch, vectors):
                    self._cache[(kind, normalized)] = vector
                    output[index] = vector

        return [vector for vector in output if vector is not None]

    def _resolve_device(self) -> str:
        if self.config.embedding_device != "auto":
            return self.config.embedding_device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 0.0:
            return 0.0
        return round(max(0.0, float(np.dot(left, right) / denominator)), 6)
