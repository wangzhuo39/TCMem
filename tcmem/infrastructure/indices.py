from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TCMemConfig
from ..utils.embedding_client import SemanticScorer


@dataclass(slots=True)
class VectorIndexItem:
    item_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VectorIndexHit:
    item_id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex(ABC):
    @abstractmethod
    def sync_items(
        self,
        items: list[VectorIndexItem],
        scorer: SemanticScorer,
        *,
        embedding_signature: dict[str, Any],
        rebuild: bool = False,
    ) -> None:
        ...

    @abstractmethod
    def search(self, query: str, scorer: SemanticScorer, *, top_k: int) -> list[VectorIndexHit]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


class NumpyVectorIndex(VectorIndex):
    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path, *, index_name: str) -> None:
        self.path = Path(path)
        self.index_name = index_name
        self._manifest: dict[str, Any] | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._embeddings: np.ndarray | None = None

    def sync_items(
        self,
        items: list[VectorIndexItem],
        scorer: SemanticScorer,
        *,
        embedding_signature: dict[str, Any],
        rebuild: bool = False,
    ) -> None:
        self._load_if_present()
        signature = _stable_json(embedding_signature)
        existing_manifest = self._manifest or {}
        existing_metadata = self._metadata or []
        existing_embeddings = self._embeddings
        reusable = (
            not rebuild
            and existing_manifest.get("schema_version") == self.SCHEMA_VERSION
            and existing_manifest.get("index_name") == self.index_name
            and existing_manifest.get("embedding_signature") == signature
            and existing_embeddings is not None
        )
        if not reusable:
            existing_metadata = []
            existing_embeddings = None

        current_entries = [_entry_from_item(item) for item in items]
        existing_by_id = {
            str(entry.get("item_id")): (index, entry)
            for index, entry in enumerate(existing_metadata)
        }

        vectors: list[np.ndarray | None] = []
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        changed = not reusable or len(current_entries) != len(existing_metadata)

        for index, entry in enumerate(current_entries):
            vector: np.ndarray | None = None
            existing = existing_by_id.get(entry["item_id"])
            if existing is not None and existing_embeddings is not None:
                existing_index, existing_entry = existing
                if existing_entry.get("text_hash") == entry["text_hash"]:
                    vector = np.asarray(existing_embeddings[existing_index], dtype="float32")
                    if existing_index != index or existing_entry != entry:
                        changed = True
            if vector is None:
                missing_indices.append(index)
                missing_texts.append(entry["text"])
                changed = True
            vectors.append(vector)

        if missing_texts:
            embedded = scorer.embed_documents(missing_texts)
            if len(embedded) != len(missing_texts):
                raise RuntimeError("Embedding scorer returned an unexpected document vector count.")
            for index, vector in zip(missing_indices, embedded):
                vectors[index] = np.asarray(vector, dtype="float32")

        matrix = np.vstack([vector for vector in vectors if vector is not None]).astype("float32") if current_entries else np.zeros((0, 0), dtype="float32")
        self._manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "index_name": self.index_name,
            "embedding_signature": signature,
            "item_count": len(current_entries),
        }
        self._metadata = current_entries
        self._embeddings = matrix
        if changed:
            self._save()

    def search(self, query: str, scorer: SemanticScorer, *, top_k: int) -> list[VectorIndexHit]:
        if top_k <= 0:
            return []
        self._load_required()
        if self._embeddings is None or self._embeddings.size == 0 or not self._metadata:
            return []
        query_vector = np.asarray(scorer.embed_query(query), dtype="float32")
        scores = _cosine_scores(query_vector, self._embeddings)
        order = np.argsort(-scores, kind="mergesort")[: min(top_k, len(self._metadata))]
        hits: list[VectorIndexHit] = []
        for raw_index in order:
            index = int(raw_index)
            entry = self._metadata[index]
            hits.append(
                VectorIndexHit(
                    item_id=str(entry["item_id"]),
                    score=round(float(scores[index]), 6),
                    text=str(entry.get("text") or ""),
                    metadata=dict(entry.get("metadata") or {}),
                )
            )
        return hits

    def clear(self) -> None:
        for path in (self.path / "manifest.json", self.path / "metadata.json", self.path / "embeddings.npy"):
            if path.exists():
                path.unlink()
        self._manifest = None
        self._metadata = None
        self._embeddings = None

    def _load_if_present(self) -> None:
        if self._manifest is not None and self._metadata is not None and self._embeddings is not None:
            return
        manifest_path = self.path / "manifest.json"
        metadata_path = self.path / "metadata.json"
        embeddings_path = self.path / "embeddings.npy"
        if not manifest_path.exists() or not metadata_path.exists() or not embeddings_path.exists():
            self._manifest = {}
            self._metadata = []
            self._embeddings = None
            return
        self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._embeddings = np.load(embeddings_path)

    def _load_required(self) -> None:
        self._load_if_present()
        if self._metadata is None or self._embeddings is None:
            raise RuntimeError(f"Vector index has not been built: {self.path}")

    def _save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "manifest.json").write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.path / "metadata.json").write_text(json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(self.path / "embeddings.npy", self._embeddings)


class ChromaVectorIndex(VectorIndex):
    def __init__(self, path: str | Path, *, collection_name: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("Chroma vector index requires chromadb.") from exc

        self.path = Path(path)
        self.collection_name = _sanitize_collection_name(collection_name)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = None

    def sync_items(
        self,
        items: list[VectorIndexItem],
        scorer: SemanticScorer,
        *,
        embedding_signature: dict[str, Any],
        rebuild: bool = False,
    ) -> None:
        collection = self._get_collection(embedding_signature=embedding_signature, rebuild=rebuild)
        if not items:
            return
        embedding_signature_text = _stable_json(embedding_signature)
        candidate_entries = [
            {
                "id": item.item_id,
                "document": item.text,
                "metadata": _chroma_metadata(
                    {
                        **item.metadata,
                        "text_hash": _hash_text(item.text),
                        "embedding_signature": embedding_signature_text,
                    }
                ),
            }
            for item in items
        ]
        existing_by_id: dict[str, dict[str, Any]] = {}
        try:
            existing = collection.get(ids=[entry["id"] for entry in candidate_entries], include=["metadatas"])
            for item_id, metadata in zip(existing.get("ids", []) or [], existing.get("metadatas", []) or []):
                existing_by_id[str(item_id)] = dict(metadata or {})
        except Exception:
            existing_by_id = {}

        pending = [
            entry
            for entry in candidate_entries
            if existing_by_id.get(entry["id"], {}).get("text_hash") != entry["metadata"].get("text_hash")
            or existing_by_id.get(entry["id"], {}).get("embedding_signature") != embedding_signature_text
        ]
        if not pending:
            return

        ids = [entry["id"] for entry in pending]
        documents = [entry["document"] for entry in pending]
        metadatas = [entry["metadata"] for entry in pending]
        embeddings = [vector.tolist() for vector in scorer.embed_documents(documents)]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def search(self, query: str, scorer: SemanticScorer, *, top_k: int) -> list[VectorIndexHit]:
        if top_k <= 0:
            return []
        collection = self._get_collection()
        count = collection.count()
        if count <= 0:
            return []
        result = collection.query(
            query_embeddings=[scorer.embed_query(query).tolist()],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[VectorIndexHit] = []
        for index, item_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            hits.append(
                VectorIndexHit(
                    item_id=str(item_id),
                    score=round(max(0.0, 1.0 - distance), 6),
                    text=str(documents[index] if index < len(documents) else ""),
                    metadata=dict(metadatas[index] if index < len(metadatas) and metadatas[index] else {}),
                )
            )
        return hits

    def clear(self) -> None:
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self._collection = None

    def _get_collection(self, embedding_signature: dict[str, Any] | None = None, rebuild: bool = False):
        if rebuild:
            self.clear()
        if self._collection is not None:
            return self._collection
        metadata = {"hnsw:space": "cosine"}
        if embedding_signature is not None:
            metadata["embedding_signature"] = _stable_json(embedding_signature)
        self._collection = self._client.get_or_create_collection(name=self.collection_name, metadata=metadata)
        return self._collection


def build_vector_index(config: TCMemConfig, *, owner_id: str, index_name: str) -> VectorIndex:
    backend = config.vector_index_backend.lower()
    if backend == "numpy":
        return NumpyVectorIndex(Path(config.vector_index_path) / owner_id / index_name, index_name=index_name)
    if backend == "chroma":
        collection_name = f"{config.chroma_collection_prefix}_{owner_id}_{index_name}"
        return ChromaVectorIndex(config.vector_index_path, collection_name=collection_name)
    if backend == "faiss":
        raise RuntimeError("Faiss backend is configured but not implemented yet; use chroma or numpy.")
    raise RuntimeError(f"Unsupported vector_index_backend: {config.vector_index_backend}")


def _entry_from_item(item: VectorIndexItem) -> dict[str, Any]:
    text = str(item.text or "")
    return {
        "item_id": str(item.item_id),
        "text": text,
        "text_hash": _hash_text(text),
        "metadata": dict(item.metadata or {}),
    }


def _hash_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _cosine_scores(query_vector: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm <= 0.0:
        return np.zeros((embeddings.shape[0],), dtype="float32")
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    denominator = embedding_norms * query_norm
    raw = embeddings @ query_vector
    return np.divide(raw, denominator, out=np.zeros_like(raw, dtype="float32"), where=denominator > 0)


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
        elif value is not None:
            clean[str(key)] = json.dumps(value, ensure_ascii=False)
    return clean


def _sanitize_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("._-")
    if len(cleaned) < 3:
        cleaned = f"idx_{cleaned}"
    return cleaned[:63]
