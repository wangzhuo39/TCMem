from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TCMemConfig:
    owner_id: str = "default"
    storage_path: str = "TCMem/data/default"
    log_path: str = "TCMem/logs"

    embedding_model: str = "BAAI/bge-m3"
    embedding_backend: str = "sentence_transformers"
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    vector_index_backend: str = "chroma"
    vector_index_path: str = "TCMem/data/default/vector_db"
    chroma_collection_prefix: str = "TCMem"
    vector_index_rebuild: bool = False

    path_a_weight: float = 0.6
    path_b_weight: float = 0.4
    path_a_semantic_weight: float = 0.7
    path_a_status_weight: float = 0.2
    path_a_chain_weight: float = 0.1
    path_b_semantic_weight: float = 0.45
    path_b_graph_weight: float = 0.55
    graph_seed_limit: int = 12
    graph_walk_depth: int = 2
    routed_task_score: float = 1.0
    unrouted_task_score: float = 0.3
    task_metadata_refresh_interval: int = 5
    task_router_entity_limit: int = 20
    prompt_path: str = ""

    active_status_score: float = 1.0
    branched_status_score: float = 0.6
    deprecated_status_score: float = 0.0
    superseded_status_score: float = 0.0
    default_status_score: float = 0.2

    active_penalty: float = 1.0
    branched_penalty: float = 0.75
    deprecated_penalty: float = 0.45
    superseded_penalty: float = 0.45

    llm_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "")
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-flash"
    llm_timeout: int = 120

    def __post_init__(self) -> None:
        self.storage_path = str(Path(self.storage_path))
        self.log_path = str(Path(self.log_path))
        self.vector_index_path = str(Path(self.vector_index_path))
        self.prompt_path = str(Path(self.prompt_path)) if self.prompt_path else ""
        backend = self.vector_index_backend.lower()
        if backend not in {"chroma", "numpy", "faiss"}:
            raise ValueError("vector_index_backend must be one of: chroma, numpy, faiss")
        if self.embedding_backend != "sentence_transformers":
            raise ValueError("Only sentence_transformers embedding backend is currently implemented")

    @property
    def state_path(self) -> Path:
        return Path(self.storage_path) / "memory_state.json"

    def embedding_signature(self) -> dict[str, Any]:
        return {
            "backend": self.embedding_backend,
            "model": self.embedding_model,
            "device": self.embedding_device,
            "batch_size": self.embedding_batch_size,
            "normalize": self.embedding_normalize,
        }

    def to_dict(self, *, include_secrets: bool = False) -> dict[str, Any]:
        data = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        if not include_secrets:
            data["llm_api_key"] = ""
        return data

    @classmethod
    def from_json(cls, path: str | Path) -> "TCMemConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
