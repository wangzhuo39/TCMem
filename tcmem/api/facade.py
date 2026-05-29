from __future__ import annotations

from pathlib import Path

from ..config import TCMemConfig
from ..core.memory_system import MemorySystem
from ..models import DialogueRecord, RetrievalResult, SessionPayload


class TCMem:
    def __init__(self, config: TCMemConfig | None = None, *, llm_client: object | None = None) -> None:
        self.system = MemorySystem(config=config, llm_client=llm_client)

    def ingest_session(self, session: SessionPayload):
        return self.system.ingest_session(session)

    def ingest_record(self, record: DialogueRecord) -> DialogueRecord:
        return self.system.ingest_record(record)

    def search(self, query: str, *, top_k: int = 10) -> RetrievalResult:
        return self.system.retrieve(query, top_k=top_k)

    def save(self, path: str | Path | None = None) -> Path:
        return self.system.save(path)

    @classmethod
    def load(cls, path: str | Path, *, config: TCMemConfig | None = None, llm_client: object | None = None) -> "TCMem":
        facade = cls.__new__(cls)
        facade.system = MemorySystem.load(path, config=config, llm_client=llm_client)
        return facade
