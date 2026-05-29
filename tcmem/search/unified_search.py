from __future__ import annotations

from ..models import RetrievalResult
from ..core.retrieval import RetrievalEngine


class UnifiedSearchService:
    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval_engine = retrieval_engine

    def search(self, query: str, *, top_k: int = 10) -> RetrievalResult:
        return self.retrieval_engine.retrieve(query, top_k=top_k)
