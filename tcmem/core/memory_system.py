from __future__ import annotations

from pathlib import Path

from ..config import TCMemConfig
from ..infrastructure.indices import VectorIndex, build_vector_index
from ..logging_utils import ModuleLogStore
from ..models import DialogueRecord, IngestionResult, RetrievalResult, SessionPayload
from ..search.unified_search import UnifiedSearchService
from ..serialization import to_primitive
from ..storage.repositories import FileSystemMemoryRepository
from ..utils.embedding_client import EmbeddingClient, SemanticScorer
from ..utils.llm_client import OpenAICompatibleLLMClient
from .graph_store import DialogueGraphStore
from .retrieval import RetrievalEngine
from .task_chain import TaskChainManager


class MemorySystem:
    def __init__(
        self,
        *,
        config: TCMemConfig | None = None,
        graph: DialogueGraphStore | None = None,
        task_manager: TaskChainManager | None = None,
        embedding_client: SemanticScorer | None = None,
        record_index: VectorIndex | None = None,
        llm_client: object | None = None,
        log_store: ModuleLogStore | None = None,
    ) -> None:
        self.config = config or TCMemConfig()
        self.log_store = log_store or ModuleLogStore(base_dir=self.config.log_path)
        self.repository = FileSystemMemoryRepository(self.config.storage_path)
        self.llm_client = llm_client or self._client_from_config_or_none()
        self.graph = graph or DialogueGraphStore()
        self.task_manager = task_manager or TaskChainManager(self.config.owner_id, llm_client=self.llm_client)
        if self.task_manager.llm_client is None:
            self.task_manager.llm_client = self.llm_client
        self.embedding_client = embedding_client or EmbeddingClient(self.config)
        self.record_index = record_index or build_vector_index(
            self.config,
            owner_id=self.config.owner_id,
            index_name="dialogue_records",
        )
        self.retrieval = RetrievalEngine(
            config=self.config,
            graph=self.graph,
            task_manager=self.task_manager,
            embedding_client=self.embedding_client,
            record_index=self.record_index,
        )
        self.search = UnifiedSearchService(self.retrieval)

    def ingest_session(self, session: SessionPayload) -> IngestionResult:
        result = IngestionResult(session_identifier=session.session_identifier, session_uuid=session.session_uuid)
        for record in self.task_manager.parse_session_records(session):
            self.ingest_record(record)
            result.record_ids.append(record.record_id)
        return result

    def ingest_record(self, record: DialogueRecord) -> DialogueRecord:
        self.task_manager.extract_record_entities(record)
        self.graph.add_record(record)
        route_decision = self.task_manager.route_record(record)
        for task_id in route_decision.routed_task_ids:
            self.task_manager.apply_record(task_id, record)
        self.retrieval.sync_record_index()
        self.log_store.log(
            "ingestion",
            "record_ingested",
            record_id=record.record_id,
            session_uuid=record.session_uuid,
            routed_task_ids=route_decision.routed_task_ids,
        )
        return record

    def retrieve(self, query: str, *, top_k: int = 10) -> RetrievalResult:
        result = self.search.search(query, top_k=top_k)
        self.log_store.log(
            "retrieval",
            "query_retrieved",
            query=query,
            routed_task_ids=result.routed_task_ids,
            hits=to_primitive(result.hits),
        )
        return result

    def export_state(self) -> dict:
        return {
            "owner_id": self.config.owner_id,
            "config": self.config.to_dict(),
            "graph": self.graph.to_state(),
            "task_chains": self.task_manager.to_state(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        state_path = Path(path) if path else self.repository.state_path
        from ..serialization import dump_json

        dump_json(state_path, self.export_state())
        return state_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        config: TCMemConfig | None = None,
        embedding_client: SemanticScorer | None = None,
        llm_client: object | None = None,
    ) -> "MemorySystem":
        from ..serialization import load_json

        state = load_json(path)
        loaded_config = config or TCMemConfig(**state.get("config", {}))
        graph = DialogueGraphStore.from_state(state.get("graph", {}))
        task_manager = TaskChainManager.from_state(state.get("task_chains", {}), llm_client=llm_client)
        return cls(
            config=loaded_config,
            graph=graph,
            task_manager=task_manager,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )

    def state_summary(self) -> dict:
        return {
            "owner_id": self.config.owner_id,
            "record_count": len(self.graph.records),
            "edge_count": len(self.graph.edges),
            "task_count": len(self.task_manager.tasks),
        }

    def _client_from_config_or_none(self) -> OpenAICompatibleLLMClient | None:
        if not self.config.llm_api_key:
            return None
        return OpenAICompatibleLLMClient(
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            model=self.config.llm_model,
            timeout=self.config.llm_timeout,
        )
