from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ..config import TCMemConfig
from ..infrastructure.indices import VectorIndex, build_vector_index
from ..logging_utils import ModuleLogStore
from ..models import ChainUpdateResult, DialogueRecord, IngestionResult, RetrievalResult, RouteDecision, SessionPayload
from ..prompts import PromptRegistry
from ..search.unified_search import UnifiedSearchService
from ..serialization import to_primitive
from ..storage.repositories import FileSystemMemoryRepository
from ..utils.embedding_client import EmbeddingClient, SemanticScorer
from ..utils.llm_client import OpenAICompatibleLLMClient
from .graph_store import DialogueGraphStore
from .retrieval import RetrievalEngine
from .task_chain import TaskChainManager


class MemorySystem:
    ROUTING_WINDOW_MINUTES = 30

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
        self.prompt_registry = PromptRegistry.load(self.config.prompt_path)
        self.graph = graph or DialogueGraphStore()
        self.task_manager = task_manager or TaskChainManager(
            self.config.owner_id,
            llm_client=self.llm_client,
            log_store=self.log_store,
            task_metadata_refresh_interval=self.config.task_metadata_refresh_interval,
            router_entity_limit=self.config.task_router_entity_limit,
            query_router_candidate_count=self.config.query_router_candidate_count,
            query_router_pool_size=self.config.query_router_pool_size,
            query_router_summary_limit=self.config.query_router_summary_limit,
            prompt_registry=self.prompt_registry,
        )
        if self.task_manager.llm_client is None:
            self.task_manager.llm_client = self.llm_client
        if getattr(self.task_manager, "log_store", None) is None:
            self.task_manager.log_store = self.log_store
        self.task_manager.task_metadata_refresh_interval = self.config.task_metadata_refresh_interval
        self.task_manager.router_entity_limit = self.config.task_router_entity_limit
        self.task_manager.query_router_candidate_count = self.config.query_router_candidate_count
        self.task_manager.query_router_pool_size = max(
            self.config.query_router_candidate_count,
            self.config.query_router_pool_size,
        )
        self.task_manager.query_router_summary_limit = self.config.query_router_summary_limit
        if self.config.prompt_path:
            self.task_manager.prompt_registry = self.prompt_registry
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
        self.pending_route_buffers: dict[str, list[DialogueRecord]] = defaultdict(list)

    def ingest_session(self, session: SessionPayload) -> IngestionResult:
        result = IngestionResult(session_identifier=session.session_identifier, session_uuid=session.session_uuid)
        for record in self.task_manager.parse_session_records(session):
            self.ingest_record(record)
            result.record_ids.append(record.record_id)
        route_decisions, updates = self._flush_pending_routes(session.session_uuid)
        self._accumulate_ingestion_result(result, route_decisions, updates)
        return result

    def ingest_record(self, record: DialogueRecord) -> DialogueRecord:
        if not self.config.task_chain_enabled:
            self.task_manager.extract_record_entities(record)
            self.graph.add_record(record)
            self.retrieval.sync_record_index()
            self.log_store.log(
                "ingestion",
                "record_ingested",
                record_id=record.record_id,
                session_uuid=record.session_uuid,
                routed_task_ids=[],
                task_chain_enabled=False,
            )
            return record

        self.task_manager.extract_record_entities(record)
        self.graph.add_record(record)
        self.retrieval.sync_record_index()

        buffer = self.pending_route_buffers[record.session_uuid]
        if not buffer:
            buffer.append(record)
        elif self._within_routing_window(buffer[-1], record):
            buffer.append(record)
        else:
            buffered_records = list(buffer)
            route_decisions, _updates = self._route_and_apply_buffer(record.session_uuid, buffered_records)
            self._log_ingestion_routes(buffered_records, route_decisions)
            buffer.clear()
            buffer.append(record)
            self.retrieval.sync_record_index()
        return record

    def retrieve(self, query: str, *, top_k: int = 10) -> RetrievalResult:
        if self.config.task_chain_enabled:
            self.flush_all_pending_routes()
        result = self.search.search(query, top_k=top_k)
        self.log_store.log(
            "retrieval",
            "query_retrieved",
            query=query,
            routed_task_ids=result.routed_task_ids,
            expanded_task_ids=result.expanded_task_ids,
            expansion_edges=result.expansion_edges,
            query_intent=to_primitive(result.query_intent),
            query_route_reason=result.query_route_reason,
            hits=to_primitive(result.hits),
        )
        return result

    def export_state(self) -> dict:
        if self.config.task_chain_enabled:
            self.flush_all_pending_routes()
        return {
            "owner_id": self.config.owner_id,
            "config": self.config.to_dict(),
            "graph": self.graph.to_state(),
            "task_chains": self.task_manager.to_state(),
        }

    def save(self, path: str | Path | None = None) -> Path:
        if self.config.task_chain_enabled:
            self.flush_all_pending_routes()
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
        task_manager = TaskChainManager.from_state(
            state.get("task_chains", {}),
            llm_client=llm_client,
            task_metadata_refresh_interval=loaded_config.task_metadata_refresh_interval,
            router_entity_limit=loaded_config.task_router_entity_limit,
            query_router_candidate_count=loaded_config.query_router_candidate_count,
            query_router_pool_size=loaded_config.query_router_pool_size,
            query_router_summary_limit=loaded_config.query_router_summary_limit,
            prompt_registry=PromptRegistry.load(loaded_config.prompt_path),
        )
        return cls(
            config=loaded_config,
            graph=graph,
            task_manager=task_manager,
            embedding_client=embedding_client,
            llm_client=llm_client,
        )

    def state_summary(self) -> dict:
        if self.config.task_chain_enabled:
            self.flush_all_pending_routes()
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

    def flush_all_pending_routes(self) -> tuple[list[RouteDecision], list[ChainUpdateResult]]:
        if not self.config.task_chain_enabled:
            self.pending_route_buffers.clear()
            return [], []
        all_decisions: list[RouteDecision] = []
        all_updates: list[ChainUpdateResult] = []
        for session_uuid in list(self.pending_route_buffers.keys()):
            route_decisions, updates = self._flush_pending_routes(session_uuid)
            all_decisions.extend(route_decisions)
            all_updates.extend(updates)
        return all_decisions, all_updates

    def _flush_pending_routes(self, session_uuid: str) -> tuple[list[RouteDecision], list[ChainUpdateResult]]:
        buffer = self.pending_route_buffers.get(session_uuid, [])
        if not buffer:
            return [], []
        buffered_records = list(buffer)
        route_decisions, updates = self._route_and_apply_buffer(session_uuid, buffered_records)
        buffer.clear()
        self._log_ingestion_routes(buffered_records, route_decisions)
        self.retrieval.sync_record_index()
        return route_decisions, updates

    def _log_ingestion_routes(
        self,
        buffered_records: list[DialogueRecord],
        route_decisions: list[RouteDecision],
    ) -> None:
        if len(buffered_records) != len(route_decisions):
            raise RuntimeError("Routing decisions must correspond to buffered records")
        for record, route_decision in zip(buffered_records, route_decisions):
            self.log_store.log(
                "ingestion",
                "record_ingested",
                record_id=record.record_id,
                session_uuid=record.session_uuid,
                routed_task_ids=route_decision.routed_task_ids,
            )

    def _route_and_apply_buffer(
        self,
        session_uuid: str,
        buffered_records: list[DialogueRecord],
    ) -> tuple[list[RouteDecision], list[ChainUpdateResult]]:
        del session_uuid
        route_decisions: list[RouteDecision] = []
        updates: list[ChainUpdateResult] = []
        full_window = list(buffered_records)
        for index, record in enumerate(full_window):
            route_decision = self.task_manager.route_record(
                record,
                routing_window=full_window,
                current_index=index,
            )
            route_decisions.append(route_decision)
            for task_id in route_decision.routed_task_ids:
                updates.append(
                    self.task_manager._apply_record_with_intent(
                        task_id,
                        record,
                        summary=route_decision.intent.summary,
                        record_intent=route_decision.intent,
                    )
                )
        return route_decisions, updates

    def _within_routing_window(self, previous: DialogueRecord, current: DialogueRecord) -> bool:
        previous_time = self._parse_record_time(previous.record_time)
        current_time = self._parse_record_time(current.record_time)
        if previous_time is None or current_time is None:
            return False
        delta_seconds = (current_time - previous_time).total_seconds()
        return 0 <= delta_seconds <= self.ROUTING_WINDOW_MINUTES * 60

    def _parse_record_time(self, value: str) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def _accumulate_ingestion_result(
        self,
        result: IngestionResult,
        route_decisions: list[RouteDecision],
        updates: list[ChainUpdateResult],
    ) -> None:
        for route_decision in route_decisions:
            for task_id in route_decision.created_task_ids:
                if task_id not in result.created_task_ids:
                    result.created_task_ids.append(task_id)
        for update in updates:
            if update.task_id not in result.updated_task_ids:
                result.updated_task_ids.append(update.task_id)
            if update.action in {"create", "override", "branch"} and update.chain_node_id:
                result.created_node_ids.append(update.chain_node_id)
            if update.action == "branch" and update.chain_node_id:
                result.branched_node_ids.append(update.chain_node_id)
            if update.action == "override" and update.reference_node_id:
                result.overridden_node_ids.append(update.reference_node_id)
