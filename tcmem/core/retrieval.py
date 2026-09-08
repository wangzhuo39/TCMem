from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import TCMemConfig
from ..infrastructure.indices import InMemoryBM25Index, VectorIndex, VectorIndexItem
from ..models import ChainNodeStatus, IntentUnderstanding, QueryRouteDecision, RetrievalResult, SearchHit, TaskChainNode
from ..utils.embedding_client import SemanticScorer
from .graph_store import DialogueGraphStore
from .task_chain import TaskChainManager


@dataclass(slots=True)
class _RecordAggregate:
    record_id: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    chain_score: float = 0.0
    graph_score: float = 0.0
    route_score: float = 0.0
    path_a_score: float = 0.0
    path_b_score: float = 0.0
    generic_score: float = 0.0
    task_chain_evidence: bool = False
    best_overall_score: float = 0.0
    depth: int | None = None
    task_id: str | None = None
    chain_node_id: str | None = None
    route_role: str = "unrouted"
    route_relation: str | None = None
    route_depth: int = 0
    reasons: list[str] = field(default_factory=list)


class RetrievalEngine:
    def __init__(
        self,
        *,
        config: TCMemConfig,
        graph: DialogueGraphStore,
        task_manager: TaskChainManager,
        embedding_client: SemanticScorer,
        record_index: VectorIndex,
        bm25_index: InMemoryBM25Index | None = None,
    ) -> None:
        self.config = config
        self.graph = graph
        self.task_manager = task_manager
        self.embedding_client = embedding_client
        self.record_index = record_index
        self.bm25_index = bm25_index or InMemoryBM25Index()

    def retrieve(self, query: str, *, top_k: int = 10) -> RetrievalResult:
        if self.config.task_chain_enabled:
            try:
                query_route = self.task_manager.route_for_query(query)
            except Exception as exc:
                query_route = QueryRouteDecision(
                    query=query,
                    routed_task_ids=[],
                    query_intent=IntentUnderstanding(
                        main_intent=query,
                        summary=query,
                        is_substantive=True,
                        reason="query_router_fallback",
                    ),
                    reason=f"query_router_fallback:{type(exc).__name__}",
                )
        else:
            query_route = QueryRouteDecision(
                query=query,
                routed_task_ids=[],
                query_intent=IntentUnderstanding(
                    main_intent=query,
                    is_substantive=True,
                    reason="task_chain_disabled",
                ),
                reason="task_chain_disabled",
            )
        routed_task_ids = list(query_route.routed_task_ids)
        expanded_task_ids, expansion_edges = (
            self.task_manager.expand_task_graph(routed_task_ids)
            if self.config.task_chain_enabled
            else ([], [])
        )
        primary_task_id_set = set(routed_task_ids)
        expanded_task_id_set = set(expanded_task_ids)
        aggregates: dict[str, _RecordAggregate] = {}

        if self.config.task_chain_enabled:
            for hit in self._path_a(query, expanded_task_ids, primary_task_id_set, expansion_edges):
                self._merge_record_hit(aggregates, hit)
        for hit in self._path_b(query, primary_task_id_set, expanded_task_id_set, expansion_edges):
            self._merge_record_hit(aggregates, hit)

        ranked = sorted(
            (self._finalize_record_hit(candidate) for candidate in aggregates.values()),
            key=self._record_rank_key,
            reverse=True,
        )[:top_k]
        return RetrievalResult(
            query=query,
            subqueries=[query],
            routed_task_ids=routed_task_ids,
            hits=ranked,
            explanation="task_chain_plus_graph_vector_bm25" if self.config.task_chain_enabled else "graph_vector_bm25_no_task_chain",
            query_intent=query_route.query_intent,
            query_route_reason=query_route.reason,
            expanded_task_ids=expanded_task_ids,
            expansion_edges=expansion_edges,
        )

    def sync_record_index(self) -> None:
        items = [
            VectorIndexItem(
                item_id=record.record_id,
                text=record.combined_content,
                metadata={
                    "session_identifier": record.session_identifier,
                    "session_uuid": record.session_uuid,
                    "record_time": record.record_time,
                },
            )
            for record in self.graph.records.values()
        ]
        self.record_index.sync_items(
            items,
            self.embedding_client,
            embedding_signature=self.config.embedding_signature(),
            rebuild=self.config.vector_index_rebuild,
        )
        self.bm25_index.sync_items(items)

    def _path_a(
        self,
        query: str,
        expanded_task_ids: list[str],
        primary_task_ids: set[str],
        expansion_edges: list[dict[str, Any]],
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for task_id in expanded_task_ids:
            task = self.task_manager.get_task(task_id)
            if task is None:
                continue
            route_role, route_score, route_relation, route_depth = self._route_metadata(
                task_id,
                primary_task_ids,
                set(expanded_task_ids),
                expansion_edges,
            )
            max_pos = max(1, len(task.nodes))
            for node in sorted(task.nodes.values(), key=lambda item: item.position, reverse=True):
                node_text = "\n".join(part for part in [node.summary, node.user_content, node.assistant_content or ""] if part)
                semantic_score = self.embedding_client.score(query, node_text)
                status_score = self._status_score(node)
                chain_score = node.position / max_pos if max_pos else 0.0
                score = (
                    self.config.path_a_semantic_weight * semantic_score
                    + self.config.path_a_status_weight * status_score
                    + self.config.path_a_chain_weight * chain_score
                )
                hits.append(
                    SearchHit(
                        item_id=node.node_id,
                        item_kind="chain_node",
                        score=score,
                        semantic_score=semantic_score,
                        chain_score=chain_score,
                        route_score=1.0,
                        reason="path_a_chain",
                        task_id=task_id,
                        source_record_id=node.source_record_id,
                        chain_node_id=node.node_id,
                        route_role=route_role,
                        route_relation=route_relation,
                        route_depth=route_depth,
                    )
                )
        return hits

    def _path_b(
        self,
        query: str,
        primary_task_ids: set[str],
        expanded_task_ids: set[str],
        expansion_edges: list[dict[str, Any]],
    ) -> list[SearchHit]:
        self.sync_record_index()
        vector_seeds = self.record_index.search(query, self.embedding_client, top_k=self.config.graph_seed_limit)
        bm25_seeds = self.bm25_index.search(query, top_k=self.config.graph_bm25_seed_limit)
        record_bm25_scores = self._record_bm25_scores(query)
        seeds = self._merge_path_b_seeds(vector_seeds, bm25_seeds)
        best_depth: dict[str, int] = {}
        best_graph_score: dict[str, float] = {}
        for seed in seeds:
            walk = self.graph.walk([seed.item_id], max_depth=self.config.graph_walk_depth)
            for record_id, depth in walk.items():
                graph_score = seed.score / (depth + 1)
                if record_id not in best_graph_score or graph_score > best_graph_score[record_id]:
                    best_graph_score[record_id] = graph_score
                    best_depth[record_id] = depth

        hits: list[SearchHit] = []
        for record_id, graph_score in best_graph_score.items():
            record = self.graph.get_record(record_id)
            if record is None:
                continue
            semantic_score = self.embedding_client.score(query, record.combined_content)
            bm25_score = record_bm25_scores.get(record_id, 0.0)
            base_score = (
                self.config.path_b_semantic_weight * semantic_score
                + self.config.path_b_bm25_weight * bm25_score
                + self.config.path_b_graph_weight * graph_score
            )
            context = (
                self._best_chain_context(record_id, primary_task_ids, expanded_task_ids, expansion_edges)
                if self.config.task_chain_enabled
                else None
            )
            penalty = float(context["penalty"]) if context is not None else 1.0
            hits.append(
                SearchHit(
                    item_id=record_id,
                    item_kind="dialogue_record",
                    score=base_score * penalty,
                    generic_score=base_score,
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    graph_score=graph_score,
                    chain_score=float(context["chain_score"]) if context is not None else 0.0,
                    route_score=float(context["route_score"]) if context is not None else 0.0,
                    depth=best_depth.get(record_id, 0),
                    reason="path_b_vector_bm25_graph" if self.config.task_chain_enabled else "path_b_vector_bm25_graph_no_task_chain",
                    task_id=str(context["task_id"]) if context is not None and context.get("task_id") else None,
                    source_record_id=record_id,
                    chain_node_id=str(context["node_id"]) if context is not None and context.get("node_id") else None,
                    route_role=str(context["route_role"]) if context is not None else "unrouted",
                    route_relation=str(context["route_relation"]) if context is not None and context.get("route_relation") else None,
                    route_depth=int(context["route_depth"]) if context is not None else 0,
                )
            )
        return hits

    def _merge_record_hit(self, aggregates: dict[str, _RecordAggregate], hit: SearchHit) -> None:
        record_id = hit.source_record_id or hit.item_id
        aggregate = aggregates.get(record_id)
        if aggregate is None:
            aggregate = _RecordAggregate(record_id=record_id)
            aggregates[record_id] = aggregate
        aggregate.semantic_score = max(aggregate.semantic_score, hit.semantic_score)
        aggregate.bm25_score = max(aggregate.bm25_score, hit.bm25_score)
        aggregate.chain_score = max(aggregate.chain_score, hit.chain_score)
        aggregate.graph_score = max(aggregate.graph_score, hit.graph_score)
        aggregate.generic_score = max(aggregate.generic_score, hit.generic_score)
        # Path A and routed/expanded Path B hits carry task-chain evidence.
        # Unrouted Path B hits remain eligible as generic fallback evidence.
        if (
            hit.task_chain_evidence
            or hit.reason == "path_a_chain"
            or hit.route_role in {"primary", "expanded"}
        ):
            aggregate.task_chain_evidence = True
        # Keep route metadata from the strongest route evidence.  The previous
        # implementation updated route_score before comparing, making the
        # comparison always true and allowing a later weaker Path-B hit to
        # overwrite the primary/expanded route label.
        if (
            hit.route_score > aggregate.route_score
            or (hit.route_score == aggregate.route_score and hit.score > aggregate.best_overall_score)
        ):
            aggregate.route_role = hit.route_role
            aggregate.route_relation = hit.route_relation
            aggregate.route_depth = hit.route_depth
        aggregate.route_score = max(aggregate.route_score, hit.route_score)
        if hit.reason == "path_a_chain":
            aggregate.path_a_score = max(aggregate.path_a_score, hit.score)
        elif hit.reason in {"path_b_vector_bm25_graph", "path_b_vector_bm25_graph_no_task_chain"}:
            aggregate.path_b_score = max(aggregate.path_b_score, hit.score)
        if hit.reason and hit.reason not in aggregate.reasons:
            aggregate.reasons.append(hit.reason)
        if hit.score > aggregate.best_overall_score:
            aggregate.best_overall_score = hit.score
            aggregate.task_id = hit.task_id or aggregate.task_id
            aggregate.chain_node_id = hit.chain_node_id or aggregate.chain_node_id
            aggregate.depth = hit.depth

    def _finalize_record_hit(self, aggregate: _RecordAggregate) -> SearchHit:
        if self.config.task_chain_enabled:
            score = (
                self.config.path_a_weight * aggregate.path_a_score
                + self.config.path_b_weight * aggregate.path_b_score
            )
        else:
            score = aggregate.path_b_score
        ordered_reasons = [
            reason
            for reason in ("path_a_chain", "path_b_vector_bm25_graph", "path_b_vector_bm25_graph_no_task_chain")
            if reason in aggregate.reasons
        ]
        ordered_reasons.extend(reason for reason in aggregate.reasons if reason not in ordered_reasons)
        return SearchHit(
            item_id=aggregate.record_id,
            item_kind="dialogue_record",
            score=score,
            semantic_score=aggregate.semantic_score,
            bm25_score=aggregate.bm25_score,
            chain_score=aggregate.chain_score,
            graph_score=aggregate.graph_score,
            route_score=aggregate.route_score,
            depth=aggregate.depth or 0,
            reason="+".join(ordered_reasons),
            task_id=aggregate.task_id,
            source_record_id=aggregate.record_id,
            chain_node_id=aggregate.chain_node_id,
            route_role=aggregate.route_role,
            route_relation=aggregate.route_relation,
            route_depth=aggregate.route_depth,
            generic_score=aggregate.generic_score,
            task_chain_evidence=aggregate.task_chain_evidence,
        )

    def _record_rank_key(self, hit: SearchHit) -> float:
        """Return the online ordering score while preserving raw evidence in hit.score."""
        if not self.config.task_chain_enabled:
            return hit.score
        if hit.task_chain_evidence:
            return hit.score + 0.05 * hit.generic_score
        return 0.5 * hit.generic_score

    def _merge_path_b_seeds(self, vector_seeds: list, bm25_seeds: list) -> list[SearchHit]:
        merged: dict[str, dict[str, float]] = {}
        for seed in vector_seeds:
            merged.setdefault(seed.item_id, {})["semantic_score"] = seed.score
        for seed in bm25_seeds:
            merged.setdefault(seed.item_id, {})["bm25_score"] = seed.score

        hits: list[SearchHit] = []
        for record_id, scores in merged.items():
            semantic_score = float(scores.get("semantic_score", 0.0))
            bm25_score = float(scores.get("bm25_score", 0.0))
            hits.append(
                SearchHit(
                    item_id=record_id,
                    item_kind="dialogue_record",
                    score=(
                        self.config.graph_vector_seed_weight * semantic_score
                        + self.config.graph_bm25_seed_weight * bm25_score
                    ),
                    semantic_score=semantic_score,
                    bm25_score=bm25_score,
                    source_record_id=record_id,
                )
            )
        return hits

    def _record_bm25_scores(self, query: str) -> dict[str, float]:
        if not self.graph.records:
            return {}
        return {
            hit.item_id: hit.score
            for hit in self.bm25_index.search(query, top_k=len(self.graph.records))
        }

    def _best_chain_context(
        self,
        record_id: str,
        primary_task_ids: set[str],
        expanded_task_ids: set[str],
        expansion_edges: list[dict[str, Any]],
    ) -> dict[str, float | str | int] | None:
        best: dict[str, float | str] | None = None
        best_key: tuple[float, float, float, float] | None = None
        for task in self.task_manager.tasks.values():
            max_pos = max(1, len(task.nodes))
            for node in task.nodes.values():
                if node.source_record_id != record_id:
                    continue
                status_score = self._status_score(node)
                chain_score = node.position / max_pos if max_pos else 0.0
                route_role, route_score, route_relation, route_depth = self._route_metadata(
                    task.task_id,
                    primary_task_ids,
                    expanded_task_ids,
                    expansion_edges,
                )
                penalty = self._path_b_penalty(node)
                key = (penalty, status_score, route_score, chain_score)
                if best_key is None or key > best_key:
                    best_key = key
                    best = {
                        "task_id": task.task_id,
                        "node_id": node.node_id,
                        "chain_score": chain_score,
                        "route_score": route_score,
                        "penalty": penalty,
                        "route_role": route_role,
                        "route_relation": route_relation,
                        "route_depth": route_depth,
                    }
        return best

    def _route_metadata(
        self,
        task_id: str,
        primary_task_ids: set[str],
        expanded_task_ids: set[str],
        expansion_edges: list[dict[str, Any]],
    ) -> tuple[str, float, str | None, int]:
        if task_id in primary_task_ids:
            return "primary", self.config.routed_task_score, None, 0
        if task_id not in expanded_task_ids:
            return "unrouted", self.config.unrouted_task_score, None, 0
        candidates = [edge for edge in expansion_edges if str(edge.get("to", "")) == task_id]
        relation_scores = {
            "child": self.config.expanded_child_task_score,
            "parent": self.config.expanded_parent_task_score,
            "branch_child": self.config.expanded_branch_child_task_score,
        }
        if not candidates:
            return "expanded", self.config.expanded_branch_child_task_score, None, 0
        edge = max(
            candidates,
            key=lambda item: (
                relation_scores.get(str(item.get("relation", "")), self.config.expanded_branch_child_task_score),
                -int(item.get("depth", 0) or 0),
            ),
        )
        relation = str(edge.get("relation", "")) or None
        score = relation_scores.get(relation or "", self.config.expanded_branch_child_task_score)
        depth = max(0, int(edge.get("depth", 0) or 0))
        return "expanded", score, relation, depth

    def _status_score(self, node: TaskChainNode) -> float:
        if node.superseded_by:
            return self.config.superseded_status_score
        return {
            ChainNodeStatus.ACTIVE: self.config.active_status_score,
            ChainNodeStatus.BRANCHED: self.config.branched_status_score,
            ChainNodeStatus.DEPRECATED: self.config.deprecated_status_score,
        }.get(node.status, self.config.default_status_score)

    def _path_b_penalty(self, node: TaskChainNode) -> float:
        if node.superseded_by:
            return self.config.superseded_penalty
        if node.status is ChainNodeStatus.DEPRECATED:
            return self.config.deprecated_penalty
        if node.status is ChainNodeStatus.BRANCHED:
            return self.config.branched_penalty
        return self.config.active_penalty
