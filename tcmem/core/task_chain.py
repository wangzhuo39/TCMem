from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from ..models import (
    ChainNodeStatus,
    ChainUpdateResult,
    DialogueRecord,
    DialogueTurn,
    IntentUnderstanding,
    QueryRouteDecision,
    RouteDecision,
    SessionPayload,
    TaskBranch,
    TaskChain,
    TaskChainNode,
    TaskStatus,
)
from ..prompts import PromptRegistry
from ..serialization import to_primitive


DEFAULT_JSON_MAX_ATTEMPTS = 5
DEFAULT_TASK_METADATA_REFRESH_INTERVAL = 5
DEFAULT_ROUTER_ENTITY_LIMIT = 20
DEFAULT_QUERY_ROUTER_CANDIDATE_COUNT = 5
DEFAULT_QUERY_ROUTER_POOL_SIZE = 48
DEFAULT_QUERY_ROUTER_SUMMARY_LIMIT = 512
DEFAULT_TASK_ROUTING_CANDIDATE_COUNT = 12
TASK_REFRESH_RECENT_NODE_LIMIT = 5
TASK_REFRESH_TEXT_LIMIT = 500


@dataclass(slots=True)
class PromptBundle:
    name: str
    system_prompt: str
    user_prompt: str


@dataclass(slots=True)
class ConflictDecision:
    action: str
    reason: str
    confidence: float
    reference_node_id: str | None = None
    branch_id: str | None = None
    branch_goal: str = ""
    summary: str = ""
    task_description_update: str | None = None


class TaskChainManager:
    def __init__(
        self,
        owner_id: str,
        llm_client: object | None = None,
        log_store: object | None = None,
        *,
        task_metadata_refresh_interval: int = DEFAULT_TASK_METADATA_REFRESH_INTERVAL,
        router_entity_limit: int = DEFAULT_ROUTER_ENTITY_LIMIT,
        query_router_candidate_count: int = DEFAULT_QUERY_ROUTER_CANDIDATE_COUNT,
        query_router_pool_size: int = DEFAULT_QUERY_ROUTER_POOL_SIZE,
        query_router_summary_limit: int = DEFAULT_QUERY_ROUTER_SUMMARY_LIMIT,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.owner_id = owner_id
        self.llm_client = llm_client
        self.log_store = log_store
        self.task_metadata_refresh_interval = task_metadata_refresh_interval
        self.router_entity_limit = router_entity_limit
        self.query_router_candidate_count = _clamp_query_candidate_count(query_router_candidate_count)
        self.query_router_pool_size = max(
            self.query_router_candidate_count,
            int(query_router_pool_size or DEFAULT_QUERY_ROUTER_POOL_SIZE),
        )
        self.query_router_summary_limit = max(
            128,
            int(query_router_summary_limit or DEFAULT_QUERY_ROUTER_SUMMARY_LIMIT),
        )
        self.prompt_registry = prompt_registry or PromptRegistry.default()
        self.tasks: dict[str, TaskChain] = {}

    def parse_session_records(self, session: SessionPayload) -> list[DialogueRecord]:
        records: list[DialogueRecord] = []
        turns = session.dialogue_turns
        base_time = self._base_record_time(session.current_time)
        index = 0
        while index < len(turns):
            current = turns[index]
            record_time = (base_time + timedelta(minutes=len(records))).strftime("%Y-%m-%d %H:%M:%S")
            if current.speaker.lower() == "user" and index + 1 < len(turns):
                next_turn = turns[index + 1]
                if next_turn.speaker.lower() == "assistant":
                    records.append(
                        DialogueRecord(
                            record_id=self._record_id(session.session_uuid, len(records)),
                            session_identifier=session.session_identifier,
                            session_uuid=session.session_uuid,
                            current_time=session.current_time,
                            record_time=record_time,
                            user_content=current.content,
                            assistant_content=next_turn.content,
                            source_turn_indexes=[index, index + 1],
                            source_turn_ids=[
                                f"{session.session_uuid}:{index}:user",
                                f"{session.session_uuid}:{index + 1}:assistant",
                            ],
                        )
                    )
                    index += 2
                    continue
            records.append(
                DialogueRecord(
                    record_id=self._record_id(session.session_uuid, len(records)),
                    session_identifier=session.session_identifier,
                    session_uuid=session.session_uuid,
                    current_time=session.current_time,
                    record_time=record_time,
                    user_content=current.content,
                    source_turn_indexes=[index],
                    source_turn_ids=[f"{session.session_uuid}:{index}:{current.speaker.lower()}"],
                )
            )
            index += 1
        return records

    def extract_record_entities(self, record: DialogueRecord) -> list[str]:
        payload = {"record": to_primitive(record)}
        rendered = self.prompt_registry.render("entity_extraction", payload_json=_json_dumps(payload))
        bundle = PromptBundle(
            name="entity_extraction",
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
        )
        max_attempts = self._json_max_attempts()
        retry_delay = self._json_retry_delay()
        for attempt in range(1, max_attempts + 1):
            parsed = self._call_llm_json(bundle)
            if isinstance(parsed, dict):
                parsed = _unwrap_schema_response(parsed, expected_keys={"entities"})
            if isinstance(parsed, dict) and isinstance(parsed.get("entities"), list):
                record.entities = self._normalize_entities([*record.entities, *parsed["entities"]])[:3]
                return record.entities
            self._log_invalid_llm_payload(
                stage="entity_extraction",
                attempt=attempt,
                max_attempts=max_attempts,
                parsed=parsed,
                record_id=record.record_id,
            )
            if attempt >= max_attempts:
                break
            if retry_delay > 0.0:
                time.sleep(retry_delay)
        return record.entities

    def understand_record_intent(
        self,
        record: DialogueRecord,
        *,
        routing_window: list[DialogueRecord] | None = None,
        current_index: int | None = None,
    ) -> IntentUnderstanding:
        parsed = self._call_llm_json(
            self._record_intent_bundle(
                record,
                routing_window=routing_window,
                current_index=current_index,
            )
        )
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"main_intent"})
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM record intent understanding returned invalid payload for record {record.record_id}")
        return self._intent_from_payload(parsed)

    def route_record(
        self,
        record: DialogueRecord,
        routing_window: list[DialogueRecord] | None = None,
        current_index: int | None = None,
    ) -> RouteDecision:
        intent = self.understand_record_intent(
            record,
            routing_window=routing_window,
            current_index=current_index,
        )
        max_attempts = self._json_max_attempts()
        retry_delay = self._json_retry_delay()
        last_decision: RouteDecision | None = None
        for attempt in range(1, max_attempts + 1):
            initial_parsed = self._call_llm_json(
                self._task_routing_bundle(
                    record,
                    intent=intent,
                    current_index=current_index,
                )
            )
            if isinstance(initial_parsed, dict):
                initial_parsed = _unwrap_schema_response(initial_parsed, expected_keys={"linked_task_ids", "new_tasks"})
            if not isinstance(initial_parsed, dict):
                raise RuntimeError(f"LLM task routing returned invalid payload for record {record.record_id}")
            initial_decision = self._route_payload_from_parsed(initial_parsed)
            parsed = self._review_task_route(
                record,
                initial_decision,
                intent=intent,
                current_index=current_index,
            )
            decision = self._finalize_route_decision(record, intent=intent, parsed=parsed)
            if decision.routed_task_ids:
                log = getattr(self.log_store, "log", None)
                if log is not None:
                    log(
                        "routing",
                        "record_routed",
                        record_id=record.record_id,
                        session_uuid=record.session_uuid,
                        main_intent=decision.main_intent,
                        linked_task_ids=decision.linked_task_ids,
                        created_task_ids=decision.created_task_ids,
                        routed_task_ids=decision.routed_task_ids,
                        intent=to_primitive(decision.intent),
                        reason=decision.reason,
                    )
                return decision
            if not intent.is_substantive:
                return decision
            if not initial_decision.get("linked_task_ids") and not initial_decision.get("new_tasks") and _is_non_task_route(decision):
                return decision
            last_decision = decision
            self._log_empty_record_route(
                record=record,
                attempt=attempt,
                max_attempts=max_attempts,
                reason=decision.reason,
                main_intent=decision.main_intent,
            )
            if attempt >= max_attempts:
                break
            if retry_delay > 0.0:
                time.sleep(retry_delay)
        reason = last_decision.reason if last_decision is not None else ""
        raise RuntimeError(f"LLM task routing returned no task for record {record.record_id} after {max_attempts} attempts: {reason}")

    def understand_query_intent(self, query: str) -> IntentUnderstanding:
        parsed = self._call_llm_json(self._query_intent_bundle(query))
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"main_intent"})
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM query intent understanding returned invalid payload")
        return self._intent_from_payload(parsed)

    def route_for_query(self, query: str) -> QueryRouteDecision:
        try:
            intent = self.understand_query_intent(query)
        except Exception as exc:
            return self._fallback_query_route(query, None, exc)
        candidate_count = min(self.query_router_candidate_count, len(self.tasks)) if self.tasks else self.query_router_candidate_count
        candidates = self._query_routing_candidates(query, intent)
        catalog = [
            self._task_summary(
                task,
                include_recent_records=False,
                text_limit=self.query_router_summary_limit,
            )
            for task in candidates
        ]
        payload = {
            "query": query,
            "intent": to_primitive(intent),
            "candidate_count": candidate_count,
            "task_catalog": catalog,
        }
        rendered = self.prompt_registry.render("query_routing", payload_json=_json_dumps(payload))
        try:
            parsed = self._call_llm_json(
                PromptBundle(
                    name="query_routing",
                    system_prompt=rendered.system_prompt,
                    user_prompt=rendered.user_prompt,
                )
            )
        except Exception as exc:
            return self._fallback_query_route(query, intent, exc, candidate_count=candidate_count)
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"routed_task_ids"})
        if not isinstance(parsed, dict):
            return self._fallback_query_route(
                query,
                intent,
                RuntimeError("LLM query routing returned invalid payload"),
                candidate_count=candidate_count,
            )
        candidate_task_ids = {task.task_id for task in candidates}
        routed = [
            str(task_id)
            for task_id in parsed.get("routed_task_ids", []) or []
            if str(task_id) in candidate_task_ids
        ]
        routed = routed[:candidate_count]
        decision = QueryRouteDecision(
            query=query,
            routed_task_ids=routed,
            query_intent=intent,
            reason=str(parsed.get("reason", "") or ""),
        )
        log = getattr(self.log_store, "log", None)
        if log is not None:
            log(
                "routing",
                "query_routed",
                query=query,
                candidate_count=candidate_count,
                candidate_pool_size=len(candidates),
                total_task_count=len(self.tasks),
                catalog_chars=len(_json_dumps(catalog)),
                routed_task_ids=routed,
                query_intent=to_primitive(intent),
                reason=decision.reason,
            )
        return decision

    def _fallback_query_route(
        self,
        query: str,
        intent: IntentUnderstanding | None,
        error: Exception,
        *,
        candidate_count: int | None = None,
    ) -> QueryRouteDecision:
        fallback_intent = intent or IntentUnderstanding(
            main_intent=query,
            summary=query,
            is_substantive=True,
            reason="query_router_fallback",
        )
        reason = f"query_router_fallback:{type(error).__name__}"
        log = getattr(self.log_store, "log", None)
        if log is not None:
            log(
                "routing",
                "query_route_fallback",
                query=query,
                candidate_count=candidate_count if candidate_count is not None else self.query_router_candidate_count,
                total_task_count=len(self.tasks),
                error_type=type(error).__name__,
                error=str(error)[:500],
                query_intent=to_primitive(fallback_intent),
                reason=reason,
            )
        return QueryRouteDecision(
            query=query,
            routed_task_ids=[],
            query_intent=fallback_intent,
            reason=reason,
        )

    def create_task_from_record(self, record: DialogueRecord, spec: dict[str, Any]) -> TaskChain:
        task_description = str(spec.get("task_description", "") or "").strip()
        if not task_description:
            raise ValueError("new task spec must include task_description")
        canonical_description = str(
            spec.get("canonical_description", task_description) or task_description
        ).strip()
        current_focus = str(spec.get("current_focus", "") or "").strip()
        parent_task_id = str(spec.get("parent_task_id", "") or "").strip() or None
        parent_branch_id = str(spec.get("parent_branch_id", "") or "").strip() or None
        task = TaskChain(
            task_id=f"task_{uuid4().hex[:8]}",
            task_description=task_description,
            owner_id=self.owner_id,
            created_at=record.current_time,
            updated_at=record.current_time,
            entities=list(record.entities),
            canonical_description=canonical_description,
            current_focus=current_focus,
            parent_task_id=parent_task_id,
            parent_branch_id=parent_branch_id,
        )
        parent = self.tasks.get(parent_task_id) if parent_task_id else None
        if parent_task_id and parent is None:
            raise ValueError(f"unknown parent_task_id {parent_task_id!r}")
        if parent_branch_id:
            if parent is None:
                raise ValueError(
                    f"parent_branch_id {parent_branch_id!r} requires an existing parent_task_id"
                )
            self._ensure_legacy_branch_registry(parent)
            if parent_branch_id not in parent.branches:
                raise ValueError(
                    f"unknown parent_branch_id {parent_branch_id!r} for task {parent_task_id}"
                )
        self.tasks[task.task_id] = task
        branch_id = str(spec.get("branch_id", "main") or "main").strip() or "main"
        branch_goal = str(spec.get("branch_goal", canonical_description) or canonical_description).strip()
        self._ensure_branch(
            task,
            branch_id,
            branch_goal=branch_goal,
            current_focus=current_focus,
            created_at=record.current_time,
        )
        if parent is not None:
            self._ensure_legacy_branch_registry(parent)
            if task.task_id not in parent.child_task_ids:
                parent.child_task_ids.append(task.task_id)
            if parent_branch_id:
                branch = parent.branches[parent_branch_id]
                if task.task_id not in branch.child_task_ids:
                    branch.child_task_ids.append(task.task_id)
        self._log_task_chain_event(
            task.task_id,
            "task_chain_created",
            task_id=task.task_id,
            owner_id=task.owner_id,
            task_description=task.task_description,
            status=task.status.value,
            created_at=task.created_at,
            source_record_id=record.record_id,
            entities=list(task.entities),
            canonical_description=task.canonical_description,
            current_focus=task.current_focus,
            parent_task_id=task.parent_task_id,
            parent_branch_id=parent_branch_id,
            branch_id=branch_id,
            branch_goal=branch_goal,
        )
        if parent_task_id and parent_task_id in self.tasks:
            self._log_task_chain_event(
                parent_task_id,
                "child_task_linked",
                parent_task_id=parent_task_id,
                child_task_id=task.task_id,
                parent_branch_id=parent_branch_id,
                relation="decomposition",
                created_at=record.current_time,
            )
        return task

    def apply_record(self, task_id: str, record: DialogueRecord, *, summary: str | None = None) -> ChainUpdateResult:
        return self._apply_record_with_intent(task_id, record, summary=summary)

    def _apply_record_with_intent(
        self,
        task_id: str,
        record: DialogueRecord,
        *,
        summary: str | None = None,
        record_intent: IntentUnderstanding | None = None,
    ) -> ChainUpdateResult:
        task = self.tasks[task_id]
        self._ensure_legacy_branch_registry(task)

        prior_application = self._get_record_application(task, record.record_id)
        if prior_application is not None:
            if record.record_id not in task.record_ids:
                task.record_ids.append(record.record_id)
            self._merge_task_entities(task, record)
            task.updated_at = record.current_time
            return self._result_from_application(task.task_id, record.record_id, prior_application)

        existing_node = self._find_node_for_record(task, record.record_id)
        if existing_node is not None:
            result = ChainUpdateResult(
                task_id=task.task_id,
                record_id=record.record_id,
                action="duplicate",
                chain_node_id=existing_node.node_id,
                reference_node_id=existing_node.node_id,
                branch_id=existing_node.branch_id,
                status=existing_node.status,
                reason="record_already_applied",
            )
            if record.record_id not in task.record_ids:
                task.record_ids.append(record.record_id)
            self._merge_task_entities(task, record)
            self._store_record_application(
                task,
                record,
                result,
                summary=existing_node.summary,
                position=existing_node.position,
            )
            task.updated_at = record.current_time
            return result

        decision = self._record_conflict(task, record, record_intent=record_intent)
        if summary:
            decision.summary = summary
        resolved_reference_node_id, target_node = self._resolve_reference_node(task, decision.reference_node_id)
        decision.reference_node_id = resolved_reference_node_id
        if decision.action in {"override", "branch", "duplicate"} and not decision.reference_node_id:
            raise ValueError(f"Conflict action {decision.action!r} requires reference_node_id for record {record.record_id}")
        if decision.reference_node_id and target_node is None:
            raise ValueError(
                f"Conflict resolution referenced unknown node_id {decision.reference_node_id!r} "
                f"for task {task.task_id} record {record.record_id}"
            )

        if decision.action == "duplicate":
            result = ChainUpdateResult(
                task_id=task.task_id,
                record_id=record.record_id,
                action="duplicate",
                chain_node_id=target_node.node_id if target_node else None,
                reference_node_id=target_node.node_id if target_node else None,
                branch_id=target_node.branch_id if target_node else None,
                status=target_node.status if target_node else None,
                reason=decision.reason,
            )
            if record.record_id not in task.record_ids:
                task.record_ids.append(record.record_id)
            self._merge_task_entities(task, record)
            self._store_record_application(
                task,
                record,
                result,
                summary=decision.summary or (target_node.summary if target_node else record.combined_content),
                position=target_node.position if target_node else 0,
            )
            if record_intent is not None:
                application = self._get_record_application(task, record.record_id)
                if application is not None:
                    application["record_intent"] = to_primitive(record_intent)
            task.updated_at = record.current_time
            self._maybe_refresh_task_metadata(task)
            return result

        if decision.action == "branch":
            requested_branch_id = (decision.branch_id or "").strip()
            if not requested_branch_id or requested_branch_id == "main":
                branch_id = f"b{task.next_branch_index}"
                task.next_branch_index += 1
            else:
                branch_id = requested_branch_id
                if branch_id.startswith("b") and branch_id[1:].isdigit():
                    task.next_branch_index = max(task.next_branch_index, int(branch_id[1:]) + 1)
            status = ChainNodeStatus.BRANCHED
        elif decision.action in {"create", "override"}:
            branch_id = decision.branch_id or (target_node.branch_id if target_node else task.active_branch_id or "main")
            status = ChainNodeStatus.ACTIVE
        else:
            raise ValueError(f"Unknown conflict action {decision.action!r} for record {record.record_id}")

        prior_active_branch_id = task.active_branch_id or task.preferred_branch_id or "main"

        self._ensure_branch(
            task,
            branch_id,
            branch_goal=decision.branch_goal or task.canonical_description or task.task_description,
            current_focus=decision.summary,
            parent_node_id=target_node.node_id if target_node is not None and decision.action == "branch" else None,
            created_at=record.current_time,
        )

        prior_head = task.nodes.get(task.branch_heads.get(branch_id, ""))
        new_node = self._make_node(task, record, decision, status, branch_id)
        if record_intent is not None:
            new_node.metadata["record_intent"] = to_primitive(record_intent)

        old_node_change: dict[str, Any] | None = None
        if target_node is not None:
            old_node_change = {
                "node_id": target_node.node_id,
                "status_before": target_node.status.value,
                "status_after": target_node.status.value,
                "superseded_by_before": target_node.superseded_by,
                "superseded_by": target_node.superseded_by,
                "next_node_ids_before": list(target_node.next_node_ids),
                "next_node_ids_after": list(target_node.next_node_ids),
            }
            self._append_edge(target_node, new_node)
        elif decision.action == "create":
            self._append_edge(prior_head, new_node)

        if decision.action == "override" and target_node is not None:
            target_node.status = ChainNodeStatus.DEPRECATED
            target_node.superseded_by = new_node.node_id

        if old_node_change is not None and target_node is not None:
            old_node_change["status_after"] = target_node.status.value
            old_node_change["superseded_by"] = target_node.superseded_by
            old_node_change["next_node_ids_after"] = list(target_node.next_node_ids)

        task.nodes[new_node.node_id] = new_node
        if record.record_id not in task.record_ids:
            task.record_ids.append(record.record_id)
        self._merge_task_entities(task, record)
        task.branch_heads[branch_id] = new_node.node_id
        task.active_branch_id = branch_id
        branch = self._ensure_branch(task, branch_id, head_node_id=new_node.node_id)
        branch.head_node_id = new_node.node_id
        branch.current_focus = _short_text(new_node.summary or task.current_focus, 256)
        branch.updated_at = record.current_time
        branch.last_activity_at = record.current_time
        task.preferred_branch_id = branch_id
        task.current_focus = branch.current_focus
        if decision.action == "branch" and prior_active_branch_id != branch_id:
            self._log_task_chain_event(
                task.task_id,
                "branch_selected",
                task_id=task.task_id,
                old_branch_id=prior_active_branch_id,
                new_branch_id=branch_id,
                updated_at=record.current_time,
                reason="conflict_resolution_branch_action",
            )
        self._apply_task_metadata_updates(task, decision)
        task.updated_at = record.current_time

        result = ChainUpdateResult(
            task_id=task.task_id,
            record_id=record.record_id,
            action=decision.action,
            chain_node_id=new_node.node_id,
            reference_node_id=target_node.node_id if target_node else None,
            branch_id=branch_id,
            status=new_node.status,
            reason=decision.reason,
        )
        self._store_record_application(task, record, result, summary=new_node.summary, position=new_node.position)
        if record_intent is not None:
            application = self._get_record_application(task, record.record_id)
            if application is not None:
                application["record_intent"] = to_primitive(record_intent)
        self._maybe_refresh_task_metadata(task)
        self._log_task_chain_event(
            task.task_id,
            "node_added",
            task_id=task.task_id,
            node_id=new_node.node_id,
            action=result.action,
            branch_id=new_node.branch_id,
            position=new_node.position,
            prev_node_ids=list(new_node.prev_node_ids),
            next_node_ids=list(new_node.next_node_ids),
            source_record_id=new_node.source_record_id,
            source_turn_ids=list(new_node.source_turn_ids),
            record_count=len(task.record_ids),
            entity_count=len(task.entities),
            status=new_node.status.value,
            reference_node_id=result.reference_node_id,
            reason=result.reason,
            summary=new_node.summary,
            record_intent=to_primitive(record_intent),
            old_node_change=old_node_change,
            updated_at=task.updated_at,
        )
        return result

    def get_task(self, task_id: str) -> TaskChain | None:
        return self.tasks.get(task_id)

    def expand_task_ids(self, task_ids: list[str], *, max_tasks: int = 20) -> list[str]:
        """Return deterministic task IDs while preserving the legacy API."""
        expanded, _edges = self.expand_task_graph(task_ids, max_tasks=max_tasks)
        return expanded

    def expand_task_graph(
        self,
        task_ids: list[str],
        *,
        max_tasks: int = 20,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Expand primary routes and expose deterministic hierarchy edges.

        The LLM still chooses a fixed-size primary route. Expansion is a
        local graph operation, so callers can distinguish model-selected
        tasks from evidence added through parent/child links.
        """
        limit = max(1, int(max_tasks or 20))
        expanded: list[str] = []
        seen: set[str] = set()
        depths: dict[str, int] = {}
        edges: list[dict[str, Any]] = []

        def add(task_id: str, *, depth: int) -> bool:
            if task_id in seen or task_id not in self.tasks or len(expanded) >= limit:
                return False
            seen.add(task_id)
            expanded.append(task_id)
            depths[task_id] = depth
            return True

        primary = [str(task_id) for task_id in task_ids]
        for task_id in primary:
            add(task_id, depth=0)
        # Breadth-first expansion keeps primary order while supporting an
        # arbitrarily deep task tree until the deterministic budget is full.
        queue: list[tuple[str, str | None]] = [(task_id, None) for task_id in expanded]
        processed: set[tuple[str, str | None]] = set()
        while queue and len(expanded) < limit:
            task_id, branch_filter = queue.pop(0)
            state_key = (task_id, branch_filter)
            if state_key in processed:
                continue
            processed.add(state_key)
            task = self.tasks.get(task_id)
            if task is None:
                continue
            source_depth = depths.get(task_id, 0)
            neighbors: list[tuple[str, str]] = []
            if task.parent_task_id:
                neighbors.append((str(task.parent_task_id), "parent"))
            child_ids = []
            for child_id in task.child_task_ids:
                child = self.tasks.get(str(child_id))
                if child is None or child.parent_task_id != task.task_id:
                    continue
                child_branch_id = child.parent_branch_id
                if branch_filter is not None and child_branch_id != branch_filter:
                    continue
                if branch_filter is None and child_branch_id:
                    branch = task.branches.get(child_branch_id)
                    if branch is not None and branch.status is not TaskStatus.ACTIVE:
                        continue
                child_ids.append((str(child_id), "branch_child" if child_branch_id else "child"))
            neighbors.extend(child_ids)
            for branch in task.branches.values():
                if branch_filter is not None and branch.branch_id != branch_filter:
                    continue
                if branch_filter is None and branch.status is not TaskStatus.ACTIVE:
                    continue
                for child_id in branch.child_task_ids:
                    child = self.tasks.get(str(child_id))
                    if child is None or child.parent_task_id != task.task_id:
                        continue
                    if branch_filter is not None and child.parent_branch_id != branch_filter:
                        continue
                    neighbors.append((str(child_id), "branch_child"))
            for neighbor, relation in neighbors:
                if add(neighbor, depth=source_depth + 1):
                    edges.append(
                        {
                            "from": task_id,
                            "to": neighbor,
                            "relation": relation,
                            "depth": source_depth + 1,
                        }
                    )
                    next_filter = branch_filter
                    if relation == "parent":
                        source = self.tasks.get(task_id)
                        next_filter = source.parent_branch_id if source is not None else None
                    queue.append((neighbor, next_filter))
        return expanded, edges

    def task_routing_profile(self, task: TaskChain) -> dict[str, Any]:
        """Return the compact, stable routing projection for one task."""
        self._ensure_legacy_branch_registry(task)
        branches = []
        for branch in sorted(task.branches.values(), key=lambda item: item.branch_id):
            branches.append(
                {
                    "branch_id": branch.branch_id,
                    "branch_goal": _short_text(branch.branch_goal, 256),
                    "current_focus": _short_text(branch.current_focus, 256),
                    "status": branch.status.value,
                    "head_node_id": branch.head_node_id,
                    "child_task_ids": list(branch.child_task_ids)[:8],
                }
            )
        return {
            "task_id": task.task_id,
            "canonical_description": _short_text(
                task.canonical_description or task.task_description,
                self.query_router_summary_limit,
            ),
            "current_focus": _short_text(task.current_focus, 256),
            "status": task.status.value,
            "entities": [_short_text(entity, 64) for entity in task.entities[: self.router_entity_limit]],
            "parent_task_id": task.parent_task_id,
            "parent_branch_id": task.parent_branch_id,
            "child_task_ids": list(task.child_task_ids)[:8],
            "branches": branches[:8],
            "updated_at": task.updated_at or task.created_at,
        }

    def get_branch(self, task_id: str, branch_id: str) -> TaskBranch | None:
        task = self.tasks.get(task_id)
        if task is None:
            return None
        self._ensure_legacy_branch_registry(task)
        return task.branches.get(branch_id)

    def create_branch(
        self,
        task_id: str,
        branch_goal: str,
        *,
        parent_node_id: str | None = None,
        branch_id: str | None = None,
        current_focus: str = "",
        current_time: str = "",
        reason: str = "",
    ) -> TaskBranch:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        goal = str(branch_goal or "").strip()
        if not goal:
            raise ValueError("branch_goal is required")
        self._ensure_legacy_branch_registry(task)
        if parent_node_id and parent_node_id not in task.nodes:
            raise ValueError(f"unknown parent_node_id {parent_node_id!r} for task {task_id}")
        requested_id = str(branch_id or "").strip()
        if requested_id:
            if requested_id in task.branches:
                raise ValueError(f"branch {requested_id!r} already exists for task {task_id}")
            resolved_id = requested_id
        else:
            resolved_id = self._next_branch_id(task)
        branch = self._ensure_branch(
            task,
            resolved_id,
            branch_goal=goal,
            current_focus=current_focus,
            parent_node_id=parent_node_id,
            created_at=current_time or task.updated_at or task.created_at,
        )
        task.status = TaskStatus.ACTIVE
        if current_time:
            task.updated_at = current_time
        self._log_task_chain_event(
            task.task_id,
            "branch_created",
            task_id=task.task_id,
            branch_id=resolved_id,
            branch_goal=branch.branch_goal,
            current_focus=branch.current_focus,
            parent_node_id=parent_node_id,
            created_at=branch.created_at,
            reason=reason,
        )
        return branch

    def select_branch(
        self,
        task_id: str,
        branch_id: str,
        *,
        current_time: str = "",
        reason: str = "",
    ) -> TaskBranch:
        """Select an existing branch for subsequent ordinary node writes."""
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        self._ensure_legacy_branch_registry(task)
        branch = task.branches.get(str(branch_id))
        if branch is None:
            raise KeyError(f"unknown branch_id {branch_id!r} for task {task_id}")
        if branch.status is TaskStatus.MERGED:
            raise ValueError(f"cannot select merged branch {branch_id!r}")
        old_branch_id = task.active_branch_id or task.preferred_branch_id or "main"
        task.active_branch_id = branch.branch_id
        task.preferred_branch_id = branch.branch_id
        if branch.current_focus:
            task.current_focus = branch.current_focus
        if current_time:
            task.updated_at = current_time
            branch.updated_at = current_time
            branch.last_activity_at = current_time
        self._log_task_chain_event(
            task.task_id,
            "branch_selected",
            task_id=task.task_id,
            old_branch_id=old_branch_id,
            new_branch_id=branch.branch_id,
            updated_at=current_time or task.updated_at,
            reason=reason,
        )
        return branch

    def update_branch_status(
        self,
        task_id: str,
        branch_id: str,
        status: TaskStatus | str,
        *,
        current_focus: str | None = None,
        current_time: str = "",
        reason: str = "",
    ) -> TaskBranch:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        self._ensure_legacy_branch_registry(task)
        branch = task.branches.get(branch_id)
        if branch is None:
            raise KeyError(f"unknown branch_id {branch_id!r} for task {task_id}")
        resolved_status = _coerce_task_status(status, strict=True)
        if resolved_status is TaskStatus.MERGED:
            raise ValueError("MERGED is managed by merge_branches()")
        old_status = branch.status
        branch.status = resolved_status
        if current_focus is not None:
            branch.current_focus = _short_text(current_focus, 256)
            if task.preferred_branch_id == branch_id or task.active_branch_id == branch_id:
                task.current_focus = branch.current_focus
        if current_time:
            branch.updated_at = current_time
            branch.last_activity_at = current_time
            task.updated_at = current_time
        self._sync_task_status_from_branches(task)
        self._log_task_chain_event(
            task.task_id,
            "branch_status_updated",
            task_id=task.task_id,
            branch_id=branch_id,
            status=branch.status.value,
            old_status=old_status.value,
            new_status=branch.status.value,
            current_focus=branch.current_focus,
            updated_at=branch.updated_at,
            reason=reason,
        )
        return branch

    def merge_branches(
        self,
        task_id: str,
        source_branch_id: str,
        target_branch_id: str,
        *,
        current_time: str = "",
        reason: str = "",
    ) -> TaskBranch:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task_id {task_id!r}")
        self._ensure_legacy_branch_registry(task)
        if source_branch_id == target_branch_id:
            raise ValueError("source_branch_id and target_branch_id must differ")
        source = task.branches.get(source_branch_id)
        target = task.branches.get(target_branch_id)
        if source is None or target is None:
            raise KeyError(f"unknown branch for merge: {source_branch_id!r}, {target_branch_id!r}")
        if source.status is TaskStatus.MERGED:
            raise ValueError(f"branch {source_branch_id!r} is already merged")
        if target.status is TaskStatus.MERGED:
            raise ValueError(f"cannot merge into merged branch {target_branch_id!r}")
        reparented_child_ids: list[str] = []
        historical_child_ids: list[str] = []
        for child_id in list(source.child_task_ids):
            child = self.tasks.get(child_id)
            if child is None:
                continue
            historical_child_ids.append(child_id)
            if child.parent_task_id == task.task_id and child.parent_branch_id == source_branch_id:
                child.parent_branch_id = target_branch_id
                if child_id not in target.child_task_ids:
                    target.child_task_ids.append(child_id)
                reparented_child_ids.append(child_id)
        source.child_task_ids = [child_id for child_id in source.child_task_ids if child_id not in reparented_child_ids]
        if historical_child_ids:
            prior_history = list(source.metadata.get("merged_child_task_ids", []) or [])
            source.metadata["merged_child_task_ids"] = list(dict.fromkeys([*prior_history, *historical_child_ids]))
        source.status = TaskStatus.MERGED
        source.merged_into_branch_id = target_branch_id
        source.updated_at = current_time or source.updated_at
        if current_time:
            source.last_activity_at = current_time
            task.updated_at = current_time
        self._sync_task_status_from_branches(task)
        self._log_task_chain_event(
            task.task_id,
            "branch_merged",
            task_id=task.task_id,
            source_branch_id=source_branch_id,
            target_branch_id=target_branch_id,
            status=source.status.value,
            merged_into_branch_id=target_branch_id,
            reparented_child_task_ids=reparented_child_ids,
            updated_at=source.updated_at,
            reason=reason,
        )
        for child_id in reparented_child_ids:
            self._log_task_chain_event(
                task.task_id,
                "child_task_reparented",
                parent_task_id=task.task_id,
                child_task_id=child_id,
                old_parent_branch_id=source_branch_id,
                new_parent_branch_id=target_branch_id,
                reason="branch_merge",
                updated_at=current_time or task.updated_at,
            )
        return target

    def create_child_task(
        self,
        parent_task_id: str,
        canonical_description: str,
        *,
        current_focus: str = "",
        entities: list[str] | None = None,
        branch_goal: str = "",
        parent_branch_id: str | None = None,
        current_time: str = "",
        task_id: str | None = None,
        reason: str = "decomposition",
    ) -> TaskChain:
        parent = self.tasks.get(parent_task_id)
        if parent is None:
            raise KeyError(f"unknown parent task_id {parent_task_id!r}")
        description = str(canonical_description or "").strip()
        if not description:
            raise ValueError("child task canonical_description is required")
        child_id = str(task_id or "").strip() or f"task_{uuid4().hex[:8]}"
        if child_id in self.tasks:
            raise ValueError(f"task {child_id!r} already exists")
        self._ensure_legacy_branch_registry(parent)
        resolved_parent_branch_id = str(parent_branch_id or "").strip() or None
        if resolved_parent_branch_id and resolved_parent_branch_id not in parent.branches:
            raise ValueError(
                f"unknown parent_branch_id {resolved_parent_branch_id!r} for task {parent_task_id}"
            )
        timestamp = current_time or parent.updated_at or parent.created_at
        child = TaskChain(
            task_id=child_id,
            task_description=description,
            owner_id=self.owner_id,
            created_at=timestamp,
            updated_at=timestamp,
            entities=self._normalize_entities(entities or []),
            canonical_description=description,
            current_focus=str(current_focus or "").strip(),
            parent_task_id=parent_task_id,
            parent_branch_id=resolved_parent_branch_id,
        )
        self.tasks[child_id] = child
        self._ensure_branch(
            child,
            "main",
            branch_goal=str(branch_goal or description).strip(),
            current_focus=child.current_focus,
            created_at=timestamp,
        )
        if child_id not in parent.child_task_ids:
            parent.child_task_ids.append(child_id)
        if resolved_parent_branch_id:
            branch = parent.branches[resolved_parent_branch_id]
            if child_id not in branch.child_task_ids:
                branch.child_task_ids.append(child_id)
            if current_time:
                branch.updated_at = current_time
                branch.last_activity_at = current_time
        if current_time:
            parent.updated_at = current_time
        self._log_task_chain_event(
            child_id,
            "task_chain_created",
            task_id=child_id,
            task_description=child.task_description,
            canonical_description=child.canonical_description,
            current_focus=child.current_focus,
            parent_task_id=parent_task_id,
            parent_branch_id=resolved_parent_branch_id,
            status=child.status.value,
            created_at=child.created_at,
            entities=list(child.entities),
            relation=reason,
        )
        self._log_task_chain_event(
            parent_task_id,
            "child_task_linked",
            parent_task_id=parent_task_id,
            child_task_id=child_id,
            parent_branch_id=resolved_parent_branch_id,
            relation=reason,
            created_at=timestamp,
        )
        return child

    def _next_branch_id(self, task: TaskChain) -> str:
        while True:
            branch_id = f"b{task.next_branch_index}"
            task.next_branch_index += 1
            if branch_id not in task.branches:
                return branch_id

    def _ensure_branch(
        self,
        task: TaskChain,
        branch_id: str,
        *,
        branch_goal: str = "",
        current_focus: str = "",
        parent_node_id: str | None = None,
        created_at: str = "",
        head_node_id: str | None = None,
        status: TaskStatus = TaskStatus.ACTIVE,
    ) -> TaskBranch:
        branch = task.branches.get(branch_id)
        if branch is None:
            branch = TaskBranch(
                branch_id=branch_id,
                task_id=task.task_id,
                branch_goal=str(branch_goal or task.canonical_description or task.task_description).strip(),
                current_focus=_short_text(current_focus, 256),
                status=status,
                head_node_id=head_node_id or task.branch_heads.get(branch_id),
                parent_node_id=parent_node_id,
                created_at=created_at or task.created_at,
                updated_at=created_at or task.updated_at or task.created_at,
                last_activity_at=created_at or task.updated_at or task.created_at,
            )
            task.branches[branch_id] = branch
        else:
            if branch_goal and not branch.branch_goal:
                branch.branch_goal = str(branch_goal).strip()
            if current_focus:
                branch.current_focus = _short_text(current_focus, 256)
            if parent_node_id and branch.parent_node_id is None:
                branch.parent_node_id = parent_node_id
            if head_node_id:
                branch.head_node_id = head_node_id
        if branch.head_node_id:
            task.branch_heads[branch_id] = branch.head_node_id
        elif branch_id in task.branch_heads:
            branch.head_node_id = task.branch_heads[branch_id]
        task.preferred_branch_id = task.preferred_branch_id or task.active_branch_id or branch_id
        return branch

    def _ensure_legacy_branch_registry(self, task: TaskChain) -> None:
        if not task.canonical_description:
            task.canonical_description = task.task_description
        if task.preferred_branch_id is None:
            task.preferred_branch_id = task.active_branch_id or "main"
        branch_ids = set(task.branch_heads) | set(task.branches)
        branch_ids.add(task.active_branch_id or "main")
        for node in task.nodes.values():
            branch_ids.add(node.branch_id or "main")
        for branch_id in sorted(branch_ids):
            self._ensure_branch(
                task,
                branch_id,
                current_focus=task.current_focus if branch_id == task.preferred_branch_id else "",
                head_node_id=task.branch_heads.get(branch_id),
            )

    def _sync_task_status_from_branches(self, task: TaskChain) -> None:
        statuses = [branch.status for branch in task.branches.values()]
        if not statuses:
            return
        if any(status is TaskStatus.ACTIVE for status in statuses):
            task.status = TaskStatus.ACTIVE
        elif any(status is TaskStatus.BLOCKED for status in statuses):
            task.status = TaskStatus.BLOCKED
        elif all(status in {TaskStatus.COMPLETED, TaskStatus.MERGED, TaskStatus.CANCELLED} for status in statuses):
            task.status = TaskStatus.COMPLETED

    def rebuild_hierarchy_links(self) -> None:
        """Normalize parent/child links and branch heads after state loading.

        Child-side parent fields are the source of truth. Legacy parent-side
        lists may be stale or absent, so they are rebuilt in task insertion
        order after all tasks and branches are available.
        """
        for task in self.tasks.values():
            self._ensure_legacy_branch_registry(task)
            task.child_task_ids = []
            for branch_id, branch in task.branches.items():
                if branch.task_id and branch.task_id != task.task_id:
                    raise ValueError(
                        f"branch {branch_id} belongs to {branch.task_id}, expected {task.task_id}"
                    )
                if branch.branch_id and branch.branch_id != branch_id:
                    raise ValueError(
                        f"branch key {branch_id} disagrees with branch id {branch.branch_id}"
                    )
                branch.task_id = task.task_id
                branch.branch_id = branch_id
                branch.child_task_ids = []

            # A stale branch head is less trustworthy than the node's branch
            # identity. Recompute heads from the latest node in each branch.
            task.branch_heads = {}
            nodes_by_branch: dict[str, list[TaskChainNode]] = {}
            for node in task.nodes.values():
                branch_id = str(node.branch_id or "main")
                node.branch_id = branch_id
                self._ensure_branch(task, branch_id)
                nodes_by_branch.setdefault(branch_id, []).append(node)
            for branch_id, branch in task.branches.items():
                nodes = nodes_by_branch.get(branch_id, [])
                if nodes:
                    head = max(nodes, key=lambda item: (item.position, item.created_at, item.node_id))
                    branch.head_node_id = head.node_id
                    task.branch_heads[branch_id] = head.node_id
                else:
                    branch.head_node_id = None

        for task in self.tasks.values():
            parent_id = task.parent_task_id
            if not parent_id:
                task.parent_branch_id = None
                continue
            parent = self.tasks.get(parent_id)
            if parent is None:
                task.parent_task_id = None
                task.parent_branch_id = None
                continue
            if parent_id == task.task_id:
                raise ValueError(f"task {task.task_id} cannot be its own parent")
            parent.child_task_ids.append(task.task_id)
            parent_branch_id = task.parent_branch_id
            if not parent_branch_id:
                continue
            branch = parent.branches.get(parent_branch_id)
            if branch is None:
                task.parent_branch_id = None
                continue
            branch.child_task_ids.append(task.task_id)

        for task in self.tasks.values():
            task.child_task_ids = list(dict.fromkeys(task.child_task_ids))
            for branch in task.branches.values():
                branch.child_task_ids = list(dict.fromkeys(branch.child_task_ids))

        # Parent links form a forest. Detect a cycle after dangling links have
        # been cleared so corrupted persisted state cannot poison expansion.
        for start in self.tasks.values():
            path: set[str] = set()
            current = start
            while current.parent_task_id:
                if current.task_id in path:
                    cycle = " -> ".join(sorted(path))
                    raise ValueError(f"task parent cycle detected: {cycle}")
                path.add(current.task_id)
                parent = self.tasks.get(current.parent_task_id)
                if parent is None:
                    break
                current = parent

    def validate_hierarchy(self) -> None:
        """Raise ``ValueError`` when persisted hierarchy invariants are broken."""
        task_ids = set(self.tasks)
        for task in self.tasks.values():
            if task.parent_task_id:
                if task.parent_task_id == task.task_id:
                    raise ValueError(f"task {task.task_id} cannot be its own parent")
                if task.parent_task_id not in task_ids:
                    raise ValueError(
                        f"task {task.task_id} references missing parent {task.parent_task_id}"
                    )
                if task.parent_branch_id and task.parent_branch_id not in self.tasks[task.parent_task_id].branches:
                    raise ValueError(
                        f"task {task.task_id} references missing parent branch {task.parent_branch_id}"
                    )
            for branch_id, branch in task.branches.items():
                if branch.task_id != task.task_id:
                    raise ValueError(
                        f"branch {branch_id} belongs to {branch.task_id}, expected {task.task_id}"
                    )
                if branch.head_node_id:
                    head = task.nodes.get(branch.head_node_id)
                    if head is None:
                        raise ValueError(
                            f"branch {branch_id} references missing head node {branch.head_node_id}"
                        )
                    if head.task_id != task.task_id or head.branch_id != branch_id:
                        raise ValueError(
                            f"branch {branch_id} head {branch.head_node_id} has inconsistent ownership"
                        )
            for node in task.nodes.values():
                if node.task_id != task.task_id:
                    raise ValueError(
                        f"node {node.node_id} belongs to {node.task_id}, expected {task.task_id}"
                    )
                if node.branch_id not in task.branches:
                    raise ValueError(
                        f"node {node.node_id} references missing branch {node.branch_id}"
                    )
            for child_id in task.child_task_ids:
                child = self.tasks.get(child_id)
                if child is None:
                    raise ValueError(f"task {task.task_id} has dangling child {child_id}")
                if child.parent_task_id != task.task_id:
                    raise ValueError(
                        f"task {task.task_id} child link {child_id} disagrees with child parent"
                    )
            for branch_id, branch in task.branches.items():
                for child_id in branch.child_task_ids:
                    child = self.tasks.get(child_id)
                    if child is None:
                        raise ValueError(f"branch {branch_id} has dangling child {child_id}")
                    if child.parent_task_id != task.task_id or child.parent_branch_id != branch_id:
                        raise ValueError(
                            f"branch {branch_id} child link {child_id} disagrees with child parent branch"
                        )
        for start in self.tasks.values():
            path: set[str] = set()
            current = start
            while current.parent_task_id:
                if current.task_id in path:
                    raise ValueError(f"task parent cycle detected at {current.task_id}")
                path.add(current.task_id)
                current = self.tasks[current.parent_task_id]

    def _merge_task_entities(self, task: TaskChain, record: DialogueRecord) -> None:
        task.entities = self._normalize_entities([*task.entities, *record.entities])

    def _apply_task_metadata_updates(self, task: TaskChain, decision: ConflictDecision) -> None:
        if decision.task_description_update is not None:
            updated = decision.task_description_update.strip()
            if updated:
                task.current_focus = _short_text(updated, 256)
                branch_id = task.preferred_branch_id or task.active_branch_id or "main"
                branch = self._ensure_branch(task, branch_id)
                branch.current_focus = task.current_focus

    def _find_node_for_record(self, task: TaskChain, record_id: str) -> TaskChainNode | None:
        for node in task.nodes.values():
            if node.source_record_id == record_id:
                return node
        return None

    def _resolve_reference_node(self, task: TaskChain, reference: str | None) -> tuple[str | None, TaskChainNode | None]:
        if not reference:
            return None, None
        direct = task.nodes.get(reference)
        if direct is not None:
            return reference, direct
        by_record = self._find_node_for_record(task, reference)
        if by_record is not None:
            return by_record.node_id, by_record
        return reference, None

    def _record_application_store(self, task: TaskChain) -> dict[str, dict[str, Any]]:
        store = task.metadata.get("record_applications")
        if not isinstance(store, dict):
            store = {}
            task.metadata["record_applications"] = store
        return store

    def _get_record_application(self, task: TaskChain, record_id: str) -> dict[str, Any] | None:
        application = self._record_application_store(task).get(record_id)
        return application if isinstance(application, dict) else None

    def _store_record_application(
        self,
        task: TaskChain,
        record: DialogueRecord,
        result: ChainUpdateResult,
        *,
        summary: str,
        position: int = 0,
    ) -> None:
        self._record_application_store(task)[record.record_id] = {
            "record_id": record.record_id,
            "user_content": record.user_content,
            "assistant_content": record.assistant_content,
            "summary": summary,
            "action": result.action,
            "reason": result.reason,
            "chain_node_id": result.chain_node_id,
            "reference_node_id": result.reference_node_id,
            "branch_id": result.branch_id,
            "status": result.status.value if result.status else None,
            "position": position,
        }

    def _result_from_application(self, task_id: str, record_id: str, application: dict[str, Any]) -> ChainUpdateResult:
        status_value = application.get("status")
        status = ChainNodeStatus(status_value) if status_value else None
        return ChainUpdateResult(
            task_id=task_id,
            record_id=record_id,
            action=str(application.get("action", "duplicate") or "duplicate"),
            chain_node_id=str(application.get("chain_node_id")) if application.get("chain_node_id") else None,
            reference_node_id=str(application.get("reference_node_id")) if application.get("reference_node_id") else None,
            branch_id=str(application.get("branch_id")) if application.get("branch_id") else None,
            status=status,
            reason=str(application.get("reason", "") or ""),
        )

    def _recent_context(self, task: TaskChain, limit: int = 20) -> list[dict[str, Any]]:
        context: list[dict[str, Any]] = []
        application_store = self._record_application_store(task)
        for record_id in task.record_ids[-limit:]:
            node = self._find_node_for_record(task, record_id)
            application = application_store.get(record_id) if not node else None
            if node is None and application:
                preferred_node_id = application.get("chain_node_id") or application.get("reference_node_id")
                if preferred_node_id in task.nodes:
                    node = task.nodes[preferred_node_id]
                    application = None
            intent_payload: dict[str, Any]
            if node is not None and isinstance(node.metadata.get("record_intent"), dict):
                intent_payload = dict(node.metadata["record_intent"])
            elif application and isinstance(application.get("record_intent"), dict):
                intent_payload = dict(application["record_intent"])
            else:
                intent_payload = {
                    "summary": "",
                    "main_intent": "",
                    "key_entities": [],
                }
            context.append(
                {
                    "record_id": record_id,
                    "node_id": node.node_id if node else (str(application.get("chain_node_id")) if application and application.get("chain_node_id") else None),
                    "user_content": node.user_content if node else str(application.get("user_content", "") if application else ""),
                    "assistant_content": node.assistant_content if node else (str(application["assistant_content"]) if application and application.get("assistant_content") is not None else None),
                    "summary": node.summary if node else str(application.get("summary", "") if application else ""),
                    "status": node.status.value if node else str(application.get("status", "") if application else ""),
                    "branch_id": node.branch_id if node else str(application.get("branch_id", "") if application else ""),
                    "position": node.position if node else int(application.get("position", 0) if application else 0),
                    "intent": {
                        "summary": str(intent_payload.get("summary", "") or ""),
                        "main_intent": str(intent_payload.get("main_intent", "") or ""),
                        "key_entities": [str(item) for item in intent_payload.get("key_entities", []) or []],
                    },
                }
            )
        return context

    def _record_conflict(
        self,
        task: TaskChain,
        record: DialogueRecord,
        *,
        record_intent: IntentUnderstanding | None = None,
    ) -> ConflictDecision:
        record_payload = to_primitive(record)
        if record_intent is not None:
            record_payload.update(
                {
                    "summary": record_intent.summary,
                    "main_intent": record_intent.main_intent,
                    "key_entities": list(record_intent.key_entities),
                }
            )
        else:
            record_payload.setdefault("summary", "")
            record_payload.setdefault("main_intent", "")
            record_payload.setdefault("key_entities", [])
        payload = {
            "task": {
                "task_id": task.task_id,
                "task_description": task.canonical_description or task.task_description,
                "current_focus": task.current_focus,
                "status": task.status.value,
                "parent_task_id": task.parent_task_id,
                "parent_branch_id": task.parent_branch_id,
                "child_task_ids": list(task.child_task_ids)[:8],
                "branches": self.task_routing_profile(task)["branches"][:8],
            },
            "record": record_payload,
            "recent_context": self._recent_context(task),
        }
        rendered = self.prompt_registry.render("record_conflict_resolution", payload_json=_json_dumps(payload))
        parsed = self._call_llm_json(
            PromptBundle(
                name="record_conflict_resolution",
                system_prompt=rendered.system_prompt,
                user_prompt=rendered.user_prompt,
            )
        )
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"action"})
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM record conflict resolution returned invalid payload for record {record.record_id}")
        decision = ConflictDecision(
            action=str(parsed.get("action", "create") or "create"),
            reason=str(parsed.get("reason", "") or ""),
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reference_node_id=str(parsed.get("reference_node_id")) if parsed.get("reference_node_id") else None,
            branch_id=str(parsed.get("branch_id")) if parsed.get("branch_id") else None,
            branch_goal=str(parsed.get("branch_goal", "") or "").strip(),
            summary=str(parsed.get("summary", "") or ""),
            task_description_update=(
                str(parsed.get("task_description_update")).strip()
                if parsed.get("task_description_update") is not None
                else None
            ),
        )
        log = getattr(self.log_store, "log", None)
        if log is not None:
            log(
                "conflicts",
                "record_conflict_decided",
                task_id=task.task_id,
                record_id=record.record_id,
                action=decision.action,
                reference_node_id=decision.reference_node_id,
                branch_id=decision.branch_id,
                branch_goal=decision.branch_goal,
                reason=decision.reason,
                summary=decision.summary,
                task_description_update=decision.task_description_update,
            )
        return decision

    def _make_node(
        self,
        task: TaskChain,
        record: DialogueRecord,
        decision: ConflictDecision,
        status: ChainNodeStatus,
        branch_id: str,
    ) -> TaskChainNode:
        task_suffix = task.task_id.removeprefix("task_")
        node_id = f"node_{task_suffix}_{task.next_node_index:04d}"
        node = TaskChainNode(
            node_id=node_id,
            task_id=task.task_id,
            user_content=record.user_content,
            assistant_content=record.assistant_content,
            summary=decision.summary or record.combined_content,
            status=status,
            branch_id=branch_id,
            position=task.next_node_index,
            created_at=record.current_time,
            source_record_id=record.record_id,
            source_turn_ids=list(record.source_turn_ids),
            metadata={},
        )
        task.next_node_index += 1
        return node

    def _append_edge(self, previous_node: TaskChainNode | None, next_node: TaskChainNode) -> None:
        if previous_node is None or previous_node.node_id == next_node.node_id:
            return
        if previous_node.node_id not in next_node.prev_node_ids:
            next_node.prev_node_ids.append(previous_node.node_id)
        if next_node.node_id not in previous_node.next_node_ids:
            previous_node.next_node_ids.append(next_node.node_id)

    def to_state(self) -> dict:
        self.validate_hierarchy()
        return {"owner_id": self.owner_id, "tasks": [to_primitive(task) for task in self.tasks.values()]}

    @classmethod
    def from_state(
        cls,
        state: dict,
        *,
        llm_client: object | None = None,
        task_metadata_refresh_interval: int = DEFAULT_TASK_METADATA_REFRESH_INTERVAL,
        router_entity_limit: int = DEFAULT_ROUTER_ENTITY_LIMIT,
        query_router_candidate_count: int = DEFAULT_QUERY_ROUTER_CANDIDATE_COUNT,
        query_router_pool_size: int = DEFAULT_QUERY_ROUTER_POOL_SIZE,
        query_router_summary_limit: int = DEFAULT_QUERY_ROUTER_SUMMARY_LIMIT,
        prompt_registry: PromptRegistry | None = None,
    ) -> "TaskChainManager":
        manager = cls(
            owner_id=state.get("owner_id", "default"),
            llm_client=llm_client,
            task_metadata_refresh_interval=task_metadata_refresh_interval,
            router_entity_limit=router_entity_limit,
            query_router_candidate_count=query_router_candidate_count,
            query_router_pool_size=query_router_pool_size,
            query_router_summary_limit=query_router_summary_limit,
            prompt_registry=prompt_registry,
        )
        for task_data in state.get("tasks", []) or []:
            task = TaskChain(
                task_id=task_data["task_id"],
                task_description=task_data.get("task_description") or task_data.get("topic", ""),
                owner_id=task_data.get("owner_id", manager.owner_id),
                status=_coerce_task_status(task_data.get("status"), strict=True),
                created_at=task_data.get("created_at", ""),
                updated_at=task_data.get("updated_at", ""),
                entities=task_data.get("entities", []) or [],
                record_ids=task_data.get("record_ids", []) or [],
                branch_heads=task_data.get("branch_heads", {}) or {},
                next_node_index=int(task_data.get("next_node_index", 1) or 1),
                next_branch_index=int(task_data.get("next_branch_index", 1) or 1),
                active_branch_id=task_data.get("active_branch_id", "main"),
                canonical_description=task_data.get("canonical_description", "") or task_data.get("task_description", ""),
                current_focus=task_data.get("current_focus", "") or "",
                parent_task_id=task_data.get("parent_task_id"),
                parent_branch_id=task_data.get("parent_branch_id"),
                child_task_ids=task_data.get("child_task_ids", []) or [],
                preferred_branch_id=task_data.get("preferred_branch_id"),
                metadata=task_data.get("metadata", {}) or {},
            )
            for branch_id, branch_data in (task_data.get("branches", {}) or {}).items():
                if not isinstance(branch_data, dict):
                    continue
                task.branches[branch_id] = TaskBranch(
                    branch_id=branch_data.get("branch_id", branch_id),
                    task_id=branch_data.get("task_id", task.task_id),
                    branch_goal=branch_data.get("branch_goal", "") or "",
                    current_focus=branch_data.get("current_focus", "") or "",
                    status=_coerce_task_status(branch_data.get("status"), strict=True),
                    head_node_id=branch_data.get("head_node_id"),
                    parent_node_id=branch_data.get("parent_node_id"),
                    child_task_ids=branch_data.get("child_task_ids", []) or [],
                    created_at=branch_data.get("created_at", "") or "",
                    updated_at=branch_data.get("updated_at", "") or "",
                    last_activity_at=branch_data.get("last_activity_at", "") or "",
                    merged_into_branch_id=branch_data.get("merged_into_branch_id"),
                    metadata=branch_data.get("metadata", {}) or {},
                )
            for node_id, node_data in (task_data.get("nodes", {}) or {}).items():
                task.nodes[node_id] = TaskChainNode(
                    node_id=node_data["node_id"],
                    task_id=node_data["task_id"],
                    user_content=node_data.get("user_content", ""),
                    assistant_content=node_data.get("assistant_content"),
                    summary=node_data.get("summary", ""),
                    status=ChainNodeStatus(node_data.get("status", ChainNodeStatus.ACTIVE.value)),
                    branch_id=node_data.get("branch_id", "main"),
                    position=int(node_data.get("position", 0) or 0),
                    prev_node_ids=node_data.get("prev_node_ids", []) or [],
                    next_node_ids=node_data.get("next_node_ids", []) or [],
                    superseded_by=node_data.get("superseded_by"),
                    created_at=node_data.get("created_at", ""),
                    source_record_id=node_data.get("source_record_id", ""),
                    source_turn_ids=node_data.get("source_turn_ids", []) or [],
                    metadata=node_data.get("metadata", {}) or {},
                )
            manager.tasks[task.task_id] = task
            manager._ensure_legacy_branch_registry(task)
        manager.rebuild_hierarchy_links()
        manager.validate_hierarchy()
        return manager

    def _call_llm_json(self, bundle: PromptBundle) -> dict[str, Any] | list[Any]:
        if self.llm_client is None:
            raise RuntimeError(f"LLM client missing for stage {bundle.name}")
        generate = getattr(self.llm_client, "generate", None)
        if generate is None:
            raise RuntimeError(f"LLM client does not expose generate() for stage {bundle.name}")
        max_attempts = self._json_max_attempts()
        retry_delay = self._json_retry_delay()
        last_error: json.JSONDecodeError | None = None
        for attempt in range(1, max_attempts + 1):
            response = generate(
                bundle.user_prompt,
                system_prompt=bundle.system_prompt,
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            response_text = str(response or "")
            try:
                return _extract_json_payload(response_text)
            except json.JSONDecodeError as exc:
                last_error = exc
                self._log_llm_json_error(
                    bundle=bundle,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    response_text=response_text,
                    error=exc,
                )
                if attempt >= max_attempts:
                    raise
                if retry_delay > 0.0:
                    time.sleep(retry_delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"LLM returned invalid JSON for stage {bundle.name}")

    def _log_llm_json_error(
        self,
        *,
        bundle: PromptBundle,
        attempt: int,
        max_attempts: int,
        response_text: str,
        error: json.JSONDecodeError,
    ) -> None:
        log = getattr(self.log_store, "log", None)
        if log is None:
            return
        context = _prompt_context(bundle.user_prompt)
        log(
            "llm_errors",
            "json_decode_failed",
            stage=bundle.name,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=type(error).__name__,
            message=str(error),
            prompt_chars=len(bundle.user_prompt),
            raw_response_excerpt=response_text[:1200],
            **context,
        )

    def _task_summary(
        self,
        task: TaskChain,
        *,
        include_entities: bool = True,
        include_recent_records: bool = True,
        text_limit: int | None = None,
    ) -> dict[str, Any]:
        recent_nodes = sorted(task.nodes.values(), key=lambda item: item.position, reverse=True)[:3]
        summary = {
            "task_id": task.task_id,
            "task_description": _short_text(
                task.canonical_description or task.task_description,
                text_limit,
            ) if text_limit else task.canonical_description or task.task_description,
            "status": task.status.value,
            "current_focus": _short_text(task.current_focus, text_limit or 256),
        }
        child_tasks = []
        for child_id in task.child_task_ids:
            child = self.tasks.get(child_id)
            if child is None or child.status is not TaskStatus.ACTIVE:
                continue
            child_tasks.append(
                {
                    "task_id": child.task_id,
                    "canonical_description": _short_text(
                        child.canonical_description or child.task_description,
                        min(text_limit or 256, 256),
                    ),
                    "current_focus": _short_text(child.current_focus, 256),
                    "status": child.status.value,
                }
            )
            if len(child_tasks) >= 8:
                break
        summary["child_tasks"] = child_tasks
        if include_entities:
            entities = task.entities[: self.router_entity_limit]
            summary["entities"] = [_short_text(entity, 64) for entity in entities] if text_limit else entities
        if include_recent_records:
            summary["parent_task_id"] = task.parent_task_id
            summary["parent_branch_id"] = task.parent_branch_id
            summary["child_task_ids"] = list(task.child_task_ids)[:8]
            summary["branches"] = self.task_routing_profile(task)["branches"][:8]
            summary["recent_records"] = [
                {
                    "source_record_id": node.source_record_id,
                    "user_content": node.user_content,
                    "assistant_content": node.assistant_content,
                }
                for node in recent_nodes
            ]
        else:
            latest_summary = next((node.summary for node in recent_nodes if node.summary), "")
            branch_profiles = self.task_routing_profile(task)["branches"]
            summary["current_focus"] = _short_text(
                task.current_focus,
                min(text_limit or self.query_router_summary_limit, 256),
            )
            summary["routing_summary"] = _short_text(
                latest_summary or task.task_description,
                min(text_limit or self.query_router_summary_limit, 256),
            )
            summary["record_count"] = len(task.record_ids)
            summary["updated_at"] = task.updated_at or task.created_at
            summary["parent_task_id"] = task.parent_task_id
            summary["parent_branch_id"] = task.parent_branch_id
            summary["child_task_ids"] = list(task.child_task_ids)[:8]
            summary["branches"] = branch_profiles[:8]
        return summary

    def _query_routing_candidates(
        self,
        query: str,
        intent: IntentUnderstanding,
    ) -> list[TaskChain]:
        if len(self.tasks) <= self.query_router_pool_size:
            candidates: list[TaskChain] = []
            seen_task_ids: set[str] = set()
            for task in self.tasks.values():
                if task.task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task.task_id)
                candidates.append(task)
            return candidates
        query_tokens = _routing_tokens(
            [
                query,
                intent.main_intent,
                intent.summary,
                intent.topic_hint,
                *intent.key_entities,
            ]
        )
        query_entities = {str(entity).casefold() for entity in intent.key_entities if str(entity).strip()}
        scored: list[tuple[tuple[int, int, int, str, str], TaskChain]] = []
        seen_task_ids: set[str] = set()
        for task in self.tasks.values():
            if task.task_id in seen_task_ids:
                continue
            seen_task_ids.add(task.task_id)
            task_tokens = _routing_tokens(
                [
                    task.canonical_description or task.task_description,
                    task.current_focus,
                    *task.entities,
                    *[
                        part
                        for branch in task.branches.values()
                        for part in (branch.branch_goal, branch.current_focus)
                    ],
                ]
            )
            task_entities = {str(entity).casefold() for entity in task.entities if str(entity).strip()}
            entity_overlap = len(query_entities & task_entities)
            token_overlap = len(query_tokens & task_tokens)
            active_bonus = int(task.status is TaskStatus.ACTIVE)
            scored.append(
                (
                    (
                        entity_overlap,
                        token_overlap,
                        active_bonus,
                        task.updated_at or task.created_at or "",
                        task.task_id,
                    ),
                    task,
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        return [task for _score, task in scored[: self.query_router_pool_size]]

    def _record_intent_bundle(
        self,
        record: DialogueRecord,
        *,
        routing_window: list[DialogueRecord] | None = None,
        current_index: int | None = None,
    ) -> PromptBundle:
        payload = {
            "current_record": self._record_router_payload(record),
            "current_index": current_index if current_index is not None else 0,
            "routing_window": [self._record_router_payload(item) for item in (routing_window or [])],
        }
        rendered = self.prompt_registry.render("record_intent_understanding", payload_json=_json_dumps(payload))
        return PromptBundle(
            name="record_intent_understanding",
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
        )

    def _query_intent_bundle(self, query: str) -> PromptBundle:
        payload = {"query": query}
        rendered = self.prompt_registry.render("query_intent_understanding", payload_json=_json_dumps(payload))
        return PromptBundle(
            name="query_intent_understanding",
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
        )

    def _task_routing_bundle(
        self,
        record: DialogueRecord,
        *,
        intent: IntentUnderstanding,
        current_index: int | None = None,
    ) -> PromptBundle:
        payload = {
            "current_record": self._record_router_payload(record),
            "current_index": current_index if current_index is not None else 0,
            "intent": to_primitive(intent),
            "task_catalog": [self._task_summary(task) for task in self._task_routing_candidates(record)],
        }
        rendered = self.prompt_registry.render("task_routing", payload_json=_json_dumps(payload))
        return PromptBundle(
            name="task_routing",
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
        )

    def _task_routing_review_bundle(
        self,
        record: DialogueRecord,
        initial_decision: dict[str, Any],
        *,
        intent: IntentUnderstanding,
        current_index: int | None = None,
    ) -> PromptBundle:
        payload = {
            "current_record": self._record_router_payload(record),
            "current_index": current_index if current_index is not None else 0,
            "intent": to_primitive(intent),
            "task_catalog": [self._task_summary(task) for task in self._task_routing_candidates(record)],
            "initial_decision": initial_decision,
        }
        rendered = self.prompt_registry.render("task_routing_review", payload_json=_json_dumps(payload))
        return PromptBundle(
            name="task_routing_review",
            system_prompt=rendered.system_prompt,
            user_prompt=rendered.user_prompt,
        )

    def _record_router_payload(self, record: DialogueRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "user_content": record.user_content,
            "assistant_content": record.assistant_content,
            "entities": list(record.entities),
        }

    def _intent_from_payload(self, parsed: dict[str, Any]) -> IntentUnderstanding:
        is_substantive_value = parsed.get("is_substantive")
        if isinstance(is_substantive_value, bool):
            is_substantive = is_substantive_value
        elif is_substantive_value is None:
            is_substantive = True
        else:
            is_substantive = str(is_substantive_value).strip().casefold() not in {"", "0", "false", "no"}
        return IntentUnderstanding(
            summary=str(parsed.get("summary", "") or "").strip(),
            main_intent=str(parsed.get("main_intent", "") or "").strip(),
            topic_hint=str(parsed.get("topic_hint", "") or "").strip(),
            key_entities=self._normalize_entities(parsed.get("key_entities", []) or []),
            is_substantive=is_substantive,
            reason=str(parsed.get("reason", "") or "").strip(),
        )

    def _route_payload_from_parsed(self, parsed: dict[str, Any]) -> dict[str, Any]:
        linked_task_ids = self._unique(
            [str(task_id) for task_id in parsed.get("linked_task_ids", []) or [] if str(task_id) in self.tasks]
        )[:5]
        new_tasks: list[dict[str, Any]] = []
        for spec in parsed.get("new_tasks", []) or []:
            if not isinstance(spec, dict):
                continue
            task_description = str(spec.get("task_description", "") or "").strip()
            if not task_description:
                continue
            normalized_spec: dict[str, Any] = {"task_description": task_description}
            for key in (
                "canonical_description",
                "current_focus",
                "branch_goal",
                "branch_id",
            ):
                value = str(spec.get(key, "") or "").strip()
                if value:
                    normalized_spec[key] = value
            parent_task_id = str(spec.get("parent_task_id", "") or "").strip()
            if parent_task_id and parent_task_id in self.tasks:
                normalized_spec["parent_task_id"] = parent_task_id
                parent_branch_id = str(spec.get("parent_branch_id", "") or "").strip()
                if parent_branch_id:
                    self._ensure_legacy_branch_registry(self.tasks[parent_task_id])
                    if parent_branch_id in self.tasks[parent_task_id].branches:
                        normalized_spec["parent_branch_id"] = parent_branch_id
            new_tasks.append(normalized_spec)
        return {
            "linked_task_ids": linked_task_ids,
            "new_tasks": new_tasks,
            "main_intent": str(parsed.get("main_intent", "") or "").strip(),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "reason": str(parsed.get("reason", "") or ""),
        }

    def _finalize_route_decision(
        self,
        record: DialogueRecord,
        *,
        intent: IntentUnderstanding,
        parsed: dict[str, Any],
    ) -> RouteDecision:
        route = self._route_payload_from_parsed(parsed)
        created_task_ids: list[str] = []
        for spec in route["new_tasks"]:
            created_task_ids.append(self.create_task_from_record(record, spec).task_id)
        routed_task_ids = self._unique([*route["linked_task_ids"], *created_task_ids])
        return RouteDecision(
            linked_task_ids=route["linked_task_ids"],
            created_task_ids=created_task_ids,
            routed_task_ids=routed_task_ids,
            main_intent=route["main_intent"],
            confidence=route["confidence"],
            reason=route["reason"],
            intent=intent,
        )

    def _review_task_route(
        self,
        record: DialogueRecord,
        initial_decision: dict[str, Any],
        *,
        intent: IntentUnderstanding,
        current_index: int | None = None,
    ) -> dict[str, Any]:
        parsed = self._call_llm_json(
            self._task_routing_review_bundle(
                record,
                initial_decision,
                intent=intent,
                current_index=current_index,
            )
        )
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"linked_task_ids", "new_tasks"})
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM task routing review returned invalid payload for record {record.record_id}")
        initial_route = self._route_payload_from_parsed(initial_decision)
        reviewed_route = self._route_payload_from_parsed(parsed)
        merged = dict(initial_route)
        for key in ("main_intent", "confidence", "reason"):
            if key in parsed:
                merged[key] = parsed[key]
        parsed_has_route = bool(parsed.get("linked_task_ids")) or bool(parsed.get("new_tasks"))
        if "linked_task_ids" in parsed and parsed_has_route:
            merged["linked_task_ids"] = reviewed_route["linked_task_ids"]
        if "new_tasks" in parsed and parsed_has_route:
            if reviewed_route["new_tasks"]:
                merged["new_tasks"] = self._merge_review_new_tasks(
                    initial_route["new_tasks"], reviewed_route["new_tasks"]
                )
            elif not initial_route["new_tasks"]:
                merged["new_tasks"] = []
        if "linked_task_ids" not in parsed or (not parsed_has_route and initial_route["linked_task_ids"]):
            merged["linked_task_ids"] = list(initial_route["linked_task_ids"])
        if "new_tasks" not in parsed or (not parsed_has_route and initial_route["new_tasks"]):
            merged["new_tasks"] = list(initial_route["new_tasks"])
        return self._route_payload_from_parsed(merged)

    def _merge_review_new_tasks(
        self,
        initial_specs: list[dict[str, Any]],
        reviewed_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Preserve draft metadata when review output is intentionally compact."""
        merged_specs: list[dict[str, Any]] = []
        used_initial: set[int] = set()
        for index, reviewed in enumerate(reviewed_specs):
            match_index: int | None = None
            for candidate_index, initial in enumerate(initial_specs):
                if candidate_index in used_initial:
                    continue
                matching_keys = (
                    "parent_task_id",
                    "parent_branch_id",
                    "canonical_description",
                    "task_description",
                )
                if any(
                    initial.get(key)
                    and reviewed.get(key)
                    and initial.get(key) == reviewed.get(key)
                    for key in matching_keys
                ):
                    match_index = candidate_index
                    break
            if match_index is None and index < len(initial_specs) and index not in used_initial:
                match_index = index
            base = dict(initial_specs[match_index]) if match_index is not None else {}
            if match_index is not None:
                used_initial.add(match_index)
            base.update(reviewed)
            merged_specs.append(base)
        return merged_specs

    def _maybe_refresh_task_metadata(self, task: TaskChain) -> None:
        interval = int(self.task_metadata_refresh_interval or 0)
        if interval <= 0 or len(task.record_ids) % interval != 0:
            return
        self.refresh_task_metadata(task)

    def refresh_task_metadata(self, task: TaskChain, *, branch_id: str | None = None) -> None:
        self._ensure_legacy_branch_registry(task)
        resolved_branch_id = str(branch_id or task.preferred_branch_id or task.active_branch_id or "main")
        branch = task.branches.get(resolved_branch_id)
        if branch is None:
            raise KeyError(f"unknown branch_id {resolved_branch_id!r} for task {task.task_id}")
        payload = {"task": self._task_refresh_payload(task, branch_id=resolved_branch_id)}
        rendered = self.prompt_registry.render("task_metadata_refresh", payload_json=_json_dumps(payload))
        parsed = self._call_llm_json(
            PromptBundle(
                name="task_metadata_refresh",
                system_prompt=rendered.system_prompt,
                user_prompt=rendered.user_prompt,
            )
        )
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"task_description", "entities"})
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM task metadata refresh returned invalid payload for task {task.task_id}")
        old_task_description = task.canonical_description or task.task_description
        old_current_focus = branch.current_focus
        old_entities = list(task.entities)
        current_focus = str(parsed.get("current_focus", parsed.get("task_description", "")) or "").strip()
        if current_focus:
            branch.current_focus = _short_text(current_focus, 256)
            if resolved_branch_id == task.preferred_branch_id or resolved_branch_id == task.active_branch_id:
                task.current_focus = branch.current_focus
        if isinstance(parsed.get("entities"), list):
            task.entities = self._normalize_entities([*task.entities, *parsed["entities"]])
        new_task_description = task.canonical_description or task.task_description
        self._log_task_chain_event(
            task.task_id,
            "metadata_refreshed",
            task_id=task.task_id,
            old_task_description=old_task_description,
            new_task_description=new_task_description,
            old_current_focus=old_current_focus,
            new_current_focus=branch.current_focus,
            branch_id=resolved_branch_id,
            old_entities=old_entities,
            new_entities=list(task.entities),
            changed=old_current_focus != branch.current_focus or old_entities != task.entities,
        )

    def _task_refresh_payload(self, task: TaskChain, *, branch_id: str | None = None) -> dict[str, Any]:
        resolved_branch_id = str(branch_id or task.preferred_branch_id or task.active_branch_id or "main")
        branch = task.branches.get(resolved_branch_id)
        recent_nodes = sorted(
            [node for node in task.nodes.values() if node.branch_id == resolved_branch_id],
            key=lambda item: item.position,
        )[-TASK_REFRESH_RECENT_NODE_LIMIT:]
        return {
            "task_id": task.task_id,
            "branch_id": resolved_branch_id,
            "branch_goal": branch.branch_goal if branch is not None else "",
            "task_description": task.canonical_description or task.task_description,
            "current_focus": branch.current_focus if branch is not None else task.current_focus,
            "status": task.status.value,
            "entities": task.entities[: self.router_entity_limit],
            "recent_records": [
                {
                    "source_record_id": node.source_record_id,
                    "summary": _short_text(node.summary, TASK_REFRESH_TEXT_LIMIT),
                }
                for node in recent_nodes
            ],
        }

    def _task_routing_candidates(self, record: DialogueRecord) -> list[TaskChain]:
        candidates: list[TaskChain] = []
        seen_task_ids: set[str] = set()
        for task in self.tasks.values():
            if task.task_id in seen_task_ids:
                continue
            seen_task_ids.add(task.task_id)
            candidates.append(task)
        if len(candidates) <= DEFAULT_TASK_ROUTING_CANDIDATE_COUNT:
            return candidates
        record_tokens = _routing_tokens(
            [
                record.user_content,
                record.assistant_content or "",
                *record.entities,
            ]
        )
        ranked = sorted(
            candidates,
            key=lambda task: self._task_routing_candidate_key(task, record_tokens),
            reverse=True,
        )
        return ranked[:DEFAULT_TASK_ROUTING_CANDIDATE_COUNT]

    def _task_routing_candidate_key(self, task: TaskChain, record_tokens: set[str]) -> tuple[int, int, str]:
        task_tokens = _routing_tokens(
            [
                task.canonical_description or task.task_description,
                task.current_focus,
                *task.entities,
                *[
                    part
                    for branch in task.branches.values()
                    for part in (branch.branch_goal, branch.current_focus)
                ],
            ]
        )
        entity_overlap = len({entity.casefold() for entity in task.entities} & record_tokens)
        token_overlap = len(task_tokens & record_tokens)
        return (
            entity_overlap,
            token_overlap,
            task.updated_at or task.created_at or "",
        )

    def _json_max_attempts(self) -> int:
        return max(1, int(getattr(self.llm_client, "json_max_attempts", DEFAULT_JSON_MAX_ATTEMPTS) or DEFAULT_JSON_MAX_ATTEMPTS))

    def _json_retry_delay(self) -> float:
        return max(0.0, float(getattr(self.llm_client, "json_retry_delay", 0.5) or 0.0))

    def _log_task_chain_event(self, chain_id: str, event: str, **payload: Any) -> None:
        log_task_chain = getattr(self.log_store, "log_task_chain", None)
        if log_task_chain is None:
            return
        log_task_chain(chain_id, event, **payload)

    def _log_empty_record_route(
        self,
        *,
        record: DialogueRecord,
        attempt: int,
        max_attempts: int,
        reason: str,
        main_intent: str = "",
    ) -> None:
        log = getattr(self.log_store, "log", None)
        if log is None:
            return
        log(
            "llm_errors",
            "empty_record_route",
            stage="task_routing",
            attempt=attempt,
            max_attempts=max_attempts,
            record_id=record.record_id,
            main_intent=main_intent,
            reason=reason,
        )

    def _log_invalid_llm_payload(
        self,
        *,
        stage: str,
        attempt: int,
        max_attempts: int,
        parsed: Any,
        record_id: str = "",
    ) -> None:
        log = getattr(self.log_store, "log", None)
        if log is None:
            return
        try:
            parsed_excerpt = json.dumps(to_primitive(parsed), ensure_ascii=False)[:1200]
        except TypeError:
            parsed_excerpt = str(parsed)[:1200]
        log(
            "llm_errors",
            "invalid_payload",
            stage=stage,
            attempt=attempt,
            max_attempts=max_attempts,
            record_id=record_id,
            parsed_excerpt=parsed_excerpt,
        )

    def _record_id(self, session_uuid: str, record_index: int) -> str:
        slug = re.sub(r"[^0-9A-Za-z]+", "_", session_uuid).strip("_") or "session"
        return f"rec_{slug}_{record_index + 1:04d}"

    def _base_record_time(self, current_time: str) -> datetime:
        text = str(current_time or "").strip()
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text.split(" ", 1)[0] if pattern == "%Y-%m-%d" else text, pattern)
            except ValueError:
                continue
        return datetime(1970, 1, 1, 0, 0, 0)

    def _normalize_entities(self, values: list[Any]) -> list[str]:
        return self._unique([str(value).strip() for value in values if str(value).strip()])

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result


def _extract_json_payload(text: str) -> dict[str, Any] | list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        if stripped.startswith(("{", "[")):
            raise exc
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


def _json_dumps(value: Any) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=False, indent=2)


def _clamp_query_candidate_count(value: int) -> int:
    return min(8, max(3, int(value or DEFAULT_QUERY_ROUTER_CANDIDATE_COUNT)))


def _coerce_task_status(value: Any, *, strict: bool = False) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value
    try:
        return TaskStatus(str(value or TaskStatus.ACTIVE.value))
    except ValueError:
        if strict:
            raise ValueError(f"invalid task status: {value!r}") from None
        return TaskStatus.ACTIVE


def _unwrap_schema_response(parsed: dict[str, Any], *, expected_keys: set[str] | None = None) -> dict[str, Any]:
    if expected_keys and any(key in parsed for key in expected_keys):
        return parsed
    nested = parsed.get("schema")
    if isinstance(nested, dict) and (not expected_keys or any(key in nested for key in expected_keys)):
        return nested
    return parsed


def _short_text(text: Any, limit: int) -> str:
    value = "" if text is None else str(text).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _is_non_task_route(decision: RouteDecision) -> bool:
    if decision.routed_task_ids:
        return False
    text = " ".join([str(decision.main_intent or ""), str(decision.reason or "")]).casefold()
    if not text.strip():
        return False
    return any(
        phrase in text
        for phrase in (
            "initiate a conversation",
            "initiating a conversation",
            "engage in a conversation",
            "engaging in a conversation",
            "greeting",
            "say hello",
            "without linking to any specific tasks",
            "does not relate to any existing tasks",
            "does not relate to any ongoing projects",
            "no specific task",
            "no ongoing project",
        )
    )


def _routing_tokens(parts: list[str]) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        text = str(part or "").casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._'-]*", text):
            if len(token) >= 3:
                tokens.add(token)
        for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
            if len(chunk) >= 2:
                tokens.add(chunk)
                tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


def _prompt_context(prompt: str) -> dict[str, Any]:
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        try:
            payload = _extract_prompt_input_json(prompt)
        except json.JSONDecodeError:
            return {}
    context: dict[str, Any] = {}
    record = None
    if isinstance(payload, dict):
        record = payload.get("record") or payload.get("current_record")
    if isinstance(record, dict):
        context["record_id"] = record.get("record_id", "")
        context["session_uuid"] = record.get("session_uuid", "")
    if isinstance(payload, dict) and "query" in payload:
        context["query_excerpt"] = str(payload.get("query") or "")[:240]
    return context


def _extract_prompt_input_json(prompt: str) -> dict[str, Any] | list[Any]:
    match = re.search(r"<[A-Za-z0-9_]*Input>\s*(\{.*?\}|\[.*?\])\s*</[A-Za-z0-9_]*Input>", prompt, flags=re.S)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"## Input\s*(\{.*?\}|\[.*?\])\s*## Output format", prompt, flags=re.S | re.I)
    if match:
        return json.loads(match.group(1))
    return _extract_json_payload(prompt)
