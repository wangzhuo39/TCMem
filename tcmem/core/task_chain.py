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
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.owner_id = owner_id
        self.llm_client = llm_client
        self.log_store = log_store
        self.task_metadata_refresh_interval = task_metadata_refresh_interval
        self.router_entity_limit = router_entity_limit
        self.query_router_candidate_count = _clamp_query_candidate_count(query_router_candidate_count)
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
        intent = self.understand_query_intent(query)
        candidate_count = min(self.query_router_candidate_count, len(self.tasks)) if self.tasks else self.query_router_candidate_count
        payload = {
            "query": query,
            "intent": to_primitive(intent),
            "candidate_count": candidate_count,
            "task_catalog": [self._task_summary(task) for task in self.tasks.values()],
        }
        rendered = self.prompt_registry.render("query_routing", payload_json=_json_dumps(payload))
        parsed = self._call_llm_json(
            PromptBundle(
                name="query_routing",
                system_prompt=rendered.system_prompt,
                user_prompt=rendered.user_prompt,
            )
        )
        if isinstance(parsed, dict):
            parsed = _unwrap_schema_response(parsed, expected_keys={"routed_task_ids"})
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM query routing returned invalid payload")
        routed = [str(task_id) for task_id in parsed.get("routed_task_ids", []) or [] if str(task_id) in self.tasks]
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
                routed_task_ids=routed,
                query_intent=to_primitive(intent),
                reason=decision.reason,
            )
        return decision

    def create_task_from_record(self, record: DialogueRecord, spec: dict[str, Any]) -> TaskChain:
        task_description = str(spec.get("task_description", "") or "").strip()
        if not task_description:
            raise ValueError("new task spec must include task_description")
        task = TaskChain(
            task_id=f"task_{uuid4().hex[:8]}",
            task_description=task_description,
            owner_id=self.owner_id,
            created_at=record.current_time,
            updated_at=record.current_time,
            entities=list(record.entities),
        )
        self.tasks[task.task_id] = task
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

    def _merge_task_entities(self, task: TaskChain, record: DialogueRecord) -> None:
        task.entities = self._normalize_entities([*task.entities, *record.entities])

    def _apply_task_metadata_updates(self, task: TaskChain, decision: ConflictDecision) -> None:
        if decision.task_description_update is not None:
            updated = decision.task_description_update.strip()
            if updated:
                task.task_description = updated

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
                "task_description": task.task_description,
                "status": task.status.value,
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
        prompt_registry: PromptRegistry | None = None,
    ) -> "TaskChainManager":
        manager = cls(
            owner_id=state.get("owner_id", "default"),
            llm_client=llm_client,
            task_metadata_refresh_interval=task_metadata_refresh_interval,
            router_entity_limit=router_entity_limit,
            query_router_candidate_count=query_router_candidate_count,
            prompt_registry=prompt_registry,
        )
        for task_data in state.get("tasks", []) or []:
            task = TaskChain(
                task_id=task_data["task_id"],
                task_description=task_data.get("task_description") or task_data.get("topic", ""),
                owner_id=task_data.get("owner_id", manager.owner_id),
                status=_coerce_task_status(task_data.get("status")),
                created_at=task_data.get("created_at", ""),
                updated_at=task_data.get("updated_at", ""),
                entities=task_data.get("entities", []) or [],
                record_ids=task_data.get("record_ids", []) or [],
                branch_heads=task_data.get("branch_heads", {}) or {},
                next_node_index=int(task_data.get("next_node_index", 1) or 1),
                next_branch_index=int(task_data.get("next_branch_index", 1) or 1),
                active_branch_id=task_data.get("active_branch_id", "main"),
                metadata=task_data.get("metadata", {}) or {},
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

    def _task_summary(self, task: TaskChain, *, include_entities: bool = True) -> dict[str, Any]:
        recent_nodes = sorted(task.nodes.values(), key=lambda item: item.position, reverse=True)[:3]
        summary = {
            "task_id": task.task_id,
            "task_description": task.task_description,
            "status": task.status.value,
            "recent_records": [
                {
                    "source_record_id": node.source_record_id,
                    "user_content": node.user_content,
                    "assistant_content": node.assistant_content,
                }
                for node in recent_nodes
            ],
        }
        if include_entities:
            summary["entities"] = task.entities[: self.router_entity_limit]
        return summary

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
            new_tasks.append({"task_description": task_description})
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
        merged = dict(initial_decision)
        merged.update(parsed)
        parsed_has_route = bool(parsed.get("linked_task_ids")) or bool(parsed.get("new_tasks"))
        if "linked_task_ids" not in parsed or (not parsed_has_route and initial_decision.get("linked_task_ids")):
            merged["linked_task_ids"] = list(initial_decision.get("linked_task_ids", []) or [])
        if "new_tasks" not in parsed or (not parsed_has_route and initial_decision.get("new_tasks")):
            merged["new_tasks"] = list(initial_decision.get("new_tasks", []) or [])
        return self._route_payload_from_parsed(merged)

    def _maybe_refresh_task_metadata(self, task: TaskChain) -> None:
        interval = int(self.task_metadata_refresh_interval or 0)
        if interval <= 0 or len(task.record_ids) % interval != 0:
            return
        self.refresh_task_metadata(task)

    def refresh_task_metadata(self, task: TaskChain) -> None:
        payload = {"task": self._task_refresh_payload(task)}
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
        old_task_description = task.task_description
        old_entities = list(task.entities)
        task_description = str(parsed.get("task_description", "") or "").strip()
        if task_description:
            task.task_description = task_description
        if isinstance(parsed.get("entities"), list):
            task.entities = self._normalize_entities(parsed["entities"])
        self._log_task_chain_event(
            task.task_id,
            "metadata_refreshed",
            task_id=task.task_id,
            old_task_description=old_task_description,
            new_task_description=task.task_description,
            old_entities=old_entities,
            new_entities=list(task.entities),
            changed=old_task_description != task.task_description or old_entities != task.entities,
        )

    def _task_refresh_payload(self, task: TaskChain) -> dict[str, Any]:
        recent_nodes = sorted(task.nodes.values(), key=lambda item: item.position)[-TASK_REFRESH_RECENT_NODE_LIMIT:]
        return {
            "task_id": task.task_id,
            "task_description": task.task_description,
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
                task.task_description,
                *task.entities,
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


def _coerce_task_status(value: Any) -> TaskStatus:
    try:
        return TaskStatus(str(value or TaskStatus.ACTIVE.value))
    except ValueError:
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
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+._'-]*", str(part or "").casefold()):
            if len(token) >= 3:
                tokens.add(token)
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
