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
    DialogueRecord,
    DialogueTurn,
    RouteDecision,
    SessionPayload,
    TaskChain,
    TaskChainNode,
)
from ..prompts import PromptRegistry
from ..serialization import to_primitive


DEFAULT_JSON_MAX_ATTEMPTS = 5
DEFAULT_TASK_METADATA_REFRESH_INTERVAL = 5
DEFAULT_ROUTER_ENTITY_LIMIT = 20
TASK_REFRESH_RECENT_NODE_LIMIT = 5
TASK_REFRESH_TEXT_LIMIT = 500


@dataclass(slots=True)
class PromptBundle:
    name: str
    system_prompt: str
    user_prompt: str


class TaskChainManager:
    def __init__(
        self,
        owner_id: str,
        llm_client: object | None = None,
        log_store: object | None = None,
        *,
        task_metadata_refresh_interval: int = DEFAULT_TASK_METADATA_REFRESH_INTERVAL,
        router_entity_limit: int = DEFAULT_ROUTER_ENTITY_LIMIT,
        prompt_registry: PromptRegistry | None = None,
    ) -> None:
        self.owner_id = owner_id
        self.llm_client = llm_client
        self.log_store = log_store
        self.task_metadata_refresh_interval = task_metadata_refresh_interval
        self.router_entity_limit = router_entity_limit
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
                record.entities = self._normalize_entities([*record.entities, *parsed["entities"]])
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

    def route_record(self, record: DialogueRecord) -> RouteDecision:
        max_attempts = self._json_max_attempts()
        retry_delay = self._json_retry_delay()
        last_decision: RouteDecision | None = None
        for attempt in range(1, max_attempts + 1):
            parsed = self._call_llm_json(self._task_routing_bundle(record))
            if isinstance(parsed, dict):
                parsed = _unwrap_schema_response(parsed, expected_keys={"linked_task_ids", "new_tasks"})
            if not isinstance(parsed, dict):
                raise RuntimeError(f"LLM task routing returned invalid payload for record {record.record_id}")
            decision = self._route_decision_from_payload(record, parsed)
            if decision.routed_task_ids:
                return decision
            last_decision = decision
            self._log_empty_record_route(record=record, attempt=attempt, max_attempts=max_attempts, reason=decision.reason)
            if attempt >= max_attempts:
                break
            if retry_delay > 0.0:
                time.sleep(retry_delay)
        reason = last_decision.reason if last_decision is not None else ""
        raise RuntimeError(f"LLM task routing returned no task for record {record.record_id} after {max_attempts} attempts: {reason}")

    def route_for_query(self, query: str) -> list[str]:
        payload = {
            "query": query,
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
        return [str(task_id) for task_id in parsed.get("routed_task_ids", []) or [] if str(task_id) in self.tasks]

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
        return task

    def apply_record(self, task_id: str, record: DialogueRecord, *, summary: str | None = None) -> TaskChainNode:
        task = self.tasks[task_id]
        branch_id = task.active_branch_id
        node_id = f"node_{task.task_id.removeprefix('task_')}_{task.next_node_index:04d}"
        previous_head = task.branch_heads.get(branch_id)
        node = TaskChainNode(
            node_id=node_id,
            task_id=task.task_id,
            user_content=record.user_content,
            assistant_content=record.assistant_content,
            summary=summary or record.combined_content,
            status=ChainNodeStatus.ACTIVE,
            branch_id=branch_id,
            position=task.next_node_index,
            prev_node_ids=[previous_head] if previous_head else [],
            created_at=record.current_time,
            source_record_id=record.record_id,
            source_turn_ids=list(record.source_turn_ids),
        )
        if previous_head and previous_head in task.nodes:
            task.nodes[previous_head].next_node_ids.append(node.node_id)
        task.nodes[node.node_id] = node
        task.branch_heads[branch_id] = node.node_id
        task.next_node_index += 1
        task.updated_at = record.current_time
        if record.record_id not in task.record_ids:
            task.record_ids.append(record.record_id)
        task.entities = self._normalize_entities([*task.entities, *record.entities])
        self._maybe_refresh_task_metadata(task)
        return node

    def get_task(self, task_id: str) -> TaskChain | None:
        return self.tasks.get(task_id)

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
        prompt_registry: PromptRegistry | None = None,
    ) -> "TaskChainManager":
        manager = cls(
            owner_id=state.get("owner_id", "default"),
            llm_client=llm_client,
            task_metadata_refresh_interval=task_metadata_refresh_interval,
            router_entity_limit=router_entity_limit,
            prompt_registry=prompt_registry,
        )
        for task_data in state.get("tasks", []) or []:
            task = TaskChain(
                task_id=task_data["task_id"],
                task_description=task_data.get("task_description") or task_data.get("topic", ""),
                owner_id=task_data.get("owner_id", manager.owner_id),
                created_at=task_data.get("created_at", ""),
                updated_at=task_data.get("updated_at", ""),
                entities=task_data.get("entities", []) or [],
                record_ids=task_data.get("record_ids", []) or [],
                branch_heads=task_data.get("branch_heads", {}) or {},
                next_node_index=int(task_data.get("next_node_index", 1) or 1),
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

    def _task_summary(self, task: TaskChain) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "task_description": task.task_description,
            "entities": task.entities[: self.router_entity_limit],
        }

    def _task_routing_bundle(self, record: DialogueRecord) -> PromptBundle:
        payload = {
            "current_record": self._record_router_payload(record),
            "task_catalog": [self._task_summary(task) for task in self.tasks.values()],
        }
        rendered = self.prompt_registry.render("task_routing", payload_json=_json_dumps(payload))
        return PromptBundle(
            name="task_routing",
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

    def _route_decision_from_payload(self, record: DialogueRecord, parsed: dict[str, Any]) -> RouteDecision:
        linked_task_ids = [str(task_id) for task_id in parsed.get("linked_task_ids", []) or [] if str(task_id) in self.tasks]
        created_task_ids: list[str] = []
        for spec in parsed.get("new_tasks", []) or []:
            if not isinstance(spec, dict):
                continue
            if not str(spec.get("task_description", "") or "").strip():
                continue
            created_task_ids.append(self.create_task_from_record(record, spec).task_id)
        routed_task_ids = self._unique([*linked_task_ids, *created_task_ids])
        return RouteDecision(
            linked_task_ids=linked_task_ids,
            created_task_ids=created_task_ids,
            routed_task_ids=routed_task_ids,
            confidence=float(parsed.get("confidence", 0.0) or 0.0),
            reason=str(parsed.get("reason", "") or ""),
        )

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
        task_description = str(parsed.get("task_description", "") or "").strip()
        if task_description:
            task.task_description = task_description
        if isinstance(parsed.get("entities"), list):
            task.entities = self._normalize_entities(parsed["entities"])

    def _task_refresh_payload(self, task: TaskChain) -> dict[str, Any]:
        recent_nodes = sorted(task.nodes.values(), key=lambda item: item.position)[-TASK_REFRESH_RECENT_NODE_LIMIT:]
        return {
            "task_id": task.task_id,
            "task_description": task.task_description,
            "entities": task.entities[: self.router_entity_limit],
            "recent_records": [
                {
                    "source_record_id": node.source_record_id,
                    "summary": _short_text(node.summary, TASK_REFRESH_TEXT_LIMIT),
                }
                for node in recent_nodes
            ],
        }

    def _json_max_attempts(self) -> int:
        return max(1, int(getattr(self.llm_client, "json_max_attempts", DEFAULT_JSON_MAX_ATTEMPTS) or DEFAULT_JSON_MAX_ATTEMPTS))

    def _json_retry_delay(self) -> float:
        return max(0.0, float(getattr(self.llm_client, "json_retry_delay", 0.5) or 0.0))

    def _log_empty_record_route(self, *, record: DialogueRecord, attempt: int, max_attempts: int, reason: str) -> None:
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
