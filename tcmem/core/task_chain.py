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
    TaskStatus,
)
from ..serialization import to_primitive


@dataclass(slots=True)
class PromptBundle:
    name: str
    system_prompt: str
    user_prompt: str


class TaskChainManager:
    def __init__(self, owner_id: str, llm_client: object | None = None, log_store: object | None = None) -> None:
        self.owner_id = owner_id
        self.llm_client = llm_client
        self.log_store = log_store
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
        parsed = self._call_llm_json(
            PromptBundle(
                name="entity_extraction",
                system_prompt="Extract compact entity strings from a dialogue record. Return JSON only.",
                user_prompt=json.dumps({"record": to_primitive(record), "schema": {"entities": ["string"]}}, ensure_ascii=False),
            )
        )
        if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
            raise RuntimeError(f"LLM entity extraction returned invalid payload for record {record.record_id}")
        record.entities = self._normalize_entities([*record.entities, *parsed["entities"]])
        return record.entities

    def route_record(self, record: DialogueRecord) -> RouteDecision:
        parsed = self._call_llm_json(
            PromptBundle(
                name="task_routing",
                system_prompt="Route a dialogue record to existing task chains or create new task specs. Return JSON only.",
                user_prompt=json.dumps(
                    {
                        "record": to_primitive(record),
                        "tasks": [self._task_summary(task) for task in self.tasks.values()],
                        "schema": {
                            "linked_task_ids": ["task id"],
                            "new_tasks": [{"task_description": "string", "topic": "string"}],
                            "confidence": 0.0,
                            "reason": "string",
                        },
                    },
                    ensure_ascii=False,
                ),
            )
        )
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM task routing returned invalid payload for record {record.record_id}")

        linked_task_ids = [str(task_id) for task_id in parsed.get("linked_task_ids", []) or [] if str(task_id) in self.tasks]
        created_task_ids: list[str] = []
        for spec in parsed.get("new_tasks", []) or []:
            if not isinstance(spec, dict):
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

    def route_for_query(self, query: str) -> list[str]:
        parsed = self._call_llm_json(
            PromptBundle(
                name="query_routing",
                system_prompt="Route a user query to relevant task chains. Return JSON only.",
                user_prompt=json.dumps(
                    {
                        "query": query,
                        "tasks": [self._task_summary(task) for task in self.tasks.values()],
                        "schema": {"routed_task_ids": ["task id"], "reason": "string"},
                    },
                    ensure_ascii=False,
                ),
            )
        )
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM query routing returned invalid payload")
        return [str(task_id) for task_id in parsed.get("routed_task_ids", []) or [] if str(task_id) in self.tasks]

    def create_task_from_record(self, record: DialogueRecord, spec: dict[str, Any]) -> TaskChain:
        task_description = str(spec.get("task_description", "") or "").strip()
        topic = str(spec.get("topic", "") or "").strip()
        if not task_description or not topic:
            raise ValueError("new task spec must include task_description and topic")
        task = TaskChain(
            task_id=f"task_{uuid4().hex[:8]}",
            task_description=task_description,
            topic=topic,
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
        return node

    def get_task(self, task_id: str) -> TaskChain | None:
        return self.tasks.get(task_id)

    def to_state(self) -> dict:
        return {"owner_id": self.owner_id, "tasks": [to_primitive(task) for task in self.tasks.values()]}

    @classmethod
    def from_state(cls, state: dict, *, llm_client: object | None = None) -> "TaskChainManager":
        manager = cls(owner_id=state.get("owner_id", "default"), llm_client=llm_client)
        for task_data in state.get("tasks", []) or []:
            task = TaskChain(
                task_id=task_data["task_id"],
                task_description=task_data["task_description"],
                topic=task_data["topic"],
                owner_id=task_data.get("owner_id", manager.owner_id),
                status=TaskStatus(task_data.get("status", TaskStatus.ACTIVE.value)),
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
        max_attempts = max(1, int(getattr(self.llm_client, "json_max_attempts", 3) or 3))
        retry_delay = max(0.0, float(getattr(self.llm_client, "json_retry_delay", 0.5) or 0.0))
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
            "topic": task.topic,
            "task_description": task.task_description,
            "status": task.status.value,
            "entities": task.entities,
            "record_count": len(task.record_ids),
        }

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


def _prompt_context(prompt: str) -> dict[str, Any]:
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return {}
    context: dict[str, Any] = {}
    record = payload.get("record") if isinstance(payload, dict) else None
    if isinstance(record, dict):
        context["record_id"] = record.get("record_id", "")
        context["session_uuid"] = record.get("session_uuid", "")
    if isinstance(payload, dict) and "query" in payload:
        context["query_excerpt"] = str(payload.get("query") or "")[:240]
    return context
