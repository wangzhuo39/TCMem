from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class ChainNodeStatus(str, Enum):
    ACTIVE = "active"
    BRANCHED = "branched"
    DEPRECATED = "deprecated"


class TaskStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class IntentUnderstanding:
    main_intent: str
    summary: str = ""
    topic_hint: str = ""
    key_entities: list[str] = field(default_factory=list)
    is_substantive: bool = True
    reason: str = ""


@dataclass(slots=True)
class DialogueTurn:
    speaker: str
    content: str
    is_query: bool = False
    query_id: str | None = None


@dataclass(slots=True)
class SessionPayload:
    current_time: str
    dialogue: list[DialogueTurn]
    session_identifier: str = ""
    session_uuid: str = ""

    def __post_init__(self) -> None:
        if not self.session_identifier:
            self.session_identifier = f"session_{uuid4().hex[:8]}"
        if not self.session_uuid:
            self.session_uuid = self.session_identifier

    @property
    def dialogue_turns(self) -> list[DialogueTurn]:
        return self.dialogue


@dataclass(slots=True)
class DialogueRecord:
    record_id: str
    session_identifier: str
    session_uuid: str
    current_time: str
    record_time: str = ""
    user_content: str = ""
    assistant_content: str | None = None
    source_turn_indexes: list[int] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.record_time:
            return
        text = str(self.current_time or "").strip()
        for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text.split(" ", 1)[0] if pattern == "%Y-%m-%d" else text, pattern)
                self.record_time = parsed.strftime("%Y-%m-%d %H:%M:%S")
                return
            except ValueError:
                continue
        self.record_time = "1970-01-01 00:00:00"

    @property
    def combined_content(self) -> str:
        if self.assistant_content:
            return f"{self.user_content}\n{self.assistant_content}"
        return self.user_content


@dataclass(slots=True)
class DialogueEdge:
    source_record_id: str
    target_record_id: str
    shared_entities: list[str] = field(default_factory=list)
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskChainNode:
    node_id: str
    task_id: str
    user_content: str
    assistant_content: str | None = None
    summary: str = ""
    status: ChainNodeStatus = ChainNodeStatus.ACTIVE
    branch_id: str = "main"
    position: int = 0
    prev_node_ids: list[str] = field(default_factory=list)
    next_node_ids: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    created_at: str = ""
    source_record_id: str = ""
    source_turn_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskChain:
    task_id: str
    task_description: str
    owner_id: str
    status: TaskStatus = TaskStatus.ACTIVE
    created_at: str = ""
    updated_at: str = ""
    entities: list[str] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)
    nodes: dict[str, TaskChainNode] = field(default_factory=dict)
    branch_heads: dict[str, str] = field(default_factory=dict)
    next_node_index: int = 1
    next_branch_index: int = 1
    active_branch_id: str = "main"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    linked_task_ids: list[str]
    created_task_ids: list[str]
    routed_task_ids: list[str]
    main_intent: str
    confidence: float
    reason: str
    intent: IntentUnderstanding

    def __post_init__(self) -> None:
        intent = self.intent
        canonical_main_intent = self.main_intent.strip() or intent.main_intent.strip()
        self.main_intent = canonical_main_intent
        self.intent = IntentUnderstanding(
            summary=intent.summary,
            main_intent=canonical_main_intent,
            topic_hint=intent.topic_hint,
            key_entities=list(intent.key_entities),
            is_substantive=intent.is_substantive,
            reason=intent.reason,
        )


@dataclass(slots=True)
class QueryRouteDecision:
    query: str
    routed_task_ids: list[str]
    query_intent: IntentUnderstanding
    reason: str


@dataclass(slots=True)
class ChainUpdateResult:
    task_id: str
    record_id: str
    action: str
    chain_node_id: str | None
    reference_node_id: str | None = None
    branch_id: str | None = None
    status: ChainNodeStatus | None = None
    reason: str = ""


@dataclass(slots=True)
class SearchHit:
    item_id: str
    item_kind: str
    score: float
    semantic_score: float = 0.0
    chain_score: float = 0.0
    graph_score: float = 0.0
    route_score: float = 0.0
    depth: int = 0
    reason: str = ""
    task_id: str | None = None
    source_record_id: str | None = None
    chain_node_id: str | None = None
    bm25_score: float = 0.0


@dataclass(slots=True)
class RetrievalResult:
    query: str
    subqueries: list[str]
    routed_task_ids: list[str]
    hits: list[SearchHit] = field(default_factory=list)
    explanation: str = ""
    query_intent: IntentUnderstanding | None = None
    query_route_reason: str = ""


@dataclass(slots=True)
class IngestionResult:
    session_identifier: str
    session_uuid: str
    record_ids: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)
    updated_task_ids: list[str] = field(default_factory=list)
    created_node_ids: list[str] = field(default_factory=list)
    branched_node_ids: list[str] = field(default_factory=list)
    overridden_node_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
