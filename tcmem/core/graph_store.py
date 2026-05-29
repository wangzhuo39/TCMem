from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

from ..models import DialogueEdge, DialogueRecord
from ..serialization import to_primitive


class DialogueGraphStore:
    def __init__(self) -> None:
        self.records: dict[str, DialogueRecord] = {}
        self.edges: list[DialogueEdge] = []
        self.outgoing: dict[str, list[DialogueEdge]] = defaultdict(list)
        self.incoming: dict[str, list[DialogueEdge]] = defaultdict(list)

    def add_record(self, record: DialogueRecord) -> DialogueRecord:
        for existing in list(self.records.values()):
            shared_entities = sorted(set(existing.entities) & set(record.entities))
            if shared_entities:
                self.add_edge(
                    DialogueEdge(
                        source_record_id=existing.record_id,
                        target_record_id=record.record_id,
                        shared_entities=shared_entities,
                        weight=float(len(shared_entities)),
                        metadata={"type": "entity_cooccurrence"},
                    )
                )
        self.records[record.record_id] = record
        return record

    def add_edge(self, edge: DialogueEdge) -> DialogueEdge:
        edge_type = edge.metadata.get("type")
        for existing in self.edges:
            if (
                existing.source_record_id == edge.source_record_id
                and existing.target_record_id == edge.target_record_id
                and existing.metadata.get("type") == edge_type
            ):
                return existing
        self.edges.append(edge)
        self.outgoing[edge.source_record_id].append(edge)
        self.incoming[edge.target_record_id].append(edge)
        return edge

    def get_record(self, record_id: str) -> DialogueRecord | None:
        return self.records.get(record_id)

    def neighbors(self, record_id: str) -> list[DialogueRecord]:
        related_ids: list[str] = []
        for edge in self.outgoing.get(record_id, []):
            related_ids.append(edge.target_record_id)
        for edge in self.incoming.get(record_id, []):
            related_ids.append(edge.source_record_id)
        seen: set[str] = set()
        records: list[DialogueRecord] = []
        for related_id in related_ids:
            if related_id in seen:
                continue
            seen.add(related_id)
            record = self.records.get(related_id)
            if record is not None:
                records.append(record)
        return records

    def walk(self, start_ids: Iterable[str], *, max_depth: int = 2) -> dict[str, int]:
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((record_id, 0) for record_id in start_ids if record_id in self.records)
        while queue:
            record_id, depth = queue.popleft()
            if record_id in visited and visited[record_id] <= depth:
                continue
            visited[record_id] = depth
            if depth >= max_depth:
                continue
            for neighbor in self.neighbors(record_id):
                if neighbor.record_id not in visited or visited[neighbor.record_id] > depth + 1:
                    queue.append((neighbor.record_id, depth + 1))
        return visited

    def to_state(self) -> dict:
        return {
            "records": [to_primitive(record) for record in self.records.values()],
            "edges": [to_primitive(edge) for edge in self.edges],
        }

    @classmethod
    def from_state(cls, state: dict) -> "DialogueGraphStore":
        store = cls()
        for record_data in state.get("records", []) or []:
            store.records[record_data["record_id"]] = DialogueRecord(
                record_id=record_data["record_id"],
                session_identifier=record_data["session_identifier"],
                session_uuid=record_data["session_uuid"],
                current_time=record_data["current_time"],
                record_time=record_data.get("record_time", ""),
                user_content=record_data.get("user_content", ""),
                assistant_content=record_data.get("assistant_content"),
                source_turn_indexes=record_data.get("source_turn_indexes", []) or [],
                source_turn_ids=record_data.get("source_turn_ids", []) or [],
                entities=record_data.get("entities", []) or [],
                metadata=record_data.get("metadata", {}) or {},
            )
        for edge_data in state.get("edges", []) or []:
            store.add_edge(
                DialogueEdge(
                    source_record_id=edge_data["source_record_id"],
                    target_record_id=edge_data["target_record_id"],
                    shared_entities=edge_data.get("shared_entities", []) or [],
                    weight=float(edge_data.get("weight", 1.0) or 1.0),
                    metadata=edge_data.get("metadata", {}) or {},
                )
            )
        return store
