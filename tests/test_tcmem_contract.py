import json
import tempfile
import unittest
from pathlib import Path
import re

import numpy as np

from tcmem import DialogueRecord, SearchHit, TCMemConfig
from tcmem.core.graph_store import DialogueGraphStore
from tcmem.core.memory_system import MemorySystem
from tcmem.core.task_chain import TaskChainManager, _extract_json_payload
from tcmem.infrastructure.indices import NumpyVectorIndex, VectorIndexItem
from tcmem.logging_utils import ModuleLogStore
from tcmem.models import TaskStatus
from tcmem.prompts import PromptRegistry


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def score(self, query: str, text: str) -> float:
        return self._cosine(self.embed_query(query), self.embed_documents([text])[0])

    def rank(self, query: str, candidates: list[tuple[str, str]], limit: int) -> list[tuple[str, float]]:
        query_vector = self.embed_query(query)
        document_vectors = self.embed_documents([text for _item_id, text in candidates])
        ranked = [
            (item_id, self._cosine(query_vector, vector))
            for (item_id, _text), vector in zip(candidates, document_vectors)
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "alpha" in lowered:
            return np.array([1.0, 0.0], dtype="float32")
        if "beta" in lowered:
            return np.array([0.0, 1.0], dtype="float32")
        return np.array([0.5, 0.5], dtype="float32")

    def _cosine(self, left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


class KeywordRescueEmbeddingClient(FakeEmbeddingClient):
    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "alpha" in lowered or "question" in lowered:
            return np.array([1.0, 0.0], dtype="float32")
        if "zanzibar" in lowered or "ledger" in lowered:
            return np.array([0.0, 1.0], dtype="float32")
        return np.array([0.5, 0.5], dtype="float32")


class FakeLLMClient:
    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "recent_context" in payload:
            return json.dumps(
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": payload["record"].get("user_content", ""),
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "default create",
                }
            )
        if "record" in payload:
            text = payload["record"].get("user_content", "")
            entities = ["alpha"] if "alpha" in text.lower() else ["beta"]
            return json.dumps({"entities": entities})
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            record_text = payload["current_record"].get("user_content", "")
            lowered = record_text.lower()
            if "hi" in lowered or "hello" in lowered:
                return json.dumps(
                    {
                        "main_intent": "initiate conversation",
                        "topic_hint": "greeting",
                        "key_entities": [],
                        "is_substantive": False,
                        "reason": "simple greeting",
                    }
                )
            entities = ["alpha"] if "alpha" in lowered else ["beta"]
            topic = entities[0]
            return json.dumps(
                {
                    "main_intent": f"start {topic} task",
                    "topic_hint": topic,
                    "key_entities": entities,
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "current_record" in payload:
            tasks = payload.get("task_catalog", [])
            record_text = payload["current_record"].get("user_content", "")
            if tasks:
                return json.dumps(
                    {
                        "main_intent": "continue existing task",
                        "linked_task_ids": [tasks[0]["task_id"]],
                        "new_tasks": [],
                        "confidence": 1.0,
                        "reason": "existing",
                    }
                )
            topic = "Alpha" if "alpha" in record_text.lower() else "Beta"
            return json.dumps(
                {
                    "main_intent": f"start {topic.lower()} task",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": f"{topic} task"}],
                    "confidence": 1.0,
                    "reason": "new",
                }
            )
        if "task" in payload:
            return json.dumps({"task_description": "Refreshed task", "entities": ["refreshed"]})
        if "query" in payload and "task_catalog" not in payload:
            query = payload["query"].lower()
            topic = "alpha" if "alpha" in query else "beta"
            return json.dumps(
                {
                    "main_intent": f"retrieve {topic} task details",
                    "topic_hint": topic,
                    "key_entities": [topic],
                    "is_substantive": True,
                    "reason": "query intent",
                }
            )
        if "query" in payload:
            tasks = payload.get("task_catalog", [])
            query = payload["query"].lower()
            routed = [
                task["task_id"]
                for task in tasks
                if task["task_description"].lower().split()[0] in query
            ]
            return json.dumps({"routed_task_ids": routed or [task["task_id"] for task in tasks[:1]], "reason": "matched"})
        raise AssertionError(f"Unexpected prompt: {prompt}")


class NoRouteLLMClient(FakeLLMClient):
    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "query" in payload and "task_catalog" in payload:
            return json.dumps({"routed_task_ids": [], "reason": "none"})
        return super().generate(prompt, **_kwargs)


class FlakyJSONLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "query" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "retrieve alpha task details",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "query intent",
                }
            )
        self.calls += 1
        if self.calls == 1:
            return '{"routed_task_ids": ["task_manual"], "reason": "'
        return json.dumps({"routed_task_ids": ["task_manual"], "reason": "ok"})


class EmptyRouteThenTaskLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.task_routing_calls = 0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "create alpha handoff task",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "initial_decision" in payload:
            return json.dumps(payload["initial_decision"])
        if "current_record" in payload:
            self.task_routing_calls += 1
            if self.task_routing_calls < 3:
                return json.dumps({"main_intent": "unsure", "linked_task_ids": [], "new_tasks": [], "confidence": 0.1, "reason": "unsure"})
            return json.dumps(
                {
                    "main_intent": "create alpha handoff task",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha handoff task"}],
                    "confidence": 1.0,
                    "reason": "created",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class InvalidEntityThenValidLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "record" in payload:
            self.calls += 1
            if self.calls == 1:
                return json.dumps({"schema": {"entities": "string"}})
            return json.dumps({"entities": ["alpha"]})
        raise AssertionError(f"Unexpected prompt: {prompt}")


class AlwaysInvalidEntityLLMClient:
    json_max_attempts = 2
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "record" in payload:
            self.calls += 1
            copied_record = dict(payload["record"])
            copied_record.pop("entities", None)
            return json.dumps(copied_record)
        raise AssertionError(f"Unexpected prompt: {prompt}")


class CapturingLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self, response: dict) -> None:
        self.response = response
        self.payloads: list[dict] = []
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    def generate(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(_kwargs))
        payload = _prompt_payload(prompt)
        self.payloads.append(payload)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            response = self.response.get("schema") if isinstance(self.response.get("schema"), dict) else self.response
            main_intent = ""
            if isinstance(response, dict):
                main_intent = str(response.get("main_intent", "") or "").strip()
            if not main_intent:
                record_text = str(payload["current_record"].get("user_content", "") or "").lower()
                if "hi" in record_text or "hello" in record_text:
                    return json.dumps(
                        {
                            "main_intent": "initiate conversation",
                            "topic_hint": "greeting",
                            "key_entities": [],
                            "is_substantive": False,
                            "reason": "simple greeting",
                        }
                    )
                main_intent = "start alpha task" if "alpha" in record_text else "start beta task"
            return json.dumps(
                {
                    "main_intent": main_intent,
                    "topic_hint": "",
                    "key_entities": [],
                    "is_substantive": True,
                    "reason": "captured intent",
                }
            )
        return json.dumps(self.response)


class TwoStageRoutingLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.payloads_by_stage: dict[str, list[dict]] = {
            "record_intent_understanding": [],
            "task_routing": [],
            "task_routing_review": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["record_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "initial_decision" in payload:
            self.payloads_by_stage["task_routing_review"].append(payload)
            reviewed = dict(payload["initial_decision"])
            reviewed["main_intent"] = "reviewed alpha intent"
            reviewed["reason"] = "reviewed route"
            return json.dumps(reviewed)
        if "current_record" in payload:
            self.payloads_by_stage["task_routing"].append(payload)
            tasks = payload.get("task_catalog", [])
            if tasks:
                return json.dumps(
                    {
                        "main_intent": "draft alpha intent",
                        "linked_task_ids": [tasks[0]["task_id"]],
                        "new_tasks": [],
                        "confidence": 0.6,
                        "reason": "draft route",
                    }
                )
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha task"}],
                    "confidence": 0.6,
                    "reason": "draft route",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class IntentAwareRoutingLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.payloads_by_stage = {
            "record_intent_understanding": [],
            "task_routing": [],
            "task_routing_review": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["record_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "main_intent": "continue alpha launch plan",
                    "topic_hint": "alpha launch",
                    "key_entities": ["alpha", "launch checklist"],
                    "is_substantive": True,
                    "reason": "same launch plan in the buffered context",
                }
            )
        if "initial_decision" in payload:
            self.payloads_by_stage["task_routing_review"].append(payload)
            return json.dumps(
                {
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha launch task"}],
                    "confidence": 0.9,
                    "reason": "keep one alpha launch task",
                }
            )
        if "intent" in payload and "task_catalog" in payload:
            self.payloads_by_stage["task_routing"].append(payload)
            return json.dumps(
                {
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha launch task"}],
                    "confidence": 0.8,
                    "reason": "draft alpha launch task",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class ReviewDropsRouteLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "initial_decision" in payload:
            return json.dumps(
                {
                    "main_intent": "reviewed but route omitted",
                    "confidence": 0.9,
                    "reason": "review only comments on intent",
                }
            )
        if "current_record" in payload:
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha task"}],
                    "confidence": 0.6,
                    "reason": "draft route",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class ReviewClearsRouteLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "initial_decision" in payload:
            return json.dumps(
                {
                    "main_intent": "reviewed but cleared route",
                    "linked_task_ids": [],
                    "new_tasks": [],
                    "confidence": 0.9,
                    "reason": "review wrongly cleared route",
                }
            )
        if "current_record" in payload:
            return json.dumps(
                {
                    "main_intent": "draft alpha intent",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha task"}],
                    "confidence": 0.6,
                    "reason": "draft route",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class ReviewKeepsGreetingUnroutedLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "initiate conversation",
                    "topic_hint": "greeting",
                    "key_entities": [],
                    "is_substantive": False,
                    "reason": "simple greeting",
                }
            )
        if "initial_decision" in payload:
            return json.dumps(
                {
                    "main_intent": "initiate conversation",
                    "linked_task_ids": [],
                    "new_tasks": [],
                    "confidence": 0.2,
                    "reason": "The draft route was kept as it correctly identifies the main intent of initiating a conversation without linking to any specific tasks.",
                }
            )
        if "current_record" in payload:
            return json.dumps(
                {
                    "main_intent": "initiate conversation",
                    "linked_task_ids": [],
                    "new_tasks": [],
                    "confidence": 0.2,
                    "reason": "simple greeting",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class ConflictAwareLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self, conflict_responses: list[dict]) -> None:
        self.conflict_responses = list(conflict_responses)
        self.payloads_by_stage: dict[str, list[dict]] = {
            "entity_extraction": [],
            "record_intent_understanding": [],
            "task_routing": [],
            "task_routing_review": [],
            "record_conflict_resolution": [],
            "query_routing": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "initial_decision" in payload:
            self.payloads_by_stage["task_routing_review"].append(payload)
            initial = dict(payload["initial_decision"])
            if initial.get("main_intent"):
                return json.dumps(initial)
            initial["main_intent"] = "continue existing task"
            return json.dumps(initial)
        if "recent_context" in payload:
            self.payloads_by_stage["record_conflict_resolution"].append(payload)
            if not self.conflict_responses:
                raise AssertionError("No conflict response queued")
            return json.dumps(self.conflict_responses.pop(0))
        if "record" in payload:
            self.payloads_by_stage["entity_extraction"].append(payload)
            return json.dumps({"entities": ["alpha"]})
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["record_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "main_intent": "continue existing task",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "record intent",
                }
            )
        if "current_record" in payload:
            self.payloads_by_stage["task_routing"].append(payload)
            tasks = payload.get("task_catalog", [])
            if tasks:
                return json.dumps(
                    {
                        "main_intent": "continue existing task",
                        "linked_task_ids": [tasks[0]["task_id"]],
                        "new_tasks": [],
                        "confidence": 1.0,
                        "reason": "existing",
                    }
                )
            return json.dumps(
                {
                    "main_intent": "create alpha task",
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha task"}],
                    "confidence": 1.0,
                    "reason": "new",
                }
            )
        if "query" in payload and "task_catalog" not in payload:
            return json.dumps(
                {
                    "main_intent": "retrieve alpha task details",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "query intent",
                }
            )
        if "query" in payload:
            self.payloads_by_stage["query_routing"].append(payload)
            return json.dumps({"routed_task_ids": [], "reason": "none"})
        raise AssertionError(f"Unexpected prompt: {prompt}")


class IntentAwareQueryLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.payloads_by_stage = {
            "query_intent_understanding": [],
            "query_routing": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "query" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["query_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "main_intent": "retrieve next alpha step",
                    "topic_hint": "alpha plan",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "asks for the next step in alpha",
                }
            )
        if "query" in payload and "task_catalog" in payload:
            self.payloads_by_stage["query_routing"].append(payload)
            return json.dumps({"routed_task_ids": ["task_alpha"], "reason": "matched alpha intent"})
        raise AssertionError(f"Unexpected prompt: {prompt}")


class ContextualRecordIntentLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.payloads_by_stage = {
            "record_intent_understanding": [],
            "task_routing": [],
            "task_routing_review": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["record_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "summary": "user tightened the WhatsApp tool requirements",
                    "main_intent": "continue selecting a WhatsApp voice transcription tool",
                    "topic_hint": "whatsapp tools",
                    "key_entities": ["WhatsApp", "voice transcription", "offline"],
                    "is_substantive": True,
                    "reason": "current record adds constraints to the same tool-selection task",
                }
            )
        if "initial_decision" in payload:
            self.payloads_by_stage["task_routing_review"].append(payload)
            return json.dumps(payload["initial_decision"])
        if "current_record" in payload and "task_catalog" in payload:
            self.payloads_by_stage["task_routing"].append(payload)
            return json.dumps(
                {
                    "linked_task_ids": ["task_whatsapp"],
                    "new_tasks": [],
                    "confidence": 1.0,
                    "reason": "same WhatsApp tool-selection task",
                }
            )
        raise AssertionError(f"Unexpected prompt: {prompt}")


class IntentSummaryOverridesConflictLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.payloads_by_stage = {
            "record_intent_understanding": [],
            "task_routing": [],
            "task_routing_review": [],
            "record_conflict_resolution": [],
        }

    def generate(self, prompt: str, **_kwargs) -> str:
        payload = _prompt_payload(prompt)
        if "current_record" in payload and "routing_window" in payload and "task_catalog" not in payload:
            self.payloads_by_stage["record_intent_understanding"].append(payload)
            return json.dumps(
                {
                    "summary": "intent summary for alpha follow up",
                    "main_intent": "continue alpha implementation",
                    "topic_hint": "alpha",
                    "key_entities": ["alpha"],
                    "is_substantive": True,
                    "reason": "same ongoing alpha task",
                }
            )
        if "initial_decision" in payload:
            self.payloads_by_stage["task_routing_review"].append(payload)
            return json.dumps(payload["initial_decision"])
        if "current_record" in payload and "task_catalog" in payload:
            self.payloads_by_stage["task_routing"].append(payload)
            return json.dumps(
                {
                    "linked_task_ids": ["task_alpha"],
                    "new_tasks": [],
                    "confidence": 1.0,
                    "reason": "continue alpha task",
                }
            )
        if "recent_context" in payload:
            self.payloads_by_stage["record_conflict_resolution"].append(payload)
            return json.dumps(
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "conflict summary should not win",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "normal progression",
                }
            )
        if "record" in payload:
            return json.dumps({"entities": ["alpha"]})
        raise AssertionError(f"Unexpected prompt: {prompt}")


def _prompt_payload(prompt: str) -> dict:
    match = re.search(r"<[A-Za-z0-9_]*Input>\s*(\{.*?\}|\[.*?\])\s*</[A-Za-z0-9_]*Input>", prompt, flags=re.S)
    if match is None:
        match = re.search(r"## Input\s*(\{.*?\}|\[.*?\])\s*## Output format", prompt, flags=re.S | re.I)
    parsed = json.loads(match.group(1)) if match else _extract_json_payload(prompt)
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected object prompt payload: {prompt}")
    return parsed


class TCMemContractTest(unittest.TestCase):
    def test_module_log_store_writes_task_chain_file_with_sanitized_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")

            log_store.log_task_chain("../task:unsafe/id", "task_chain_created", task_id="../task:unsafe/id")

            task_log_path = log_store.run_dir / "task_chains" / "task_unsafe_id.jsonl"
            self.assertTrue(task_log_path.exists())
            payload = json.loads(task_log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["module"], "task_chain")
        self.assertEqual(payload["event"], "task_chain_created")
        self.assertEqual(payload["payload"]["task_id"], "../task:unsafe/id")

    def test_task_chain_manager_logs_creation_and_node_append_to_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            manager = TaskChainManager(owner_id="owner", llm_client=FakeLLMClient(), log_store=log_store)
            record = DialogueRecord(
                record_id="rec_1",
                session_identifier="session",
                session_uuid="session",
                current_time="2026-05-30 10:00:00",
                user_content="alpha request",
                assistant_content="alpha answer",
                source_turn_ids=["session:0:user", "session:1:assistant"],
                entities=["alpha"],
            )

            task = manager.create_task_from_record(record, {"task_description": "Alpha task", "topic": "Alpha work"})
            result = manager.apply_record(task.task_id, record)
            entries = [
                json.loads(line)
                for line in log_store.task_chain_path_for(task.task_id).read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([entry["event"] for entry in entries], ["task_chain_created", "node_added"])
        self.assertEqual(entries[0]["payload"]["task_id"], task.task_id)
        self.assertEqual(entries[0]["payload"]["source_record_id"], "rec_1")
        self.assertEqual(entries[0]["payload"]["task_description"], "Alpha task")
        self.assertEqual(entries[0]["payload"]["status"], "active")
        self.assertEqual(entries[1]["payload"]["node_id"], result.chain_node_id)
        self.assertEqual(entries[1]["payload"]["position"], 1)
        self.assertEqual(entries[1]["payload"]["source_turn_ids"], ["session:0:user", "session:1:assistant"])

    def test_task_chain_manager_logs_successful_record_and_query_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            manager = TaskChainManager(owner_id="owner", llm_client=FakeLLMClient(), log_store=log_store)
            record = self._record("rec_1", "alpha request", entities=["alpha"])

            decision = manager.route_record(record)
            routed_task_id = decision.routed_task_ids[0]
            query_decision = manager.route_for_query("alpha question")
            entries = [
                json.loads(line)
                for line in log_store.path_for("routing").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([entry["event"] for entry in entries], ["record_routed", "query_routed"])
        self.assertEqual(entries[0]["payload"]["record_id"], "rec_1")
        self.assertEqual(entries[0]["payload"]["main_intent"], "start alpha task")
        self.assertEqual(entries[0]["payload"]["routed_task_ids"], [routed_task_id])
        self.assertEqual(entries[0]["payload"]["reason"], "new")
        self.assertEqual(query_decision.query_intent.main_intent, "retrieve alpha task details")
        self.assertEqual(entries[1]["payload"]["query"], "alpha question")
        self.assertIn(routed_task_id, entries[1]["payload"]["routed_task_ids"])
        self.assertEqual(entries[1]["payload"]["reason"], "matched")

    def test_task_routing_runs_second_llm_review_and_logs_reviewed_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            llm = TwoStageRoutingLLMClient()
            manager = TaskChainManager(owner_id="owner", llm_client=llm, log_store=log_store)
            record = self._record("rec_1", "alpha request", entities=["alpha"])

            decision = manager.route_record(record)
            entries = [
                json.loads(line)
                for line in log_store.path_for("routing").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(decision.main_intent, "reviewed alpha intent")
        self.assertEqual(decision.reason, "reviewed route")
        self.assertEqual(len(llm.payloads_by_stage["task_routing"]), 1)
        self.assertEqual(len(llm.payloads_by_stage["task_routing_review"]), 1)
        review_payload = llm.payloads_by_stage["task_routing_review"][0]
        self.assertEqual(review_payload["initial_decision"]["main_intent"], "draft alpha intent")
        self.assertEqual(entries[-1]["payload"]["main_intent"], "reviewed alpha intent")
        self.assertEqual(entries[-1]["payload"]["reason"], "reviewed route")

    def test_task_routing_review_falls_back_to_initial_route_when_review_omits_route_fields(self) -> None:
        llm = ReviewDropsRouteLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)

        decision = manager.route_record(self._record("rec_1", "alpha request", entities=["alpha"]))

        self.assertEqual(decision.main_intent, "reviewed but route omitted")
        self.assertEqual(len(decision.created_task_ids), 1)
        self.assertEqual(decision.routed_task_ids, decision.created_task_ids)

    def test_task_routing_review_does_not_clear_non_empty_initial_route(self) -> None:
        llm = ReviewClearsRouteLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)

        decision = manager.route_record(self._record("rec_1", "alpha request", entities=["alpha"]))

        self.assertEqual(decision.main_intent, "reviewed but cleared route")
        self.assertEqual(len(decision.created_task_ids), 1)
        self.assertEqual(decision.routed_task_ids, decision.created_task_ids)

    def test_task_routing_allows_low_confidence_greeting_to_remain_unrouted(self) -> None:
        llm = ReviewKeepsGreetingUnroutedLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)

        decision = manager.route_record(self._record("rec_hello", "Hi chat"))

        self.assertEqual(decision.main_intent, "initiate conversation")
        self.assertEqual(decision.routed_task_ids, [])
        self.assertEqual(decision.created_task_ids, [])
        self.assertEqual(decision.linked_task_ids, [])

    def test_create_task_ignores_topic_field_and_sets_status(self) -> None:
        manager = TaskChainManager("unit", llm_client=FakeLLMClient())
        record = self._record("rec_alpha", "alpha project decision", entities=["alpha"])

        explicit = manager.create_task_from_record(
            record,
            {"task_description": "Alpha task", "topic": "project planning"},
        )
        fallback = manager.create_task_from_record(record, {"task_description": "Fallback task"})

        self.assertFalse(hasattr(explicit, "topic"))
        self.assertEqual(explicit.status, TaskStatus.ACTIVE)
        self.assertFalse(hasattr(fallback, "topic"))
        self.assertEqual(fallback.status, TaskStatus.ACTIVE)

    def test_apply_record_uses_conflict_resolution_for_override_branch_and_duplicate(self) -> None:
        llm = ConflictAwareLLMClient(
            [
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "first active goal",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "first record",
                },
                {
                    "action": "override",
                    "reference_node_id": "rec_1",
                    "branch_id": "main",
                    "summary": "replacement active goal",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "replaces first goal",
                },
                {
                    "action": "branch",
                    "reference_node_id": "rec_2",
                    "branch_id": "",
                    "summary": "parallel constraint",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "coexisting constraint",
                },
                {
                    "action": "duplicate",
                    "reference_node_id": "rec_3",
                    "branch_id": "b1",
                    "summary": "repeat",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "same meaning",
                },
            ]
        )
        manager = TaskChainManager("unit", llm_client=llm, task_metadata_refresh_interval=0)
        task = manager.create_task_from_record(self._record("seed", "alpha seed"), {"task_description": "Alpha task"})

        first = manager.apply_record(task.task_id, self._record("rec_1", "alpha first goal"))
        override = manager.apply_record(task.task_id, self._record("rec_2", "alpha replacement goal"))
        branch = manager.apply_record(task.task_id, self._record("rec_3", "alpha parallel constraint"))
        duplicate = manager.apply_record(task.task_id, self._record("rec_4", "alpha parallel constraint again"))

        self.assertEqual(first.action, "create")
        self.assertEqual(override.action, "override")
        self.assertEqual(branch.action, "branch")
        self.assertEqual(duplicate.action, "duplicate")
        self.assertEqual(duplicate.chain_node_id, branch.chain_node_id)
        self.assertEqual(len(task.nodes), 3)

        first_node = task.nodes[first.chain_node_id]
        override_node = task.nodes[override.chain_node_id]
        branch_node = task.nodes[branch.chain_node_id]
        self.assertEqual(first_node.status.value, "deprecated")
        self.assertEqual(first_node.superseded_by, override_node.node_id)
        self.assertEqual(branch_node.status.value, "branched")
        self.assertEqual(branch_node.branch_id, "b1")
        self.assertEqual(task.branch_heads["main"], override_node.node_id)
        self.assertEqual(task.branch_heads["b1"], branch_node.node_id)

        conflict_payloads = llm.payloads_by_stage["record_conflict_resolution"]
        self.assertEqual(conflict_payloads[1]["recent_context"][0]["record_id"], "rec_1")
        self.assertEqual(conflict_payloads[2]["recent_context"][-1]["record_id"], "rec_2")

    def test_record_intent_prompt_receives_closed_routing_window_and_routing_prompts_receive_intent(self) -> None:
        llm = ConflictAwareLLMClient(
            [
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "first",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "first",
                },
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "second",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "second",
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            system = MemorySystem(
                config=self._config(tmpdir),
                embedding_client=FakeEmbeddingClient(),
                llm_client=llm,
            )
            system.ingest_record(
                self._record(
                    "rec_1",
                    "alpha first",
                    record_time="2026-05-30 10:00:00",
                )
            )
            system.ingest_record(
                self._record(
                    "rec_2",
                    "alpha second",
                    record_time="2026-05-30 10:05:00",
                )
            )

            self.assertEqual(llm.payloads_by_stage["task_routing"], [])
            decisions, updates = system.flush_all_pending_routes()

        self.assertEqual(len(decisions), 2)
        self.assertEqual(len(updates), 2)
        intent_payloads = llm.payloads_by_stage["record_intent_understanding"]
        routing_payloads = llm.payloads_by_stage["task_routing"]
        review_payloads = llm.payloads_by_stage["task_routing_review"]
        self.assertEqual(len(intent_payloads), 2)
        self.assertEqual([payload["current_index"] for payload in intent_payloads], [0, 1])
        self.assertEqual(
            [[item["record_id"] for item in payload["routing_window"]] for payload in intent_payloads],
            [["rec_1", "rec_2"], ["rec_1", "rec_2"]],
        )
        self.assertEqual([payload["current_index"] for payload in routing_payloads], [0, 1])
        self.assertTrue(all("routing_window" not in payload for payload in routing_payloads))
        self.assertTrue(all("routing_window" not in payload for payload in review_payloads))
        self.assertEqual([payload["intent"]["main_intent"] for payload in routing_payloads], ["continue existing task"] * 2)
        self.assertEqual([payload["intent"]["main_intent"] for payload in review_payloads], ["continue existing task"] * 2)
        self.assertEqual(len(review_payloads), 2)

    def test_task_chain_manager_logs_metadata_refresh_to_task_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            manager = TaskChainManager(owner_id="owner", llm_client=FakeLLMClient(), log_store=log_store)
            record = DialogueRecord(
                record_id="rec_1",
                session_identifier="session",
                session_uuid="session",
                current_time="2026-05-30 10:00:00",
                user_content="alpha request",
                assistant_content="alpha answer",
                entities=["alpha"],
            )
            task = manager.create_task_from_record(record, {"task_description": "Alpha task"})

            manager.refresh_task_metadata(task)
            entries = [
                json.loads(line)
                for line in log_store.task_chain_path_for(task.task_id).read_text(encoding="utf-8").splitlines()
            ]
            refresh_entry = entries[-1]

        self.assertEqual(refresh_entry["event"], "metadata_refreshed")
        self.assertEqual(refresh_entry["payload"]["task_id"], task.task_id)
        self.assertEqual(refresh_entry["payload"]["old_task_description"], "Alpha task")
        self.assertEqual(refresh_entry["payload"]["new_task_description"], "Refreshed task")
        self.assertEqual(refresh_entry["payload"]["old_entities"], ["alpha"])
        self.assertEqual(refresh_entry["payload"]["new_entities"], ["refreshed"])
        self.assertTrue(refresh_entry["payload"]["changed"])

    def test_conflict_payload_omits_topic_and_applies_task_description_update(self) -> None:
        llm = ConflictAwareLLMClient(
            [
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "alpha first",
                    "task_description_update": "Updated alpha task",
                    "confidence": 1.0,
                    "reason": "explicit metadata update",
                },
            ]
        )
        manager = TaskChainManager("unit", llm_client=llm, task_metadata_refresh_interval=0)
        task = manager.create_task_from_record(
            self._record("seed", "alpha seed"),
            {"task_description": "Alpha task", "topic": "alpha topic"},
        )

        manager.apply_record(task.task_id, self._record("rec_1", "alpha first"))

        self.assertEqual(task.task_description, "Updated alpha task")
        conflict_payload = llm.payloads_by_stage["record_conflict_resolution"][0]
        self.assertNotIn("topic", conflict_payload["task"])
        self.assertEqual(conflict_payload["task"]["status"], "active")
        self.assertEqual(set(conflict_payload["task"]), {"task_id", "task_description", "status"})

    def test_record_conflict_payload_uses_task_scope_record_intent_and_recent_context_intent(self) -> None:
        from tcmem import IntentUnderstanding

        llm = ConflictAwareLLMClient(
            [
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "seed conflict summary",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "seed create",
                },
                {
                    "action": "create",
                    "reference_node_id": "",
                    "branch_id": "main",
                    "summary": "follow conflict summary",
                    "task_description_update": None,
                    "confidence": 1.0,
                    "reason": "follow create",
                },
            ]
        )
        manager = TaskChainManager("unit", llm_client=llm, task_metadata_refresh_interval=0)
        task = manager.create_task_from_record(self._record("seed", "alpha seed"), {"task_description": "Alpha task"})
        seed_intent = IntentUnderstanding(
            summary="seed intent summary",
            main_intent="continue alpha implementation",
            key_entities=["alpha"],
            is_substantive=True,
            reason="seed reason",
        )
        follow_intent = IntentUnderstanding(
            summary="follow intent summary",
            main_intent="continue alpha implementation",
            key_entities=["alpha", "beta"],
            is_substantive=True,
            reason="follow reason",
        )

        manager._apply_record_with_intent(
            task.task_id,
            self._record("rec_1", "alpha first"),
            summary=seed_intent.summary,
            record_intent=seed_intent,
        )
        manager._apply_record_with_intent(
            task.task_id,
            self._record("rec_2", "alpha second"),
            summary=follow_intent.summary,
            record_intent=follow_intent,
        )

        payload = llm.payloads_by_stage["record_conflict_resolution"][1]
        self.assertEqual(set(payload["task"]), {"task_id", "task_description", "status"})
        self.assertEqual(payload["record"]["summary"], "follow intent summary")
        self.assertEqual(payload["record"]["main_intent"], "continue alpha implementation")
        self.assertEqual(payload["record"]["key_entities"], ["alpha", "beta"])
        self.assertEqual(payload["recent_context"][0]["intent"]["summary"], "seed intent summary")
        self.assertEqual(payload["recent_context"][0]["intent"]["main_intent"], "continue alpha implementation")

    def test_default_prompts_use_handwritten_txt_templates_for_tcmem_stages(self) -> None:
        registry = PromptRegistry.default()

        for name in [
            "entity_extraction",
            "task_routing",
            "record_conflict_resolution",
            "query_routing",
            "task_metadata_refresh",
        ]:
            rendered = registry.render(name, payload_json='{"ok": true}')
            self.assertTrue(rendered.system_prompt)
            self.assertIn("{{payload_json}}", Path(f"tcmem/prompts/{name}.txt").read_text(encoding="utf-8"))
            self.assertIn('{"ok": true}', rendered.user_prompt)
            self.assertIn("## Input", rendered.user_prompt)
            self.assertIn("## Output format", rendered.user_prompt)
        review_rendered = registry.render("task_routing_review", payload_json='{"ok": true}')
        self.assertTrue(review_rendered.system_prompt)
        self.assertIn('{"ok": true}', review_rendered.user_prompt)
        self.assertIn("## Input", review_rendered.user_prompt)
        self.assertIn("## Output format", review_rendered.user_prompt)

    def test_prompt_registry_renders_intent_prompts_and_intent_aware_routing_contracts(self) -> None:
        registry = PromptRegistry.default()

        record_rendered = registry.render("record_intent_understanding", payload_json='{"current_record": {"record_id": "rec_1"}}')
        query_rendered = registry.render("query_intent_understanding", payload_json='{"query": "alpha"}')

        self.assertTrue(record_rendered.system_prompt)
        self.assertIn('"record_id": "rec_1"', record_rendered.user_prompt)
        self.assertTrue(query_rendered.system_prompt)
        self.assertIn('"query": "alpha"', query_rendered.user_prompt)
        record_intent = Path("tcmem/prompts/record_intent_understanding.txt").read_text(encoding="utf-8")
        self.assertIn('"summary": "..."', record_intent)
        self.assertIn("summary must capture the current record", record_intent)
        self.assertIn("main_intent must describe the broader ongoing task", record_intent)

        task_routing = Path("tcmem/prompts/task_routing.txt").read_text(encoding="utf-8")
        task_review = Path("tcmem/prompts/task_routing_review.txt").read_text(encoding="utf-8")
        query_routing = Path("tcmem/prompts/query_routing.txt").read_text(encoding="utf-8")

        self.assertIn("intent", task_routing)
        self.assertNotIn("routing_window", task_routing)
        self.assertIn("Treat the provided `intent` object as the canonical upstream understanding", task_routing)
        self.assertIn("Use `intent.summary` as the best description of what the current record says right now", task_routing)
        self.assertIn("Use `intent.main_intent` to decide which broader ongoing task", task_routing)
        self.assertIn("Prioritize `intent.main_intent`", task_routing)
        self.assertIn("Prioritize `intent.key_entities`", task_routing)
        self.assertNotIn("First identify the single main intent of `current_record`", task_routing)
        self.assertIn("intent", task_review)
        self.assertNotIn("routing_window", task_review)
        self.assertIn("initial_decision", task_review)
        self.assertIn("Read `initial_decision` first", task_review)
        self.assertIn("Treat the provided `intent` object as the canonical upstream understanding", task_review)
        self.assertIn("Use `intent.summary` to understand what changed in the current record", task_review)
        self.assertIn("Use `intent.main_intent` to judge whether the draft stays in the same broader ongoing task", task_review)
        self.assertIn("intent", query_routing)
        self.assertIn("Treat the provided `intent` object as the canonical upstream understanding", query_routing)
        self.assertIn("Prioritize `intent.main_intent`", query_routing)
        self.assertIn("Prioritize `intent.key_entities`", query_routing)
        self.assertNotIn("parse the core intent of the user query", query_routing.lower())

    def test_task_routing_prompt_limits_overbroad_multi_task_links(self) -> None:
        prompt_text = Path("tcmem/prompts/task_routing.txt").read_text(encoding="utf-8")

        self.assertLess(len(prompt_text), 4500)
        self.assertIn("main_intent", prompt_text)
        self.assertIn("canonical upstream understanding", prompt_text)
        self.assertIn("Default routing should link 1-3 existing tasks", prompt_text)
        self.assertIn("A record must not be linked to more than 5 existing tasks", prompt_text)
        self.assertIn("Weak shared words are not sufficient evidence", prompt_text)
        self.assertIn("Do not link a task merely because it shares the same user, business domain, or broad background", prompt_text)
        self.assertIn("Use `intent.summary` to understand the current record locally", prompt_text)
        self.assertIn("Use `intent.main_intent` to preserve task continuity across records", prompt_text)
        self.assertIn("Default to one main task", prompt_text)
        self.assertIn("If unsure between one existing task and multiple existing tasks, choose the single strongest task", prompt_text)
        self.assertIn("Do not create a new task for a goal change, new constraint, reprioritization, or execution detail inside the same ongoing project", prompt_text)
        self.assertIn("Same domain does not imply the same task", prompt_text)
        self.assertIn("A new deliverable, artifact, or work product should usually start a new task", prompt_text)
        self.assertIn("Tool selection and tool setup or implementation are different deliverables", prompt_text)
        self.assertIn("Wealth management planning and remittance app design are different deliverables", prompt_text)
        self.assertIn("App asset gathering and business-systems learning are different deliverables", prompt_text)
        self.assertIn("Financial analysis and operational risk mitigation are different deliverables", prompt_text)
        self.assertIn("Return multiple existing tasks only when the current record explicitly advances multiple deliverables", prompt_text)

    def test_task_routing_review_prompt_requires_checking_initial_decision(self) -> None:
        prompt_text = Path("tcmem/prompts/task_routing_review.txt").read_text(encoding="utf-8")

        self.assertLess(len(prompt_text), 4500)
        self.assertIn("initial_decision", prompt_text)
        self.assertIn("Check whether the draft route matches the record's main intent", prompt_text)
        self.assertIn("Use `intent.summary` to understand what the current record changed or added", prompt_text)
        self.assertIn("Use `intent.main_intent` to decide whether the record is still inside the same broader task", prompt_text)
        self.assertIn('"main_intent": "..."', prompt_text)
        self.assertIn("Same domain does not imply the same task", prompt_text)
        self.assertIn("Correct the draft when the record switches deliverables even if the business domain overlaps", prompt_text)
        self.assertIn("Do not keep a draft that merges tool selection with tool setup", prompt_text)

    def test_prompt_templates_remove_task_chain_topic_contract(self) -> None:
        task_routing = Path("tcmem/prompts/task_routing.txt").read_text(encoding="utf-8")
        query_routing = Path("tcmem/prompts/query_routing.txt").read_text(encoding="utf-8")
        conflict = Path("tcmem/prompts/record_conflict_resolution.txt").read_text(encoding="utf-8")
        refresh = Path("tcmem/prompts/task_metadata_refresh.txt").read_text(encoding="utf-8")

        self.assertIn('"main_intent": "..."', task_routing)
        self.assertNotIn('"topic": "..."', task_routing)
        self.assertNotIn("topic is a weak semantic label", query_routing)
        self.assertNotIn('"topic_update": null', conflict)
        self.assertNotIn('"topic": "..."', refresh)
        self.assertIn("Do not compare the first record against the task description", conflict)
        self.assertIn("Old items must come from recent_context", conflict)
        self.assertIn("If unsure between create and override, choose create", conflict)
        self.assertIn("If unsure between branch and override, choose branch", conflict)
        self.assertIn("If unsure between create and branch, choose create", conflict)

    def test_split_txt_prompt_loads_system_and_user_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir)
            (prompt_dir / "query_routing.txt").write_text(
                """
--- system ---
Custom query router system.
--- user ---
payload={{payload_json}}
""".strip(),
                encoding="utf-8",
            )

            rendered = PromptRegistry.from_path(prompt_dir).render("query_routing", payload_json='{"query": "alpha"}')

        self.assertEqual(rendered.system_prompt, "Custom query router system.")
        self.assertEqual(rendered.user_prompt, 'payload={"query": "alpha"}')

    def test_realmem_default_prompts_match_realmembench_eval_style(self) -> None:
        registry = PromptRegistry.default()

        answer = registry.render("realmem_answer_generation", question="question", evidence_text="memory")
        judge = registry.render(
            "realmem_qa_judge",
            question="question",
            gold_memory_text="gold memory",
            reference_answer="reference",
            candidate_answer="candidate",
        )

        self.assertEqual(answer.system_prompt, "")
        self.assertIn("Memories:\nmemory", answer.user_prompt)
        self.assertIn("Query: question", answer.user_prompt)
        self.assertEqual(judge.system_prompt, "")
        self.assertIn("Your task is to evaluate the consistency", judge.user_prompt)
        self.assertIn("### Input Data", judge.user_prompt)
        self.assertIn("4. Candidate Answer: candidate", judge.user_prompt)

    def test_prompt_registry_loads_custom_yaml_and_renders_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompts.yaml"
            prompt_path.write_text(
                """
custom_stage:
  system: custom system
  user: |
    payload={{payload_json}}
    question={{question}}
""".strip(),
                encoding="utf-8",
            )

            registry = PromptRegistry.from_path(prompt_path)
            rendered = registry.render("custom_stage", payload_json='{"ok": true}', question="hello")

        self.assertEqual(rendered.system_prompt, "custom system")
        self.assertIn('payload={"ok": true}', rendered.user_prompt)
        self.assertIn("question=hello", rendered.user_prompt)

    def test_task_chain_manager_uses_custom_prompt_registry(self) -> None:
        registry = PromptRegistry.from_mapping(
            {
                "entity_extraction": {
                    "system": "CUSTOM ENTITY SYSTEM",
                    "user": "{{payload_json}}",
                }
            }
        )
        llm = CapturingLLMClient({"entities": ["alpha"]})
        manager = TaskChainManager("unit", llm_client=llm, prompt_registry=registry)

        manager.extract_record_entities(self._record("rec_alpha", "alpha entity"))

        self.assertEqual(llm.kwargs[-1]["system_prompt"], "CUSTOM ENTITY SYSTEM")

    def test_default_embedding_model_is_bge_m3(self) -> None:
        self.assertEqual(TCMemConfig().embedding_model, "BAAI/bge-m3")

    def test_config_to_dict_redacts_api_key_by_default(self) -> None:
        config = TCMemConfig(llm_api_key="secret")

        self.assertEqual(config.to_dict()["llm_api_key"], "")
        self.assertEqual(config.to_dict(include_secrets=True)["llm_api_key"], "secret")

    def test_search_hit_and_config_support_bm25_path_b_scoring(self) -> None:
        config = TCMemConfig(path_b_bm25_weight=0.25)
        clamped_config = TCMemConfig(path_b_bm25_weight=-1)
        hit = SearchHit(item_id="rec", item_kind="dialogue_record", score=1.0)
        positional_hit = SearchHit("rec", "dialogue_record", 1.0, 0.9, 0.8, 0.7, 0.6)

        self.assertEqual(config.path_b_bm25_weight, 0.25)
        self.assertEqual(clamped_config.path_b_bm25_weight, 0.0)
        self.assertEqual(hit.bm25_score, 0.0)
        self.assertEqual(positional_hit.semantic_score, 0.9)
        self.assertEqual(positional_hit.chain_score, 0.8)
        self.assertEqual(positional_hit.graph_score, 0.7)
        self.assertEqual(positional_hit.route_score, 0.6)
        self.assertEqual(positional_hit.bm25_score, 0.0)
        self.assertTrue(config.task_chain_enabled)

    def _config(self, tmpdir: str) -> TCMemConfig:
        return TCMemConfig(
            owner_id="unit",
            storage_path=str(Path(tmpdir) / "state"),
            log_path=str(Path(tmpdir) / "logs"),
            vector_index_backend="numpy",
            vector_index_path=str(Path(tmpdir) / "vectors"),
        )

    def _record(
        self,
        record_id: str,
        text: str,
        *,
        entities: list[str] | None = None,
        record_time: str = "",
    ) -> DialogueRecord:
        return DialogueRecord(
            record_id=record_id,
            session_identifier="case",
            session_uuid="session",
            current_time="2026-05-29",
            record_time=record_time,
            user_content=text,
            entities=list(entities or []),
        )

    def test_numpy_vector_index_persists_and_reloads_record_vectors(self) -> None:
        scorer = FakeEmbeddingClient()
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "record_index"
            index = NumpyVectorIndex(index_path, index_name="dialogue_records")
            index.sync_items(
                [
                    VectorIndexItem("rec_alpha", "alpha memory", {"session_uuid": "alpha_session"}),
                    VectorIndexItem("rec_beta", "beta memory", {"session_uuid": "beta_session"}),
                ],
                scorer,
                embedding_signature={"model": "fake"},
            )

            self.assertTrue((index_path / "embeddings.npy").exists())
            reloaded = NumpyVectorIndex(index_path, index_name="dialogue_records")
            hits = reloaded.search("alpha question", scorer, top_k=2)

        self.assertEqual([hit.item_id for hit in hits], ["rec_alpha", "rec_beta"])
        self.assertEqual(hits[0].metadata["session_uuid"], "alpha_session")

    def test_in_memory_bm25_index_ranks_keyword_hits_and_normalizes_scores(self) -> None:
        from tcmem.infrastructure.indices import InMemoryBM25Index

        index = InMemoryBM25Index()
        self.assertEqual(index.search("zanzibar ledger", top_k=3), [])

        index.sync_items(
            [
                VectorIndexItem("rec_low", "zanzibar archive"),
                VectorIndexItem("rec_high", "ledger ledger zanzibar ledger archive"),
                VectorIndexItem("rec_none", "alpha generic project note"),
            ]
        )

        self.assertEqual(index.search("", top_k=3), [])
        hits = index.search("zanzibar ledger", top_k=3)

        self.assertEqual([hit.item_id for hit in hits], ["rec_high", "rec_low"])
        self.assertGreater(hits[0].score, hits[1].score)
        self.assertEqual(hits[0].score, 1.0)
        self.assertTrue(all(0.0 <= hit.score <= 1.0 for hit in hits))
        self.assertLess(hits[1].score, 1.0)

        index.sync_items(
            [
                VectorIndexItem("rec_empty", "!!!"),
                VectorIndexItem("rec_mixed", "ledger ledger zanzibar ledger archive"),
                VectorIndexItem("rec_none", "alpha generic project note"),
            ]
        )

        mixed_hits = index.search("zanzibar ledger", top_k=3)

        self.assertEqual([hit.item_id for hit in mixed_hits], ["rec_mixed"])
        self.assertEqual(mixed_hits[0].score, 1.0)
        self.assertTrue(all(0.0 <= hit.score <= 1.0 for hit in mixed_hits))

    def test_memory_system_retrieves_through_persistent_vector_index_and_task_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            system = MemorySystem(
                config=self._config(tmpdir),
                embedding_client=FakeEmbeddingClient(),
                llm_client=FakeLLMClient(),
            )
            system.ingest_record(self._record("rec_alpha", "alpha project decision"))
            system.ingest_record(self._record("rec_beta", "beta unrelated note"))

            result = system.retrieve("alpha question", top_k=1)

        self.assertEqual(result.hits[0].source_record_id, "rec_alpha")
        self.assertIn("path_b_vector_bm25_graph", result.hits[0].reason)
        self.assertTrue(result.routed_task_ids)

    def test_memory_system_no_task_chain_mode_extracts_entities_without_task_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            config.task_chain_enabled = False
            config.path_a_weight = 0.0
            config.path_b_weight = 1.0
            llm = FakeLLMClient()
            system = MemorySystem(
                config=config,
                embedding_client=FakeEmbeddingClient(),
                llm_client=llm,
            )
            alpha = self._record("rec_alpha", "alpha project decision")
            beta = self._record("rec_beta", "beta unrelated note")
            system.ingest_record(alpha)
            system.ingest_record(beta)
            system.task_manager.route_for_query = lambda _query: self.fail("no-task-chain retrieval must not route queries")

            result = system.retrieve("alpha question", top_k=1)

        self.assertEqual(alpha.entities, ["alpha"])
        self.assertEqual(beta.entities, ["beta"])
        self.assertEqual(result.routed_task_ids, [])
        self.assertEqual(result.query_route_reason, "task_chain_disabled")
        self.assertEqual(result.hits[0].source_record_id, "rec_alpha")
        self.assertIsNone(result.hits[0].task_id)
        self.assertEqual(result.hits[0].route_score, 0.0)
        self.assertIn("no_task_chain", result.hits[0].reason)

    def test_graph_path_uses_bm25_seed_when_vector_seed_misses_keyword_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_config = self._config(tmpdir)
            base_config.path_a_weight = 0.0
            base_config.path_b_weight = 1.0
            base_config.graph_seed_limit = 0
            base_config.path_b_semantic_weight = 0.0
            base_config.path_b_graph_weight = 0.0
            base_config.path_b_bm25_weight = 1.0
            control_config = TCMemConfig(**base_config.to_dict(include_secrets=True))
            control_config.graph_bm25_seed_limit = 0
            enabled_config = TCMemConfig(**base_config.to_dict(include_secrets=True))
            enabled_config.graph_bm25_seed_limit = 1

            control_system = MemorySystem(
                config=control_config,
                embedding_client=KeywordRescueEmbeddingClient(),
                llm_client=NoRouteLLMClient(),
            )
            enabled_system = MemorySystem(
                config=enabled_config,
                embedding_client=KeywordRescueEmbeddingClient(),
                llm_client=NoRouteLLMClient(),
            )
            for system in (control_system, enabled_system):
                system.ingest_record(self._record("rec_alpha", "alpha question bridge note", entities=["shared"]))
                system.ingest_record(self._record("rec_kw", "zanzibar ledger compliance detail", entities=["shared"]))
                system.ingest_record(self._record("rec_none", "beta generic note"))

            control_result = control_system.retrieve("zanzibar ledger", top_k=5)
            enabled_result = enabled_system.retrieve("zanzibar ledger", top_k=5)

        self.assertNotIn("rec_kw", [hit.source_record_id for hit in control_result.hits])
        record_ids = [hit.source_record_id for hit in enabled_result.hits]
        self.assertIn("rec_kw", record_ids)
        keyword_hit = next(hit for hit in enabled_result.hits if hit.source_record_id == "rec_kw")
        alpha_hit = next(hit for hit in enabled_result.hits if hit.source_record_id == "rec_alpha")
        self.assertIn("path_b_vector_bm25_graph", keyword_hit.reason)
        self.assertGreater(keyword_hit.bm25_score, 0.0)
        self.assertEqual(alpha_hit.bm25_score, 0.0)

    def test_path_b_ranking_uses_explicit_bm25_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            config.path_a_weight = 0.0
            config.path_b_weight = 1.0
            config.graph_seed_limit = 0
            config.graph_bm25_seed_limit = 2
            config.path_b_semantic_weight = 0.0
            config.path_b_graph_weight = 0.0
            config.path_b_bm25_weight = 1.0
            system = MemorySystem(
                config=config,
                embedding_client=FakeEmbeddingClient(),
                llm_client=NoRouteLLMClient(),
            )
            system.ingest_record(self._record("rec_low", "zanzibar archive"))
            system.ingest_record(self._record("rec_high", "ledger ledger zanzibar ledger archive"))
            system.ingest_record(self._record("rec_none", "alpha generic project note"))

            result = system.retrieve("zanzibar ledger", top_k=2)

        self.assertEqual([hit.source_record_id for hit in result.hits], ["rec_high", "rec_low"])
        self.assertGreater(result.hits[0].score, result.hits[1].score)
        self.assertGreater(result.hits[0].bm25_score, result.hits[1].bm25_score)
        self.assertEqual(result.hits[0].score, result.hits[0].bm25_score)
        self.assertEqual(result.hits[1].score, result.hits[1].bm25_score)

    def test_path_b_graph_neighbor_without_lexical_match_keeps_zero_bm25_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            config.path_a_weight = 0.0
            config.path_b_weight = 1.0
            config.graph_seed_limit = 0
            config.graph_bm25_seed_limit = 1
            config.path_b_semantic_weight = 0.0
            config.path_b_graph_weight = 1.0
            config.path_b_bm25_weight = 0.0
            system = MemorySystem(
                config=config,
                embedding_client=FakeEmbeddingClient(),
                llm_client=FakeLLMClient(),
            )
            system.ingest_record(self._record("rec_kw", "zanzibar ledger compliance detail", entities=["shared"]))
            system.ingest_record(self._record("rec_neighbor", "alpha planning note", entities=["shared"]))

            result = system.retrieve("zanzibar ledger", top_k=5)

        neighbor_hit = next(hit for hit in result.hits if hit.source_record_id == "rec_neighbor")
        self.assertEqual(neighbor_hit.bm25_score, 0.0)

    def test_query_routing_requires_llm_client_without_fallback(self) -> None:
        manager = TaskChainManager("unit", llm_client=None)

        with self.assertRaisesRegex(RuntimeError, "LLM client missing for stage query_intent_understanding"):
            manager.route_for_query("alpha question")

    def test_query_routing_retries_json_decode_error_and_logs_raw_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            llm = FlakyJSONLLMClient()
            manager = TaskChainManager("unit", llm_client=llm, log_store=log_store)
            record = self._record("rec_alpha", "alpha project decision", entities=["alpha"])
            task = manager.create_task_from_record(record, {"task_description": "Alpha task"})
            task.task_id = "task_manual"
            manager.tasks = {"task_manual": task}

            decision = manager.route_for_query("alpha question")

            errors = (Path(tmpdir) / "run" / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(decision.routed_task_ids, ["task_manual"])
        self.assertEqual(llm.calls, 2)
        payload = json.loads(errors[0])["payload"]
        self.assertEqual(payload["stage"], "query_routing")
        self.assertEqual(payload["attempt"], 1)
        self.assertIn("raw_response_excerpt", payload)

    def test_record_routing_retries_empty_route_until_a_task_is_selected(self) -> None:
        llm = EmptyRouteThenTaskLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha process handoff")

        decision = manager.route_record(record)

        self.assertEqual(llm.task_routing_calls, 3)
        self.assertEqual(len(decision.routed_task_ids), 1)
        self.assertEqual(manager.tasks[decision.routed_task_ids[0]].task_description, "Alpha handoff task")

    def test_record_routing_accepts_schema_wrapped_response(self) -> None:
        llm = CapturingLLMClient(
            {
                "schema": {
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": "Alpha schema task"}],
                    "confidence": 1.0,
                    "reason": "wrapped",
                }
            }
        )
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha schema handoff")

        decision = manager.route_record(record)

        self.assertEqual(len(decision.routed_task_ids), 1)
        self.assertEqual(manager.tasks[decision.routed_task_ids[0]].task_description, "Alpha schema task")

    def test_record_routing_uses_intent_stage_and_materializes_new_tasks_once(self) -> None:
        llm = IntentAwareRoutingLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)

        decision = manager.route_record(
            self._record("rec_1", "alpha launch follow up", entities=["alpha"]),
            routing_window=[
                self._record("rec_0", "kick off alpha launch", entities=["alpha"]),
                self._record("rec_1", "alpha launch follow up", entities=["alpha"]),
            ],
            current_index=1,
        )

        self.assertEqual(decision.intent.main_intent, "continue alpha launch plan")
        self.assertEqual(decision.main_intent, "continue alpha launch plan")
        self.assertEqual(len(manager.tasks), 1)
        self.assertEqual(len(decision.created_task_ids), 1)
        self.assertEqual(llm.payloads_by_stage["task_routing"][0]["intent"]["main_intent"], "continue alpha launch plan")
        self.assertEqual(
            llm.payloads_by_stage["task_routing_review"][0]["intent"]["main_intent"],
            "continue alpha launch plan",
        )
        self.assertNotIn("routing_window", llm.payloads_by_stage["task_routing"][0])
        self.assertNotIn("routing_window", llm.payloads_by_stage["task_routing_review"][0])

    def test_record_intent_understanding_separates_record_summary_from_task_intent(self) -> None:
        llm = ContextualRecordIntentLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)
        task = manager.create_task_from_record(
            self._record("seed", "Need a WhatsApp tool", entities=["WhatsApp"]),
            {"task_description": "Select a WhatsApp voice transcription tool"},
        )
        task.task_id = "task_whatsapp"
        manager.tasks = {"task_whatsapp": task}

        decision = manager.route_record(
            self._record("rec_1", "It must work offline and stay cheap", entities=["WhatsApp", "offline"]),
            routing_window=[
                self._record("rec_0", "Need a WhatsApp voice note tool", entities=["WhatsApp"]),
                self._record("rec_1", "It must work offline and stay cheap", entities=["WhatsApp", "offline"]),
            ],
            current_index=1,
        )

        self.assertEqual(decision.intent.summary, "user tightened the WhatsApp tool requirements")
        self.assertEqual(decision.intent.main_intent, "continue selecting a WhatsApp voice transcription tool")
        self.assertEqual(
            llm.payloads_by_stage["task_routing"][0]["intent"]["summary"],
            "user tightened the WhatsApp tool requirements",
        )
        self.assertEqual(
            llm.payloads_by_stage["task_routing"][0]["intent"]["main_intent"],
            "continue selecting a WhatsApp voice transcription tool",
        )

    def test_routed_record_writes_node_summary_from_intent_summary(self) -> None:
        llm = IntentSummaryOverridesConflictLLMClient()
        manager = TaskChainManager("unit", llm_client=llm, task_metadata_refresh_interval=0)
        task = manager.create_task_from_record(
            self._record("seed", "alpha seed", entities=["alpha"]),
            {"task_description": "Alpha task"},
        )
        task.task_id = "task_alpha"
        manager.tasks = {"task_alpha": task}

        system = MemorySystem(
            config=self._config(tempfile.mkdtemp()),
            task_manager=manager,
            embedding_client=FakeEmbeddingClient(),
            llm_client=llm,
        )
        system.ingest_record(
            self._record(
                "rec_1",
                "alpha follow up",
                entities=["alpha"],
                record_time="2026-05-30 10:00:00",
            )
        )
        system.flush_all_pending_routes()

        task = manager.tasks["task_alpha"]
        created_nodes = [node for node in task.nodes.values() if node.source_record_id == "rec_1"]
        self.assertEqual(len(created_nodes), 1)
        self.assertEqual(created_nodes[0].summary, "intent summary for alpha follow up")
        self.assertEqual(
            created_nodes[0].metadata["record_intent"]["main_intent"],
            "continue alpha implementation",
        )
        self.assertEqual(
            created_nodes[0].metadata["record_intent"]["summary"],
            "intent summary for alpha follow up",
        )

    def test_entity_extraction_prefers_top_level_fields_over_prompt_schema(self) -> None:
        llm = CapturingLLMClient({"entities": ["alpha"], "schema": {"entities": "string"}})
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha entity")

        entities = manager.extract_record_entities(record)

        self.assertEqual(entities, ["alpha"])

    def test_entity_extraction_retries_invalid_payload_shape(self) -> None:
        llm = InvalidEntityThenValidLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha entity")

        entities = manager.extract_record_entities(record)

        self.assertEqual(llm.calls, 2)
        self.assertEqual(entities, ["alpha"])

    def test_entity_extraction_keeps_empty_entities_after_invalid_retries(self) -> None:
        llm = AlwaysInvalidEntityLLMClient()
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha entity")

        entities = manager.extract_record_entities(record)

        self.assertEqual(llm.calls, 2)
        self.assertEqual(entities, [])

    def test_router_task_summary_exposes_status_entities_and_recent_records(self) -> None:
        llm = CapturingLLMClient({"routed_task_ids": ["task_manual"], "reason": "matched"})
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha project decision", entities=[f"e{i}" for i in range(30)])
        task = manager.create_task_from_record(record, {"task_description": "Alpha task"})
        task.task_id = "task_manual"
        manager.tasks = {"task_manual": task}
        manager.apply_record(task.task_id, self._record("rec_1", "alpha first", entities=["alpha"]))
        second = self._record("rec_2", "alpha second", entities=["alpha"])
        second.assistant_content = "assistant second"
        manager.apply_record(task.task_id, second)
        manager.apply_record(task.task_id, self._record("rec_3", "alpha third", entities=["alpha"]))
        manager.apply_record(task.task_id, self._record("rec_4", "alpha fourth", entities=["alpha"]))

        manager.route_for_query("alpha question")

        task_summary = llm.payloads[-1]["task_catalog"][0]
        self.assertEqual(set(task_summary), {"task_id", "task_description", "status", "entities", "recent_records"})
        self.assertEqual(task_summary["status"], "active")
        self.assertEqual(task_summary["entities"], [f"e{i}" for i in range(20)])
        self.assertEqual([item["source_record_id"] for item in task_summary["recent_records"]], ["rec_4", "rec_3", "rec_2"])
        self.assertEqual(
            set(task_summary["recent_records"][0]),
            {"source_record_id", "user_content", "assistant_content"},
        )
        self.assertEqual(task_summary["recent_records"][2]["assistant_content"], "assistant second")

    def test_query_routing_sends_candidate_count_and_limits_returned_ids(self) -> None:
        llm = CapturingLLMClient(
            {
                "routed_task_ids": [f"task_{index}" for index in range(10)],
                "reason": "ranked candidates",
            }
        )
        manager = TaskChainManager("unit", llm_client=llm, query_router_candidate_count=3)
        for index in range(10):
            task = manager.create_task_from_record(
                self._record(f"rec_{index}", f"topic {index}"),
                {"task_description": f"Topic {index} task"},
            )
            task.task_id = f"task_{index}"
            manager.tasks[task.task_id] = task

        decision = manager.route_for_query("topic question")

        self.assertEqual(llm.payloads[-1]["candidate_count"], 3)
        self.assertEqual(decision.routed_task_ids, ["task_0", "task_1", "task_2"])

    def test_query_routing_returns_query_route_decision_and_logs_query_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            llm = IntentAwareQueryLLMClient()
            manager = TaskChainManager("owner", llm_client=llm, log_store=log_store)
            task = manager.create_task_from_record(
                self._record("rec_alpha", "alpha planning", entities=["alpha"]),
                {"task_description": "Alpha task"},
            )
            task.task_id = "task_alpha"
            manager.tasks = {"task_alpha": task}

            decision = manager.route_for_query("What is the next alpha step?")
            entries = [
                json.loads(line)
                for line in log_store.path_for("routing").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "query_routed"
            ]

        self.assertEqual(decision.routed_task_ids, ["task_alpha"])
        self.assertEqual(decision.query_intent.main_intent, "retrieve next alpha step")
        self.assertEqual(llm.payloads_by_stage["query_routing"][0]["intent"]["main_intent"], "retrieve next alpha step")
        self.assertEqual(entries[0]["payload"]["query_intent"]["main_intent"], "retrieve next alpha step")

    def test_memory_system_logs_query_intent_in_retrieval_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TCMemConfig(log_path=tmpdir, vector_index_backend="numpy")
            llm = FakeLLMClient()
            system = MemorySystem(
                config=config,
                llm_client=llm,
                embedding_client=FakeEmbeddingClient(),
                record_index=NumpyVectorIndex(Path(tmpdir) / "vector_index", index_name="records"),
            )
            record = self._record("rec_alpha", "alpha planning", entities=["alpha"])
            system.task_manager.extract_record_entities(record)
            task = system.task_manager.create_task_from_record(record, {"task_description": "Alpha task"})
            task.task_id = "task_alpha"
            system.task_manager.tasks = {"task_alpha": task}
            system.graph.add_record(record)

            result = system.retrieve("alpha question", top_k=3)
            entries = [
                json.loads(line)
                for line in system.log_store.path_for("retrieval").read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(result.query_intent is not None)
        self.assertEqual(entries[-1]["payload"]["query_intent"]["main_intent"], result.query_intent.main_intent)
        self.assertEqual(entries[-1]["payload"]["query_route_reason"], result.query_route_reason)

    def test_docs_describe_intent_aware_routing_flow(self) -> None:
        readme = Path("README.md").read_text(encoding="utf-8")
        architecture = Path("docs/architecture.md").read_text(encoding="utf-8")

        self.assertIn("record_intent_understanding", readme)
        self.assertIn("query_intent_understanding", readme)
        self.assertIn("query_intent", readme)
        self.assertIn("record_intent_understanding", architecture)
        self.assertIn("query_intent_understanding", architecture)

    def test_record_routing_limits_task_catalog_to_recent_relevant_candidates(self) -> None:
        llm = CapturingLLMClient(
            {
                "linked_task_ids": ["task_19"],
                "new_tasks": [],
                "confidence": 1.0,
                "reason": "recent alpha task",
            }
        )
        manager = TaskChainManager("unit", llm_client=llm)
        for index in range(20):
            task = manager.create_task_from_record(
                self._record(f"rec_{index}", f"alpha topic {index}", entities=["alpha"]),
                {"task_description": f"Alpha task {index}"},
            )
            task.task_id = f"task_{index}"
            task.updated_at = f"2026-05-30 10:{index:02d}:00"
            manager.tasks[task.task_id] = task

        decision = manager.route_record(self._record("rec_current", "alpha follow up", entities=["alpha"]))

        task_catalog = llm.payloads[-1]["task_catalog"]
        self.assertEqual(len(task_catalog), 12)
        self.assertEqual([item["task_id"] for item in task_catalog[:3]], ["task_19", "task_18", "task_17"])
        self.assertTrue(all("recent_records" in item for item in task_catalog[:3]))
        self.assertTrue(all("entities" in item for item in task_catalog[:3]))
        self.assertEqual(decision.routed_task_ids, ["task_19"])

    def test_record_task_routing_task_catalog_includes_entities(self) -> None:
        llm = ConflictAwareLLMClient([])
        manager = TaskChainManager("unit", llm_client=llm)
        task = manager.create_task_from_record(
            self._record("seed", "alpha seed", entities=["alpha"]),
            {"task_description": "Alpha task"},
        )
        task.task_id = "task_alpha"
        manager.tasks = {"task_alpha": task}

        manager.route_record(self._record("rec_1", "alpha follow up", entities=["alpha"]))

        routing_task = llm.payloads_by_stage["task_routing"][0]["task_catalog"][0]
        review_task = llm.payloads_by_stage["task_routing_review"][0]["task_catalog"][0]
        self.assertEqual(set(routing_task), {"task_id", "task_description", "status", "entities", "recent_records"})
        self.assertEqual(set(review_task), {"task_id", "task_description", "status", "entities", "recent_records"})
        self.assertEqual(routing_task["entities"], ["alpha"])
        self.assertEqual(review_task["entities"], ["alpha"])

    def test_entity_extraction_keeps_only_top_three_entities(self) -> None:
        llm = CapturingLLMClient({"entities": ["alpha", "beta", "gamma", "delta"]})
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha beta gamma delta entity")

        entities = manager.extract_record_entities(record)

        self.assertEqual(entities, ["alpha", "beta", "gamma"])

    def test_record_routing_deduplicates_and_caps_linked_task_ids(self) -> None:
        llm = CapturingLLMClient(
            {
                "linked_task_ids": ["task_0", "task_0", "task_1", "task_2", "task_3", "task_4", "task_5"],
                "new_tasks": [],
                "confidence": 1.0,
                "reason": "overbroad duplicate route",
            }
        )
        manager = TaskChainManager("unit", llm_client=llm)
        for index in range(6):
            task = manager.create_task_from_record(
                self._record(f"rec_{index}", f"alpha topic {index}", entities=["alpha"]),
                {"task_description": f"Alpha task {index}"},
            )
            task.task_id = f"task_{index}"
            manager.tasks[task.task_id] = task

        decision = manager.route_record(self._record("rec_current", "alpha follow up", entities=["alpha"]))

        self.assertEqual(decision.linked_task_ids, ["task_0", "task_1", "task_2", "task_3", "task_4"])
        self.assertEqual(decision.routed_task_ids, ["task_0", "task_1", "task_2", "task_3", "task_4"])

    def test_create_task_has_no_topic_attribute(self) -> None:
        manager = TaskChainManager("unit", llm_client=FakeLLMClient())
        task = manager.create_task_from_record(
            self._record("rec_alpha", "alpha seed"),
            {"task_description": "Investigate AI-powered WhatsApp Business API integrations for voice notes"},
        )

        self.assertFalse(hasattr(task, "topic"))

    def test_record_router_payload_omits_stable_session_and_metadata_fields(self) -> None:
        llm = CapturingLLMClient(
            {
                "linked_task_ids": [],
                "new_tasks": [{"task_description": "Alpha task"}],
                "confidence": 1.0,
                "reason": "new",
            }
        )
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha project decision", entities=["alpha"])
        record.assistant_content = "assistant reply"
        record.metadata["unused"] = "value"

        manager.route_record(record)

        routed_record = llm.payloads[-1]["current_record"]
        self.assertEqual(set(routed_record), {"record_id", "user_content", "assistant_content", "entities"})

    def test_record_route_decision_preserves_main_intent(self) -> None:
        llm = CapturingLLMClient(
            {
                "main_intent": "continue alpha planning",
                "linked_task_ids": [],
                "new_tasks": [{"task_description": "Alpha task"}],
                "confidence": 1.0,
                "reason": "new",
            }
        )
        manager = TaskChainManager("unit", llm_client=llm)

        decision = manager.route_record(self._record("rec_alpha", "alpha project decision", entities=["alpha"]))

        self.assertEqual(decision.main_intent, "continue alpha planning")
        self.assertEqual(decision.intent.main_intent, "continue alpha planning")

    def test_route_and_retrieval_models_preserve_intent_fields(self) -> None:
        from tcmem import IntentUnderstanding, QueryRouteDecision
        from tcmem.models import RetrievalResult, RouteDecision

        default_intent = IntentUnderstanding(main_intent="continue alpha planning")
        intent = IntentUnderstanding(
            summary="user narrowed the alpha requirements",
            main_intent="continue alpha planning",
            topic_hint="alpha roadmap",
            key_entities=["alpha", "milestone"],
            is_substantive=True,
            reason="same delivery plan",
        )

        route = RouteDecision(
            linked_task_ids=["task_1"],
            created_task_ids=[],
            routed_task_ids=["task_1"],
            main_intent=intent.main_intent,
            confidence=0.9,
            reason="matched active alpha task",
            intent=intent,
        )
        normalized_route = RouteDecision(
            linked_task_ids=["task_1"],
            created_task_ids=[],
            routed_task_ids=["task_1"],
            main_intent="",
            confidence=0.9,
            reason="matched active alpha task",
            intent=intent,
        )
        query_route = QueryRouteDecision(
            query="What is next for alpha?",
            routed_task_ids=["task_1"],
            query_intent=intent,
            reason="matched active alpha task",
        )
        result = RetrievalResult(
            query="What is next for alpha?",
            subqueries=["What is next for alpha?"],
            routed_task_ids=["task_1"],
            hits=[],
            query_intent=intent,
            query_route_reason="matched active alpha task",
        )
        legacy_result = RetrievalResult(
            "What is next for alpha?",
            ["What is next for alpha?"],
            ["task_1"],
            [],
            "matched active alpha task",
        )

        self.assertTrue(default_intent.is_substantive)
        self.assertEqual(intent.summary, "user narrowed the alpha requirements")
        self.assertEqual(route.intent.main_intent, "continue alpha planning")
        self.assertEqual(route.intent.summary, "user narrowed the alpha requirements")
        self.assertEqual(normalized_route.main_intent, "continue alpha planning")
        self.assertEqual(query_route.query_intent.topic_hint, "alpha roadmap")
        self.assertEqual(result.query_intent.key_entities, ["alpha", "milestone"])
        self.assertEqual(result.query_route_reason, "matched active alpha task")
        self.assertEqual(legacy_result.explanation, "matched active alpha task")
        self.assertIsNone(legacy_result.query_intent)

    def test_task_metadata_refreshes_on_configured_interval(self) -> None:
        manager = TaskChainManager("unit", llm_client=FakeLLMClient(), task_metadata_refresh_interval=2)
        task = manager.create_task_from_record(self._record("rec_alpha", "alpha seed"), {"task_description": "Alpha task"})

        manager.apply_record(task.task_id, self._record("rec_1", "alpha first"))
        self.assertEqual(task.task_description, "Alpha task")

        manager.apply_record(task.task_id, self._record("rec_2", "alpha second"))

        self.assertEqual(task.task_description, "Refreshed task")
        self.assertEqual(task.entities, ["refreshed"])

    def test_state_round_trip_preserves_task_status_and_ignores_legacy_topic(self) -> None:
        manager = TaskChainManager("unit", llm_client=FakeLLMClient())
        task = manager.create_task_from_record(
            self._record("rec_alpha", "alpha seed"),
            {"task_description": "Alpha task"},
        )
        task.status = TaskStatus.BLOCKED

        state = manager.to_state()
        state["tasks"][0]["topic"] = "legacy topic"
        reloaded = TaskChainManager.from_state(state, llm_client=FakeLLMClient())
        reloaded_task = reloaded.tasks[task.task_id]

        self.assertFalse(hasattr(reloaded_task, "topic"))
        self.assertEqual(reloaded_task.task_description, "Alpha task")
        self.assertEqual(reloaded_task.status, TaskStatus.BLOCKED)

    def test_state_round_trip_preserves_graph_and_task_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            system = MemorySystem(config=config, embedding_client=FakeEmbeddingClient(), llm_client=FakeLLMClient())
            system.ingest_record(self._record("rec_alpha", "alpha project decision"))
            state_path = system.save()

            loaded = MemorySystem.load(
                state_path,
                config=config,
                embedding_client=FakeEmbeddingClient(),
                llm_client=FakeLLMClient(),
            )

        self.assertEqual(loaded.state_summary()["record_count"], 1)
        self.assertGreaterEqual(loaded.state_summary()["task_count"], 1)

    def test_graph_walk_preserves_special_graph_structure(self) -> None:
        graph = DialogueGraphStore()
        graph.add_record(self._record("rec_1", "alpha one", entities=["alpha"]))
        graph.add_record(self._record("rec_2", "alpha two", entities=["alpha"]))

        walked = graph.walk(["rec_1"], max_depth=1)

        self.assertEqual(walked["rec_1"], 0)
        self.assertEqual(walked["rec_2"], 1)


if __name__ == "__main__":
    unittest.main()
