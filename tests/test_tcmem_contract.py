import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tcmem import DialogueRecord, TCMemConfig
from tcmem.core.graph_store import DialogueGraphStore
from tcmem.core.memory_system import MemorySystem
from tcmem.core.task_chain import TaskChainManager
from tcmem.infrastructure.indices import NumpyVectorIndex, VectorIndexItem


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


class FakeLLMClient:
    def generate(self, prompt: str, **_kwargs) -> str:
        payload = json.loads(prompt)
        if "record" in payload and "entities" in payload.get("schema", {}):
            text = payload["record"].get("user_content", "")
            entities = ["alpha"] if "alpha" in text.lower() else ["beta"]
            return json.dumps({"entities": entities})
        if "record" in payload and "new_tasks" in payload.get("schema", {}):
            tasks = payload.get("tasks", [])
            record_text = payload["record"].get("user_content", "")
            if tasks:
                return json.dumps({"linked_task_ids": [tasks[0]["task_id"]], "new_tasks": [], "confidence": 1.0, "reason": "existing"})
            topic = "Alpha" if "alpha" in record_text.lower() else "Beta"
            return json.dumps(
                {
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": f"{topic} task", "topic": topic}],
                    "confidence": 1.0,
                    "reason": "new",
                }
            )
        if "query" in payload:
            tasks = payload.get("tasks", [])
            query = payload["query"].lower()
            routed = [
                task["task_id"]
                for task in tasks
                if task["topic"].lower() in query or task["task_description"].lower().split()[0] in query
            ]
            return json.dumps({"routed_task_ids": routed or [task["task_id"] for task in tasks[:1]], "reason": "matched"})
        raise AssertionError(f"Unexpected prompt: {prompt}")


class TCMemContractTest(unittest.TestCase):
    def test_default_embedding_model_is_bge_m3(self) -> None:
        self.assertEqual(TCMemConfig().embedding_model, "BAAI/bge-m3")

    def _config(self, tmpdir: str) -> TCMemConfig:
        return TCMemConfig(
            owner_id="unit",
            storage_path=str(Path(tmpdir) / "state"),
            log_path=str(Path(tmpdir) / "logs"),
            vector_index_backend="numpy",
            vector_index_path=str(Path(tmpdir) / "vectors"),
        )

    def _record(self, record_id: str, text: str, *, entities: list[str] | None = None) -> DialogueRecord:
        return DialogueRecord(
            record_id=record_id,
            session_identifier="case",
            session_uuid="session",
            current_time="2026-05-29",
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
        self.assertIn("path_b_vector_graph", result.hits[0].reason)
        self.assertTrue(result.routed_task_ids)

    def test_query_routing_requires_llm_client_without_fallback(self) -> None:
        manager = TaskChainManager("unit", llm_client=None)

        with self.assertRaisesRegex(RuntimeError, "LLM client missing for stage query_routing"):
            manager.route_for_query("alpha question")

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
        self.assertEqual(loaded.state_summary()["task_count"], 1)

    def test_graph_walk_preserves_special_graph_structure(self) -> None:
        graph = DialogueGraphStore()
        graph.add_record(self._record("rec_1", "alpha one", entities=["alpha"]))
        graph.add_record(self._record("rec_2", "alpha two", entities=["alpha"]))

        walked = graph.walk(["rec_1"], max_depth=1)

        self.assertEqual(walked["rec_1"], 0)
        self.assertEqual(walked["rec_2"], 1)


if __name__ == "__main__":
    unittest.main()
