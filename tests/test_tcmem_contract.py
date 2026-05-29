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
        if "record" in payload:
            text = payload["record"].get("user_content", "")
            entities = ["alpha"] if "alpha" in text.lower() else ["beta"]
            return json.dumps({"entities": entities})
        if "current_record" in payload:
            tasks = payload.get("task_catalog", [])
            record_text = payload["current_record"].get("user_content", "")
            if tasks:
                return json.dumps({"linked_task_ids": [tasks[0]["task_id"]], "new_tasks": [], "confidence": 1.0, "reason": "existing"})
            topic = "Alpha" if "alpha" in record_text.lower() else "Beta"
            return json.dumps(
                {
                    "linked_task_ids": [],
                    "new_tasks": [{"task_description": f"{topic} task"}],
                    "confidence": 1.0,
                    "reason": "new",
                }
            )
        if "task" in payload:
            return json.dumps({"task_description": "Refreshed task", "entities": ["refreshed"]})
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


class FlakyJSONLLMClient:
    json_max_attempts = 5
    json_retry_delay = 0.0

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, **_kwargs) -> str:
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
        if "current_record" in payload:
            self.task_routing_calls += 1
            if self.task_routing_calls < 3:
                return json.dumps({"linked_task_ids": [], "new_tasks": [], "confidence": 0.1, "reason": "unsure"})
            return json.dumps(
                {
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
        return json.dumps(self.response)


def _prompt_payload(prompt: str) -> dict:
    match = re.search(r"<[A-Za-z0-9_]*Input>\s*(\{.*?\}|\[.*?\])\s*</[A-Za-z0-9_]*Input>", prompt, flags=re.S)
    if match is None:
        match = re.search(r"## Input\s*(\{.*?\}|\[.*?\])\s*## Output format", prompt, flags=re.S | re.I)
    parsed = json.loads(match.group(1)) if match else _extract_json_payload(prompt)
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected object prompt payload: {prompt}")
    return parsed


class TCMemContractTest(unittest.TestCase):
    def test_default_prompts_use_handwritten_txt_templates_for_tcmem_stages(self) -> None:
        registry = PromptRegistry.default()

        for name in [
            "entity_extraction",
            "task_routing",
            "query_routing",
            "task_metadata_refresh",
        ]:
            rendered = registry.render(name, payload_json='{"ok": true}')
            self.assertTrue(rendered.system_prompt)
            self.assertIn("{{payload_json}}", Path(f"tcmem/prompts/{name}.txt").read_text(encoding="utf-8"))
            self.assertIn('{"ok": true}', rendered.user_prompt)
            self.assertIn("## Input", rendered.user_prompt)
            self.assertIn("## Output format", rendered.user_prompt)

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
                llm_client=FakeLLMClient(),
            )
            enabled_system = MemorySystem(
                config=enabled_config,
                embedding_client=KeywordRescueEmbeddingClient(),
                llm_client=FakeLLMClient(),
            )
            for system in (control_system, enabled_system):
                system.ingest_record(self._record("rec_alpha", "alpha question bridge note"))
                system.ingest_record(self._record("rec_kw", "zanzibar ledger compliance detail"))

            control_result = control_system.retrieve("alpha question", top_k=5)
            enabled_result = enabled_system.retrieve("alpha question", top_k=5)

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
                llm_client=FakeLLMClient(),
            )
            system.ingest_record(self._record("rec_low", "zanzibar archive"))
            system.ingest_record(self._record("rec_high", "ledger ledger zanzibar ledger archive"))

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

        with self.assertRaisesRegex(RuntimeError, "LLM client missing for stage query_routing"):
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

            routed = manager.route_for_query("alpha question")

            errors = (Path(tmpdir) / "run" / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(routed, ["task_manual"])
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

    def test_router_task_summary_omits_unused_fields_and_limits_entities(self) -> None:
        llm = CapturingLLMClient({"routed_task_ids": ["task_manual"], "reason": "matched"})
        manager = TaskChainManager("unit", llm_client=llm)
        record = self._record("rec_alpha", "alpha project decision", entities=[f"e{i}" for i in range(30)])
        task = manager.create_task_from_record(record, {"task_description": "Alpha task"})
        task.task_id = "task_manual"
        manager.tasks = {"task_manual": task}

        manager.route_for_query("alpha question")

        task_summary = llm.payloads[-1]["task_catalog"][0]
        self.assertEqual(set(task_summary), {"task_id", "task_description", "entities"})
        self.assertEqual(task_summary["entities"], [f"e{i}" for i in range(20)])

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

        routed = manager.route_for_query("topic question")

        self.assertEqual(llm.payloads[-1]["candidate_count"], 3)
        self.assertEqual(routed, ["task_0", "task_1", "task_2"])

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

    def test_task_metadata_refreshes_on_configured_interval(self) -> None:
        manager = TaskChainManager("unit", llm_client=FakeLLMClient(), task_metadata_refresh_interval=2)
        task = manager.create_task_from_record(self._record("rec_alpha", "alpha seed"), {"task_description": "Alpha task"})

        manager.apply_record(task.task_id, self._record("rec_1", "alpha first"))
        self.assertEqual(task.task_description, "Alpha task")

        manager.apply_record(task.task_id, self._record("rec_2", "alpha second"))

        self.assertEqual(task.task_description, "Refreshed task")
        self.assertEqual(task.entities, ["refreshed"])

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
