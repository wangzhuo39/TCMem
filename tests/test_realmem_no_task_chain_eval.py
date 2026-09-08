import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tcmem.config import TCMemConfig
from tcmem.core.memory_system import MemorySystem
from tcmem.core.task_chain import _extract_json_payload
from tcmem.evals import realmem_top_session
from tcmem.infrastructure.indices import NumpyVectorIndex
from tcmem.models import DialogueRecord


def _prompt_payload(prompt: str) -> dict:
    match = re.search(r"<[A-Za-z0-9_]*Input>\s*(\{.*?\}|\[.*?\])\s*</[A-Za-z0-9_]*Input>", prompt, flags=re.S)
    if match is None:
        match = re.search(r"## Input\s*(\{.*?\}|\[.*?\])\s*## Output format", prompt, flags=re.S | re.I)
    parsed = json.loads(match.group(1)) if match else _extract_json_payload(prompt)
    if not isinstance(parsed, dict):
        raise AssertionError(f"Expected object prompt payload: {prompt}")
    return parsed


class FakeEmbeddingClient:
    def _vector(self, text):
        return np.asarray([1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0], dtype="float32")

    def embed_query(self, query):
        return self._vector(query)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def score(self, query, text):
        query_has_alpha = "alpha" in query.lower()
        text_has_alpha = "alpha" in text.lower()
        return 1.0 if query_has_alpha == text_has_alpha else 0.1


class EntityOnlyLLMClient:
    def __init__(self):
        self.payloads = []

    def generate(self, prompt, **_kwargs):
        payload = _prompt_payload(prompt)
        self.payloads.append(payload)
        if "record" in payload:
            return json.dumps({"entities": ["alpha"]})
        raise AssertionError(f"Unexpected task-chain prompt: {prompt}")


class RealMemNoTaskChainEvalTest(unittest.TestCase):
    def _system(self, tmpdir, llm):
        config = TCMemConfig(
            storage_path=str(Path(tmpdir) / "state"),
            log_path=str(Path(tmpdir) / "logs"),
            vector_index_backend="numpy",
            vector_index_path=str(Path(tmpdir) / "vector_index"),
            task_chain_enabled=False,
        )
        return MemorySystem(
            config=config,
            llm_client=llm,
            embedding_client=FakeEmbeddingClient(),
            record_index=NumpyVectorIndex(Path(tmpdir) / "vector_index", index_name="records"),
        )

    def test_disabled_task_chain_ingestion_keeps_entities_graph_edges_and_no_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = EntityOnlyLLMClient()
            system = self._system(tmpdir, llm)
            first = DialogueRecord(
                record_id="rec_1",
                session_identifier="s1",
                session_uuid="uuid_1",
                current_time="2026-06-01",
                user_content="alpha planning",
            )
            second = DialogueRecord(
                record_id="rec_2",
                session_identifier="s2",
                session_uuid="uuid_2",
                current_time="2026-06-01",
                user_content="alpha follow up",
            )

            system.ingest_record(first)
            system.ingest_record(second)

            self.assertEqual(first.entities, ["alpha"])
            self.assertEqual(second.entities, ["alpha"])
            self.assertGreater(system.state_summary()["edge_count"], 0)
            self.assertEqual(system.state_summary()["task_count"], 0)
            self.assertEqual(len(llm.payloads), 2)

    def test_disabled_task_chain_retrieval_uses_path_b_without_query_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = EntityOnlyLLMClient()
            system = self._system(tmpdir, llm)
            system.ingest_record(
                DialogueRecord(
                    record_id="rec_1",
                    session_identifier="s1",
                    session_uuid="uuid_1",
                    current_time="2026-06-01",
                    user_content="alpha planning",
                )
            )

            result = system.retrieve("alpha planning", top_k=5)

            self.assertEqual(result.routed_task_ids, [])
            self.assertEqual(result.query_route_reason, "task_chain_disabled")
            self.assertTrue(result.hits)
            self.assertTrue(all("path_b_vector_bm25_graph_no_task_chain" in hit.reason for hit in result.hits))
            self.assertTrue(all(hit.chain_score == 0.0 for hit in result.hits))
            self.assertTrue(all(hit.route_score == 0.0 for hit in result.hits))
            self.assertTrue(all(hit.task_id is None for hit in result.hits))
            self.assertEqual(len(llm.payloads), 1)

    def test_standard_eval_options_keep_task_chain_enabled(self):
        args = realmem_top_session.parse_args([])

        options = realmem_top_session.evaluation_options_from_args(args)

        self.assertEqual(options.mode, "tcmem_top_session")
        self.assertTrue(options.task_chain_enabled)

    def test_config_builder_can_disable_task_chains_for_ablation(self):
        args = realmem_top_session.parse_args(["--vector-index-backend", "numpy"])

        config = realmem_top_session.build_tcmem_config(
            args,
            output_dir=Path("out"),
            log_store_base_dir="logs",
            runtime=realmem_top_session.RuntimeConfig(api_key="key"),
            task_chain_enabled=False,
        )

        self.assertFalse(config.task_chain_enabled)
        self.assertEqual(config.vector_index_backend, "numpy")

    def test_no_task_chain_eval_options_force_ablation_mode(self):
        from tcmem.evals import realmem_no_task_chain

        args = realmem_no_task_chain.parse_args([])

        options = realmem_no_task_chain.evaluation_options_from_args(args)

        self.assertEqual(options.mode, "tcmem_no_task_chain_online")
        self.assertFalse(options.task_chain_enabled)
        self.assertEqual(options.ablation_design["llm_usage"]["ingest"], ["entity_extraction"])
        self.assertEqual(options.ablation_design["llm_usage"]["retrieval"], [])
        self.assertIn("query_before_ingest", options.ablation_design["online_order"])
        self.assertIn("vector_bm25_graph_walk", options.ablation_design["retrieval_flow"])
        self.assertEqual(options.run_name_prefix, "tcmem_realmem_no_task_chain")

    def test_render_report_includes_mode(self):
        report = realmem_top_session.render_report(
            run_config={
                "mode": "tcmem_no_task_chain_online",
                "model": "m",
                "retrieval_record_k": 1,
                "session_ks": [1],
                "with_qa": False,
                "tcmem_config": {},
            },
            dataset_summary={"person_name": "p", "session_count": 1, "query_count": 1, "record_count": 1},
            metrics_summary={"query_count": 1},
        )

        self.assertIn("- mode: tcmem_no_task_chain_online", report)

    def test_render_report_includes_no_task_chain_ablation_design(self):
        from tcmem.evals import realmem_no_task_chain

        options = realmem_no_task_chain.evaluation_options_from_args(realmem_no_task_chain.parse_args([]))
        report = realmem_top_session.render_report(
            run_config={
                "mode": options.mode,
                "model": "m",
                "retrieval_record_k": 1,
                "session_ks": [1],
                "with_qa": False,
                "tcmem_config": {"embedding_model": "e", "vector_index_backend": "numpy"},
                "ablation_design": options.ablation_design,
            },
            dataset_summary={"person_name": "p", "session_count": 1, "query_count": 1, "record_count": 1},
            metrics_summary={"query_count": 1},
        )

        self.assertIn("## Ablation Design", report)
        self.assertIn("query_before_ingest", report)
        self.assertIn("LLM ingest: entity_extraction", report)
        self.assertIn("LLM retrieval: none", report)
        self.assertIn("vector_bm25_graph_walk", report)


if __name__ == "__main__":
    unittest.main()
