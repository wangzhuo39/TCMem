import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tcmem.config import TCMemConfig
from tcmem.evals.realmem_top_session import (
    PairRecord,
    build_progress_payload,
    compute_retrieval_metrics,
    generate_answer,
    judge_qa_score,
    log_query_result,
    parse_args,
    ranked_sessions_from_traces,
    run_evaluation,
    save_latest_state,
    save_query_snapshot,
    should_save_record_state,
)
from tcmem.logging_utils import ModuleLogStore
from tcmem.models import DialogueRecord
from tcmem.prompts import PromptRegistry


class RealMemTopSessionEvalTest(unittest.TestCase):
    def test_generate_answer_uses_custom_prompt_registry(self) -> None:
        class CapturingClient:
            def __init__(self) -> None:
                self.prompt = ""
                self.system_prompt = ""

            def generate(self, prompt: str, **kwargs) -> str:
                self.prompt = prompt
                self.system_prompt = kwargs["system_prompt"]
                return "answer"

        registry = PromptRegistry.from_mapping(
            {
                "realmem_answer_generation": {
                    "system": "CUSTOM QA SYSTEM",
                    "user": "Q={{question}}\nMEM={{evidence_text}}",
                }
            }
        )
        client = CapturingClient()

        answer = generate_answer(client, "question text", "session memory", prompt_registry=registry)

        self.assertEqual(answer, "answer")
        self.assertEqual(client.system_prompt, "CUSTOM QA SYSTEM")
        self.assertIn("Q=question text", client.prompt)
        self.assertIn("MEM=session memory", client.prompt)

    def test_ranked_sessions_from_traces_deduplicates_sessions_by_first_record_rank(self) -> None:
        traces = [
            {"source_record_id": "r1", "source_session_uuid": "s1", "score": 0.9},
            {"source_record_id": "r2", "source_session_uuid": "s1", "score": 0.8},
            {"source_record_id": "r3", "source_session_uuid": "gold", "score": 0.7},
        ]

        ranked = ranked_sessions_from_traces(traces)

        self.assertEqual([item["session_uuid"] for item in ranked], ["s1", "gold"])
        self.assertEqual(ranked[0]["record_count"], 2)
        self.assertEqual(ranked[0]["record_ids"], ["r1", "r2"])

    def test_metrics_score_top_session_recall(self) -> None:
        metrics = compute_retrieval_metrics(
            retrieved_session_uuids=["s1", "gold"],
            gold_session_uuids=["gold"],
            ks=[1, 2],
        )

        self.assertEqual(metrics["recall_all@1"], 0.0)
        self.assertEqual(metrics["recall_all@2"], 1.0)
        self.assertNotIn("recall_frac@1", metrics)
        self.assertNotIn("recall_frac@2", metrics)

    def test_parse_args_uses_tcmem_embedding_default(self) -> None:
        args = parse_args([])

        self.assertEqual(args.embedding_model, TCMemConfig().embedding_model)
        self.assertEqual(args.embedding_model, "BAAI/bge-m3")

    def test_progress_payload_reports_record_query_and_session_position(self) -> None:
        pair = PairRecord(
            session_index=1,
            record_index=2,
            user_turn_index=4,
            assistant_turn_index=5,
            record=DialogueRecord(
                record_id="rec_s2_0003",
                session_identifier="session-2",
                session_uuid="s2",
                current_time="2026-05-29",
                user_content="hello",
            ),
        )

        payload = build_progress_payload(
            pair=pair,
            processed_records=3,
            total_records=10,
            evaluated_queries=1,
            total_queries=2,
            elapsed_seconds=12.345,
            query_id="Q-0001",
        )

        self.assertEqual(payload["processed_records"], 3)
        self.assertEqual(payload["total_records"], 10)
        self.assertEqual(payload["record_progress"], 0.3)
        self.assertEqual(payload["evaluated_queries"], 1)
        self.assertEqual(payload["total_queries"], 2)
        self.assertEqual(payload["query_progress"], 0.5)
        self.assertEqual(payload["session_number"], 2)
        self.assertEqual(payload["record_number_in_session"], 3)
        self.assertEqual(payload["session_uuid"], "s2")
        self.assertEqual(payload["record_id"], "rec_s2_0003")
        self.assertEqual(payload["query_id"], "Q-0001")
        self.assertEqual(payload["elapsed_seconds"], 12.35)

    def test_query_result_log_preserves_generation_and_qa_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            result = {
                "query_id": "Q-0001",
                "question": "question",
                "retrieval_metrics": {"recall_all@20": 1.0},
                "qa_score": 3,
                "qa_reason": "uses all relevant memory",
                "generation_result": {
                    "generated_answer": "answer",
                    "evidence_used": "session text",
                    "evidence_session_uuids": ["s1"],
                },
            }

            log_query_result(log_store, "query_completed", result)

            entries = (Path(tmpdir) / "run" / "query_results.jsonl").read_text(encoding="utf-8").splitlines()
        payload = json.loads(entries[-1])["payload"]
        self.assertEqual(payload["query_id"], "Q-0001")
        self.assertEqual(payload["generation_result"]["generated_answer"], "answer")
        self.assertEqual(payload["qa_score"], 3)

    def test_save_latest_state_writes_graph_and_task_chain_snapshot(self) -> None:
        class FakeSystem:
            def save(self, path):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(
                    json.dumps({"graph": {"records": []}, "task_chains": {"tasks": []}}),
                    encoding="utf-8",
                )
                return Path(path)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = save_latest_state(FakeSystem(), Path(tmpdir))

            state = json.loads(state_path.read_text(encoding="utf-8"))

        self.assertEqual(state_path.name, "memory_state_latest.json")
        self.assertIn("graph", state)
        self.assertIn("task_chains", state)

    def test_save_query_snapshot_writes_state_vector_index_and_manifest(self) -> None:
        class FakeSystem:
            def __init__(self, vector_index_path: Path) -> None:
                self.saved_paths: list[Path] = []
                self.vector_index_path = vector_index_path

            def save(self, path):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "config": {
                                "storage_path": "old-state",
                                "vector_index_path": str(self.vector_index_path),
                                "vector_index_backend": "numpy",
                            },
                            "graph": {"records": [{"record_id": "rec_before_query"}]},
                            "task_chains": {"tasks": []},
                        }
                    ),
                    encoding="utf-8",
                )
                self.saved_paths.append(target)
                return target

        with tempfile.TemporaryDirectory() as tmpdir:
            vector_index_path = Path(tmpdir) / "source_vector_index"
            vector_index_path.mkdir()
            (vector_index_path / "manifest.json").write_text('{"item_count": 1}', encoding="utf-8")
            system = FakeSystem(vector_index_path)

            snapshot = save_query_snapshot(
                system,
                Path(tmpdir),
                "Q/0001",
                question="What should I do next?",
                metadata={"retrieval_record_k": 20},
            )

            state_path = Path(snapshot["memory_state_path"])
            state = json.loads(state_path.read_text(encoding="utf-8"))
            manifest = json.loads(Path(snapshot["manifest_path"]).read_text(encoding="utf-8"))
            copied_vector_manifest_exists = (Path(snapshot["vector_index_path"]) / "manifest.json").exists()

        self.assertEqual(Path(snapshot["snapshot_dir"]).name, "Q_0001")
        self.assertEqual(state_path.name, "memory_state.json")
        self.assertEqual(state_path.parent.parent.name, "query_snapshots")
        self.assertEqual(system.saved_paths, [state_path])
        self.assertEqual(state["graph"]["records"][0]["record_id"], "rec_before_query")
        self.assertEqual(state["config"]["storage_path"], snapshot["snapshot_dir"])
        self.assertEqual(state["config"]["vector_index_path"], snapshot["vector_index_path"])
        self.assertIn("task_chains", state)
        self.assertTrue(copied_vector_manifest_exists)
        self.assertEqual(manifest["query_id"], "Q/0001")
        self.assertEqual(manifest["question"], "What should I do next?")
        self.assertEqual(manifest["metadata"]["retrieval_record_k"], 20)
        self.assertTrue(manifest["vector_index"]["copied"])

    def test_run_evaluation_saves_query_snapshot_before_query_evaluation(self) -> None:
        class FakeClient:
            def __init__(self, **_kwargs) -> None:
                pass

        class FakeSystem:
            def __init__(self, **kwargs) -> None:
                self.config = kwargs["config"]
                self.ingested_record_ids: list[str] = []
                self.prompt_registry = PromptRegistry.from_mapping({})
                vector_index_path = Path(self.config.vector_index_path)
                vector_index_path.mkdir(parents=True, exist_ok=True)
                (vector_index_path / "manifest.json").write_text('{"item_count": 1}', encoding="utf-8")

            def ingest_record(self, record: DialogueRecord) -> DialogueRecord:
                self.ingested_record_ids.append(record.record_id)
                return record

            def save(self, path):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "graph": {"records": [{"record_id": record_id} for record_id in self.ingested_record_ids]},
                            "config": self.config.to_dict(),
                            "task_chains": {"tasks": []},
                        }
                    ),
                    encoding="utf-8",
                )
                return target

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_path = base / "dataset.json"
            output_dir = base / "output"
            log_dir = base / "logs"
            dataset_path.write_text(
                json.dumps(
                    {
                        "_metadata": {"person_name": "unit"},
                        "dialogues": [
                            {
                                "session_identifier": "session-1",
                                "session_uuid": "s1",
                                "current_time": "2026-05-29",
                                "dialogue_turns": [
                                    {"speaker": "User", "content": "alpha setup"},
                                    {"speaker": "Assistant", "content": "alpha reply"},
                                    {
                                        "speaker": "User",
                                        "content": "What should I do next?",
                                        "is_query": True,
                                        "query_id": "Q/0001",
                                    },
                                    {
                                        "speaker": "Assistant",
                                        "content": "Use alpha.",
                                        "memory_session_uuids": ["s1"],
                                        "memory_used": [{"session_uuid": "s1", "content": "alpha setup"}],
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_evaluate_query_top_session(*, system, example, **_kwargs):
                snapshot_dir = output_dir / "query_snapshots" / "Q_0001"
                state_path = snapshot_dir / "memory_state.json"
                self.assertTrue(state_path.exists())
                self.assertTrue((snapshot_dir / "vector_index" / "manifest.json").exists())
                self.assertTrue((snapshot_dir / "manifest.json").exists())
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual([item["record_id"] for item in state["graph"]["records"]], ["rec_s1_0001"])
                self.assertEqual(state["config"]["vector_index_path"], str(snapshot_dir / "vector_index"))
                return {
                    "query_id": example.query_id,
                    "question": example.question,
                    "gold_session_uuids": example.gold_session_uuids,
                    "retrieved_session_uuids": ["s1"],
                    "ranked_session_uuids": ["s1"],
                    "ranked_sessions": [],
                    "retrieval_metrics": {"recall_any@1": 1.0, "recall_all@1": 1.0, "ndcg@1": 1.0},
                    "qa_score": None,
                    "qa_reason": "",
                    "retrieval_result": {"query_id": example.query_id, "question": example.question, "ranked_items": []},
                }

            args = parse_args(
                [
                    "--dataset",
                    str(dataset_path),
                    "--api-key",
                    "test-key",
                    "--run-name",
                    "run",
                    "--output-dir",
                    str(output_dir),
                    "--log-dir",
                    str(log_dir),
                    "--session-ks",
                    "1",
                    "--retrieval-record-k",
                    "1",
                    "--max-queries",
                    "1",
                    "--vector-index-backend",
                    "numpy",
                ]
            )

            with patch("tcmem.evals.realmem_top_session.OpenAICompatibleLLMClient", FakeClient), patch(
                "tcmem.evals.realmem_top_session.MemorySystem",
                FakeSystem,
            ), patch("tcmem.evals.realmem_top_session.evaluate_query_top_session", fake_evaluate_query_top_session):
                run_evaluation(args)

            metrics = json.loads((output_dir / "metrics_results.json").read_text(encoding="utf-8"))
            retrieval_results = json.loads((output_dir / "retrieval_results.json").read_text(encoding="utf-8"))

        self.assertEqual(
            metrics["detailed_results"][0]["memory_state_path"],
            str(output_dir / "query_snapshots" / "Q_0001" / "memory_state.json"),
        )
        self.assertEqual(
            retrieval_results["Q/0001"]["memory_state_path"],
            str(output_dir / "query_snapshots" / "Q_0001" / "memory_state.json"),
        )
        self.assertEqual(
            retrieval_results["Q/0001"]["vector_index_path"],
            str(output_dir / "query_snapshots" / "Q_0001" / "vector_index"),
        )

    def test_should_save_record_state_uses_positive_interval(self) -> None:
        self.assertFalse(should_save_record_state(9, 10))
        self.assertTrue(should_save_record_state(10, 10))
        self.assertFalse(should_save_record_state(10, 0))

    def test_qa_judge_retries_json_decode_error_and_logs_raw_response(self) -> None:
        class FlakyJudgeClient:
            json_max_attempts = 2
            json_retry_delay = 0.0

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, *_args, **_kwargs) -> str:
                self.calls += 1
                if self.calls == 1:
                    return '{"score": 3, "reason": "'
                return json.dumps({"score": 3, "reason": "complete"})

        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            client = FlakyJudgeClient()

            result = judge_qa_score(
                client,
                question="question",
                gold_memory_text="memory",
                reference_answer="reference",
                candidate_answer="candidate",
                log_store=log_store,
                query_id="Q-0001",
            )

            errors = (Path(tmpdir) / "run" / "llm_errors.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(result["score"], 3)
        self.assertEqual(client.calls, 2)
        payload = json.loads(errors[0])["payload"]
        self.assertEqual(payload["stage"], "qa_judge")
        self.assertEqual(payload["query_id"], "Q-0001")


if __name__ == "__main__":
    unittest.main()
