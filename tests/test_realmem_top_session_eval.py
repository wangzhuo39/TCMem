import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from tcmem.config import TCMemConfig
from tcmem.evals.realmem_top_session import (
    PairRecord,
    QueryExample,
    build_session_task_chain_trace,
    build_progress_payload,
    compute_retrieval_metrics,
    evaluate_query_top_session,
    extract_query_examples,
    generate_answer,
    judge_memory_metrics,
    judge_qa_score,
    log_query_result,
    parse_args,
    ranked_sessions_from_traces,
    realmem_official_results_from_detailed,
    resolve_runtime_config,
    load_resume_query_results,
    resolve_resume_results_dir,
    validate_resume_prefix,
    run_evaluation,
    save_latest_state,
    save_query_snapshot,
    should_print_progress,
    should_save_record_state,
    summarize_metrics,
)
from tcmem.logging_utils import ModuleLogStore
from tcmem.models import DialogueRecord, RetrievalResult, SearchHit, TaskBranch, TaskChain, TaskChainNode, TaskStatus
from tcmem.prompts import PromptRegistry


class RealMemTopSessionEvalTest(unittest.TestCase):
    def test_resume_query_results_are_loaded_from_append_only_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            log_dir = Path(tmpdir) / "logs" / "run"
            results.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            (results / "memory_state_latest.json").write_text('{"graph": {"records": []}}', encoding="utf-8")
            (results / "manifest.json").write_text(json.dumps({"log_dir": str(log_dir)}), encoding="utf-8")
            (log_dir / "query_results.jsonl").write_text(
                json.dumps({"event": "query_completed", "payload": {"query_id": "Q-1", "value": 1}}) + "\n"
                + json.dumps({"event": "query_completed", "payload": {"query_id": "Q-1", "value": 2}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(load_resume_query_results(results), [{"query_id": "Q-1", "value": 2}])
            self.assertEqual(resolve_resume_results_dir(results)[1], results / "memory_state_latest.json")

    def test_validate_resume_prefix_rejects_non_prefix_state(self) -> None:
        record_a = DialogueRecord(
            record_id="r-a", session_identifier="s", session_uuid="s", current_time="", record_time="",
            user_content="a", assistant_content="b",
        )
        record_b = DialogueRecord(
            record_id="r-b", session_identifier="s", session_uuid="s", current_time="", record_time="",
            user_content="c", assistant_content="d",
        )
        pair_a = PairRecord(0, 0, 0, 1, record_a)
        pair_b = PairRecord(0, 1, 2, 3, record_b)
        with self.assertRaises(ValueError):
            validate_resume_prefix(
                pair_records=[pair_a, pair_b],
                state={"graph": {"records": [{"record_id": "r-b"}]}},
                completed_results=[],
            )

    def test_extract_query_examples_uses_immediate_assistant_memory_used(self) -> None:
        dataset = {
            "dialogues": [
                {
                    "session_identifier": "session-a",
                    "session_uuid": "session-a-uuid",
                    "current_time": "2026-01-01",
                    "dialogue_turns": [
                        {"speaker": "User", "content": "context"},
                        {"speaker": "Assistant", "content": "old answer", "memory_used": []},
                        {"speaker": "User", "content": "question", "is_query": True, "query_id": "Q-1"},
                        {
                            "speaker": "Assistant",
                            "content": "answer",
                            "memory_session_uuids": ["wrong-legacy-value"],
                            "memory_used": [
                                {"session_uuid": "gold-a", "content": "memory a"},
                                {"session_uuid": "gold-a", "content": "memory a duplicate"},
                            ],
                        },
                    ],
                }
            ]
        }

        examples = extract_query_examples(dataset)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].query_id, "Q-1")
        self.assertEqual(examples[0].reference_answer, "answer")
        self.assertEqual(examples[0].gold_session_uuids, ["gold-a"])
        self.assertEqual([item["session_uuid"] for item in examples[0].memory_used], ["gold-a", "gold-a"])

    def test_session_task_chain_trace_exposes_gold_retrieval_and_hierarchy(self) -> None:
        root = TaskChain(
            task_id="task-root",
            task_description="Plan the trip",
            canonical_description="Plan the trip",
            owner_id="test",
            branches={"main": TaskBranch(branch_id="main", task_id="task-root", branch_goal="choose destination")},
            nodes={
                "node-r1": TaskChainNode(
                    node_id="node-r1",
                    task_id="task-root",
                    user_content="trip context",
                    branch_id="main",
                    source_record_id="record-1",
                )
            },
            record_ids=["record-1"],
        )
        child = TaskChain(
            task_id="task-child",
            task_description="Book transport",
            canonical_description="Book transport",
            owner_id="test",
            parent_task_id="task-root",
            parent_branch_id="main",
            branches={"main": TaskBranch(branch_id="main", task_id="task-child", branch_goal="compare trains")},
            nodes={
                "node-r2": TaskChainNode(
                    node_id="node-r2",
                    task_id="task-child",
                    user_content="transport context",
                    branch_id="main",
                    source_record_id="record-2",
                )
            },
            record_ids=["record-2"],
        )
        system = type("FakeSystem", (), {})()
        system.graph = type("FakeGraph", (), {})()
        system.graph.records = {
            "record-1": DialogueRecord("record-1", "session-a", "session-a", "2026-01-01", user_content="a"),
            "record-2": DialogueRecord("record-2", "session-b", "session-b", "2026-01-01", user_content="b"),
            "record-3": DialogueRecord("record-3", "session-c", "session-c", "2026-01-01", user_content="c"),
        }
        system.task_manager = type("FakeTaskManager", (), {})()
        system.task_manager.tasks = {"task-root": root, "task-child": child}

        trace = build_session_task_chain_trace(
            system=system,
            gold_session_uuids=["session-a", "session-b", "session-c"],
            memory_used=[{"session_uuid": "session-a", "content": "gold a"}],
            ranked_sessions=[
                {"session_uuid": "session-b", "score": 0.8, "record_ids": ["record-2"], "record_count": 1},
            ],
            routed_task_ids=["task-root"],
            expanded_task_ids=["task-root", "task-child"],
            traces=[
                {
                    "source_session_uuid": "session-b",
                    "source_record_id": "record-2",
                    "task_id": "task-child",
                    "chain_node_id": "node-r2",
                    "route_role": "expanded",
                    "route_relation": "parent",
                    "route_depth": 1,
                    "reason": "path_a",
                }
            ],
        )

        by_session = {item["session_uuid"]: item for item in trace["sessions"]}
        self.assertEqual(by_session["session-a"]["chain_root_task_ids"], ["task-root"])
        self.assertEqual(by_session["session-b"]["chain_root_task_ids"], ["task-root"])
        self.assertFalse(by_session["session-c"]["in_any_task_chain"])
        self.assertEqual(by_session["session-b"]["task_chain_memberships"][0]["route_evidence"]["route_roles"], ["expanded"])
        self.assertFalse(by_session["session-b"]["task_chain_memberships"][0]["route_evidence"]["task_routed"])
        self.assertTrue(by_session["session-b"]["task_chain_memberships"][0]["route_evidence"]["task_expanded"])
        self.assertEqual(trace["summary"]["gold_sessions_without_task_chain"], ["session-c"])
        self.assertFalse(trace["summary"]["gold_sessions_share_one_root_chain"])

    def test_evaluate_query_persists_session_task_chain_trace_in_result_and_log(self) -> None:
        root = TaskChain(
            task_id="task-root",
            task_description="root goal",
            owner_id="test",
            branches={"main": TaskBranch(branch_id="main", task_id="task-root")},
            nodes={
                "node-r1": TaskChainNode(
                    node_id="node-r1",
                    task_id="task-root",
                    user_content="context",
                    branch_id="main",
                    source_record_id="record-1",
                )
            },
            record_ids=["record-1"],
        )

        class FakeSystem:
            def __init__(self) -> None:
                self.graph = type("FakeGraph", (), {})()
                self.graph.records = {
                    "record-1": DialogueRecord("record-1", "session-a", "session-a", "2026-01-01", user_content="a")
                }
                self.graph.get_record = self.graph.records.get
                self.task_manager = type("FakeTaskManager", (), {})()
                self.task_manager.tasks = {"task-root": root}

            def retrieve(self, _question: str, *, top_k: int) -> RetrievalResult:
                if top_k != 5:
                    raise AssertionError(f"Expected top_k=5, got {top_k}")
                return RetrievalResult(
                    query="question",
                    subqueries=[],
                    routed_task_ids=["task-root"],
                    expanded_task_ids=["task-root"],
                    expansion_edges=[],
                    hits=[
                        SearchHit(
                            item_id="record-1",
                            item_kind="record",
                            score=0.9,
                            task_id="task-root",
                            source_record_id="record-1",
                            chain_node_id="node-r1",
                            route_role="primary",
                            route_relation="primary",
                        )
                    ],
                )

        example = QueryExample(
            query_id="Q-1",
            session_identifier="session-q",
            session_uuid="session-q",
            current_time="2026-01-01",
            turn_index=0,
            question="question",
            reference_answer="answer",
            gold_session_uuids=["session-a"],
            gold_memory_text="memory",
            memory_used=[{"session_uuid": "session-a", "content": "memory"}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            log_store = ModuleLogStore(base_dir=tmpdir, run_name="run")
            result = evaluate_query_top_session(
                system=FakeSystem(),
                example=example,
                retrieval_record_k=5,
                session_ks=[1],
                session_text_by_uuid={"session-a": "memory"},
                evidence_top_k=1,
                with_qa=False,
                log_store=log_store,
            )
            log_path = Path(tmpdir) / "run" / "session_task_chain.jsonl"
            logged = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(result["gold_memory_used"][0]["session_uuid"], "session-a")
        self.assertEqual(result["retrieval_result"]["session_task_chain_trace"][0]["session_uuid"], "session-a")
        self.assertTrue(result["gold_session_task_chain_summary"]["gold_sessions_share_one_root_chain"])
        self.assertEqual(logged["event"], "query_session_trace")
        self.assertEqual(logged["payload"]["query_id"], "Q-1")

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

    def test_realmem_official_results_export_uses_question_keys_and_chunk_items(self) -> None:
        detailed = [
            {
                "query_id": "Q-0001",
                "question": "What should I do next?",
                "retrieval_result": {
                    "ranked_items": [
                        {
                            "source_record_id": "rec_1",
                            "source_session_identifier": "Session:A",
                            "source_session_uuid": "uuid-a",
                            "content_excerpt": "alpha memory",
                            "score": 0.9,
                        },
                        {
                            "source_record_id": "rec_2",
                            "source_session_identifier": "Session:A",
                            "source_session_uuid": "uuid-a",
                            "content_excerpt": "alpha follow-up memory",
                            "score": 0.85,
                        },
                        {
                            "source_record_id": "rec_3",
                            "source_session_identifier": "Session:B",
                            "source_session_uuid": "uuid-b",
                            "content_excerpt": "beta memory",
                            "score": 0.8,
                        },
                    ]
                },
            }
        ]

        official = realmem_official_results_from_detailed(detailed)

        self.assertEqual(list(official), ["What should I do next?"])
        ranked_items = official["What should I do next?"]["ranked_items"]
        self.assertEqual(ranked_items[0]["res_type"], "chunk")
        self.assertEqual(ranked_items[0]["chunk_id"], "Session:A")
        self.assertEqual(ranked_items[0]["content"], "alpha memory")
        self.assertEqual(ranked_items[0]["rank"], 1)
        self.assertEqual(ranked_items[1]["chunk_id"], "Session:B")
        self.assertEqual(ranked_items[1]["rank"], 3)

    def test_parse_args_uses_tcmem_embedding_default(self) -> None:
        args = parse_args([])

        self.assertEqual(args.embedding_model, TCMemConfig().embedding_model)
        self.assertEqual(args.embedding_model, "BAAI/bge-m3")
        self.assertEqual(args.progress_every_records, 1)

    def test_runtime_config_separates_build_and_eval_models_with_shared_json_client_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "runtime_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "api_key": "json-key",
                        "base_url": "https://example.test/v1",
                        "model": "json-default-model",
                        "timeout": 33,
                    }
                ),
                encoding="utf-8",
            )
            args = parse_args(
                [
                    "--config",
                    str(config_path),
                    "--build-model",
                    "gpt-4o-mini",
                    "--eval-model",
                    "gpt-4o",
                ]
            )

            runtime = resolve_runtime_config(args)

        self.assertEqual(runtime.api_key, "json-key")
        self.assertEqual(runtime.base_url, "https://example.test/v1")
        self.assertEqual(runtime.timeout, 33)
        self.assertEqual(runtime.model, "json-default-model")
        self.assertEqual(runtime.build_model, "gpt-4o-mini")
        self.assertEqual(runtime.eval_model, "gpt-4o")

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

    def test_run_evaluation_mirrors_progress_into_output_dir(self) -> None:
        class FakeClient:
            def __init__(self, **_kwargs) -> None:
                pass

        class FakeSystem:
            def __init__(self, **kwargs) -> None:
                self.config = kwargs["config"]
                self.prompt_registry = PromptRegistry.from_mapping({})

            def ingest_record(self, record: DialogueRecord) -> DialogueRecord:
                return record

            def save(self, path):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(
                        {
                            "graph": {"records": []},
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
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
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
                    "--vector-index-backend",
                    "numpy",
                ]
            )

            with patch("tcmem.evals.realmem_top_session.OpenAICompatibleLLMClient", FakeClient), patch(
                "tcmem.evals.realmem_top_session.MemorySystem",
                FakeSystem,
            ):
                run_evaluation(args)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            progress_lines = (output_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            latest = json.loads((output_dir / "progress_latest.json").read_text(encoding="utf-8"))
            official = json.loads((output_dir / "realmem_official_retrieval_results.json").read_text(encoding="utf-8"))
            trace = json.loads((output_dir / "session_task_chain_trace.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["artifacts"]["progress"], str(output_dir / "progress.jsonl"))
        self.assertEqual(manifest["artifacts"]["progress_latest"], str(output_dir / "progress_latest.json"))
        self.assertEqual(
            manifest["artifacts"]["realmem_official_retrieval"],
            str(output_dir / "realmem_official_retrieval_results.json"),
        )
        self.assertGreaterEqual(len(progress_lines), 3)
        self.assertEqual(json.loads(progress_lines[0])["event"], "run_started")
        self.assertEqual(latest["event"], "run_finished")
        self.assertEqual(latest["payload"]["record_progress"], 1.0)
        self.assertEqual(official, {})
        self.assertEqual(trace, {})
        self.assertEqual(
            manifest["artifacts"]["session_task_chain_trace"],
            str(output_dir / "session_task_chain_trace.json"),
        )

    def test_run_evaluation_uses_build_model_for_memory_and_eval_model_for_qa(self) -> None:
        class FakeClient:
            def __init__(self, **kwargs) -> None:
                self.model = kwargs["model"]

        class FakeSystem:
            def __init__(self, **kwargs) -> None:
                self.config = kwargs["config"]
                self.llm_client = kwargs["llm_client"]
                self.prompt_registry = PromptRegistry.from_mapping({})
                self.ingested_record_ids: list[str] = []
                self.assert_build_client()

            def assert_build_client(self) -> None:
                if self.llm_client.model != "gpt-4o-mini":
                    raise AssertionError(f"Expected build client model gpt-4o-mini, got {self.llm_client.model}")

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
            config_path = base / "runtime_config.json"
            dataset_path = base / "dataset.json"
            output_dir = base / "output"
            log_dir = base / "logs"
            config_path.write_text(
                json.dumps(
                    {
                        "api_key": "json-key",
                        "base_url": "https://example.test/v1",
                        "model": "json-default-model",
                        "timeout": 33,
                    }
                ),
                encoding="utf-8",
            )
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
                                        "query_id": "Q-0001",
                                    },
                                    {
                                        "speaker": "Assistant",
                                        "content": "Use alpha.",
                                        "memory_session_uuids": ["s1"],
                                    },
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            def fake_evaluate_query_top_session(*, client, qa_model_name, example, **_kwargs):
                self.assertEqual(client.model, "gpt-4o")
                self.assertEqual(qa_model_name, "gpt-4o")
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
                    "--config",
                    str(config_path),
                    "--build-model",
                    "gpt-4o-mini",
                    "--eval-model",
                    "gpt-4o",
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

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["config"]["model"], "json-default-model")
        self.assertEqual(manifest["config"]["build_model"], "gpt-4o-mini")
        self.assertEqual(manifest["config"]["eval_model"], "gpt-4o")
        self.assertEqual(manifest["config"]["tcmem_config"]["llm_model"], "gpt-4o-mini")

    def test_should_save_record_state_uses_positive_interval(self) -> None:
        self.assertFalse(should_save_record_state(9, 10))
        self.assertTrue(should_save_record_state(10, 10))
        self.assertFalse(should_save_record_state(10, 0))

    def test_should_print_progress_includes_first_last_and_interval(self) -> None:
        self.assertTrue(should_print_progress(1, 100, 10))
        self.assertFalse(should_print_progress(9, 100, 10))
        self.assertTrue(should_print_progress(10, 100, 10))
        self.assertTrue(should_print_progress(100, 100, 10))
        self.assertFalse(should_print_progress(9, 100, 0))

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

    def test_memory_judge_uses_registry_prompt_and_parses_mem_metrics(self) -> None:
        class CapturingClient:
            json_max_attempts = 1
            json_retry_delay = 0.0

            def __init__(self) -> None:
                self.prompt = ""
                self.system_prompt = ""

            def generate(self, prompt: str, **kwargs) -> str:
                self.prompt = prompt
                self.system_prompt = kwargs["system_prompt"]
                return json.dumps(
                    {
                        "Mem_recall": 0.5,
                        "Mem_helpful_score": 2,
                        "Mem_hits": ["gold memory"],
                        "Mem_helpful_reason": "retrieved memory helps",
                    }
                )

        registry = PromptRegistry.from_mapping(
            {
                "realmem_memory_judge": {
                    "system": "CUSTOM MEM SYSTEM",
                    "user": "Q={{question}}\nGT={{groundtruth_memory}}\nRET={{retrieved_memory}}",
                }
            }
        )
        client = CapturingClient()

        result = judge_memory_metrics(
            client,
            question="question",
            groundtruth_memory="gold memory",
            retrieved_memory="retrieved memory",
            prompt_registry=registry,
            query_id="Q-0001",
        )

        self.assertEqual(client.system_prompt, "CUSTOM MEM SYSTEM")
        self.assertIn("Q=question", client.prompt)
        self.assertEqual(result["Mem_recall"], 0.5)
        self.assertEqual(result["Mem_helpful_score"], 2)
        self.assertEqual(result["Mem_hits"], ["gold memory"])

    def test_summarize_metrics_includes_memory_recall_and_helpfulness(self) -> None:
        summary = summarize_metrics(
            [
                {
                    "retrieval_metrics": {"recall_any@1": 1.0, "recall_all@1": 1.0, "ndcg@1": 1.0},
                    "qa_score": 3,
                    "Mem_recall": 1.0,
                    "Mem_helpful_score": 2,
                },
                {
                    "retrieval_metrics": {"recall_any@1": 0.0, "recall_all@1": 0.0, "ndcg@1": 0.0},
                    "qa_score": None,
                    "Mem_recall": 0.5,
                    "Mem_helpful_score": 1,
                },
                {
                    "retrieval_metrics": {"recall_any@1": 0.0, "recall_all@1": 0.0, "ndcg@1": 0.0},
                    "qa_score": None,
                    "mem_error": "judge failed",
                },
            ],
            [1],
        )

        self.assertEqual(summary["average_mem_recall"], 0.75)
        self.assertEqual(summary["average_mem_helpful_score"], 1.5)
        self.assertEqual(summary["mem_helpful_score_distribution"], {"0": 0, "1": 1, "2": 1})
        self.assertEqual(summary["mem_failed_count"], 1)


if __name__ == "__main__":
    unittest.main()
