import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tcmem import IntentUnderstanding, QueryRouteDecision
from tcmem.config import TCMemConfig
from tcmem.evals import realmem_retrieval_sweep as sweep


class RealMemRetrievalSweepTest(unittest.TestCase):
    EXPECTED_SWEEP_FIELDS = {
        "path_a_weight",
        "path_b_weight",
        "path_a_semantic_weight",
        "path_a_status_weight",
        "path_a_chain_weight",
        "path_b_semantic_weight",
        "path_b_bm25_weight",
        "path_b_graph_weight",
        "graph_seed_limit",
        "graph_bm25_seed_limit",
        "graph_vector_seed_weight",
        "graph_bm25_seed_weight",
        "graph_walk_depth",
        "routed_task_score",
        "unrouted_task_score",
        "active_status_score",
        "branched_status_score",
        "deprecated_status_score",
        "superseded_status_score",
        "default_status_score",
        "active_penalty",
        "branched_penalty",
        "deprecated_penalty",
        "superseded_penalty",
        "task_metadata_refresh_interval",
        "task_router_entity_limit",
        "query_router_candidate_count",
    }

    def test_parse_args_uses_online_exhaustive_defaults(self) -> None:
        args = sweep.parse_args([])

        self.assertEqual(args.mode, "online")
        self.assertEqual(args.session_ks, "5,10,20,30")
        self.assertEqual(args.retrieval_record_k, 200)
        self.assertEqual(args.evidence_top_k, 20)
        self.assertEqual(args.state_save_every_records, 5)
        self.assertEqual(args.objective, "recall_all@20")
        self.assertEqual(args.ablation, "none")
        self.assertFalse(args.baseline_only)
        self.assertTrue(args.include_baseline)
        self.assertTrue(args.keep_trial_states)
        self.assertEqual(args.preset, "exhaustive")
        self.assertFalse(hasattr(args, "base_config"))

    def test_parse_args_rejects_base_config(self) -> None:
        with self.assertRaises(SystemExit):
            sweep.parse_args(["--base-config", "x.json"])

    def test_sweep_field_coverage_matches_retrieval_and_online_state_config(self) -> None:
        self.assertEqual(sweep.SWEEP_FIELD_NAMES, self.EXPECTED_SWEEP_FIELDS)

    def test_expand_trial_overrides_supports_cartesian_grid(self) -> None:
        trials = sweep.expand_trial_overrides(
            {
                "path-a-weight": [0.4, 0.6],
                "graph_seed_limit": [12, 24],
                "graph_walk_depth": 2,
            }
        )

        self.assertEqual(
            trials,
            [
                {"path_a_weight": 0.4, "graph_seed_limit": 12, "graph_walk_depth": 2},
                {"path_a_weight": 0.4, "graph_seed_limit": 24, "graph_walk_depth": 2},
                {"path_a_weight": 0.6, "graph_seed_limit": 12, "graph_walk_depth": 2},
                {"path_a_weight": 0.6, "graph_seed_limit": 24, "graph_walk_depth": 2},
            ],
        )

    def test_expand_trial_overrides_accepts_explicit_trials(self) -> None:
        trials = sweep.expand_trial_overrides(
            {
                "trials": [
                    {"path-a-weight": 0.4, "path-b-weight": 0.6},
                    {"graph-seed-limit": 24},
                ]
            }
        )

        self.assertEqual(trials, [{"path_a_weight": 0.4, "path_b_weight": 0.6}, {"graph_seed_limit": 24}])

    def test_config_from_overrides_rejects_unknown_parameter(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown retrieval config parameter"):
            sweep.tcmem_config_from_overrides({"not_a_real_parameter": 1}, base=TCMemConfig())

    def test_preset_trial_counts_are_stable(self) -> None:
        self.assertEqual(len(sweep.preset_trial_overrides("small")), 32)
        self.assertEqual(len(sweep.preset_trial_overrides("medium")), 144)
        self.assertEqual(len(sweep.preset_trial_overrides("large")), 432)

    def test_exhaustive_preset_starts_with_baseline_and_covers_every_field(self) -> None:
        trials = sweep.preset_trial_overrides("exhaustive")
        covered = {key for trial in trials for key in trial}

        self.assertEqual(trials[0], {})
        self.assertGreater(len(trials), len(sweep.preset_trial_overrides("large")))
        self.assertEqual(covered, self.EXPECTED_SWEEP_FIELDS)
        self.assertNotIn("recall_frac", json.dumps(trials))

    def test_checkpoint_writes_partial_artifacts_and_best_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            manifest = {"run_name": "sweep", "config": {"session_ks": [10, 20]}}
            trials = [
                {
                    "trial_id": 1,
                    "objective": "recall_all@20",
                    "objective_value": 0.5,
                    "summary": {"query_count": 2, "recall_all@10": 0.5, "recall_all@20": 0.5},
                    "retrieval_config": {"path_a_weight": 0.8, "path_b_weight": 0.2},
                    "overrides": {"path_a_weight": 0.8},
                },
                {
                    "trial_id": 2,
                    "objective": "recall_all@20",
                    "objective_value": 1.0,
                    "summary": {"query_count": 2, "recall_all@10": 1.0, "recall_all@20": 1.0},
                    "retrieval_config": {"path_a_weight": 0.4, "path_b_weight": 0.6},
                    "overrides": {"path_a_weight": 0.4},
                },
            ]

            result = sweep.write_sweep_checkpoint(
                output_dir=output_dir,
                manifest=manifest,
                trials=trials,
                session_ks=[10, 20],
                completed_trials=2,
                total_trials=2,
                status="running",
            )

            self.assertEqual(result["best_trial"]["trial_id"], 2)
            self.assertTrue((output_dir / "sweep_results_partial.json").exists())
            self.assertTrue((output_dir / "sweep_results_partial.csv").exists())
            best_config = json.loads((output_dir / "best_retrieval_config_partial.json").read_text(encoding="utf-8"))
            self.assertEqual(best_config["path_b_weight"], 0.6)

    def test_rank_trials_prefers_higher_recall_all_tiebreakers_before_lower_trial_id(self) -> None:
        ranked = sweep.rank_trials(
            [
                {
                    "trial_id": 1,
                    "objective": "recall_all@20",
                    "objective_value": 0.8,
                    "summary": {"recall_all@30": 0.2, "recall_all@10": 1.0, "recall_all@5": 1.0, "ndcg@30": 1.0},
                },
                {
                    "trial_id": 2,
                    "objective": "recall_all@20",
                    "objective_value": 0.8,
                    "summary": {"recall_all@30": 0.9, "recall_all@10": 0.0, "recall_all@5": 0.0, "ndcg@30": 0.0},
                },
                {
                    "trial_id": 3,
                    "objective": "recall_all@20",
                    "objective_value": 0.8,
                    "summary": {"recall_all@30": 0.9, "recall_all@10": 0.0, "recall_all@5": 0.0, "ndcg@30": 0.0},
                },
                {"trial_id": 4, "objective": "recall_all@20", "objective_value": 0.7, "summary": {}},
            ],
            session_ks=[5, 10, 20, 30],
            objective="recall_all@20",
        )

        self.assertEqual([trial["trial_id"] for trial in ranked], [2, 3, 1, 4])
        self.assertEqual([trial["rank"] for trial in ranked], [1, 2, 3, 4])

    def test_write_trials_csv_excludes_recall_frac_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trials.csv"
            sweep.write_trials_csv(
                path,
                [
                    {
                        "rank": 1,
                        "trial_id": 1,
                        "objective": "recall_all@1",
                        "objective_value": 1.0,
                        "summary": {
                            "query_count": 1,
                            "recall_any@1": 1.0,
                            "recall_all@1": 1.0,
                            "recall_frac@1": 1.0,
                            "ndcg@1": 1.0,
                        },
                        "retrieval_config": {},
                    }
                ],
                session_ks=[1],
            )

            csv_text = path.read_text(encoding="utf-8")

        self.assertIn("recall_any@1", csv_text)
        self.assertIn("recall_all@1", csv_text)
        self.assertIn("ndcg@1", csv_text)
        self.assertNotIn("recall_frac", csv_text)

    def test_parse_args_accepts_runtime_config_options(self) -> None:
        args = sweep.parse_args(["--config", "local.json", "--api-key", "key", "--base-url", "http://localhost", "--model", "mini"])

        self.assertEqual(args.config, "local.json")
        self.assertEqual(args.api_key, "key")
        self.assertEqual(args.base_url, "http://localhost")
        self.assertEqual(args.model, "mini")

    def test_route_cache_requires_llm_when_missing_routes(self) -> None:
        class FakeTaskManager:
            llm_client = None

        class FakeSystem:
            task_manager = FakeTaskManager()

        class Example:
            query_id = "Q-0001"
            question = "question"

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(SystemExit, "Missing LLM client"):
                sweep.load_or_build_route_cache(
                    system=FakeSystem(),
                    examples=[Example()],
                    path=Path(tmpdir) / "routes.json",
                )

    def test_load_route_cache_accepts_legacy_and_rich_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "legacy_routes.json"
            rich_path = Path(tmpdir) / "rich_routes.json"

            legacy_path.write_text(json.dumps({"Q-1": ["task_1"]}), encoding="utf-8")
            rich_path.write_text(
                json.dumps(
                    {
                        "Q-2": {
                            "routed_task_ids": ["task_2"],
                            "query_intent": {
                                "main_intent": "retrieve beta status",
                                "topic_hint": "beta",
                                "key_entities": ["beta"],
                                "is_substantive": True,
                                "reason": "asks for beta status",
                            },
                            "reason": "matched beta task",
                        }
                    }
                ),
                encoding="utf-8",
            )

            legacy = sweep.load_route_cache(legacy_path)
            rich = sweep.load_route_cache(rich_path)

        self.assertEqual(legacy["Q-1"].routed_task_ids, ["task_1"])
        self.assertEqual(legacy["Q-1"].query_intent.reason, "loaded_from_legacy_route_cache")
        self.assertEqual(rich["Q-2"].query_intent.main_intent, "retrieve beta status")
        self.assertEqual(rich["Q-2"].reason, "matched beta task")

    def test_append_sweep_event_writes_jsonl_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            events_path = Path(tmpdir) / "sweep_events.jsonl"

            sweep.append_sweep_event(
                events_path,
                "query_evaluated",
                trial_id=2,
                query_index=3,
                query_count=10,
                query_id="Q-0003",
                elapsed_seconds=1.234,
            )

            payload = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(payload["event"], "query_evaluated")
        self.assertEqual(payload["trial_id"], 2)
        self.assertEqual(payload["query_index"], 3)
        self.assertEqual(payload["query_count"], 10)
        self.assertEqual(payload["query_id"], "Q-0003")
        self.assertEqual(payload["elapsed_seconds"], 1.23)

    def test_evaluate_offline_trial_logs_query_started_before_retrieval_finishes(self) -> None:
        class FakeTaskManager:
            def __init__(self) -> None:
                self.route_for_query = lambda query: QueryRouteDecision(
                    query=query,
                    routed_task_ids=["task-1"],
                    query_intent=IntentUnderstanding(main_intent="cached task"),
                    reason="cached",
                )

        class FakeRetrieval:
            def __init__(self, config: TCMemConfig) -> None:
                self.config = config

        class FakeSystem:
            def __init__(self) -> None:
                self.config = TCMemConfig()
                self.retrieval = FakeRetrieval(self.config)
                self.task_manager = FakeTaskManager()

        examples = [
            SimpleNamespace(
                query_id="Q-0001",
                question="Where did we leave the keys?",
                gold_session_uuids=["session-1"],
            )
        ]
        route_cache = {
            "Q-0001": QueryRouteDecision(
                query="Where did we leave the keys?",
                routed_task_ids=["task-1"],
                query_intent=IntentUnderstanding(main_intent="find keys"),
                reason="cached",
            )
        }
        observed_events: list[dict[str, object]] = []
        original_append = sweep.append_sweep_event
        original_evaluate_query = sweep.evaluate_query

        def fake_append(path: Path, event: str, **payload: object) -> None:
            observed_events.append({"event": event, **payload})

        def fake_evaluate_query(**_: object) -> dict[str, object]:
            started_events = [entry for entry in observed_events if entry["event"] == "query_started"]
            self.assertEqual(len(started_events), 1)
            self.assertEqual(started_events[0]["query_id"], "Q-0001")
            return {
                "query_id": "Q-0001",
                "retrieval_metrics": {
                    "recall_any@10": 1.0,
                    "recall_all@10": 1.0,
                    "ndcg@10": 1.0,
                },
            }

        try:
            sweep.append_sweep_event = fake_append
            sweep.evaluate_query = fake_evaluate_query
            result = sweep.evaluate_offline_trial(
                trial_id=1,
                total_trials=2,
                system=FakeSystem(),
                examples=examples,
                overrides={"path_a_weight": 0.8},
                base_config=TCMemConfig(),
                route_cache=route_cache,
                retrieval_record_k=100,
                session_ks=[10],
                objective="recall_all@10",
                events_path=Path("/tmp/unused.jsonl"),
            )
        finally:
            sweep.append_sweep_event = original_append
            sweep.evaluate_query = original_evaluate_query

        self.assertEqual(result["objective_value"], 1.0)
        self.assertEqual([entry["event"] for entry in observed_events[:3]], ["trial_started", "query_started", "query_evaluated"])
        self.assertNotIn("average_qa_score", result["summary"])
        self.assertNotIn("qa_score_distribution", result["summary"])
        self.assertNotIn("qa_failed_count", result["summary"])

    def test_evaluate_online_trial_evaluates_query_before_ingesting_current_record(self) -> None:
        calls: list[tuple[str, str]] = []
        example = SimpleNamespace(
            query_id="Q-0001",
            question="Where did we leave the keys?",
            gold_session_uuids=["session-1"],
        )
        pair_records = [
            SimpleNamespace(record="record-before", query_example=None),
            SimpleNamespace(record="record-current", query_example=example),
        ]

        class FakeSystem:
            def __init__(self, config: TCMemConfig, **_: object) -> None:
                self.config = config

            def ingest_record(self, record: str) -> None:
                calls.append(("ingest", record))

            def save(self, path: Path) -> Path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                return path

        original_evaluate_query = sweep.evaluate_query

        def fake_evaluate_query(**kwargs: object) -> dict[str, object]:
            self.assertEqual(calls, [("ingest", "record-before")])
            return {
                "query_id": "Q-0001",
                "retrieval_metrics": {
                    "recall_any@1": 1.0,
                    "recall_all@1": 1.0,
                    "ndcg@1": 1.0,
                },
            }

        try:
            sweep.evaluate_query = fake_evaluate_query
            with tempfile.TemporaryDirectory() as tmpdir:
                result = sweep.evaluate_online_trial(
                    trial_id=1,
                    total_trials=1,
                    pair_records=pair_records,
                    examples=[example],
                    overrides={},
                    base_config=TCMemConfig(),
                    runtime=SimpleNamespace(api_key="", base_url="http://localhost", model="mini", timeout=30),
                    retrieval_record_k=100,
                    session_ks=[1],
                    objective="recall_all@1",
                    output_dir=Path(tmpdir),
                    run_name="run",
                    state_save_every_records=1,
                    keep_trial_states=True,
                    memory_system_factory=FakeSystem,
                )
        finally:
            sweep.evaluate_query = original_evaluate_query

        self.assertEqual(calls, [("ingest", "record-before"), ("ingest", "record-current")])
        self.assertEqual(result["evaluation_mode"], "online_incremental_sweep")
        self.assertEqual(result["objective_value"], 1.0)
        self.assertNotIn("average_qa_score", result["summary"])
        self.assertNotIn("qa_score_distribution", result["summary"])
        self.assertNotIn("qa_failed_count", result["summary"])

    def test_include_baseline_then_max_trials_keeps_only_baseline(self) -> None:
        args = sweep.parse_args(["--preset", "small", "--include-baseline", "--max-trials", "1"])
        original_preset = sweep.preset_trial_overrides

        try:
            sweep.preset_trial_overrides = lambda _name: [{"path_a_weight": 0.8}]
            self.assertEqual(sweep.load_grid_spec(args), [{}])
        finally:
            sweep.preset_trial_overrides = original_preset

    def test_online_baseline_only_run_uses_single_empty_trial_and_does_not_load_state(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        pair = SimpleNamespace(record="record-current", query_example=example)
        calls: list[dict[str, object]] = []
        original_extract = sweep.extract_query_examples
        original_build_pairs = sweep.build_pair_records
        original_evaluate_online = sweep.evaluate_online_trial
        original_load = sweep.MemorySystem.__dict__["load"]

        def fake_evaluate_online_trial(**kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": {},
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "online_incremental_sweep",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.build_pair_records = lambda _dataset, _examples: [pair]
            sweep.evaluate_online_trial = fake_evaluate_online_trial
            sweep.MemorySystem.load = classmethod(lambda *_args, **_kwargs: self.fail("online mode must not load state"))
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--output-dir",
                            str(Path(tmpdir) / "out"),
                            "--baseline-only",
                        ]
                    )
                )
        finally:
            sweep.extract_query_examples = original_extract
            sweep.build_pair_records = original_build_pairs
            sweep.evaluate_online_trial = original_evaluate_online
            sweep.MemorySystem.load = original_load

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["overrides"], {})
        self.assertEqual(result["manifest"]["evaluation_mode"], "online_incremental_sweep")
        self.assertTrue(result["manifest"]["safe_for_final_benchmark"])

    def test_offline_run_manifest_marks_leakage_risk_without_real_state_work(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        calls: list[dict[str, object]] = []
        load_calls: list[dict[str, object]] = []

        class FakeTaskManager:
            llm_client = object()

        class FakeSystem:
            config = TCMemConfig()
            task_manager = FakeTaskManager()

        original_extract = sweep.extract_query_examples
        original_load = sweep.MemorySystem.__dict__["load"]
        original_route_cache = sweep.load_or_build_route_cache
        original_evaluate_offline = sweep.evaluate_offline_trial

        def fake_load(cls, path: str, *, config: object = None, llm_client: object = None) -> FakeSystem:
            load_calls.append({"path": path, "config": config, "llm_client": llm_client})
            return FakeSystem()

        def fake_evaluate_offline_trial(**kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": {},
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "offline_retrieval_tuning",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.MemorySystem.load = classmethod(fake_load)
            sweep.load_or_build_route_cache = lambda **_kwargs: {
                "Q-0001": QueryRouteDecision(
                    query="question",
                    routed_task_ids=[],
                    query_intent=IntentUnderstanding(main_intent="question"),
                    reason="cached",
                )
            }
            sweep.evaluate_offline_trial = fake_evaluate_offline_trial
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--mode",
                            "offline",
                            "--dataset",
                            str(dataset_path),
                            "--state-path",
                            str(Path(tmpdir) / "state.json"),
                            "--output-dir",
                            str(Path(tmpdir) / "out"),
                            "--baseline-only",
                        ]
                    )
                )
        finally:
            sweep.extract_query_examples = original_extract
            sweep.MemorySystem.load = original_load
            sweep.load_or_build_route_cache = original_route_cache
            sweep.evaluate_offline_trial = original_evaluate_offline

        self.assertEqual(len(calls), 1)
        self.assertIsNone(load_calls[0]["config"])
        self.assertEqual(result["manifest"]["evaluation_mode"], "offline_retrieval_tuning")
        self.assertFalse(result["manifest"]["safe_for_final_benchmark"])
        self.assertIn("leakage", result["manifest"]["leakage_warning"].lower())

    def test_offline_no_task_chain_ablation_skips_route_cache_and_disables_task_chain(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        calls: list[dict[str, object]] = []

        class FakeTaskManager:
            llm_client = None

        class FakeSystem:
            config = TCMemConfig()
            task_manager = FakeTaskManager()

        original_extract = sweep.extract_query_examples
        original_load = sweep.MemorySystem.__dict__["load"]
        original_route_cache = sweep.load_or_build_route_cache
        original_evaluate_offline = sweep.evaluate_offline_trial

        def fake_evaluate_offline_trial(**kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            base_config = kwargs["base_config"]
            route_cache = kwargs["route_cache"]
            self.assertFalse(base_config.task_chain_enabled)
            self.assertEqual(base_config.path_a_weight, 0.0)
            self.assertEqual(base_config.path_b_weight, 1.0)
            self.assertEqual(route_cache["Q-0001"].routed_task_ids, [])
            self.assertEqual(route_cache["Q-0001"].reason, "task_chain_disabled")
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": base_config.to_dict(),
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "offline_retrieval_tuning",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.MemorySystem.load = classmethod(lambda *_args, **_kwargs: FakeSystem())
            sweep.load_or_build_route_cache = lambda **_kwargs: self.fail("no-task-chain ablation must not build route cache")
            sweep.evaluate_offline_trial = fake_evaluate_offline_trial
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--mode",
                            "offline",
                            "--ablation",
                            "no-task-chain",
                            "--dataset",
                            str(dataset_path),
                            "--state-path",
                            str(Path(tmpdir) / "state.json"),
                            "--output-dir",
                            str(Path(tmpdir) / "out"),
                            "--baseline-only",
                        ]
                    )
                )
        finally:
            sweep.extract_query_examples = original_extract
            sweep.MemorySystem.load = original_load
            sweep.load_or_build_route_cache = original_route_cache
            sweep.evaluate_offline_trial = original_evaluate_offline

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["manifest"]["ablation"], "no-task-chain")

    def test_online_no_task_chain_ablation_passes_disabled_base_config_to_trial(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        pair = SimpleNamespace(record="record-current", query_example=example)
        calls: list[dict[str, object]] = []
        original_extract = sweep.extract_query_examples
        original_build_pairs = sweep.build_pair_records
        original_evaluate_online = sweep.evaluate_online_trial

        def fake_evaluate_online_trial(**kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            base_config = kwargs["base_config"]
            self.assertFalse(base_config.task_chain_enabled)
            self.assertEqual(base_config.path_a_weight, 0.0)
            self.assertEqual(base_config.path_b_weight, 1.0)
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": base_config.to_dict(),
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "online_incremental_sweep",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.build_pair_records = lambda _dataset, _examples: [pair]
            sweep.evaluate_online_trial = fake_evaluate_online_trial
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--output-dir",
                            str(Path(tmpdir) / "out"),
                            "--ablation",
                            "no-task-chain",
                            "--baseline-only",
                        ]
                    )
                )
        finally:
            sweep.extract_query_examples = original_extract
            sweep.build_pair_records = original_build_pairs
            sweep.evaluate_online_trial = original_evaluate_online

        self.assertEqual(len(calls), 1)
        self.assertEqual(result["manifest"]["ablation"], "no-task-chain")
        self.assertTrue(result["manifest"]["safe_for_final_benchmark"])

    def test_final_manifest_lists_all_expected_artifact_paths(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        pair = SimpleNamespace(record="record-current", query_example=example)
        original_extract = sweep.extract_query_examples
        original_build_pairs = sweep.build_pair_records
        original_evaluate_online = sweep.evaluate_online_trial

        def fake_evaluate_online_trial(**kwargs: object) -> dict[str, object]:
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": {},
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "online_incremental_sweep",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.build_pair_records = lambda _dataset, _examples: [pair]
            sweep.evaluate_online_trial = fake_evaluate_online_trial
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                output_dir = Path(tmpdir) / "out"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--output-dir",
                            str(output_dir),
                            "--baseline-only",
                        ]
                    )
                )
                artifacts = result["manifest"]["artifacts"]
                for key in (
                    "sweep_results",
                    "sweep_csv",
                    "best_retrieval_config",
                    "sweep_events",
                    "sweep_manifest",
                    "sweep_results_partial",
                    "sweep_csv_partial",
                    "best_retrieval_config_partial",
                ):
                    self.assertIn(key, artifacts)
                    self.assertTrue(artifacts[key])

                for path in artifacts.values():
                    self.assertTrue(Path(path).exists(), path)
        finally:
            sweep.extract_query_examples = original_extract
            sweep.build_pair_records = original_build_pairs
            sweep.evaluate_online_trial = original_evaluate_online

    def test_checkpoint_disabled_manifest_artifacts_all_exist(self) -> None:
        dataset = {"dialogues": []}
        example = SimpleNamespace(
            query_id="Q-0001",
            question="question",
            gold_session_uuids=["session-1"],
        )
        pair = SimpleNamespace(record="record-current", query_example=example)
        original_extract = sweep.extract_query_examples
        original_build_pairs = sweep.build_pair_records
        original_evaluate_online = sweep.evaluate_online_trial

        def fake_evaluate_online_trial(**kwargs: object) -> dict[str, object]:
            return {
                "trial_id": kwargs["trial_id"],
                "objective": kwargs["objective"],
                "objective_value": 1.0,
                "overrides": kwargs["overrides"],
                "retrieval_config": {},
                "summary": {"query_count": 1, "recall_all@20": 1.0},
                "evaluation_mode": "online_incremental_sweep",
            }

        try:
            sweep.extract_query_examples = lambda _dataset: [example]
            sweep.build_pair_records = lambda _dataset, _examples: [pair]
            sweep.evaluate_online_trial = fake_evaluate_online_trial
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_path = Path(tmpdir) / "dataset.json"
                output_dir = Path(tmpdir) / "out"
                dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
                result = sweep.run_sweep(
                    sweep.parse_args(
                        [
                            "--dataset",
                            str(dataset_path),
                            "--output-dir",
                            str(output_dir),
                            "--baseline-only",
                            "--checkpoint-every",
                            "0",
                        ]
                    )
                )
                artifacts = result["manifest"]["artifacts"]
                for path in artifacts.values():
                    self.assertTrue(Path(path).exists(), path)
        finally:
            sweep.extract_query_examples = original_extract
            sweep.build_pair_records = original_build_pairs
            sweep.evaluate_online_trial = original_evaluate_online


if __name__ == "__main__":
    unittest.main()
