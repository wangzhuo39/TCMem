import json
import tempfile
import unittest
from pathlib import Path

from tcmem.evals.realmem_mem_metrics_from_logs import (
    build_mem_eval_prompt,
    build_retrieved_memory,
    load_logged_results,
    summarize_mem_metrics,
)


class RealMemMemMetricsFromLogsTest(unittest.TestCase):
    def test_load_logged_results_merges_query_results_and_generation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            query_payload = {
                "query_id": "Q-0001",
                "question": "question",
                "ranked_session_uuids": ["s1"],
            }
            log_entry = {"module": "query_results", "event": "query_completed", "payload": query_payload}
            (run_dir / "query_results.jsonl").write_text(json.dumps(log_entry) + "\n", encoding="utf-8")
            (run_dir / "generation_results.json").write_text(
                json.dumps({"Q-0001": {"generated_answer": "answer", "evidence_used": "evidence"}}),
                encoding="utf-8",
            )

            loaded = load_logged_results(run_dir)

        self.assertEqual(loaded["Q-0001"]["question"], "question")
        self.assertEqual(loaded["Q-0001"]["generation_result"]["generated_answer"], "answer")

    def test_build_retrieved_memory_uses_ranked_sessions_top_k_before_evidence_text(self) -> None:
        result = {
            "ranked_sessions": [
                {"session_uuid": "s1"},
                {"session_uuid": "s2"},
            ],
            "generation_result": {
                "evidence_used": "old evidence that should not be used when ranked_sessions exist",
            },
        }
        session_text_by_uuid = {"s1": "User: first", "s2": "User: second"}

        evidence = build_retrieved_memory(result, session_text_by_uuid=session_text_by_uuid, top_k=1)

        self.assertIn("session_uuid=s1", evidence)
        self.assertIn("User: first", evidence)
        self.assertNotIn("session_uuid=s2", evidence)
        self.assertNotIn("old evidence", evidence)

    def test_summarize_mem_metrics_averages_valid_scores(self) -> None:
        summary = summarize_mem_metrics(
            [
                {"Mem_recall": 1.0, "Mem_helpful_score": 2},
                {"Mem_recall": 0.5, "Mem_helpful_score": 1},
                {"error": "judge failed"},
            ]
        )

        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["evaluated_query_count"], 2)
        self.assertEqual(summary["failed_query_count"], 1)
        self.assertEqual(summary["average_mem_recall"], 0.75)
        self.assertEqual(summary["average_mem_helpful_score"], 1.5)
        self.assertEqual(summary["mem_helpful_score_distribution"], {"0": 0, "1": 1, "2": 1})

    def test_build_mem_eval_prompt_preserves_json_output_schema(self) -> None:
        prompt = build_mem_eval_prompt(
            question="question",
            groundtruth_memory="gold",
            retrieved_memory="retrieved",
        )

        self.assertIn('"Mem_recall": float', prompt)
        self.assertIn("<question>: question", prompt)
        self.assertIn("<groundtruth_memory>: gold", prompt)


if __name__ == "__main__":
    unittest.main()
