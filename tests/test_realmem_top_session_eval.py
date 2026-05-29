import unittest

from tcmem.config import TCMemConfig
from tcmem.evals.realmem_top_session import (
    PairRecord,
    build_progress_payload,
    compute_retrieval_metrics,
    parse_args,
    ranked_sessions_from_traces,
)
from tcmem.models import DialogueRecord


class RealMemTopSessionEvalTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
