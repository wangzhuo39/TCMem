import unittest

from tcmem.config import TCMemConfig
from tcmem.evals.realmem_top_session import (
    compute_retrieval_metrics,
    parse_args,
    ranked_sessions_from_traces,
)


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


if __name__ == "__main__":
    unittest.main()
