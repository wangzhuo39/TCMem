import json
import tempfile
import unittest
from pathlib import Path

from tcmem.evals.realmem_top_session import compute_retrieval_metrics
from tcmem.evals.realmem_task_chain_ablation_from_logs import (
    build_task_chain_ablation,
    load_ablation_weights,
    rerank_without_task_chain,
)


class RealMemTaskChainAblationFromLogsTest(unittest.TestCase):
    def test_rerank_without_task_chain_drops_path_a_only_and_recomputes_path_b_score(self) -> None:
        logged = {
            "retrieval_result": {
                "ranked_items": [
                    {
                        "source_record_id": "r1",
                        "source_session_uuid": "s1",
                        "reason": "path_a_chain+path_b_vector_bm25_graph",
                        "semantic_score": 0.2,
                        "bm25_score": 0.0,
                        "graph_score": 0.0,
                    },
                    {
                        "source_record_id": "r2",
                        "source_session_uuid": "s2",
                        "reason": "path_b_vector_bm25_graph",
                        "semantic_score": 0.1,
                        "bm25_score": 1.0,
                        "graph_score": 0.0,
                    },
                    {
                        "source_record_id": "r3",
                        "source_session_uuid": "s3",
                        "reason": "path_a_chain",
                        "semantic_score": 1.0,
                        "bm25_score": 1.0,
                        "graph_score": 1.0,
                    },
                ]
            }
        }

        reranked = rerank_without_task_chain(
            logged,
            path_b_semantic_weight=0.5,
            path_b_bm25_weight=0.5,
            path_b_graph_weight=0.0,
        )

        self.assertEqual([item["source_record_id"] for item in reranked], ["r2", "r1"])
        self.assertAlmostEqual(reranked[0]["score"], 0.55)
        self.assertEqual(reranked[0]["reason"], "path_b_vector_bm25_graph_no_task_chain_log_rerank")
        self.assertEqual(reranked[0]["route_score"], 0.0)
        self.assertEqual(reranked[0]["chain_score"], 0.0)

    def test_build_task_chain_ablation_reports_original_ablated_and_delta_metrics(self) -> None:
        logged_results = {
            "Q-0001": {
                "query_id": "Q-0001",
                "question": "question",
                "gold_session_uuids": ["gold"],
                "ranked_session_uuids": ["wrong", "gold"],
                "retrieval_metrics": compute_retrieval_metrics(
                    retrieved_session_uuids=["wrong", "gold"],
                    gold_session_uuids=["gold"],
                    ks=[1, 2],
                ),
                "retrieval_result": {
                    "ranked_items": [
                        {
                            "source_record_id": "r_wrong",
                            "source_session_uuid": "wrong",
                            "reason": "path_a_chain+path_b_vector_bm25_graph",
                            "semantic_score": 0.2,
                            "bm25_score": 0.0,
                            "graph_score": 0.0,
                        },
                        {
                            "source_record_id": "r_gold",
                            "source_session_uuid": "gold",
                            "reason": "path_b_vector_bm25_graph",
                            "semantic_score": 0.1,
                            "bm25_score": 1.0,
                            "graph_score": 0.0,
                        },
                    ]
                },
            }
        }

        result = build_task_chain_ablation(
            logged_results,
            session_ks=[1, 2],
            path_b_semantic_weight=0.5,
            path_b_bm25_weight=0.5,
            path_b_graph_weight=0.0,
            include_items=False,
        )

        self.assertEqual(result["summary"]["query_count"], 1)
        self.assertEqual(result["summary"]["original"]["recall_all@1"], 0.0)
        self.assertEqual(result["summary"]["no_task_chain_log_rerank"]["recall_all@1"], 1.0)
        self.assertEqual(result["summary"]["delta_no_task_chain_minus_original"]["recall_all@1"], 1.0)
        self.assertEqual(result["detailed_results"][0]["ablated_ranked_session_uuids"], ["gold", "wrong"])
        self.assertNotIn("ablated_ranked_items", result["detailed_results"][0])

    def test_load_ablation_weights_reads_manifest_tcmem_config_with_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "config": {
                            "tcmem_config": {
                                "path_b_semantic_weight": 0.3,
                                "path_b_bm25_weight": 0.4,
                                "path_b_graph_weight": 0.5,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            weights = load_ablation_weights(manifest)

        self.assertEqual(weights["path_b_semantic_weight"], 0.3)
        self.assertEqual(weights["path_b_bm25_weight"], 0.4)
        self.assertEqual(weights["path_b_graph_weight"], 0.5)


if __name__ == "__main__":
    unittest.main()
