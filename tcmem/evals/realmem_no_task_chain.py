from __future__ import annotations

import argparse
import json
from typing import Any

from ..serialization import to_primitive
from . import realmem_top_session
from .realmem_top_session import EvaluationOptions


NO_TASK_CHAIN_ABLATION_DESIGN = {
    "name": "online_no_task_chain_with_entity_graph",
    "online_order": [
        "parse_dataset",
        "build_query_examples_and_pair_records",
        "query_before_ingest",
        "ingest_current_record_after_query",
    ],
    "ingest_flow": [
        "llm_entity_extraction",
        "entity_graph_add_record",
        "vector_index_sync",
        "bm25_index_sync",
    ],
    "retrieval_flow": [
        "no_query_task_routing",
        "no_path_a",
        "vector_bm25_graph_walk",
        "path_b_score_only",
    ],
    "llm_usage": {
        "ingest": ["entity_extraction"],
        "retrieval": [],
    },
    "disabled_components": [
        "record_task_routing",
        "task_chain_creation",
        "task_chain_node_writes",
        "task_metadata_refresh",
        "query_task_routing",
        "path_a_chain_retrieval",
        "path_b_chain_context_penalty",
    ],
}


def evaluation_options_from_args(_args: argparse.Namespace) -> EvaluationOptions:
    return EvaluationOptions(
        mode="tcmem_no_task_chain_online",
        task_chain_enabled=False,
        ablation_design=NO_TASK_CHAIN_ABLATION_DESIGN,
        run_name_prefix="tcmem_realmem_no_task_chain",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return realmem_top_session.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    return realmem_top_session.run_evaluation(args, options=evaluation_options_from_args(args))


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
