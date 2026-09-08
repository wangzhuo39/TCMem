from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..config import TCMemConfig
from ..serialization import dump_json, to_primitive
from .realmem_mem_metrics_from_logs import discover_run_files, load_logged_results
from .realmem_top_session import (
    _parse_ks,
    compute_retrieval_metrics,
    ranked_sessions_from_traces,
)


ABLATION_NAME = "no_task_chain_log_rerank"
APPROXIMATION_WARNING = (
    "This is a log-only approximation. It reuses the already logged retrieval "
    "candidates, removes Path A/task-chain fields, and reranks Path B candidates "
    "with the Path B scoring formula. It does not regenerate candidates that "
    "would have appeared in a fresh no-task-chain retrieval run."
)


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl_payloads(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        payload = entry.get("payload", entry) if isinstance(entry, dict) else entry
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} does not contain a JSON object payload")
        payloads.append(payload)
    return payloads


def _merge_dict(target: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value


def _record_key(item: dict[str, Any]) -> str:
    return str(item.get("source_record_id") or item.get("item_id") or "").strip()


def _merge_ranked_item_fields(existing: list[dict[str, Any]], update: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not existing:
        return [dict(item) for item in update if isinstance(item, dict)]
    by_record = {_record_key(item): item for item in existing if _record_key(item)}
    for item in update:
        if not isinstance(item, dict):
            continue
        key = _record_key(item)
        if key in by_record:
            _merge_dict(by_record[key], item)
    return existing


def _merge_retrieval_payload(target: dict[str, Any], retrieval_payload: dict[str, Any]) -> None:
    retrieval_result = dict(target.get("retrieval_result") or {})
    existing_items = [
        dict(item)
        for item in (retrieval_result.get("ranked_items") or retrieval_result.get("hits") or [])
        if isinstance(item, dict)
    ]
    update_items = [
        dict(item)
        for item in (retrieval_payload.get("ranked_items") or retrieval_payload.get("hits") or [])
        if isinstance(item, dict)
    ]
    if update_items:
        retrieval_result["ranked_items"] = _merge_ranked_item_fields(existing_items, update_items)
    for key in (
        "query_id",
        "question",
        "routed_task_ids",
        "expanded_task_ids",
        "expansion_edges",
        "ranked_sessions",
        "ranked_session_uuids",
        "retrieved_session_uuids",
        "gold_session_uuids",
        "gold_memory_used",
        "session_task_chain_trace",
        "gold_session_task_chain_summary",
        "retrieval_record_k",
        "query_intent",
        "query_route_reason",
    ):
        if key in retrieval_payload and retrieval_payload[key] not in (None, ""):
            retrieval_result[key] = retrieval_payload[key]
    target["retrieval_result"] = retrieval_result


def _load_result_artifacts(result_dir: Path | None) -> dict[str, dict[str, Any]]:
    if result_dir is None:
        return {}
    results: dict[str, dict[str, Any]] = {}

    retrieval_data = _read_json(result_dir / "retrieval_results.json")
    if isinstance(retrieval_data, dict):
        for key, value in retrieval_data.items():
            if not isinstance(value, dict):
                continue
            qid = str(value.get("query_id") or key).strip()
            if not qid:
                continue
            target = results.setdefault(qid, {})
            _merge_retrieval_payload(target, value)

    metrics_data = _read_json(result_dir / "metrics_results.json")
    if isinstance(metrics_data, dict):
        for value in metrics_data.get("detailed_results", []) or []:
            if not isinstance(value, dict):
                continue
            qid = str(value.get("query_id") or "").strip()
            if not qid:
                continue
            _merge_dict(results.setdefault(qid, {}), value)
    return results


def load_ablation_inputs(
    run_dir: str | Path | None = None,
    *,
    query_results_file: str | Path | None = None,
    retrieval_file: str | Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    files = discover_run_files(run_dir, query_results_file=query_results_file, retrieval_file=retrieval_file)
    results = _load_result_artifacts(files.result_dir)
    logged_results = load_logged_results(
        files.run_dir,
        query_results_file=files.query_results_file,
        retrieval_file=files.retrieval_file,
    )
    for qid, value in logged_results.items():
        _merge_dict(results.setdefault(qid, {}), value)

    qid_by_question = {
        str(item.get("question") or item.get("query") or "").strip(): qid
        for qid, item in results.items()
        if str(item.get("question") or item.get("query") or "").strip()
    }
    for payload in _iter_jsonl_payloads(files.retrieval_file):
        qid = str(payload.get("query_id") or "").strip()
        if not qid:
            qid = qid_by_question.get(str(payload.get("query") or "").strip(), "")
        if not qid:
            continue
        _merge_retrieval_payload(results.setdefault(qid, {}), payload)

    artifacts = {
        "run_dir": str(files.run_dir),
        "log_dir": str(files.log_dir) if files.log_dir else None,
        "result_dir": str(files.result_dir) if files.result_dir else None,
        "manifest_file": str(files.manifest_file) if files.manifest_file else None,
        "query_results_file": str(files.query_results_file) if files.query_results_file else None,
        "retrieval_file": str(files.retrieval_file) if files.retrieval_file else None,
    }
    return results, artifacts


def load_ablation_weights(manifest_file: str | Path | None) -> dict[str, float]:
    defaults = TCMemConfig()
    weights = {
        "path_b_semantic_weight": defaults.path_b_semantic_weight,
        "path_b_bm25_weight": defaults.path_b_bm25_weight,
        "path_b_graph_weight": defaults.path_b_graph_weight,
    }
    path = Path(manifest_file) if manifest_file else None
    manifest = _read_json(path)
    if not isinstance(manifest, dict):
        return weights
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    tcmem_config = config.get("tcmem_config") if isinstance(config.get("tcmem_config"), dict) else {}
    for key in weights:
        if key in tcmem_config:
            weights[key] = float(tcmem_config[key])
    return weights


def _has_path_b(item: dict[str, Any]) -> bool:
    return "path_b" in str(item.get("reason") or "")


def _float_field(item: dict[str, Any], key: str) -> float:
    try:
        return float(item.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def rerank_without_task_chain(
    logged: dict[str, Any],
    *,
    path_b_semantic_weight: float,
    path_b_bm25_weight: float,
    path_b_graph_weight: float,
) -> list[dict[str, Any]]:
    retrieval_result = logged.get("retrieval_result") if isinstance(logged.get("retrieval_result"), dict) else {}
    ranked_items = retrieval_result.get("ranked_items") or retrieval_result.get("hits") or logged.get("ranked_items") or []
    reranked: list[dict[str, Any]] = []
    for item in ranked_items:
        if not isinstance(item, dict) or not _has_path_b(item):
            continue
        score = (
            path_b_semantic_weight * _float_field(item, "semantic_score")
            + path_b_bm25_weight * _float_field(item, "bm25_score")
            + path_b_graph_weight * _float_field(item, "graph_score")
        )
        ablated = dict(item)
        ablated["score"] = score
        ablated["chain_score"] = 0.0
        ablated["route_score"] = 0.0
        ablated["task_id"] = None
        ablated["chain_node_id"] = None
        ablated["reason"] = "path_b_vector_bm25_graph_no_task_chain_log_rerank"
        reranked.append(ablated)
    return sorted(reranked, key=lambda item: (-float(item.get("score") or 0.0), _record_key(item)))


def _ranked_session_uuids_from_logged(logged: dict[str, Any]) -> list[str]:
    retrieval_result = logged.get("retrieval_result") if isinstance(logged.get("retrieval_result"), dict) else {}
    values = (
        logged.get("ranked_session_uuids")
        or logged.get("retrieved_session_uuids")
        or retrieval_result.get("ranked_session_uuids")
        or retrieval_result.get("retrieved_session_uuids")
        or []
    )
    if values:
        return _unique_strings(values)
    ranked_sessions = logged.get("ranked_sessions") or retrieval_result.get("ranked_sessions") or []
    return _unique_strings([item.get("session_uuid") for item in ranked_sessions if isinstance(item, dict)])


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _gold_session_uuids(logged: dict[str, Any]) -> list[str]:
    retrieval_result = logged.get("retrieval_result") if isinstance(logged.get("retrieval_result"), dict) else {}
    return _unique_strings(logged.get("gold_session_uuids") or retrieval_result.get("gold_session_uuids") or [])


def _summarize_retrieval_metrics(detailed_results: list[dict[str, Any]], *, metrics_key: str, session_ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in session_ks:
        for prefix in ("recall_any", "recall_all", "ndcg"):
            key = f"{prefix}@{k}"
            values = [
                float(item[metrics_key][key])
                for item in detailed_results
                if isinstance(item.get(metrics_key), dict) and key in item[metrics_key]
            ]
            summary[key] = round(sum(values) / len(values), 4) if values else 0.0
    return summary


def _delta_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in left.items():
        if "@" not in key or key not in right:
            continue
        result[key] = round(float(value) - float(right[key]), 4)
    return result


def build_task_chain_ablation(
    logged_results: dict[str, dict[str, Any]],
    *,
    session_ks: list[int],
    path_b_semantic_weight: float,
    path_b_bm25_weight: float,
    path_b_graph_weight: float,
    include_items: bool = False,
    query_ids: list[str] | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    selected = [(qid, logged_results[qid]) for qid in sorted(logged_results)]
    if query_ids:
        wanted = set(query_ids)
        selected = [(qid, item) for qid, item in selected if qid in wanted]
    if max_queries is not None:
        selected = selected[:max_queries]

    detailed_results: list[dict[str, Any]] = []
    for qid, logged in selected:
        gold_session_uuids = _gold_session_uuids(logged)
        original_ranked_session_uuids = _ranked_session_uuids_from_logged(logged)
        original_metrics = compute_retrieval_metrics(
            retrieved_session_uuids=original_ranked_session_uuids,
            gold_session_uuids=gold_session_uuids,
            ks=session_ks,
        )
        ablated_items = rerank_without_task_chain(
            logged,
            path_b_semantic_weight=path_b_semantic_weight,
            path_b_bm25_weight=path_b_bm25_weight,
            path_b_graph_weight=path_b_graph_weight,
        )
        ablated_ranked_sessions = ranked_sessions_from_traces(ablated_items)
        ablated_ranked_session_uuids = [str(item["session_uuid"]) for item in ablated_ranked_sessions]
        ablated_metrics = compute_retrieval_metrics(
            retrieved_session_uuids=ablated_ranked_session_uuids,
            gold_session_uuids=gold_session_uuids,
            ks=session_ks,
        )
        retrieval_result = logged.get("retrieval_result") if isinstance(logged.get("retrieval_result"), dict) else {}
        ranked_items = retrieval_result.get("ranked_items") or retrieval_result.get("hits") or logged.get("ranked_items") or []
        path_b_candidate_count = sum(1 for item in ranked_items if isinstance(item, dict) and _has_path_b(item))
        detail = {
            "query_id": qid,
            "question": str(logged.get("question") or logged.get("query") or retrieval_result.get("question") or ""),
            "gold_session_uuids": gold_session_uuids,
            "original_ranked_session_uuids": original_ranked_session_uuids,
            "ablated_ranked_session_uuids": ablated_ranked_session_uuids,
            "original_metrics": original_metrics,
            "ablated_metrics": ablated_metrics,
            "logged_candidate_count": len(ranked_items) if isinstance(ranked_items, list) else 0,
            "path_b_candidate_count": path_b_candidate_count,
            "dropped_path_a_only_count": max(0, (len(ranked_items) if isinstance(ranked_items, list) else 0) - path_b_candidate_count),
        }
        if include_items:
            detail["ablated_ranked_items"] = ablated_items
            detail["ablated_ranked_sessions"] = ablated_ranked_sessions
        detailed_results.append(detail)

    original = _summarize_retrieval_metrics(detailed_results, metrics_key="original_metrics", session_ks=session_ks)
    ablated = _summarize_retrieval_metrics(detailed_results, metrics_key="ablated_metrics", session_ks=session_ks)
    summary = {
        "ablation": ABLATION_NAME,
        "approximation_warning": APPROXIMATION_WARNING,
        "query_count": len(detailed_results),
        "session_ks": session_ks,
        "path_b_weights": {
            "path_b_semantic_weight": path_b_semantic_weight,
            "path_b_bm25_weight": path_b_bm25_weight,
            "path_b_graph_weight": path_b_graph_weight,
        },
        "average_logged_candidate_count": _average(detailed_results, "logged_candidate_count"),
        "average_path_b_candidate_count": _average(detailed_results, "path_b_candidate_count"),
        "average_dropped_path_a_only_count": _average(detailed_results, "dropped_path_a_only_count"),
        "original": original,
        ABLATION_NAME: ablated,
        "delta_no_task_chain_minus_original": _delta_metrics(ablated, original),
    }
    return {"summary": summary, "detailed_results": detailed_results}


def _average(items: list[dict[str, Any]], key: str) -> float:
    values = [float(item.get(key) or 0.0) for item in items]
    return round(sum(values) / len(values), 4) if values else 0.0


def _infer_session_ks(results: dict[str, dict[str, Any]], manifest_file: str | Path | None) -> list[int]:
    manifest = _read_json(Path(manifest_file) if manifest_file else None)
    if isinstance(manifest, dict):
        config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
        session_ks = config.get("session_ks")
        if isinstance(session_ks, list) and session_ks:
            return [int(item) for item in session_ks]
    metric_keys: set[int] = set()
    pattern = re.compile(r"^(?:recall_any|recall_all|ndcg)@([0-9]+)$")
    for logged in results.values():
        metrics = logged.get("retrieval_metrics") if isinstance(logged.get("retrieval_metrics"), dict) else {}
        for key in metrics:
            match = pattern.match(str(key))
            if match:
                metric_keys.add(int(match.group(1)))
    return sorted(metric_keys) or [5, 10, 20, 30]


def evaluate_task_chain_ablation_from_logs(args: argparse.Namespace) -> dict[str, Any]:
    logged_results, artifacts = load_ablation_inputs(
        args.run_dir,
        query_results_file=args.query_results_file,
        retrieval_file=args.retrieval_file,
    )
    if not logged_results:
        raise SystemExit("No logged query results were found.")
    weights = load_ablation_weights(artifacts.get("manifest_file"))
    if args.path_b_semantic_weight is not None:
        weights["path_b_semantic_weight"] = args.path_b_semantic_weight
    if args.path_b_bm25_weight is not None:
        weights["path_b_bm25_weight"] = args.path_b_bm25_weight
    if args.path_b_graph_weight is not None:
        weights["path_b_graph_weight"] = args.path_b_graph_weight

    session_ks = _parse_ks(args.session_ks) if args.session_ks else _infer_session_ks(logged_results, artifacts.get("manifest_file"))
    output = build_task_chain_ablation(
        logged_results,
        session_ks=session_ks,
        path_b_semantic_weight=weights["path_b_semantic_weight"],
        path_b_bm25_weight=weights["path_b_bm25_weight"],
        path_b_graph_weight=weights["path_b_graph_weight"],
        include_items=args.include_items,
        query_ids=args.query_id,
        max_queries=args.max_queries,
    )
    output["artifacts"] = artifacts
    out_file = Path(args.out_file) if args.out_file else _default_output_file(artifacts)
    dump_json(out_file, output)
    return {"out_file": str(out_file), "summary": output["summary"]}


def _default_output_file(artifacts: dict[str, Any]) -> Path:
    result_dir = artifacts.get("result_dir")
    if result_dir:
        return Path(result_dir) / "task_chain_ablation_from_logs.json"
    return Path(artifacts.get("run_dir") or ".") / "task_chain_ablation_from_logs.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approximate no-task-chain ablation by reranking logged TCMem candidates.")
    parser.add_argument("--run-dir", default=None, help="A result/results/<run> or result/logs/<run> directory.")
    parser.add_argument("--query-results-file", default=None)
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--session-ks", "--ks", dest="session_ks", default=None)
    parser.add_argument("--query-id", action="append", default=[], help="Evaluate only this query id; may be repeated.")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--path-b-semantic-weight", type=float, default=None)
    parser.add_argument("--path-b-bm25-weight", type=float, default=None)
    parser.add_argument("--path-b-graph-weight", type=float, default=None)
    parser.add_argument("--include-items", action="store_true", help="Include reranked record-level items in the output JSON.")
    return parser.parse_args(argv)


def main() -> None:
    result = evaluate_task_chain_ablation_from_logs(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
