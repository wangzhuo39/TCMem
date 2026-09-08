#!/usr/bin/env python3
"""Generate a compact semantic audit for a Full-baseline checkpoint.

This is intentionally heuristic: it highlights records/queries for human
inspection and never changes experiment results or state.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")


def load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def tokens(*values: Any) -> set[str]:
    return {m.group(0).casefold() for value in values for m in TOKEN_RE.finditer(str(value or ""))}


def query_events(results_dir: Path) -> list[dict[str, Any]]:
    manifest = load(results_dir / "manifest.json", {})
    candidates = []
    if manifest.get("log_dir"):
        candidates.append(Path(str(manifest["log_dir"])) / "query_results.jsonl")
    candidates.extend(results_dir.glob("../logs/**/query_results.jsonl"))
    events: dict[str, dict[str, Any]] = {}
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") != "query_completed":
                continue
            payload = item.get("payload") or {}
            if payload.get("query_id"):
                events[str(payload["query_id"])] = payload
    return list(events.values())


def audit(results_dir: Path) -> dict[str, Any]:
    state = load(results_dir / "memory_state_latest.json", {})
    graph_records = {
        str(item.get("record_id")): item
        for item in (state.get("graph") or {}).get("records", [])
        if isinstance(item, dict) and item.get("record_id")
    }
    tasks = (state.get("task_chains") or {}).get("tasks", [])
    task_rows = []
    low_overlap = []
    for task in tasks:
        description = task.get("canonical_description") or task.get("task_description") or ""
        task_tokens = tokens(description, task.get("current_focus"), *(task.get("entities") or []))
        nodes = task.get("nodes") or {}
        node_rows = []
        for node in sorted(nodes.values(), key=lambda value: int(value.get("position", 0))):
            text = " ".join(str(node.get(key, "")) for key in ("user_content", "assistant_content", "summary"))
            overlap = len(task_tokens & tokens(text)) / max(1, len(tokens(text)))
            row = {
                "record_id": node.get("source_record_id"),
                "node_id": node.get("node_id"),
                "position": node.get("position"),
                "branch_id": node.get("branch_id"),
                "overlap": round(overlap, 4),
                "user": str(node.get("user_content", ""))[:240],
                "assistant": str(node.get("assistant_content", ""))[:240],
            }
            node_rows.append(row)
            if overlap < 0.02 and len(tokens(text)) >= 8:
                low_overlap.append({"task_id": task.get("task_id"), "task_description": description, **row})
        task_rows.append({
            "task_id": task.get("task_id"),
            "description": description,
            "status": task.get("status"),
            "node_count": len(nodes),
            "branch_count": len(task.get("branches") or {}),
            "nodes": node_rows[:8],
        })

    misses = []
    low_recall = []
    for payload in query_events(results_dir):
        gold = set(str(value) for value in payload.get("gold_session_uuids", []) or [])
        ranked = [str(value) for value in payload.get("ranked_session_uuids", []) or []]
        if gold and not gold.intersection(ranked):
            misses.append({
                "query_id": payload.get("query_id"),
                "question": str(payload.get("question", ""))[:400],
                "gold": sorted(gold),
                "top_ranked": ranked[:10],
                "reason": payload.get("retrieval_result", {}).get("explanation", ""),
            })
        metrics = payload.get("retrieval_metrics") or {}
        if metrics.get("recall_all@5", 1.0) < 1.0 or metrics.get("ndcg@5", 1.0) < 0.8:
            retrieval = payload.get("retrieval_result") or {}
            low_recall.append({
                "query_id": payload.get("query_id"),
                "question": str(payload.get("question", ""))[:400],
                "metrics": {key: metrics.get(key) for key in ("recall_all@5", "recall_all@10", "ndcg@5")},
                "gold": sorted(gold),
                "top_ranked": ranked[:10],
                "routed_task_ids": retrieval.get("routed_task_ids", []),
                "expanded_task_ids": retrieval.get("expanded_task_ids", []),
                "route_reason": retrieval.get("query_route_reason", ""),
            })
    return {
        "record_count": len(graph_records),
        "task_count": len(tasks),
        "query_count": len(query_events(results_dir)),
        "task_status_counts": dict(Counter(str(task.get("status")) for task in tasks)),
        "branch_count": sum(len(task.get("branches") or {}) for task in tasks),
        "tasks": task_rows,
        "low_lexical_overlap_nodes": low_overlap,
        "query_any_recall_misses": misses,
        "query_low_recall_or_ndcg": low_recall,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["# Full baseline semantic audit", "", f"- records: {report['record_count']}", f"- tasks: {report['task_count']}", f"- queries: {report['query_count']}", f"- branches: {report['branch_count']}", f"- task statuses: {report['task_status_counts']}", "", "## Task samples", ""]
    for task in report["tasks"]:
        lines.append(f"- **{task['task_id']}** ({task['status']}, nodes={task['node_count']}): {task['description']}")
        for node in task["nodes"][:3]:
            lines.append(f"  - {node['record_id']} overlap={node['overlap']}: user={node['user']}")
    lines.extend(["", "## Low lexical-overlap nodes for review", ""])
    for row in report["low_lexical_overlap_nodes"]:
        lines.append(f"- {row['task_id']} / {row['record_id']} overlap={row['overlap']}: {row['user']}")
    lines.extend(["", "## Query low recall or nDCG for human review", ""])
    for row in report["query_low_recall_or_ndcg"]:
        lines.append(f"- {row['query_id']} metrics={row['metrics']} routed={row['routed_task_ids']} expanded={row['expanded_task_ids']}: {row['question']}")
    if not report["query_low_recall_or_ndcg"]:
        lines.append("- none in the available checkpoint")
    lines.extend(["", "## Query any-recall misses", ""])
    for row in report["query_any_recall_misses"]:
        lines.append(f"- {row['query_id']}: gold={row['gold']} top={row['top_ranked']}; {row['question']}")
    if not report["query_any_recall_misses"]:
        lines.append("- none in the available checkpoint")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.results_dir)
    output = args.output or args.results_dir / "semantic_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("record_count", "task_count", "query_count", "branch_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
