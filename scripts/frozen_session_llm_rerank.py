#!/usr/bin/env python3
"""Rerank frozen RealMemBench candidates without rebuilding task chains."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tcmem.utils.llm_client import OpenAICompatibleLLMClient

KS = (5, 10, 20, 30)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def details(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    return {str(item["query_id"]): item for item in payload.get("detailed_results", [])}


def number(item: dict[str, Any], key: str) -> float:
    try:
        return float(item.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def candidate_items(detail: dict[str, Any]) -> list[dict[str, Any]]:
    result = detail.get("retrieval_result") or {}
    values = result.get("ranked_items") or detail.get("ranked_items") or []
    return [item for item in values if isinstance(item, dict) and item.get("source_session_uuid")]


def build_candidates(
    full: dict[str, Any],
    no_chain: dict[str, Any],
    limit: int = 80,
    pool: str = "union",
    exclude_session: str | None = None,
) -> list[dict[str, Any]]:
    by_session: dict[str, dict[str, Any]] = {}
    arms = [("full", full)] if pool == "full" else [("no_task_chain", no_chain)] if pool == "no_task_chain" else [("full", full), ("no_task_chain", no_chain)]
    for arm, detail in arms:
        for rank, item in enumerate(candidate_items(detail), start=1):
            sid = str(item["source_session_uuid"])
            if exclude_session and sid == exclude_session:
                continue
            row = by_session.setdefault(
                sid,
                {
                    "session_uuid": sid,
                    "full_rank": 999,
                    "no_task_chain_rank": 999,
                    "full_score": 0.0,
                    "no_task_chain_score": 0.0,
                    "semantic_score": 0.0,
                    "bm25_score": 0.0,
                    "graph_score": 0.0,
                    "chain_score": 0.0,
                    "route_roles": [],
                    "task_ids": [],
                    "excerpts": [],
                    "record_scores": [],
                },
            )
            rank_key = "full_rank" if arm == "full" else "no_task_chain_rank"
            score_key = "full_score" if arm == "full" else "no_task_chain_score"
            row[rank_key] = min(int(row[rank_key]), rank)
            row[score_key] = max(float(row[score_key]), number(item, "score"))
            row["record_scores"].append(number(item, "score"))
            for key in ("semantic_score", "bm25_score", "graph_score", "chain_score"):
                row[key] = max(float(row[key]), number(item, key))
            role = str(item.get("route_role") or "").strip()
            if role and role not in row["route_roles"]:
                row["route_roles"].append(role)
            task_id = str(item.get("task_id") or "").strip()
            if task_id and task_id not in row["task_ids"]:
                row["task_ids"].append(task_id)
            excerpt = str(item.get("content_excerpt") or "").strip()
            if excerpt and excerpt not in row["excerpts"]:
                row["excerpts"].append(excerpt)
    # Candidate generation is frozen: use only sessions already surfaced by
    # either completed arm.  Keep a generous rank-based prefix so an LLM can
    # repair ordering without seeing the gold labels or the entire corpus.
    rows = list(by_session.values())
    rows.sort(key=lambda row: (min(row["full_rank"], row["no_task_chain_rank"]), row["session_uuid"]))
    rows = rows[: max(1, int(limit))]
    for row in rows:
        row["excerpts"] = row["excerpts"][:3]
        row["task_chain_evidence"] = bool(row["task_ids"] or row["chain_score"] > 0)
    return rows


def rrf(row: dict[str, Any]) -> float:
    return (1.0 / (60 + row["full_rank"]) if row["full_rank"] < 999 else 0.0) + (
        1.0 / (60 + row["no_task_chain_rank"]) if row["no_task_chain_rank"] < 999 else 0.0
    )


def deterministic_score(row: dict[str, Any], mode: str) -> float:
    if mode == "full_raw":
        return row["full_score"]
    if mode == "rrf":
        return rrf(row)
    if mode == "semantic":
        return row["semantic_score"]
    if mode == "path_b":
        return 0.5 * row["semantic_score"] + 0.2 * row["bm25_score"] + 0.3 * row["graph_score"]
    if mode == "evidence_sum":
        # The row carries the max score only; this mode is implemented by the
        # session-level fallback below when record-level evidence is retained.
        return row["full_score"]
    if mode == "task_chain_fallback":
        # Keep the task-chain arm primary. A candidate surfaced only by the
        # no-task-chain arm is retained as a low-weight recall fallback, while
        # shared candidates receive only a small generic-retrieval tie-break.
        # This uses frozen outputs only; it does not rebuild task chains.
        if row["full_rank"] < 999:
            return row["full_score"] + 0.05 * row["no_task_chain_score"]
        return 0.5 * row["no_task_chain_score"]
    raise ValueError(f"unknown deterministic mode: {mode}")


def prompt_for(query: str, rows: list[dict[str, Any]], excerpt_chars: int = 400) -> str:
    candidates = []
    for index, row in enumerate(rows):
        candidates.append(
            {
                "candidate": f"C{index:03d}",
                "session_uuid": row["session_uuid"],
                "evidence": " | ".join(row["excerpts"][:2])[:excerpt_chars],
                "retrieval_metadata": {
                    "full_rank": row["full_rank"],
                    "no_task_chain_rank": row["no_task_chain_rank"],
                    "task_chain_evidence": row["task_chain_evidence"],
                },
            }
        )
    return (
        "Rank frozen memory-session candidates for the user query. Do not invent facts "
        "and do not use retrieval ranks as relevance labels. A candidate is relevant "
        "when its evidence directly answers the query or supplies a necessary prior "
        "constraint, preference, decision, or task state. Penalize merely shared words, "
        "unrelated tasks, and stale adjacent topics. Return every candidate exactly once.\n\n"
        "Output JSON only with this schema: {\"ranked\":[{\"candidate\":\"C000\","
        "\"relevance\":0,\"evidence_types\":[\"direct|constraint|preference|state|none\"],"
        "\"confidence\":0.0}],\"query_type\":\"...\"}. Relevance is an integer "
        "0..4, where 4 is directly needed and 0 is unrelated.\n\n"
        f"USER QUERY:\n{query}\n\nCANDIDATES:\n{json.dumps(candidates, ensure_ascii=False)}"
    )


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("LLM response did not contain a JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("ranked"), list):
        raise ValueError("LLM response has no ranked list")
    return value


def llm_scores(
    client: OpenAICompatibleLLMClient,
    query: str,
    rows: list[dict[str, Any]],
    *,
    max_attempts: int = 1,
    max_tokens: int = 700,
    excerpt_chars: int = 400,
) -> tuple[dict[str, float], dict[str, Any]]:
    last_error = ""
    for attempt in range(max(1, max_attempts)):
        try:
            raw = client.generate(
                prompt_for(query, rows, excerpt_chars),
                system_prompt="You are a precise retrieval evaluator. Follow the JSON schema exactly.",
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            payload = parse_json(raw)
            scores: dict[str, float] = {}
            for item in payload["ranked"]:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("candidate") or "")
                try:
                    value = max(0.0, min(4.0, float(item.get("relevance", 0))))
                except (TypeError, ValueError):
                    value = 0.0
                scores[candidate] = value
            return scores, {"query_type": payload.get("query_type"), "raw_count": len(payload["ranked"])}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return {}, {"error": last_error}


def rank_rows(rows: list[dict[str, Any]], mode: str, scores: dict[str, float]) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        if mode == "llm":
            score = float(scores.get(f"C{index:03d}", 0.0))
            score += 0.04 * min(1.0, row["chain_score"])
        elif mode == "evidence_sum":
            values = sorted([float(value) for value in row.get("record_scores", [])], reverse=True)
            score = sum(weight * value for weight, value in zip((1.0, 0.75, 0.5), values))
        else:
            score = deterministic_score(row, mode)
        ranked.append((score, row["session_uuid"], row))
    return [row for _, _, row in sorted(ranked, key=lambda value: (-value[0], value[1]))]


def ndcg(ranked: list[str], gold: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum((1.0 if sid in gold_set else 0.0) / math.log2(index + 2) for index, sid in enumerate(ranked[:k]))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def query_metrics(ranked: list[str], gold: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        top = ranked[:k]
        result[f"recall_any@{k}"] = float(bool(set(top) & set(gold)))
        result[f"recall_all@{k}"] = sum(sid in top for sid in gold) / len(gold) if gold else 0.0
        result[f"ndcg@{k}"] = ndcg(ranked, gold, k)
    return result


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"query_count": len(items)}
    for key in [f"{prefix}@{k}" for k in KS for prefix in ("recall_any", "recall_all", "ndcg")]:
        values = [float(item["metrics"][key]) for item in items]
        result[key] = round(sum(values) / len(values), 4) if values else 0.0
    return result


def save(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--no-task-chain-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["full_raw", "path_b", "semantic", "rrf", "evidence_sum", "task_chain_fallback", "llm"],
        default="llm",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=80)
    parser.add_argument("--query-id", action="append", default=None)
    parser.add_argument("--pool", choices=["full", "no_task_chain", "union"], default="union")
    parser.add_argument("--exclude-current-session", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--excerpt-chars", type=int, default=400)
    args = parser.parse_args()
    full = details(args.full_results)
    no_chain = details(args.no_task_chain_results)
    previous = read_json(args.out) if args.resume and args.out.exists() else None
    done = {str(item["query_id"]): item for item in (previous or {}).get("detailed_results", [])}
    client = None
    if args.mode == "llm":
        env = load_env(args.env_file or Path(".llm_runtime_qwen/grok.env"))
        client = OpenAICompatibleLLMClient(
            api_key=env["OPENAI_API_KEY"],
            base_url=env["OPENAI_BASE_URL"],
            model=env["OPENAI_MODEL"],
            timeout=int(env.get("OPENAI_TIMEOUT", "180")),
        )
    output = {
        "schema_version": 1,
        "method": "frozen_candidate_session_rerank",
        "mode": args.mode,
        "pool": args.pool,
        "exclude_current_session": args.exclude_current_session,
        "task_chain_rebuilt": False,
        "llm_used_for_retrieval": args.mode == "llm",
        "details": {"full_results": str(args.full_results), "no_task_chain_results": str(args.no_task_chain_results)},
        "detailed_results": list(done.values()),
    }
    query_set = set(full) & set(no_chain)
    if args.query_id:
        query_set &= set(args.query_id)
    for query_id in sorted(query_set, key=lambda value: int(value.split("-")[-1])):
        if query_id in done:
            continue
        full_detail = full[query_id]
        rows = build_candidates(
            full_detail,
            no_chain[query_id],
            args.candidate_limit,
            args.pool,
            str(full_detail.get("session_uuid") or "") if args.exclude_current_session else None,
        )
        scores: dict[str, float] = {}
        llm_meta: dict[str, Any] = {}
        if client is not None:
            scores, llm_meta = llm_scores(
                client,
                str(full_detail.get("question") or ""),
                rows,
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
                excerpt_chars=args.excerpt_chars,
            )
        ranked_rows = rank_rows(rows, args.mode, scores)
        ranked = [row["session_uuid"] for row in ranked_rows]
        gold = [str(value) for value in full_detail.get("gold_session_uuids", [])]
        output["detailed_results"].append(
            {
                "query_id": query_id,
                "question": full_detail.get("question", ""),
                "gold_session_uuids": gold,
                "candidate_session_count": len(rows),
                "gold_sessions_in_candidate_pool": sorted(set(gold) & set(ranked)),
                "ranked_session_uuids": ranked,
                "metrics": query_metrics(ranked, gold),
                "llm": llm_meta,
                "candidates": rows,
            }
        )
        output["detailed_results"].sort(key=lambda item: int(item["query_id"].split("-")[-1]))
        output["summary"] = summarize(output["detailed_results"])
        save(args.out, output)
        print(query_id, "candidates", len(rows), "llm_error", llm_meta.get("error", ""), flush=True)
    output["summary"] = summarize(output["detailed_results"])
    save(args.out, output)
    print(json.dumps(output["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
