from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import TCMemConfig
from ..logging_utils import ModuleLogStore
from ..models import DialogueRecord, DialogueTurn, RetrievalResult, SearchHit, SessionPayload
from ..prompts import PromptRegistry
from ..serialization import dump_json, to_primitive
from ..core.memory_system import MemorySystem
from ..utils.llm_client import OpenAICompatibleLLMClient


DEFAULT_KS = [5, 10, 20]


@dataclass(slots=True)
class RuntimeConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeout: int = 120


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    mode: str = "tcmem_top_session"
    task_chain_enabled: bool = True


@dataclass(slots=True)
class QueryExample:
    query_id: str
    session_identifier: str
    session_uuid: str
    current_time: str
    turn_index: int
    question: str
    reference_answer: str
    gold_session_uuids: list[str]
    gold_memory_text: str
    memory_used: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PairRecord:
    session_index: int
    record_index: int
    user_turn_index: int
    assistant_turn_index: int | None
    record: DialogueRecord
    query_example: QueryExample | None = None


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    config_path = Path(args.config)
    if not config_path.exists() and str(args.config) == "zhuo/runtime_config.json":
        fallback = Path("..") / "zhuo" / "runtime_config.json"
        if fallback.exists():
            config_path = fallback
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    return RuntimeConfig(
        api_key=args.api_key or data.get("api_key") or "",
        base_url=args.base_url or data.get("base_url") or "https://api.deepseek.com",
        model=args.model or data.get("model") or "deepseek-v4-flash",
        timeout=args.timeout or int(data.get("timeout", 120) or 120),
    )


def evaluation_options_from_args(_args: argparse.Namespace) -> EvaluationOptions:
    return EvaluationOptions()


def build_tcmem_config(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    log_store_base_dir: str | Path,
    runtime: RuntimeConfig,
    task_chain_enabled: bool,
) -> TCMemConfig:
    return TCMemConfig(
        owner_id=args.owner_id,
        storage_path=str(output_dir / "state"),
        log_path=str(log_store_base_dir),
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        vector_index_backend=args.vector_index_backend,
        vector_index_path=str(output_dir / "vector_index"),
        graph_seed_limit=args.graph_seed_limit,
        graph_walk_depth=args.graph_walk_depth,
        path_a_weight=args.path_a_weight,
        path_b_weight=args.path_b_weight,
        llm_api_key=runtime.api_key,
        llm_base_url=runtime.base_url,
        llm_model=runtime.model,
        llm_timeout=runtime.timeout,
        prompt_path=args.prompt_path,
        task_chain_enabled=task_chain_enabled,
    )


def _parse_ks(value: str) -> list[int]:
    result = [int(part.strip()) for part in str(value or "").split(",") if part.strip()]
    if not result:
        raise ValueError("--session-ks must contain at least one integer")
    return result


def _short(text: Any, limit: int = 260) -> str:
    value = "" if text is None else str(text).replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _dcg(relevances: list[float], k: int) -> float:
    selected = relevances[:k]
    if not selected:
        return 0.0
    score = selected[0]
    for index, relevance in enumerate(selected[1:], start=2):
        score += relevance / math.log2(index + 1)
    return score


def compute_retrieval_metrics(
    *,
    retrieved_session_uuids: list[str],
    gold_session_uuids: list[str],
    ks: list[int] | None = None,
) -> dict[str, float]:
    ks = ks or DEFAULT_KS
    retrieved = _unique(retrieved_session_uuids)
    gold = set(_unique(gold_session_uuids))
    metrics: dict[str, float] = {}
    for k in ks:
        top_k = retrieved[:k]
        recalled = set(top_k) & gold
        metrics[f"recall_any@{k}"] = float(bool(recalled)) if gold else 0.0
        metrics[f"recall_all@{k}"] = float(gold.issubset(set(top_k))) if gold else 0.0
        ranked_relevances = [1.0 if session_uuid in gold else 0.0 for session_uuid in top_k]
        ideal_relevances = [1.0] * min(len(gold), k)
        ideal = _dcg(ideal_relevances, k)
        metrics[f"ndcg@{k}"] = (_dcg(ranked_relevances, k) / ideal) if ideal else 0.0
    return metrics


def extract_query_examples(dataset: dict[str, Any]) -> list[QueryExample]:
    examples: list[QueryExample] = []
    query_index = 1
    for dialogue in dataset.get("dialogues", []) or []:
        turns = dialogue.get("dialogue_turns", []) or []
        for turn_index, turn in enumerate(turns):
            if turn.get("speaker") != "User" or turn.get("is_query") is not True:
                continue
            next_turn = turns[turn_index + 1] if turn_index + 1 < len(turns) else {}
            if next_turn.get("speaker") != "Assistant":
                next_turn = {}
            memory_used = [item for item in (next_turn.get("memory_used") or []) if isinstance(item, dict)]
            gold_uuids = [str(uuid) for uuid in (next_turn.get("memory_session_uuids") or []) if str(uuid)]
            if not gold_uuids:
                gold_uuids = [str(item.get("session_uuid")) for item in memory_used if item.get("session_uuid")]
            gold_memory_text = "\n".join(
                str(item.get("content", "")).strip()
                for item in memory_used
                if item.get("content")
            ).strip()
            examples.append(
                QueryExample(
                    query_id=str(turn.get("query_id") or f"Q-{query_index:04d}"),
                    session_identifier=str(dialogue.get("session_identifier", "")),
                    session_uuid=str(dialogue.get("session_uuid", "")),
                    current_time=str(dialogue.get("current_time", "")),
                    turn_index=turn_index,
                    question=str(turn.get("content", "")).strip(),
                    reference_answer=str(next_turn.get("content", "")).strip(),
                    gold_session_uuids=_unique(gold_uuids),
                    gold_memory_text=gold_memory_text,
                    memory_used=memory_used,
                )
            )
            query_index += 1
    return examples


def build_pair_records(dataset: dict[str, Any], examples: list[QueryExample]) -> list[PairRecord]:
    example_by_location = {(example.session_uuid, example.turn_index): example for example in examples}
    pair_records: list[PairRecord] = []
    for session_index, dialogue in enumerate(dataset.get("dialogues", []) or []):
        session = _session_from_dialogue(dialogue)
        records = _parse_session_records(session)
        for record_index, record in enumerate(records):
            user_turn_index = record.source_turn_indexes[0] if record.source_turn_indexes else 0
            assistant_turn_index = record.source_turn_indexes[1] if len(record.source_turn_indexes) > 1 else None
            pair_records.append(
                PairRecord(
                    session_index=session_index,
                    record_index=record_index,
                    user_turn_index=user_turn_index,
                    assistant_turn_index=assistant_turn_index,
                    record=record,
                    query_example=example_by_location.get((session.session_uuid, user_turn_index)),
                )
            )
    return pair_records


def build_progress_payload(
    *,
    pair: PairRecord,
    processed_records: int,
    total_records: int,
    evaluated_queries: int,
    total_queries: int,
    elapsed_seconds: float,
    query_id: str | None = None,
) -> dict[str, Any]:
    def ratio(done: int, total: int) -> float:
        return round(done / total, 4) if total > 0 else 0.0

    payload: dict[str, Any] = {
        "processed_records": processed_records,
        "total_records": total_records,
        "record_progress": ratio(processed_records, total_records),
        "evaluated_queries": evaluated_queries,
        "total_queries": total_queries,
        "query_progress": ratio(evaluated_queries, total_queries),
        "session_number": pair.session_index + 1,
        "session_uuid": pair.record.session_uuid,
        "session_identifier": pair.record.session_identifier,
        "record_number_in_session": pair.record_index + 1,
        "record_id": pair.record.record_id,
        "user_turn_index": pair.user_turn_index,
        "assistant_turn_index": pair.assistant_turn_index,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }
    if query_id:
        payload["query_id"] = query_id
    return payload


def log_query_result(log_store: ModuleLogStore, event: str, result: dict[str, Any]) -> None:
    log_store.log("query_results", event, **result)


def save_latest_state(system: MemorySystem, output_dir: Path) -> Path:
    return system.save(output_dir / "memory_state_latest.json")


def save_query_snapshot(
    system: MemorySystem,
    output_dir: Path,
    query_id: str,
    *,
    question: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slug = _query_snapshot_slug(query_id)
    snapshot_dir = output_dir / "query_snapshots" / slug
    memory_state_path = snapshot_dir / "memory_state.json"
    vector_index_path = snapshot_dir / "vector_index"
    manifest_path = snapshot_dir / "manifest.json"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    state_path = system.save(memory_state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state_config = dict(state.get("config") or {})
    source_vector_index_text = str(
        state_config.get("vector_index_path")
        or getattr(getattr(system, "config", None), "vector_index_path", "")
        or ""
    ).strip()
    source_vector_index_path = Path(source_vector_index_text) if source_vector_index_text else None
    if state_config:
        state_config["storage_path"] = str(snapshot_dir)
        state_config["vector_index_path"] = str(vector_index_path)
        state["config"] = state_config
        dump_json(state_path, state)

    _sync_record_index_if_available(system)
    vector_index_copied = _copy_vector_index_snapshot(source_vector_index_path, vector_index_path)
    snapshot = {
        "schema_version": 1,
        "query_id": str(query_id),
        "question": str(question or ""),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot_dir": str(snapshot_dir),
        "memory_state_path": str(state_path),
        "manifest_path": str(manifest_path),
        "vector_index_path": str(vector_index_path),
        "vector_index": {
            "source_path": str(source_vector_index_path) if source_vector_index_path is not None else "",
            "snapshot_path": str(vector_index_path),
            "copied": vector_index_copied,
        },
        "metadata": metadata or {},
    }
    dump_json(manifest_path, snapshot)
    return snapshot


def save_query_state(system: MemorySystem, output_dir: Path, query_id: str) -> Path:
    snapshot = save_query_snapshot(system, output_dir, query_id)
    return Path(snapshot["memory_state_path"])


def _query_snapshot_slug(query_id: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", str(query_id or "").strip()).strip("._-")
    return slug or "query"


def _sync_record_index_if_available(system: MemorySystem) -> None:
    retrieval = getattr(system, "retrieval", None)
    sync_record_index = getattr(retrieval, "sync_record_index", None)
    if callable(sync_record_index):
        sync_record_index()


def _copy_vector_index_snapshot(source_path: Path | None, target_path: Path) -> bool:
    if source_path is None or not source_path.exists():
        return False
    source_resolved = source_path.resolve()
    target_resolved = target_path.resolve()
    if source_resolved == target_resolved:
        return True
    try:
        if target_resolved.is_relative_to(source_resolved):
            return False
    except AttributeError:
        if str(target_resolved).startswith(f"{source_resolved}/"):
            return False
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path / source_path.name)
    return True


def should_save_record_state(processed_records: int, every_records: int) -> bool:
    return every_records > 0 and processed_records > 0 and processed_records % every_records == 0


def build_session_text_by_uuid(dataset: dict[str, Any]) -> dict[str, str]:
    session_text_by_uuid: dict[str, str] = {}
    for dialogue in dataset.get("dialogues", []) or []:
        session_uuid = str(dialogue.get("session_uuid") or "").strip()
        if not session_uuid:
            continue
        lines: list[str] = []
        for turn in dialogue.get("dialogue_turns", []) or []:
            speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
            content = str(turn.get("content") or "").strip()
            if content:
                lines.append(f"{speaker}: {content}")
        session_text_by_uuid[session_uuid] = "\n".join(lines).strip()
    return session_text_by_uuid


def ranked_sessions_from_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    index_by_session: dict[str, int] = {}
    for rank, trace in enumerate(traces, start=1):
        session_uuid = str(trace.get("source_session_uuid") or "").strip()
        if not session_uuid:
            continue
        record_id = str(trace.get("source_record_id") or trace.get("item_id") or "").strip()
        existing_index = index_by_session.get(session_uuid)
        if existing_index is None:
            index_by_session[session_uuid] = len(ranked)
            ranked.append(
                {
                    "session_uuid": session_uuid,
                    "score": float(trace.get("score") or 0.0),
                    "first_record_rank": rank,
                    "representative_record_id": record_id,
                    "record_count": 1,
                    "record_ids": [record_id] if record_id else [],
                }
            )
            continue
        session = ranked[existing_index]
        session["record_count"] = int(session["record_count"]) + 1
        if record_id:
            session["record_ids"].append(record_id)
    return ranked


def construct_session_evidence(
    *,
    ranked_sessions: list[dict[str, Any]],
    session_text_by_uuid: dict[str, str],
    top_k: int,
) -> str:
    chunks: list[str] = []
    for rank, session in enumerate(ranked_sessions[:top_k], start=1):
        session_uuid = str(session.get("session_uuid") or "").strip()
        session_text = str(session_text_by_uuid.get(session_uuid) or "").strip()
        if not session_uuid or not session_text:
            continue
        chunks.append(f"---- idx {rank} | session_uuid={session_uuid} ----\n{session_text}")
    return "\n\n".join(chunks)


def traces_from_retrieval(
    retrieval: RetrievalResult,
    system: MemorySystem,
    gold_session_uuids: list[str],
) -> list[dict[str, Any]]:
    return [hit_to_trace(hit, system, gold_session_uuids) for hit in retrieval.hits]


def hit_to_trace(hit: SearchHit, system: MemorySystem, gold_session_uuids: list[str]) -> dict[str, Any]:
    record = system.graph.get_record(hit.source_record_id or hit.item_id)
    return {
        "item_id": hit.item_id,
        "item_kind": hit.item_kind,
        "score": hit.score,
        "semantic_score": hit.semantic_score,
        "chain_score": hit.chain_score,
        "graph_score": hit.graph_score,
        "route_score": hit.route_score,
        "reason": hit.reason,
        "task_id": hit.task_id,
        "chain_node_id": hit.chain_node_id,
        "source_record_id": hit.source_record_id or hit.item_id,
        "source_session_uuid": record.session_uuid if record else None,
        "source_session_identifier": record.session_identifier if record else None,
        "source_turn_indexes": list(record.source_turn_indexes) if record else [],
        "source_turn_ids": list(record.source_turn_ids) if record else [],
        "content_excerpt": _short(record.combined_content if record else "", 420),
        "matched_gold": bool(record and record.session_uuid in set(gold_session_uuids)),
    }


def dedupe_ranked_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for trace in traces:
        key = str(trace.get("source_record_id") or trace.get("item_id") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trace)
    return deduped


def evaluate_query_top_session(
    *,
    system: MemorySystem,
    example: QueryExample,
    retrieval_record_k: int,
    session_ks: list[int],
    session_text_by_uuid: dict[str, str],
    evidence_top_k: int,
    client: OpenAICompatibleLLMClient | None = None,
    qa_model_name: str = "",
    with_qa: bool = False,
    log_store: ModuleLogStore | None = None,
    prompt_registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    retrieval = system.retrieve(example.question, top_k=retrieval_record_k)
    traces = dedupe_ranked_traces(traces_from_retrieval(retrieval, system, example.gold_session_uuids))
    ranked_sessions = ranked_sessions_from_traces(traces)
    ranked_session_uuids = [str(item["session_uuid"]) for item in ranked_sessions]
    retrieval_metrics = compute_retrieval_metrics(
        retrieved_session_uuids=ranked_session_uuids,
        gold_session_uuids=example.gold_session_uuids,
        ks=session_ks,
    )
    result: dict[str, Any] = {
        "query_id": example.query_id,
        "question": example.question,
        "session_uuid": example.session_uuid,
        "turn_index": example.turn_index,
        "gold_session_uuids": example.gold_session_uuids,
        "retrieved_session_uuids": ranked_session_uuids,
        "ranked_session_uuids": ranked_session_uuids,
        "ranked_sessions": ranked_sessions,
        "retrieval_record_k": retrieval_record_k,
        "retrieval_metrics": retrieval_metrics,
        "qa_score": None,
        "qa_reason": "",
        "retrieval_result": {
            "query_id": example.query_id,
            "question": example.question,
            "routed_task_ids": retrieval.routed_task_ids,
            "ranked_items": traces,
            "retrieved_session_uuids": ranked_session_uuids,
            "ranked_session_uuids": ranked_session_uuids,
            "ranked_sessions": ranked_sessions,
            "gold_session_uuids": example.gold_session_uuids,
            "retrieval_record_k": retrieval_record_k,
        },
    }
    if with_qa:
        if client is None:
            raise RuntimeError("--with-qa requires an API client")
        evidence_text = construct_session_evidence(
            ranked_sessions=ranked_sessions,
            session_text_by_uuid=session_text_by_uuid,
            top_k=evidence_top_k,
        )
        generated_answer = generate_answer(client, example.question, evidence_text, prompt_registry=prompt_registry)
        judge_result = judge_qa_score(
            client,
            question=example.question,
            gold_memory_text=example.gold_memory_text,
            reference_answer=example.reference_answer,
            candidate_answer=generated_answer,
            log_store=log_store,
            query_id=example.query_id,
            prompt_registry=prompt_registry,
        )
        result["qa_score"] = judge_result["score"]
        result["qa_reason"] = judge_result["reason"]
        result["generation_result"] = {
            "query_id": example.query_id,
            "question": example.question,
            "generated_answer": generated_answer,
            "evidence_used": evidence_text,
            "ranked_sessions": ranked_sessions[:evidence_top_k],
            "evidence_session_uuids": [item["session_uuid"] for item in ranked_sessions[:evidence_top_k]],
            "model": qa_model_name,
        }
    return result


def generate_answer(
    client: OpenAICompatibleLLMClient,
    question: str,
    evidence_text: str,
    *,
    prompt_registry: PromptRegistry | None = None,
) -> str:
    registry = prompt_registry or PromptRegistry.default()
    payload = {"reference_memory": evidence_text, "query": question}
    rendered = registry.render(
        "realmem_answer_generation",
        payload_json=_prompt_json(payload),
        question=question,
        evidence_text=evidence_text,
    )
    return client.generate(rendered.user_prompt, system_prompt=rendered.system_prompt, temperature=0.2, max_tokens=1200).strip()


def judge_qa_score(
    client: OpenAICompatibleLLMClient,
    *,
    question: str,
    gold_memory_text: str,
    reference_answer: str,
    candidate_answer: str,
    log_store: ModuleLogStore | None = None,
    query_id: str = "",
    prompt_registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    registry = prompt_registry or PromptRegistry.default()
    payload = {
        "query": question,
        "user_related_memory": gold_memory_text,
        "reference_answer": reference_answer,
        "candidate_answer": candidate_answer,
    }
    rendered = registry.render(
        "realmem_qa_judge",
        payload_json=_prompt_json(payload),
        question=question,
        gold_memory_text=gold_memory_text,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
    )
    parsed = _generate_json_with_retries(
        client,
        rendered.user_prompt,
        system_prompt=rendered.system_prompt,
        temperature=0.0,
        max_tokens=1200,
        stage="qa_judge",
        log_store=log_store,
        context={"query_id": query_id},
    )
    if not isinstance(parsed, dict):
        raise RuntimeError("QA judge returned non-object JSON")
    score = int(parsed.get("score", -1))
    if score not in {0, 1, 2, 3}:
        raise ValueError(f"QA judge score must be 0, 1, 2, or 3; got {score!r}")
    return {"score": score, "reason": str(parsed.get("reason", "") or "")}


def _prompt_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _generate_json_with_retries(
    client: OpenAICompatibleLLMClient,
    prompt: str,
    *,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    stage: str,
    log_store: ModuleLogStore | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | list[Any]:
    max_attempts = max(1, int(getattr(client, "json_max_attempts", 5) or 5))
    retry_delay = max(0.0, float(getattr(client, "json_retry_delay", 0.5) or 0.0))
    last_error: json.JSONDecodeError | None = None
    for attempt in range(1, max_attempts + 1):
        raw = client.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        response_text = str(raw or "")
        try:
            return _extract_json_payload(response_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            _log_llm_json_error(
                log_store=log_store,
                stage=stage,
                attempt=attempt,
                max_attempts=max_attempts,
                response_text=response_text,
                error=exc,
                prompt=prompt,
                context=context,
            )
            if attempt >= max_attempts:
                raise
            if retry_delay > 0.0:
                time.sleep(retry_delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"LLM returned invalid JSON for stage {stage}")


def _log_llm_json_error(
    *,
    log_store: ModuleLogStore | None,
    stage: str,
    attempt: int,
    max_attempts: int,
    response_text: str,
    error: json.JSONDecodeError,
    prompt: str,
    context: dict[str, Any] | None = None,
) -> None:
    if log_store is None:
        return
    log_store.log(
        "llm_errors",
        "json_decode_failed",
        stage=stage,
        attempt=attempt,
        max_attempts=max_attempts,
        error_type=type(error).__name__,
        message=str(error),
        prompt_chars=len(prompt),
        raw_response_excerpt=response_text[:1200],
        **(context or {}),
    )


def summarize_metrics(detailed_results: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in ks:
        for prefix in ("recall_any", "recall_all", "ndcg"):
            key = f"{prefix}@{k}"
            values = [float(item["retrieval_metrics"][key]) for item in detailed_results if key in item.get("retrieval_metrics", {})]
            summary[key] = round(sum(values) / len(values), 4) if values else 0.0
    qa_scores = [int(item["qa_score"]) for item in detailed_results if isinstance(item.get("qa_score"), int)]
    summary["average_qa_score"] = round(sum(qa_scores) / len(qa_scores), 4) if qa_scores else None
    summary["qa_score_distribution"] = {str(score): qa_scores.count(score) for score in range(4)}
    summary["query_count"] = len(detailed_results)
    summary["qa_failed_count"] = sum(1 for item in detailed_results if item.get("qa_score") is None)
    return summary


def default_retrieval_record_k(session_ks: list[int]) -> int:
    max_session_k = max(session_ks) if session_ks else 20
    return max(200, max_session_k * 10)


def run_evaluation(args: argparse.Namespace, *, options: EvaluationOptions | None = None) -> dict[str, Any]:
    options = options or evaluation_options_from_args(args)
    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    examples = extract_query_examples(dataset)
    pair_records = build_pair_records(dataset, examples)
    session_text_by_uuid = build_session_text_by_uuid(dataset)
    session_ks = _parse_ks(args.session_ks)
    retrieval_record_k = args.retrieval_record_k or default_retrieval_record_k(session_ks)

    run_name = args.run_name or f"tcmem_realmem_top_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else Path("result/results") / run_name
    log_store = ModuleLogStore(base_dir=args.log_dir, run_name=run_name)
    runtime = resolve_runtime_config(args)
    if not runtime.api_key:
        raise SystemExit("Missing API key for TCMem RealMem evaluation.")
    client = OpenAICompatibleLLMClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        model=runtime.model,
        timeout=runtime.timeout,
    )
    config = build_tcmem_config(
        args,
        output_dir=output_dir,
        log_store_base_dir=log_store.base_dir,
        runtime=runtime,
        task_chain_enabled=options.task_chain_enabled,
    )
    system = MemorySystem(config=config, llm_client=client, log_store=log_store)
    if args.max_queries is not None:
        examples = examples[: args.max_queries]
    example_ids = {example.query_id for example in examples}
    dataset_summary = {
        "person_name": dataset.get("_metadata", {}).get("person_name", ""),
        "session_count": len(dataset.get("dialogues", []) or []),
        "query_count": len(examples),
        "record_count": len(pair_records),
    }
    run_config = {
        "dataset": str(dataset_path),
        "base_url": runtime.base_url,
        "model": runtime.model,
        "session_ks": session_ks,
        "retrieval_record_k": retrieval_record_k,
        "evidence_top_k": args.evidence_top_k,
        "with_qa": args.with_qa,
        "max_queries": args.max_queries,
        "mode": options.mode,
        "tcmem_config": config.to_dict(),
    }
    log_store.log("dataset", "dataset_loaded", **dataset_summary)
    log_store.log("dataset", "run_config", **run_config)

    retrieval_results: dict[str, Any] = {}
    generation_results: dict[str, Any] = {}
    detailed_results: list[dict[str, Any]] = []
    evaluated = 0
    failed = 0
    started_at = time.time()
    processed_records = 0
    current_session_uuid = ""
    log_store.log(
        "progress",
        "run_started",
        processed_records=processed_records,
        total_records=len(pair_records),
        record_progress=0.0,
        evaluated_queries=evaluated,
        total_queries=len(examples),
        query_progress=0.0,
        elapsed_seconds=0.0,
    )
    for index, pair in enumerate(pair_records, start=1):
        if pair.record.session_uuid != current_session_uuid:
            current_session_uuid = pair.record.session_uuid
            log_store.log(
                "progress",
                "session_started",
                **build_progress_payload(
                    pair=pair,
                    processed_records=processed_records,
                    total_records=len(pair_records),
                    evaluated_queries=evaluated,
                    total_queries=len(examples),
                    elapsed_seconds=time.time() - started_at,
                ),
            )

        example = pair.query_example
        if example is not None and example.query_id in example_ids:
            log_store.log(
                "progress",
                "query_started",
                **build_progress_payload(
                    pair=pair,
                    processed_records=processed_records,
                    total_records=len(pair_records),
                    evaluated_queries=evaluated,
                    total_queries=len(examples),
                    elapsed_seconds=time.time() - started_at,
                    query_id=example.query_id,
                ),
            )
            query_snapshot: dict[str, Any] | None = None
            query_state_path: Path | None = None
            try:
                query_snapshot = save_query_snapshot(
                    system,
                    output_dir,
                    example.query_id,
                    question=example.question,
                    metadata={
                        "retrieval_record_k": retrieval_record_k,
                        "session_ks": session_ks,
                        "evidence_top_k": args.evidence_top_k,
                        "with_qa": args.with_qa,
                        "model": runtime.model,
                        "base_url": runtime.base_url,
                        "processed_records": processed_records,
                        "evaluated_queries": evaluated,
                        "tcmem_config": config.to_dict(),
                    },
                )
                query_state_path = Path(query_snapshot["memory_state_path"])
                log_store.log(
                    "progress",
                    "query_snapshot_saved",
                    **build_progress_payload(
                        pair=pair,
                        processed_records=processed_records,
                        total_records=len(pair_records),
                        evaluated_queries=evaluated,
                        total_queries=len(examples),
                        elapsed_seconds=time.time() - started_at,
                        query_id=example.query_id,
                    ),
                    memory_state_path=str(query_state_path),
                    query_snapshot_dir=str(query_snapshot["snapshot_dir"]),
                    query_snapshot_manifest_path=str(query_snapshot["manifest_path"]),
                    vector_index_path=str(query_snapshot["vector_index_path"]),
                    vector_index_copied=bool(query_snapshot.get("vector_index", {}).get("copied")),
                )
                result = evaluate_query_top_session(
                    system=system,
                    example=example,
                    retrieval_record_k=retrieval_record_k,
                    session_ks=session_ks,
                    session_text_by_uuid=session_text_by_uuid,
                    evidence_top_k=args.evidence_top_k,
                    client=client,
                    qa_model_name=runtime.model,
                    with_qa=args.with_qa,
                    log_store=log_store,
                    prompt_registry=system.prompt_registry,
                )
                result["memory_state_path"] = str(query_state_path)
                result["query_snapshot_dir"] = str(query_snapshot["snapshot_dir"])
                result["query_snapshot_manifest_path"] = str(query_snapshot["manifest_path"])
                result["vector_index_path"] = str(query_snapshot["vector_index_path"])
                if isinstance(result.get("retrieval_result"), dict):
                    result["retrieval_result"]["memory_state_path"] = str(query_state_path)
                    result["retrieval_result"]["query_snapshot_dir"] = str(query_snapshot["snapshot_dir"])
                    result["retrieval_result"]["query_snapshot_manifest_path"] = str(query_snapshot["manifest_path"])
                    result["retrieval_result"]["vector_index_path"] = str(query_snapshot["vector_index_path"])
                detailed_results.append(result)
                retrieval_results[example.query_id] = result["retrieval_result"]
                if "generation_result" in result:
                    generation_results[example.query_id] = result["generation_result"]
                evaluated += 1
                summary_so_far = summarize_metrics(detailed_results, session_ks)
                log_query_result(log_store, "query_completed", result)
                log_store.log("metrics", "cumulative_metrics", query_id=example.query_id, **summary_so_far)
                log_store.log(
                    "progress",
                    "query_evaluated",
                    **build_progress_payload(
                        pair=pair,
                        processed_records=processed_records,
                        total_records=len(pair_records),
                        evaluated_queries=evaluated,
                        total_queries=len(examples),
                        elapsed_seconds=time.time() - started_at,
                        query_id=example.query_id,
                    ),
                )
                save_latest_state(system, output_dir)
                if args.verbose:
                    elapsed = time.time() - started_at
                    print(f"[query] {evaluated}/{len(examples)} {example.query_id} recall_all@{max(session_ks)}={result['retrieval_metrics'][f'recall_all@{max(session_ks)}']:.1f} elapsed={elapsed:.1f}s", flush=True)
            except Exception as exc:
                failed += 1
                log_store.log("errors", "query_failed", query_id=example.query_id, error_type=type(exc).__name__, message=str(exc))
                log_store.log(
                    "progress",
                    "query_failed",
                    **build_progress_payload(
                        pair=pair,
                        processed_records=processed_records,
                        total_records=len(pair_records),
                        evaluated_queries=evaluated,
                        total_queries=len(examples),
                        elapsed_seconds=time.time() - started_at,
                        query_id=example.query_id,
                    ),
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                failure_result = {
                    "query_id": example.query_id,
                    "question": example.question,
                    "gold_session_uuids": example.gold_session_uuids,
                    "retrieved_session_uuids": [],
                    "ranked_session_uuids": [],
                    "ranked_sessions": [],
                    "retrieval_metrics": compute_retrieval_metrics(
                        retrieved_session_uuids=[],
                        gold_session_uuids=example.gold_session_uuids,
                        ks=session_ks,
                    ),
                    "qa_score": None,
                    "qa_reason": str(exc),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
                if query_snapshot is not None and query_state_path is not None:
                    failure_result["memory_state_path"] = str(query_state_path)
                    failure_result["query_snapshot_dir"] = str(query_snapshot["snapshot_dir"])
                    failure_result["query_snapshot_manifest_path"] = str(query_snapshot["manifest_path"])
                    failure_result["vector_index_path"] = str(query_snapshot["vector_index_path"])
                detailed_results.append(failure_result)
                log_query_result(log_store, "query_failed", failure_result)
                save_latest_state(system, output_dir)
                if args.fail_fast:
                    raise

        if args.max_queries is not None and evaluated >= args.max_queries:
            break
        system.ingest_record(pair.record)
        processed_records = index
        if should_save_record_state(processed_records, args.state_save_every_records):
            save_latest_state(system, output_dir)
        log_store.log(
            "progress",
            "record_ingested",
            **build_progress_payload(
                pair=pair,
                processed_records=processed_records,
                total_records=len(pair_records),
                evaluated_queries=evaluated,
                total_queries=len(examples),
                elapsed_seconds=time.time() - started_at,
            ),
        )
        if args.verbose and index % 50 == 0:
            print(f"[ingest] records={index}/{len(pair_records)} queries={evaluated}/{len(examples)}", flush=True)

    metrics_summary = summarize_metrics(detailed_results, session_ks)
    metrics_summary["failed_query_count"] = failed
    manifest = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "log_dir": str(log_store.run_dir),
        "output_dir": str(output_dir),
        "config": run_config,
        "summary": metrics_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = system.save(output_dir / "memory_state.json")
    save_latest_state(system, output_dir)
    paths = {
        "manifest": output_dir / "manifest.json",
        "state": state_path,
        "retrieval": output_dir / "retrieval_results.json",
        "generation": output_dir / "generation_results.json",
        "metrics": output_dir / "metrics_results.json",
        "report": output_dir / "realmem_top_session_report.md",
    }
    dump_json(paths["retrieval"], retrieval_results)
    dump_json(paths["generation"], generation_results)
    dump_json(paths["metrics"], {"summary": metrics_summary, "detailed_results": detailed_results})
    report = render_report(run_config=run_config, dataset_summary=dataset_summary, metrics_summary=metrics_summary)
    paths["report"].write_text(report, encoding="utf-8")
    manifest["artifacts"] = {key: str(path) for key, path in paths.items()}
    dump_json(paths["manifest"], manifest)
    log_store.log("dataset", "run_finished", output_dir=str(output_dir), evaluated_queries=evaluated, failed_queries=failed)
    log_store.log(
        "progress",
        "run_finished",
        processed_records=processed_records,
        total_records=len(pair_records),
        record_progress=round(processed_records / len(pair_records), 4) if pair_records else 0.0,
        evaluated_queries=evaluated,
        total_queries=len(examples),
        query_progress=round(evaluated / len(examples), 4) if examples else 0.0,
        failed_queries=failed,
        elapsed_seconds=round(time.time() - started_at, 2),
    )
    return {"manifest": manifest, "summary": metrics_summary}


def render_report(*, run_config: dict[str, Any], dataset_summary: dict[str, Any], metrics_summary: dict[str, Any]) -> str:
    lines = [
        "# TCMem RealMem Top-Session Report",
        "",
        "## Dataset",
        "",
        f"- person: {dataset_summary.get('person_name')}",
        f"- sessions: {dataset_summary.get('session_count')}",
        f"- queries: {dataset_summary.get('query_count')}",
        f"- records: {dataset_summary.get('record_count')}",
        "",
        "## Config",
        "",
        f"- mode: {run_config.get('mode')}",
        f"- model: {run_config.get('model')}",
        f"- embedding_model: {run_config.get('tcmem_config', {}).get('embedding_model')}",
        f"- vector_index_backend: {run_config.get('tcmem_config', {}).get('vector_index_backend')}",
        f"- retrieval_record_k: {run_config.get('retrieval_record_k')}",
        f"- session_ks: {run_config.get('session_ks')}",
        f"- with_qa: {run_config.get('with_qa')}",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TCMemConfig()
    parser = argparse.ArgumentParser(description="Run RealMemBench evaluation with top-session recall for TCMem.")
    parser.add_argument("--dataset", default="../RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json")
    parser.add_argument("--config", default="zhuo/runtime_config.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--owner-id", default="realmem")
    parser.add_argument("--session-ks", "--ks", dest="session_ks", default="5,10,20")
    parser.add_argument("--retrieval-record-k", type=int, default=None)
    parser.add_argument("--evidence-top-k", type=int, default=20)
    parser.add_argument("--embedding-model", default=defaults.embedding_model)
    parser.add_argument("--embedding-device", default=defaults.embedding_device)
    parser.add_argument("--embedding-batch-size", type=int, default=defaults.embedding_batch_size)
    parser.add_argument("--vector-index-backend", choices=["chroma", "numpy"], default=defaults.vector_index_backend)
    parser.add_argument("--graph-seed-limit", type=int, default=defaults.graph_seed_limit)
    parser.add_argument("--graph-walk-depth", type=int, default=defaults.graph_walk_depth)
    parser.add_argument("--path-a-weight", type=float, default=defaults.path_a_weight)
    parser.add_argument("--path-b-weight", type=float, default=defaults.path_b_weight)
    parser.add_argument("--prompt-path", default=defaults.prompt_path)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--with-qa", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-dir", default="result/logs")
    parser.add_argument("--state-save-every-records", type=int, default=10)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


def _session_from_dialogue(dialogue: dict[str, Any]) -> SessionPayload:
    turns = [
        DialogueTurn(
            speaker=str(turn.get("speaker", "")),
            content=str(turn.get("content", "")),
            is_query=bool(turn.get("is_query", False)),
            query_id=turn.get("query_id"),
        )
        for turn in dialogue.get("dialogue_turns", []) or []
    ]
    return SessionPayload(
        current_time=str(dialogue.get("current_time", "")),
        dialogue=turns,
        session_identifier=str(dialogue.get("session_identifier", "")),
        session_uuid=str(dialogue.get("session_uuid", "")),
    )


def _parse_session_records(session: SessionPayload) -> list[DialogueRecord]:
    records: list[DialogueRecord] = []
    turns = session.dialogue_turns
    index = 0
    base_record_time = _base_record_time(session.current_time)
    while index < len(turns):
        current = turns[index]
        record_id = _record_id(session.session_uuid, len(records))
        record_time = (base_record_time + timedelta(minutes=len(records))).strftime("%Y-%m-%d %H:%M:%S")
        if current.speaker.lower() == "user" and index + 1 < len(turns):
            next_turn = turns[index + 1]
            if next_turn.speaker.lower() == "assistant":
                records.append(
                    DialogueRecord(
                        record_id=record_id,
                        session_identifier=session.session_identifier,
                        session_uuid=session.session_uuid,
                        current_time=session.current_time,
                        record_time=record_time,
                        user_content=current.content,
                        assistant_content=next_turn.content,
                        source_turn_indexes=[index, index + 1],
                        source_turn_ids=[
                            f"{session.session_uuid}:{index}:user",
                            f"{session.session_uuid}:{index + 1}:assistant",
                        ],
                    )
                )
                index += 2
                continue
        records.append(
            DialogueRecord(
                record_id=record_id,
                session_identifier=session.session_identifier,
                session_uuid=session.session_uuid,
                current_time=session.current_time,
                record_time=record_time,
                user_content=current.content,
                source_turn_indexes=[index],
                source_turn_ids=[f"{session.session_uuid}:{index}:{current.speaker.lower()}"],
            )
        )
        index += 1
    return records


def _record_id(session_uuid: str, record_index: int) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", session_uuid).strip("_") or "session"
    return f"rec_{slug}_{record_index + 1:04d}"


def _base_record_time(current_time: str) -> datetime:
    text = str(current_time or "").strip()
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text.split(" ", 1)[0] if pattern == "%Y-%m-%d" else text, pattern)
        except ValueError:
            continue
    return datetime(1970, 1, 1, 0, 0, 0)


def _extract_json_payload(text: str) -> dict[str, Any] | list[Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        if stripped.startswith(("{", "[")):
            raise exc
        match = re.search(r"(\{.*\}|\[.*\])", stripped, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(1))


if __name__ == "__main__":
    main()
