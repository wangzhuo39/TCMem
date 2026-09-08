from __future__ import annotations

import argparse
import json
import math
import os
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
    build_model: str = "deepseek-v4-flash"
    eval_model: str = "deepseek-v4-flash"
    timeout: int = 120


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    mode: str = "tcmem_top_session"
    task_chain_enabled: bool = True
    ablation_design: dict[str, Any] = field(default_factory=dict)
    run_name_prefix: str = "tcmem_realmem_top_session"


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
    config_value = getattr(args, "config", "zhuo/runtime_config.json")
    config_path = Path(config_value)
    if not config_path.exists() and str(config_value) == "zhuo/runtime_config.json":
        fallback = Path("..") / "zhuo" / "runtime_config.json"
        if fallback.exists():
            config_path = fallback
    data: dict[str, Any] = {}
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
    model = (
        getattr(args, "model", None)
        or data.get("model")
        or os.getenv("OPENAI_MODEL")
        or "deepseek-v4-flash"
    )
    return RuntimeConfig(
        api_key=getattr(args, "api_key", None) or data.get("api_key") or os.getenv("OPENAI_API_KEY") or "",
        base_url=(
            getattr(args, "base_url", None)
            or data.get("base_url")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        ),
        model=model,
        build_model=(
            getattr(args, "build_model", None)
            or data.get("build_model")
            or data.get("graph_model")
            or os.getenv("OPENAI_BUILD_MODEL")
            or model
        ),
        eval_model=(
            getattr(args, "eval_model", None)
            or data.get("eval_model")
            or data.get("qa_model")
            or os.getenv("OPENAI_EVAL_MODEL")
            or model
        ),
        timeout=getattr(args, "timeout", None) or int(data.get("timeout", os.getenv("OPENAI_TIMEOUT", 120)) or 120),
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
        llm_model=runtime.build_model,
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
            # RealMemBench defines answer evidence by the immediate assistant
            # turn's memory_used list.  memory_session_uuids is only a fallback
            # for legacy examples that lack item-level evidence.
            gold_uuids = [str(item.get("session_uuid")) for item in memory_used if item.get("session_uuid")]
            if not gold_uuids:
                gold_uuids = [str(uuid) for uuid in (next_turn.get("memory_session_uuids") or []) if str(uuid)]
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


def log_progress(log_store: ModuleLogStore, output_dir: Path, event: str, **payload: Any) -> None:
    primitive_payload = to_primitive(payload)
    log_store.log("progress", event, **primitive_payload)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": "progress",
        "event": event,
        "payload": primitive_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    dump_json(output_dir / "progress_latest.json", entry)


def save_latest_state(system: MemorySystem, output_dir: Path) -> Path:
    return system.save(output_dir / "memory_state_latest.json")


def preflight_llm_clients(*clients: OpenAICompatibleLLMClient) -> None:
    """Fail before a long run if a configured model cannot serve a tiny request."""
    checked: set[tuple[str, str]] = set()
    for client in clients:
        # Test doubles and offline adapters may intentionally implement no
        # network method; the real OpenAI-compatible client always has one.
        if not callable(getattr(client, "generate", None)):
            continue
        base_url = str(getattr(getattr(client, "client", None), "base_url", ""))
        key = (base_url, str(getattr(client, "model", "")))
        if key in checked:
            continue
        client.generate(
            "Reply with OK.",
            system_prompt="This is a runtime availability check.",
            temperature=0.0,
            max_tokens=1,
        )
        checked.add(key)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def resolve_resume_results_dir(resume_from: str | Path) -> tuple[Path, Path]:
    source = Path(resume_from)
    if source.is_file():
        return source.parent, source
    if (source / "memory_state_latest.json").exists():
        return source, source / "memory_state_latest.json"
    if (source / "results" / "memory_state_latest.json").exists():
        return source / "results", source / "results" / "memory_state_latest.json"
    raise ValueError(f"No memory_state_latest.json found under resume path: {source}")


def load_resume_query_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load the latest completed result per query from artifacts or append-only logs."""
    by_query_id: dict[str, dict[str, Any]] = {}
    metrics = _read_json_object(results_dir / "metrics_results.json")
    for result in metrics.get("detailed_results", []) or []:
        if isinstance(result, dict) and result.get("query_id"):
            by_query_id[str(result["query_id"])] = result

    candidates: list[Path] = []
    manifest = _read_json_object(results_dir / "manifest.json")
    log_dir_text = str(manifest.get("log_dir") or "").strip()
    if log_dir_text:
        candidates.append(Path(log_dir_text) / "query_results.jsonl")
    candidates.extend(results_dir.parent.glob("logs/**/query_results.jsonl"))
    candidates.extend(results_dir.glob("**/query_results.jsonl"))
    seen_paths: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            payload = entry.get("payload") or {}
            if entry.get("event") == "query_completed" and isinstance(payload, dict) and payload.get("query_id"):
                by_query_id[str(payload["query_id"])] = payload
    return list(by_query_id.values())


def validate_resume_prefix(
    *,
    pair_records: list[PairRecord],
    state: dict[str, Any],
    completed_results: list[dict[str, Any]],
) -> int:
    graph_records = (state.get("graph") or {}).get("records") or []
    state_record_ids = [str(record.get("record_id") or "") for record in graph_records if isinstance(record, dict)]
    expected_ids = [pair.record.record_id for pair in pair_records]
    if len(state_record_ids) > len(expected_ids) or state_record_ids != expected_ids[: len(state_record_ids)]:
        raise ValueError("Resume state is not an exact prefix of the requested dataset records")

    query_prior_record_counts = {
        pair.query_example.query_id: index - 1
        for index, pair in enumerate(pair_records, start=1)
        if pair.query_example is not None
    }
    for result in completed_results:
        query_id = str(result.get("query_id") or "")
        prior_count = query_prior_record_counts.get(query_id)
        if prior_count is None:
            raise ValueError(f"Resume result query is absent from requested dataset: {query_id}")
        if prior_count > len(state_record_ids):
            raise ValueError(f"Resume result {query_id} is ahead of the saved memory state")
    return len(state_record_ids)


def checkpoint_result_artifacts(
    *,
    output_dir: Path,
    retrieval_results: dict[str, Any],
    generation_results: dict[str, Any],
    session_task_chain_results: dict[str, Any],
    detailed_results: list[dict[str, Any]],
    session_ks: list[int],
    status: str = "running",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_metrics(detailed_results, session_ks)
    dump_json(output_dir / "retrieval_results.json", retrieval_results)
    dump_json(output_dir / "realmem_official_retrieval_results.json", realmem_official_results_from_detailed(detailed_results))
    dump_json(output_dir / "generation_results.json", generation_results)
    dump_json(output_dir / "session_task_chain_trace.json", session_task_chain_results)
    dump_json(
        output_dir / "metrics_results.json",
        {"status": status, "summary": summary, "detailed_results": detailed_results},
    )


def write_invalid_run_artifacts(
    *,
    output_dir: Path,
    log_store: ModuleLogStore,
    run_name: str,
    dataset_path: Path,
    run_config: dict[str, Any],
    summary: dict[str, Any],
    processed_records: int,
    total_records: int,
    evaluated_queries: int,
    total_queries: int,
    error: BaseException,
    error_stage: str,
) -> dict[str, Any]:
    """Persist an invalid-run marker without flushing or saving MemorySystem state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    error_payload = {
        "stage": error_stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "processed_records": processed_records,
        "total_records": total_records,
        "evaluated_queries": evaluated_queries,
        "total_queries": total_queries,
    }
    paths = {
        "manifest": output_dir / "manifest.json",
        "retrieval": output_dir / "retrieval_results.json",
        "generation": output_dir / "generation_results.json",
        "session_task_chain_trace": output_dir / "session_task_chain_trace.json",
        "metrics": output_dir / "metrics_results.json",
        "report": output_dir / "realmem_top_session_report.md",
        "progress": output_dir / "progress.jsonl",
        "progress_latest": output_dir / "progress_latest.json",
    }
    # Successful-query artifacts are checkpointed eagerly.  Never destroy them
    # merely because a later record or API request interrupted the run.
    for key in ("retrieval", "generation", "session_task_chain_trace"):
        if not paths[key].exists():
            dump_json(paths[key], {})
    existing_metrics = _read_json_object(paths["metrics"])
    dump_json(
        paths["metrics"],
        {
            "status": "invalid",
            "summary": summary,
            "error": error_payload,
            "detailed_results": existing_metrics.get("detailed_results", []),
        },
    )
    paths["report"].write_text(
        render_report(
            run_config=run_config,
            dataset_summary={
                "record_count": total_records,
                "query_count": total_queries,
                "session_count": 0,
                "person_name": "",
            },
            metrics_summary={"status": "invalid", **summary},
        )
        + "\n\nRun status: invalid\n"
        + f"Error stage: {error_stage}\n"
        + f"Error type: {type(error).__name__}\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "status": "invalid",
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "log_dir": str(log_store.run_dir),
        "output_dir": str(output_dir),
        "config": run_config,
        "summary": summary,
        "error": error_payload,
        "processed_records": processed_records,
        "evaluated_queries": evaluated_queries,
        "artifacts": {key: str(path) for key, path in paths.items()},
    }
    dump_json(paths["manifest"], manifest)
    log_store.log("dataset", "run_invalid", **error_payload, output_dir=str(output_dir))
    log_progress(
        log_store,
        output_dir,
        "run_invalid",
        processed_records=processed_records,
        total_records=total_records,
        evaluated_queries=evaluated_queries,
        total_queries=total_queries,
        error_stage=error_stage,
        error_type=type(error).__name__,
        message=str(error),
    )
    return {"status": "invalid", "manifest": manifest, "summary": summary}


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


def should_print_progress(current: int, total: int, every_records: int) -> bool:
    if current <= 0:
        return False
    if current == 1:
        return True
    if total > 0 and current >= total:
        return True
    return every_records > 0 and current % every_records == 0


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


def ranked_sessions_from_traces(
    traces: list[dict[str, Any]],
    *,
    task_chain_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate record traces into session scores for the active retrieval mode.

    Full mode keeps the strongest task-chain-aware record primary while adding
    a small generic Path-B tie-break.  Sessions with no chain evidence remain
    available through a lower-weight generic fallback.  The ablated mode uses
    the top-three generic record evidence sum, matching the frozen baseline.
    """
    sessions: dict[str, dict[str, Any]] = {}
    for rank, trace in enumerate(traces, start=1):
        session_uuid = str(trace.get("source_session_uuid") or "").strip()
        if not session_uuid:
            continue
        record_id = str(trace.get("source_record_id") or trace.get("item_id") or "").strip()
        try:
            score = float(trace.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        generic_value = trace.get("generic_score")
        try:
            generic_score = score if generic_value is None else float(generic_value or 0.0)
        except (TypeError, ValueError):
            generic_score = 0.0
        route_role = str(trace.get("route_role") or "").strip()
        reason = str(trace.get("reason") or "")
        has_chain_evidence = bool(trace.get("task_chain_evidence")) or route_role in {
            "primary",
            "expanded",
        } or "path_a" in reason
        session = sessions.get(session_uuid)
        if session is None:
            session = {
                "session_uuid": session_uuid,
                "score": 0.0,
                "first_record_rank": rank,
                "representative_record_id": record_id,
                "record_count": 0,
                "record_ids": [],
                "record_scores": [],
                "full_max_score": 0.0,
                "generic_max_score": 0.0,
                "task_chain_evidence": False,
            }
            sessions[session_uuid] = session
        session["record_count"] = int(session["record_count"]) + 1
        session["first_record_rank"] = min(int(session["first_record_rank"]), rank)
        if record_id and record_id not in session["record_ids"]:
            session["record_ids"].append(record_id)
        session["record_scores"].append(score)
        session["full_max_score"] = max(float(session["full_max_score"]), score)
        session["generic_max_score"] = max(float(session["generic_max_score"]), generic_score)
        session["task_chain_evidence"] = bool(session["task_chain_evidence"] or has_chain_evidence)

    ranked = list(sessions.values())
    for session in ranked:
        if task_chain_enabled:
            if session["task_chain_evidence"]:
                session["score"] = float(session["full_max_score"]) + 0.05 * float(session["generic_max_score"])
            else:
                session["score"] = 0.5 * float(session["generic_max_score"])
        else:
            values = sorted((float(value) for value in session["record_scores"]), reverse=True)
            session["score"] = sum(weight * value for weight, value in zip((1.0, 0.75, 0.5), values))
        # This is an implementation detail for the aggregation formula; the
        # record-level scores remain available through record_count/record_ids.
        session.pop("record_scores", None)
    ranked.sort(key=lambda item: (-float(item["score"]), int(item["first_record_rank"]), item["session_uuid"]))
    return ranked


def _status_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _task_chain_root(manager: Any, task_id: str) -> tuple[str | None, str, list[str]]:
    """Resolve a task's root while retaining corruption diagnostics."""
    tasks = getattr(manager, "tasks", {}) or {}
    current_id = str(task_id or "").strip()
    visited: list[str] = []
    seen: set[str] = set()
    while current_id:
        if current_id in seen:
            return None, "cycle", visited
        seen.add(current_id)
        visited.append(current_id)
        task = tasks.get(current_id)
        if task is None:
            return None, "missing_task", visited
        parent_id = str(getattr(task, "parent_task_id", None) or "").strip()
        if not parent_id:
            return current_id, "ok", visited
        if parent_id not in tasks:
            return None, "missing_parent", visited
        current_id = parent_id
    return None, "missing_task", visited


def _build_task_chain_record_index(system: MemorySystem) -> dict[str, list[dict[str, Any]]]:
    """Index task/branch/node ownership by session UUID for query diagnostics."""
    graph = getattr(system, "graph", None)
    records = getattr(graph, "records", {}) or {}
    manager = getattr(system, "task_manager", None)
    tasks = getattr(manager, "tasks", {}) or {}
    index: dict[str, list[dict[str, Any]]] = {}
    for raw_task_id, task in tasks.items():
        task_id = str(raw_task_id or getattr(task, "task_id", "")).strip()
        if not task_id:
            continue
        record_ids: list[str] = []
        for record_id in getattr(task, "record_ids", []) or []:
            record_ids.append(str(record_id))
        nodes = getattr(task, "nodes", {}) or {}
        for node in nodes.values():
            source_record_id = str(getattr(node, "source_record_id", "") or "").strip()
            if source_record_id:
                record_ids.append(source_record_id)
        task_record_ids = _unique(record_ids)
        records_by_session: dict[str, list[str]] = {}
        node_ids_by_record: dict[str, list[str]] = {}
        branch_ids_by_record: dict[str, list[str]] = {}
        for record_id in task_record_ids:
            record = records.get(record_id)
            if record is None:
                continue
            session_uuid = str(getattr(record, "session_uuid", "") or "").strip()
            if not session_uuid:
                continue
            records_by_session.setdefault(session_uuid, []).append(record_id)
            matching_nodes = [
                node
                for node in nodes.values()
                if str(getattr(node, "source_record_id", "") or "").strip() == record_id
            ]
            node_ids_by_record[record_id] = [
                str(getattr(node, "node_id", "") or "").strip()
                for node in matching_nodes
                if str(getattr(node, "node_id", "") or "").strip()
            ]
            branch_ids_by_record[record_id] = _unique(
                [str(getattr(node, "branch_id", "") or "main").strip() for node in matching_nodes]
            )
        root_id, chain_integrity, ancestry = _task_chain_root(manager, task_id)
        branches = getattr(task, "branches", {}) or {}
        task_description = str(
            getattr(task, "canonical_description", "")
            or getattr(task, "task_description", "")
            or ""
        ).strip()
        parent_task_id = str(getattr(task, "parent_task_id", None) or "").strip() or None
        parent_branch_id = str(getattr(task, "parent_branch_id", None) or "").strip() or None
        for session_uuid, session_record_ids in records_by_session.items():
            branch_ids = _unique(
                branch_id
                for record_id in session_record_ids
                for branch_id in branch_ids_by_record.get(record_id, [])
            )
            branch_details: list[dict[str, Any]] = []
            for branch_id in branch_ids:
                branch = branches.get(branch_id)
                branch_details.append(
                    {
                        "branch_id": branch_id,
                        "branch_goal": _short(getattr(branch, "branch_goal", "") if branch else "", 260),
                        "current_focus": _short(getattr(branch, "current_focus", "") if branch else "", 260),
                        "status": _status_value(getattr(branch, "status", "") if branch else ""),
                        "head_node_id": getattr(branch, "head_node_id", None) if branch else None,
                        "parent_node_id": getattr(branch, "parent_node_id", None) if branch else None,
                        "merged_into_branch_id": getattr(branch, "merged_into_branch_id", None) if branch else None,
                    }
                )
            index.setdefault(session_uuid, []).append(
                {
                    "task_id": task_id,
                    "task_role": "child" if parent_task_id else "root",
                    "task_description": _short(task_description, 360),
                    "task_status": _status_value(getattr(task, "status", "")),
                    "task_current_focus": _short(getattr(task, "current_focus", ""), 260),
                    "parent_task_id": parent_task_id,
                    "parent_branch_id": parent_branch_id,
                    "chain_root_task_id": root_id,
                    "chain_integrity": chain_integrity,
                    "ancestry_task_ids": ancestry,
                    "record_ids": _unique(session_record_ids),
                    "node_ids": _unique(
                        node_id
                        for record_id in session_record_ids
                        for node_id in node_ids_by_record.get(record_id, [])
                    ),
                    "branch_ids": branch_ids,
                    "branches": branch_details,
                }
            )
    for memberships in index.values():
        memberships.sort(key=lambda item: (str(item.get("chain_root_task_id") or ""), str(item.get("task_id") or "")))
    return index


def build_session_task_chain_trace(
    *,
    system: MemorySystem,
    gold_session_uuids: list[str],
    memory_used: list[dict[str, Any]],
    ranked_sessions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    routed_task_ids: list[str] | None = None,
    expanded_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build a compact, query-level mapping from evidence sessions to task chains."""
    gold = _unique([str(value or "").strip() for value in gold_session_uuids])
    gold_counts: dict[str, int] = {}
    gold_memory_entries: list[dict[str, Any]] = []
    for memory_index, item in enumerate(memory_used):
        if not isinstance(item, dict):
            continue
        session_uuid = str(item.get("session_uuid") or "").strip()
        if not session_uuid:
            continue
        gold_counts[session_uuid] = gold_counts.get(session_uuid, 0) + 1
        gold_memory_entries.append(
            {
                "memory_index": memory_index,
                "session_uuid": session_uuid,
                "content_excerpt": _short(item.get("content", ""), 220),
            }
        )

    traces_by_session: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        session_uuid = str(trace.get("source_session_uuid") or "").strip()
        if session_uuid:
            traces_by_session.setdefault(session_uuid, []).append(trace)
    ranked_by_session: dict[str, tuple[int, dict[str, Any]]] = {}
    for rank, item in enumerate(ranked_sessions, start=1):
        session_uuid = str(item.get("session_uuid") or "").strip()
        if session_uuid and session_uuid not in ranked_by_session:
            ranked_by_session[session_uuid] = (rank, item)

    routed_tasks = set(_unique([str(task_id or "").strip() for task_id in (routed_task_ids or [])]))
    expanded_tasks = set(_unique([str(task_id or "").strip() for task_id in (expanded_task_ids or [])]))

    session_uuids = _unique(gold + [str(item.get("session_uuid") or "").strip() for item in ranked_sessions])
    record_index = _build_task_chain_record_index(system)
    session_entries: list[dict[str, Any]] = []
    for session_uuid in session_uuids:
        ranked = ranked_by_session.get(session_uuid)
        session_traces = traces_by_session.get(session_uuid, [])
        memberships = []
        for membership in record_index.get(session_uuid, []):
            item = dict(membership)
            task_id = str(item.get("task_id") or "")
            route_traces = [trace for trace in session_traces if str(trace.get("task_id") or "") == task_id]
            item["route_evidence"] = {
                "hit_count": len(route_traces),
                "task_routed": task_id in routed_tasks,
                "task_expanded": task_id in expanded_tasks,
                "record_ids": _unique(str(trace.get("source_record_id") or "").strip() for trace in route_traces),
                "chain_node_ids": _unique(str(trace.get("chain_node_id") or "").strip() for trace in route_traces),
                "route_roles": _unique(str(trace.get("route_role") or "").strip() for trace in route_traces),
                "route_relations": _unique(str(trace.get("route_relation") or "").strip() for trace in route_traces),
                "route_depths": sorted({int(trace.get("route_depth") or 0) for trace in route_traces}),
                "reasons": _unique(str(trace.get("reason") or "").strip() for trace in route_traces),
            }
            memberships.append(item)
        root_ids = _unique(
            str(item.get("chain_root_task_id") or "").strip()
            for item in memberships
            if item.get("chain_root_task_id")
        )
        session_entries.append(
            {
                "session_uuid": session_uuid,
                "is_gold": session_uuid in set(gold),
                "gold_memory_used_count": gold_counts.get(session_uuid, 0),
                "retrieved_rank": ranked[0] if ranked else None,
                "retrieved_score": float(ranked[1].get("score") or 0.0) if ranked else None,
                "retrieved_record_ids": list(ranked[1].get("record_ids") or []) if ranked else [],
                "retrieved_record_count": int(ranked[1].get("record_count") or 0) if ranked else 0,
                "retrieval_hit_count": len(session_traces),
                "task_chain_memberships": memberships,
                "chain_root_task_ids": root_ids,
                "in_any_task_chain": bool(memberships),
            }
        )

    gold_entries = [item for item in session_entries if item["is_gold"]]
    root_sets = [set(item["chain_root_task_ids"]) for item in gold_entries]
    common_roots = sorted(set.intersection(*root_sets)) if root_sets else []
    union_roots = sorted(set().union(*root_sets)) if root_sets else []
    chain_groups = [
        {
            "chain_root_task_id": root_id,
            "gold_session_uuids": [item["session_uuid"] for item in gold_entries if root_id in item["chain_root_task_ids"]],
        }
        for root_id in union_roots
    ]
    summary = {
        "gold_session_count": len(gold_entries),
        "gold_sessions_retrieved_count": sum(1 for item in gold_entries if item["retrieved_rank"] is not None),
        "gold_sessions_in_any_task_chain_count": sum(1 for item in gold_entries if item["in_any_task_chain"]),
        "gold_sessions_with_valid_root_count": sum(1 for item in gold_entries if item["chain_root_task_ids"]),
        "gold_sessions_without_task_chain": [item["session_uuid"] for item in gold_entries if not item["in_any_task_chain"]],
        "gold_sessions_without_valid_root": [item["session_uuid"] for item in gold_entries if not item["chain_root_task_ids"]],
        "gold_chain_root_task_ids": union_roots,
        "gold_sessions_common_root_task_ids": common_roots,
        "gold_sessions_share_common_root_chain": bool(gold_entries) and bool(common_roots),
        "gold_sessions_share_one_root_chain": bool(gold_entries) and len(common_roots) == 1,
        "gold_chain_groups": chain_groups,
        "retrieved_session_count": len(ranked_sessions),
        "retrieved_sessions_in_any_task_chain_count": sum(1 for item in session_entries if item["retrieved_rank"] and item["in_any_task_chain"]),
    }
    return {
        "gold_memory_used": gold_memory_entries,
        "sessions": session_entries,
        "summary": summary,
    }


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
        "depth": hit.depth,
        "bm25_score": hit.bm25_score,
        "route_role": hit.route_role,
        "route_relation": hit.route_relation,
        "route_depth": hit.route_depth,
        "generic_score": hit.generic_score,
        "task_chain_evidence": hit.task_chain_evidence,
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


def realmem_official_results_from_detailed(detailed_results: list[dict[str, Any]]) -> dict[str, Any]:
    official: dict[str, Any] = {}
    for result in detailed_results:
        question = str(result.get("question") or "").strip()
        if not question:
            continue
        retrieval_result = result.get("retrieval_result") if isinstance(result.get("retrieval_result"), dict) else {}
        ranked_items = retrieval_result.get("ranked_items") or []
        official_items: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for rank, item in enumerate([item for item in ranked_items if isinstance(item, dict)], start=1):
            chunk_id = str(item.get("source_session_identifier") or "").strip()
            if not chunk_id:
                continue
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            official_items.append(
                {
                    "res_type": "chunk",
                    "chunk_id": chunk_id,
                    "content": str(item.get("content_excerpt") or "").strip(),
                    "rank": rank,
                    "score": float(item.get("score") or 0.0),
                    "source_record_id": str(item.get("source_record_id") or item.get("item_id") or "").strip(),
                    "source_session_uuid": str(item.get("source_session_uuid") or "").strip(),
                }
            )
        official[question] = {"question": question, "ranked_items": official_items}
    return official


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
    task_chain_enabled = bool(getattr(getattr(system, "config", None), "task_chain_enabled", True))
    ranked_sessions = ranked_sessions_from_traces(
        traces,
        task_chain_enabled=task_chain_enabled,
    )
    ranked_session_uuids = [str(item["session_uuid"]) for item in ranked_sessions]
    session_task_chain_trace = build_session_task_chain_trace(
        system=system,
        gold_session_uuids=example.gold_session_uuids,
        memory_used=example.memory_used,
        ranked_sessions=ranked_sessions,
        traces=traces,
        routed_task_ids=retrieval.routed_task_ids,
        expanded_task_ids=retrieval.expanded_task_ids,
    )
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
        "gold_memory_used": session_task_chain_trace["gold_memory_used"],
        "session_task_chain_trace": session_task_chain_trace["sessions"],
        "gold_session_task_chain_summary": session_task_chain_trace["summary"],
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
            "expanded_task_ids": retrieval.expanded_task_ids,
            "expansion_edges": retrieval.expansion_edges,
            "query_intent": to_primitive(retrieval.query_intent) if retrieval.query_intent else None,
            "query_route_reason": retrieval.query_route_reason,
            "ranked_items": traces,
            "retrieved_session_uuids": ranked_session_uuids,
            "ranked_session_uuids": ranked_session_uuids,
            "ranked_sessions": ranked_sessions,
            "gold_session_uuids": example.gold_session_uuids,
            "gold_memory_used": session_task_chain_trace["gold_memory_used"],
            "session_task_chain_trace": session_task_chain_trace["sessions"],
            "gold_session_task_chain_summary": session_task_chain_trace["summary"],
            "retrieval_record_k": retrieval_record_k,
        },
    }
    if log_store is not None:
        log_store.log(
            "session_task_chain",
            "query_session_trace",
            query_id=example.query_id,
            question=example.question,
            gold_memory_used=session_task_chain_trace["gold_memory_used"],
            session_task_chain_trace=session_task_chain_trace["sessions"],
            gold_session_task_chain_summary=session_task_chain_trace["summary"],
        )
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
        memory_result = judge_memory_metrics(
            client,
            question=example.question,
            groundtruth_memory=example.gold_memory_text,
            retrieved_memory=evidence_text,
            log_store=log_store,
            query_id=example.query_id,
            prompt_registry=prompt_registry,
        )
        result["Mem_recall"] = memory_result["Mem_recall"]
        result["Mem_helpful_score"] = memory_result["Mem_helpful_score"]
        result["Mem_hits"] = memory_result["Mem_hits"]
        result["Mem_helpful_reason"] = memory_result["Mem_helpful_reason"]
        result["generation_result"] = {
            "query_id": example.query_id,
            "question": example.question,
            "generated_answer": generated_answer,
            "evidence_used": evidence_text,
            "ranked_sessions": ranked_sessions[:evidence_top_k],
            "evidence_session_uuids": [item["session_uuid"] for item in ranked_sessions[:evidence_top_k]],
            "model": qa_model_name,
            "Mem_recall": memory_result["Mem_recall"],
            "Mem_helpful_score": memory_result["Mem_helpful_score"],
            "Mem_hits": memory_result["Mem_hits"],
            "Mem_helpful_reason": memory_result["Mem_helpful_reason"],
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


def judge_memory_metrics(
    client: OpenAICompatibleLLMClient,
    *,
    question: str,
    groundtruth_memory: str,
    retrieved_memory: str,
    log_store: ModuleLogStore | None = None,
    query_id: str = "",
    prompt_registry: PromptRegistry | None = None,
) -> dict[str, Any]:
    registry = prompt_registry or PromptRegistry.default()
    rendered = registry.render(
        "realmem_memory_judge",
        question=question,
        groundtruth_memory=groundtruth_memory,
        retrieved_memory=retrieved_memory,
    )
    parsed = _generate_json_with_retries(
        client,
        rendered.user_prompt,
        system_prompt=rendered.system_prompt,
        temperature=0.0,
        max_tokens=1200,
        stage="memory_judge",
        log_store=log_store,
        context={"query_id": query_id},
    )
    if not isinstance(parsed, dict):
        raise RuntimeError("Memory judge returned non-object JSON")
    mem_recall = float(parsed.get("Mem_recall", -1.0))
    if mem_recall < 0.0 or mem_recall > 1.0:
        raise ValueError(f"Mem_recall must be between 0 and 1; got {mem_recall!r}")
    helpful_score = int(parsed.get("Mem_helpful_score", -1))
    if helpful_score not in {0, 1, 2}:
        raise ValueError(f"Mem_helpful_score must be 0, 1, or 2; got {helpful_score!r}")
    hits = parsed.get("Mem_hits") if isinstance(parsed.get("Mem_hits"), list) else []
    return {
        "Mem_recall": mem_recall,
        "Mem_helpful_score": helpful_score,
        "Mem_hits": [str(item) for item in hits],
        "Mem_helpful_reason": str(parsed.get("Mem_helpful_reason", "") or ""),
    }


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
    mem_recalls = [
        float(item["Mem_recall"])
        for item in detailed_results
        if isinstance(item.get("Mem_recall"), (int, float))
    ]
    mem_helpful_scores = [
        int(item["Mem_helpful_score"])
        for item in detailed_results
        if isinstance(item.get("Mem_helpful_score"), int)
    ]
    summary["average_mem_recall"] = round(sum(mem_recalls) / len(mem_recalls), 4) if mem_recalls else None
    summary["average_mem_helpful_score"] = (
        round(sum(mem_helpful_scores) / len(mem_helpful_scores), 4) if mem_helpful_scores else None
    )
    summary["mem_helpful_score_distribution"] = {
        str(score): mem_helpful_scores.count(score) for score in range(3)
    }
    summary["query_count"] = len(detailed_results)
    summary["qa_failed_count"] = sum(1 for item in detailed_results if item.get("qa_score") is None)
    summary["mem_failed_count"] = sum(1 for item in detailed_results if item.get("mem_error"))
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

    run_name = args.run_name or f"{options.run_name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) if args.output_dir else Path("result/results") / run_name
    log_store = ModuleLogStore(base_dir=args.log_dir, run_name=run_name)
    runtime = resolve_runtime_config(args)
    if not runtime.api_key:
        raise SystemExit("Missing API key for TCMem RealMem evaluation.")
    build_client = OpenAICompatibleLLMClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        model=runtime.build_model,
        timeout=runtime.timeout,
    )
    eval_client = OpenAICompatibleLLMClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        model=runtime.eval_model,
        timeout=runtime.timeout,
    )
    if not getattr(args, "skip_llm_preflight", False):
        preflight_llm_clients(build_client, eval_client)
    config = build_tcmem_config(
        args,
        output_dir=output_dir,
        log_store_base_dir=log_store.base_dir,
        runtime=runtime,
        task_chain_enabled=options.task_chain_enabled,
    )
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
        "build_model": runtime.build_model,
        "eval_model": runtime.eval_model,
        "session_ks": session_ks,
        "retrieval_record_k": retrieval_record_k,
        "evidence_top_k": args.evidence_top_k,
        "with_qa": args.with_qa,
        "max_queries": args.max_queries,
        "mode": options.mode,
        "tcmem_config": config.to_dict(),
    }
    if options.ablation_design:
        run_config["ablation_design"] = options.ablation_design

    resume_from = getattr(args, "resume_from", None)
    completed_results: list[dict[str, Any]] = []
    processed_records = 0
    if resume_from:
        resume_results_dir, resume_state_path = resolve_resume_results_dir(resume_from)
        resume_state = _read_json_object(resume_state_path)
        completed_results = load_resume_query_results(resume_results_dir)
        processed_records = validate_resume_prefix(
            pair_records=pair_records,
            state=resume_state,
            completed_results=completed_results,
        )
        source_manifest = _read_json_object(resume_results_dir / "manifest.json")
        source_config = source_manifest.get("config") or {}
        for key in ("dataset", "mode", "session_ks", "retrieval_record_k", "evidence_top_k", "with_qa"):
            if key in source_config and source_config.get(key) != run_config.get(key):
                raise ValueError(
                    f"Resume config mismatch for {key}: source={source_config.get(key)!r}, current={run_config.get(key)!r}"
                )
        system = MemorySystem.load(resume_state_path, config=config, llm_client=build_client)
        system.log_store = log_store
        system.task_manager.log_store = log_store
        system.retrieval.sync_record_index()
        run_config["resume"] = {
            "source": str(resume_results_dir),
            "state_path": str(resume_state_path),
            "restored_records": processed_records,
            "restored_queries": len(completed_results),
        }
    else:
        system = MemorySystem(config=config, llm_client=build_client, log_store=log_store)
    log_store.log("dataset", "dataset_loaded", **dataset_summary)
    log_store.log("dataset", "run_config", **run_config)

    detailed_results = list(completed_results)
    retrieval_results = {
        str(result["query_id"]): result["retrieval_result"]
        for result in detailed_results
        if result.get("query_id") and isinstance(result.get("retrieval_result"), dict)
    }
    generation_results = {
        str(result["query_id"]): result["generation_result"]
        for result in detailed_results
        if result.get("query_id") and isinstance(result.get("generation_result"), dict)
    }
    session_task_chain_results = {
        str(result["query_id"]): {
            "query_id": result["query_id"],
            "question": result.get("question", ""),
            "gold_memory_used": result.get("gold_memory_used", []),
            "session_task_chain_trace": result.get("session_task_chain_trace", []),
            "gold_session_task_chain_summary": result.get("gold_session_task_chain_summary", {}),
        }
        for result in detailed_results
        if result.get("query_id")
    }
    completed_query_ids = {str(result.get("query_id")) for result in detailed_results if result.get("query_id")}
    evaluated = len(detailed_results)
    failed = sum(1 for result in detailed_results if result.get("error_type"))
    started_at = time.time()
    current_session_uuid = ""
    if resume_from:
        checkpoint_result_artifacts(
            output_dir=output_dir,
            retrieval_results=retrieval_results,
            generation_results=generation_results,
            session_task_chain_results=session_task_chain_results,
            detailed_results=detailed_results,
            session_ks=session_ks,
        )
    log_progress(
        log_store,
        output_dir,
        "run_resumed" if resume_from else "run_started",
        processed_records=processed_records,
        total_records=len(pair_records),
        record_progress=0.0,
        evaluated_queries=evaluated,
        total_queries=len(examples),
        query_progress=0.0,
        elapsed_seconds=0.0,
    )
    if args.verbose:
        print(
            f"[run] mode={options.mode} records={len(pair_records)} queries={len(examples)} "
            f"retrieval_record_k={retrieval_record_k} evidence_top_k={args.evidence_top_k} "
            f"output_dir={output_dir}",
            flush=True,
        )
    for index, pair in enumerate(pair_records, start=1):
        if index <= processed_records:
            continue
        if pair.record.session_uuid != current_session_uuid:
            current_session_uuid = pair.record.session_uuid
            log_progress(
                log_store,
                output_dir,
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
            if args.verbose:
                print(
                    f"[session] {pair.session_index + 1}/{len(dataset.get('dialogues', []) or [])} "
                    f"session_uuid={pair.record.session_uuid} record_index={index}/{len(pair_records)}",
                    flush=True,
                )

        example = pair.query_example
        if (
            example is not None
            and example.query_id in example_ids
            and example.query_id not in completed_query_ids
        ):
            if args.verbose:
                print(
                    f"[query-start] {evaluated + 1}/{len(examples)} {example.query_id} "
                    f"record_index={index}/{len(pair_records)} elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )
            log_progress(
                log_store,
                output_dir,
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
            query_snapshot_completed = False
            query_evaluation_completed = False
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
                        "build_model": runtime.build_model,
                        "eval_model": runtime.eval_model,
                        "base_url": runtime.base_url,
                        "processed_records": processed_records,
                        "evaluated_queries": evaluated,
                        "tcmem_config": config.to_dict(),
                    },
                )
                query_state_path = Path(query_snapshot["memory_state_path"])
                query_snapshot_completed = True
                log_progress(
                    log_store,
                    output_dir,
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
                    client=eval_client,
                    qa_model_name=runtime.eval_model,
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
                session_task_chain_results[example.query_id] = {
                    "query_id": example.query_id,
                    "question": example.question,
                    "gold_memory_used": result.get("gold_memory_used", []),
                    "session_task_chain_trace": result.get("session_task_chain_trace", []),
                    "gold_session_task_chain_summary": result.get("gold_session_task_chain_summary", {}),
                }
                if "generation_result" in result:
                    generation_results[example.query_id] = result["generation_result"]
                evaluated += 1
                summary_so_far = summarize_metrics(detailed_results, session_ks)
                log_query_result(log_store, "query_completed", result)
                log_store.log("metrics", "cumulative_metrics", query_id=example.query_id, **summary_so_far)
                log_progress(
                    log_store,
                    output_dir,
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
                query_evaluation_completed = True
                checkpoint_result_artifacts(
                    output_dir=output_dir,
                    retrieval_results=retrieval_results,
                    generation_results=generation_results,
                    session_task_chain_results=session_task_chain_results,
                    detailed_results=detailed_results,
                    session_ks=session_ks,
                )
                save_latest_state(system, output_dir)
                if args.verbose:
                    elapsed = time.time() - started_at
                    print(f"[query] {evaluated}/{len(examples)} {example.query_id} recall_all@{max(session_ks)}={result['retrieval_metrics'][f'recall_all@{max(session_ks)}']:.1f} elapsed={elapsed:.1f}s", flush=True)
            except Exception as exc:
                if not query_snapshot_completed:
                    summary = summarize_metrics(detailed_results, session_ks)
                    summary["failed_query_count"] = failed
                    return write_invalid_run_artifacts(
                        output_dir=output_dir,
                        log_store=log_store,
                        run_name=run_name,
                        dataset_path=dataset_path,
                        run_config=run_config,
                        summary=summary,
                        processed_records=processed_records,
                        total_records=len(pair_records),
                        evaluated_queries=evaluated,
                        total_queries=len(examples),
                        error=exc,
                        error_stage="query_snapshot",
                    )
                failed += 1
                failed_trace = build_session_task_chain_trace(
                    system=system,
                    gold_session_uuids=example.gold_session_uuids,
                    memory_used=example.memory_used,
                    ranked_sessions=[],
                    traces=[],
                )
                log_store.log("errors", "query_failed", query_id=example.query_id, error_type=type(exc).__name__, message=str(exc))
                log_progress(
                    log_store,
                    output_dir,
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
                    "gold_memory_used": failed_trace["gold_memory_used"],
                    "session_task_chain_trace": failed_trace["sessions"],
                    "gold_session_task_chain_summary": failed_trace["summary"],
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
                session_task_chain_results[example.query_id] = {
                    "query_id": example.query_id,
                    "question": example.question,
                    "gold_memory_used": failed_trace["gold_memory_used"],
                    "session_task_chain_trace": failed_trace["sessions"],
                    "gold_session_task_chain_summary": failed_trace["summary"],
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
                if args.fail_fast:
                    raise

        if args.max_queries is not None and evaluated >= args.max_queries:
            break
        if args.verbose and should_print_progress(index, len(pair_records), args.progress_every_records):
            print(
                f"[ingest-start] record={index}/{len(pair_records)} "
                f"session_uuid={pair.record.session_uuid} record_id={pair.record.record_id} "
                f"queries={evaluated}/{len(examples)} elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )
        try:
            system.ingest_record(pair.record)
            processed_records = index
            if should_save_record_state(processed_records, args.state_save_every_records):
                save_latest_state(system, output_dir)
            log_progress(
                log_store,
                output_dir,
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
        except Exception as exc:
            summary = summarize_metrics(detailed_results, session_ks)
            summary["failed_query_count"] = failed
            return write_invalid_run_artifacts(
                output_dir=output_dir,
                log_store=log_store,
                run_name=run_name,
                dataset_path=dataset_path,
                run_config=run_config,
                summary=summary,
                processed_records=processed_records,
                total_records=len(pair_records),
                evaluated_queries=evaluated,
                total_queries=len(examples),
                error=exc,
                error_stage="record_ingestion",
            )
        if args.verbose and should_print_progress(index, len(pair_records), args.progress_every_records):
            print(
                f"[ingest-done] records={index}/{len(pair_records)} queries={evaluated}/{len(examples)} "
                f"elapsed={time.time() - started_at:.1f}s",
                flush=True,
            )

    metrics_summary = summarize_metrics(detailed_results, session_ks)
    metrics_summary["failed_query_count"] = failed
    manifest = {
        "schema_version": 1,
        "status": "finished",
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset_path),
        "log_dir": str(log_store.run_dir),
        "output_dir": str(output_dir),
        "config": run_config,
        "summary": metrics_summary,
        "processed_records": processed_records,
        "evaluated_queries": evaluated,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        state_path = system.save(output_dir / "memory_state.json")
        save_latest_state(system, output_dir)
    except Exception as exc:
        return write_invalid_run_artifacts(
            output_dir=output_dir,
            log_store=log_store,
            run_name=run_name,
            dataset_path=dataset_path,
            run_config=run_config,
            summary=metrics_summary,
            processed_records=processed_records,
            total_records=len(pair_records),
            evaluated_queries=evaluated,
            total_queries=len(examples),
            error=exc,
            error_stage="final_state_save",
        )
    paths = {
        "manifest": output_dir / "manifest.json",
        "state": state_path,
        "retrieval": output_dir / "retrieval_results.json",
        "realmem_official_retrieval": output_dir / "realmem_official_retrieval_results.json",
        "generation": output_dir / "generation_results.json",
        "session_task_chain_trace": output_dir / "session_task_chain_trace.json",
        "metrics": output_dir / "metrics_results.json",
        "report": output_dir / "realmem_top_session_report.md",
        "progress": output_dir / "progress.jsonl",
        "progress_latest": output_dir / "progress_latest.json",
    }
    dump_json(paths["retrieval"], retrieval_results)
    dump_json(paths["realmem_official_retrieval"], realmem_official_results_from_detailed(detailed_results))
    dump_json(paths["generation"], generation_results)
    dump_json(paths["session_task_chain_trace"], session_task_chain_results)
    dump_json(paths["metrics"], {"summary": metrics_summary, "detailed_results": detailed_results})
    report = render_report(run_config=run_config, dataset_summary=dataset_summary, metrics_summary=metrics_summary)
    paths["report"].write_text(report, encoding="utf-8")
    manifest["artifacts"] = {key: str(path) for key, path in paths.items()}
    dump_json(paths["manifest"], manifest)
    log_store.log("dataset", "run_finished", output_dir=str(output_dir), evaluated_queries=evaluated, failed_queries=failed)
    log_progress(
        log_store,
        output_dir,
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
    ]
    ablation_design = run_config.get("ablation_design")
    if isinstance(ablation_design, dict) and ablation_design:
        llm_usage = ablation_design.get("llm_usage") if isinstance(ablation_design.get("llm_usage"), dict) else {}
        lines.extend(
            [
                "## Ablation Design",
                "",
                f"- name: {ablation_design.get('name', '')}",
                f"- online_order: {_format_design_steps(ablation_design.get('online_order'))}",
                f"- ingest_flow: {_format_design_steps(ablation_design.get('ingest_flow'))}",
                f"- retrieval_flow: {_format_design_steps(ablation_design.get('retrieval_flow'))}",
                f"- LLM ingest: {_format_design_steps(llm_usage.get('ingest'))}",
                f"- LLM retrieval: {_format_design_steps(llm_usage.get('retrieval'))}",
                f"- disabled_components: {_format_design_steps(ablation_design.get('disabled_components'))}",
                "",
            ]
        )
    lines.extend(["## Metrics", ""])
    for key, value in metrics_summary.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def _format_design_steps(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "none"
    if value in (None, ""):
        return "none"
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = TCMemConfig()
    parser = argparse.ArgumentParser(description="Run RealMemBench evaluation with top-session recall for TCMem.")
    parser.add_argument("--dataset", default="../RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json")
    parser.add_argument("--config", default="zhuo/runtime_config.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--build-model", "--graph-model", dest="build_model", default=None)
    parser.add_argument("--eval-model", "--qa-model", dest="eval_model", default=None)
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
    parser.add_argument("--progress-every-records", type=int, default=1)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a previous run directory or memory_state_latest.json checkpoint.",
    )
    parser.add_argument(
        "--skip-llm-preflight",
        action="store_true",
        help="Skip the tiny model availability request (for offline tests only).",
    )
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
