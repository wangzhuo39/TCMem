from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..serialization import dump_json, to_primitive
from ..utils.llm_client import OpenAICompatibleLLMClient
from .realmem_top_session import (
    _generate_json_with_retries,
    build_session_text_by_uuid,
    construct_session_evidence,
    extract_query_examples,
    resolve_runtime_config,
)


MEM_EVAL_PROMPT = """Your task is to evaluate the consistency between the [retrieved memory] and the [ground-truth memory], and whether the retrieved memory is helpful.

### Input Data
* <question>: {question}
* <groundtruth_memory>: {groundtruth_memory}
* <retrieved_memory>: {retrieved_memory}

---

### Evaluation Dimensions
#### 1. Memory Recall
Mem_recall: Semantics-aware memory recall calculation (0-1)
step1: For each groundtruth_memory, check in sequence whether its semantics are contained in any retrieved_memory.
step2: Count how many groundtruth_memory items are covered (hits_cnt).
step3: Compute the final recall score as hits_cnt / total number of groundtruth_memory items.

#### 2. Memory Helpfulness
Mem_helpful_score: The helpfulness of the retrieved memory for answering the question
Score 0: retrieved_memory contains mutually conflicting or contradictory memories, which not only fail to help answer the question but may also cause confusion.
Score 1: retrieved_memory is somewhat helpful for answering the question (can provide partial supporting evidence).
Score 2: retrieved_memory is very helpful for answering the question (can provide comprehensive supporting evidence).

---

### Output Format
Please provide your evaluation results using the following structure:

```json
{
  "Mem_recall": float,
  "Mem_helpful_score": int,
  "Mem_hits": ["..."],
  "Mem_helpful_reason": "..."
}
```
"""


@dataclass(frozen=True, slots=True)
class RunFiles:
    run_dir: Path
    log_dir: Path | None = None
    result_dir: Path | None = None
    manifest_file: Path | None = None
    dataset_file: Path | None = None
    query_results_file: Path | None = None
    generation_file: Path | None = None
    generation_jsonl_file: Path | None = None
    queries_file: Path | None = None
    retrieval_file: Path | None = None


def _path_bases(run_dir: Path | None = None) -> list[Path]:
    tcmem_root = Path(__file__).resolve().parents[2]
    bases = [Path.cwd(), tcmem_root, tcmem_root.parent]
    if run_dir is not None:
        bases.insert(0, run_dir)
    result: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        try:
            resolved = base.resolve()
        except OSError:
            resolved = base
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def _resolve_existing_path(value: str | Path | None, *, bases: list[Path]) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if path.exists():
        return path.resolve()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate.resolve()
    return path


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


def _merge_generation_payload(target: dict[str, Any], generation: dict[str, Any]) -> None:
    generation_result = dict(target.get("generation_result") or {})
    _merge_dict(generation_result, generation)
    target["generation_result"] = generation_result
    for key in ("question", "generated_answer", "evidence_used", "evidence_session_uuids", "ranked_sessions", "model"):
        if key in generation and key not in target:
            target[key] = generation[key]


def _ranked_session_uuids_from_hits(hits: list[dict[str, Any]]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        session_uuid = str(hit.get("source_session_uuid") or hit.get("session_uuid") or "").strip()
        if session_uuid and session_uuid not in seen:
            ranked.append(session_uuid)
            seen.add(session_uuid)
    return ranked


def discover_run_files(
    run_dir: str | Path | None,
    *,
    dataset: str | Path | None = None,
    query_results_file: str | Path | None = None,
    generation_file: str | Path | None = None,
    generation_jsonl_file: str | Path | None = None,
    queries_file: str | Path | None = None,
    retrieval_file: str | Path | None = None,
) -> RunFiles:
    base_run_dir = Path(run_dir or ".").expanduser()
    if base_run_dir.is_file():
        base_run_dir = base_run_dir.parent
    bases = _path_bases(base_run_dir)
    resolved_run_dir = _resolve_existing_path(base_run_dir, bases=bases) or base_run_dir
    resolved_run_dir = resolved_run_dir.resolve() if resolved_run_dir.exists() else resolved_run_dir

    manifest_file = resolved_run_dir / "manifest.json"
    manifest = _read_json(manifest_file)
    if not isinstance(manifest, dict):
        manifest_file = None
        manifest = {}

    log_dir = None
    result_dir = resolved_run_dir
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    if manifest.get("log_dir"):
        log_dir = _resolve_existing_path(manifest.get("log_dir"), bases=_path_bases(resolved_run_dir))
    if manifest.get("output_dir"):
        result_dir = _resolve_existing_path(manifest.get("output_dir"), bases=_path_bases(resolved_run_dir)) or resolved_run_dir
    elif (resolved_run_dir / "query_results.jsonl").exists():
        log_dir = resolved_run_dir
        result_dir = _infer_output_dir_from_dataset_log(resolved_run_dir) or resolved_run_dir

    if log_dir is None and (resolved_run_dir / "query_results.jsonl").exists():
        log_dir = resolved_run_dir
    if log_dir is None and (resolved_run_dir / "generation.jsonl").exists():
        log_dir = resolved_run_dir

    dataset_file = _resolve_existing_path(dataset, bases=_path_bases(resolved_run_dir))
    if dataset_file is None:
        if isinstance(manifest.get("dataset"), str):
            dataset_file = _resolve_existing_path(manifest.get("dataset"), bases=_path_bases(resolved_run_dir))
        if dataset_file is None and log_dir is not None:
            dataset_file = _infer_dataset_from_dataset_log(log_dir)

    q_file = _resolve_existing_path(query_results_file, bases=_path_bases(resolved_run_dir))
    if q_file is None and log_dir is not None:
        candidate = log_dir / "query_results.jsonl"
        q_file = candidate if candidate.exists() else None

    gen_file = _resolve_existing_path(generation_file, bases=_path_bases(resolved_run_dir))
    if gen_file is None and artifacts.get("generation"):
        gen_file = _resolve_existing_path(artifacts.get("generation"), bases=_path_bases(resolved_run_dir))
    if gen_file is None and result_dir is not None:
        candidate = result_dir / "generation_results.json"
        gen_file = candidate if candidate.exists() else None
    if gen_file is None:
        candidate = resolved_run_dir / "generation_results.json"
        gen_file = candidate if candidate.exists() else None

    gen_jsonl = _resolve_existing_path(generation_jsonl_file, bases=_path_bases(resolved_run_dir))
    if gen_jsonl is None and log_dir is not None:
        candidate = log_dir / "generation.jsonl"
        gen_jsonl = candidate if candidate.exists() else None

    qs_file = _resolve_existing_path(queries_file, bases=_path_bases(resolved_run_dir))
    if qs_file is None and log_dir is not None:
        candidate = log_dir / "queries.jsonl"
        qs_file = candidate if candidate.exists() else None

    ret_file = _resolve_existing_path(retrieval_file, bases=_path_bases(resolved_run_dir))
    if ret_file is None and log_dir is not None:
        candidate = log_dir / "retrieval.jsonl"
        ret_file = candidate if candidate.exists() else None

    return RunFiles(
        run_dir=resolved_run_dir,
        log_dir=log_dir,
        result_dir=result_dir,
        manifest_file=manifest_file,
        dataset_file=dataset_file,
        query_results_file=q_file,
        generation_file=gen_file,
        generation_jsonl_file=gen_jsonl,
        queries_file=qs_file,
        retrieval_file=ret_file,
    )


def _infer_dataset_from_dataset_log(log_dir: Path) -> Path | None:
    for payload in _iter_jsonl_payloads(log_dir / "dataset.jsonl"):
        dataset = payload.get("dataset")
        if dataset:
            return _resolve_existing_path(dataset, bases=_path_bases(log_dir))
    return None


def _infer_output_dir_from_dataset_log(log_dir: Path) -> Path | None:
    for payload in _iter_jsonl_payloads(log_dir / "dataset.jsonl"):
        output_dir = payload.get("output_dir")
        if output_dir:
            return _resolve_existing_path(output_dir, bases=_path_bases(log_dir))
    return None


def load_logged_results(
    run_dir: str | Path | None = None,
    *,
    query_results_file: str | Path | None = None,
    generation_file: str | Path | None = None,
    generation_jsonl_file: str | Path | None = None,
    queries_file: str | Path | None = None,
    retrieval_file: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    files = discover_run_files(
        run_dir,
        query_results_file=query_results_file,
        generation_file=generation_file,
        generation_jsonl_file=generation_jsonl_file,
        queries_file=queries_file,
        retrieval_file=retrieval_file,
    )
    results: dict[str, dict[str, Any]] = {}

    for payload in _iter_jsonl_payloads(files.queries_file):
        qid = str(payload.get("query_id") or "").strip()
        if qid:
            _merge_dict(results.setdefault(qid, {}), payload)

    for payload in _iter_jsonl_payloads(files.query_results_file):
        qid = str(payload.get("query_id") or "").strip()
        if qid:
            _merge_dict(results.setdefault(qid, {}), payload)

    for payload in _iter_jsonl_payloads(files.retrieval_file):
        qid = str(payload.get("query_id") or "").strip()
        if not qid:
            continue
        target = results.setdefault(qid, {})
        retrieval_result = dict(target.get("retrieval_result") or {})
        retrieval_result.setdefault("ranked_items", payload.get("hits") or payload.get("ranked_items") or [])
        retrieval_result.setdefault("gold_session_uuids", target.get("gold_session_uuids") or payload.get("gold_session_uuids") or [])
        target["retrieval_result"] = retrieval_result
        if "ranked_session_uuids" not in target:
            target["ranked_session_uuids"] = _ranked_session_uuids_from_hits(retrieval_result.get("ranked_items") or [])

    for payload in _iter_jsonl_payloads(files.generation_jsonl_file):
        qid = str(payload.get("query_id") or payload.get("id") or "").strip()
        if qid:
            _merge_generation_payload(results.setdefault(qid, {}), payload)

    generation_data = _read_json(files.generation_file)
    if isinstance(generation_data, dict):
        for key, value in generation_data.items():
            if not isinstance(value, dict):
                continue
            qid = str(value.get("query_id") or value.get("id") or key).strip()
            if qid:
                _merge_generation_payload(results.setdefault(qid, {}), value)

    return results


def _evidence_blocks(text: str, top_k: int) -> str:
    text = str(text or "").strip()
    if not text or top_k <= 0:
        return ""
    starts: list[int] = []
    for line_start, line in _iter_lines_with_offsets(text):
        if line.strip().startswith("---- idx "):
            starts.append(line_start)
    if not starts:
        return text
    blocks: list[str] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        blocks.append(text[start:end].strip())
    return "\n\n".join(blocks[:top_k])


def _iter_lines_with_offsets(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        lines.append((offset, line))
        offset += len(line)
    return lines


def _ranked_sessions_from_uuid_list(session_uuids: list[str]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for session_uuid in session_uuids:
        value = str(session_uuid or "").strip()
        if value and value not in seen:
            ranked.append({"session_uuid": value})
            seen.add(value)
    return ranked


def build_retrieved_memory(
    result: dict[str, Any],
    *,
    session_text_by_uuid: dict[str, str],
    top_k: int,
) -> str:
    generation_result = result.get("generation_result") if isinstance(result.get("generation_result"), dict) else {}
    retrieval_result = result.get("retrieval_result") if isinstance(result.get("retrieval_result"), dict) else {}

    ranked_sessions = (
        result.get("ranked_sessions")
        or generation_result.get("ranked_sessions")
        or retrieval_result.get("ranked_sessions")
    )
    if not ranked_sessions:
        ranked_session_uuids = (
            result.get("ranked_session_uuids")
            or result.get("retrieved_session_uuids")
            or generation_result.get("evidence_session_uuids")
            or []
        )
        ranked_sessions = _ranked_sessions_from_uuid_list(ranked_session_uuids)
    if isinstance(ranked_sessions, list) and session_text_by_uuid:
        evidence = construct_session_evidence(
            ranked_sessions=[item for item in ranked_sessions if isinstance(item, dict)],
            session_text_by_uuid=session_text_by_uuid,
            top_k=top_k,
        )
        if evidence:
            return evidence

    for key in ("evidence_used", "evidence_text", "evidence_excerpt"):
        evidence_text = result.get(key) or generation_result.get(key)
        if evidence_text:
            return _evidence_blocks(str(evidence_text), top_k)

    ranked_items = retrieval_result.get("ranked_items") or retrieval_result.get("hits") or result.get("ranked_items") or []
    blocks: list[str] = []
    seen_sessions: set[str] = set()
    for item in ranked_items:
        if not isinstance(item, dict):
            continue
        session_uuid = str(item.get("source_session_uuid") or item.get("session_uuid") or "").strip()
        if session_uuid and session_uuid in seen_sessions:
            continue
        content = str(item.get("content") or item.get("content_excerpt") or "").strip()
        if not content:
            continue
        if session_uuid:
            seen_sessions.add(session_uuid)
        blocks.append(f"---- idx {len(blocks) + 1} | session_uuid={session_uuid} ----\n{content}")
        if len(blocks) >= top_k:
            break
    return "\n\n".join(blocks)


def build_groundtruth_memory(example: Any) -> str:
    chunks: list[str] = []
    for index, item in enumerate(getattr(example, "memory_used", []) or [], start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        session_uuid = str(item.get("session_uuid") or "").strip()
        chunks.append(f"---- gt {index} | session_uuid={session_uuid} ----\n{content}")
    if not chunks and getattr(example, "gold_memory_text", ""):
        chunks.append(str(example.gold_memory_text).strip())
    return "\n\n".join(chunks)


def build_mem_eval_prompt(*, question: str, groundtruth_memory: str, retrieved_memory: str) -> str:
    return (
        MEM_EVAL_PROMPT.replace("{question}", question)
        .replace("{groundtruth_memory}", groundtruth_memory)
        .replace("{retrieved_memory}", retrieved_memory)
    )


def normalize_mem_judge_result(parsed: dict[str, Any]) -> dict[str, Any]:
    recall = float(parsed.get("Mem_recall", parsed.get("mem_recall", -1)))
    if recall < 0.0 or recall > 1.0:
        raise ValueError(f"Mem_recall must be between 0 and 1; got {recall!r}")
    helpful = int(parsed.get("Mem_helpful_score", parsed.get("mem_helpful_score", -1)))
    if helpful not in {0, 1, 2}:
        raise ValueError(f"Mem_helpful_score must be 0, 1, or 2; got {helpful!r}")
    hits = parsed.get("Mem_hits", parsed.get("mem_hits", []))
    if not isinstance(hits, list):
        hits = [str(hits)]
    reason = str(parsed.get("Mem_helpful_reason", parsed.get("mem_helpful_reason", "")) or "")
    return {
        "Mem_recall": recall,
        "Mem_helpful_score": helpful,
        "Mem_hits": [str(item) for item in hits],
        "Mem_helpful_reason": reason,
    }


def judge_mem_metrics(
    client: OpenAICompatibleLLMClient,
    *,
    question: str,
    groundtruth_memory: str,
    retrieved_memory: str,
    query_id: str = "",
) -> dict[str, Any]:
    prompt = build_mem_eval_prompt(
        question=question,
        groundtruth_memory=groundtruth_memory,
        retrieved_memory=retrieved_memory,
    )
    parsed = _generate_json_with_retries(
        client,
        prompt,
        system_prompt="",
        temperature=0.0,
        max_tokens=1600,
        stage="mem_judge",
        context={"query_id": query_id},
    )
    if not isinstance(parsed, dict):
        raise RuntimeError("Mem judge returned non-object JSON")
    return normalize_mem_judge_result(parsed)


def summarize_mem_metrics(detailed_results: list[dict[str, Any]]) -> dict[str, Any]:
    recalls = [float(item["Mem_recall"]) for item in detailed_results if isinstance(item.get("Mem_recall"), (int, float))]
    helpful_scores = [
        int(item["Mem_helpful_score"])
        for item in detailed_results
        if isinstance(item.get("Mem_helpful_score"), int)
    ]
    failed_count = sum(1 for item in detailed_results if item.get("error"))
    return {
        "query_count": len(detailed_results),
        "evaluated_query_count": len(recalls),
        "failed_query_count": failed_count,
        "average_mem_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "average_mem_helpful_score": round(sum(helpful_scores) / len(helpful_scores), 4) if helpful_scores else None,
        "mem_helpful_score_distribution": {str(score): helpful_scores.count(score) for score in range(3)},
    }


def _example_indexes(dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    examples = extract_query_examples(dataset)
    by_query_id = {example.query_id: example for example in examples}
    by_question = {example.question.strip(): example for example in examples if example.question.strip()}
    return by_query_id, by_question


def _select_logged_items(
    logged_results: dict[str, dict[str, Any]],
    *,
    query_ids: list[str],
    max_queries: int | None,
) -> list[tuple[str, dict[str, Any]]]:
    selected = [(qid, logged_results[qid]) for qid in sorted(logged_results)]
    if query_ids:
        wanted = set(query_ids)
        selected = [(qid, item) for qid, item in selected if qid in wanted]
    if max_queries is not None:
        selected = selected[:max_queries]
    return selected


def evaluate_mem_metrics_from_logs(args: argparse.Namespace) -> dict[str, Any]:
    files = discover_run_files(
        args.run_dir,
        dataset=args.dataset,
        query_results_file=args.query_results_file,
        generation_file=args.generation_file,
        generation_jsonl_file=args.generation_jsonl_file,
        queries_file=args.queries_file,
        retrieval_file=args.retrieval_file,
    )
    if files.dataset_file is None or not files.dataset_file.exists():
        raise SystemExit("Could not infer dataset path. Pass --dataset explicitly.")

    logged_results = load_logged_results(
        files.run_dir,
        query_results_file=files.query_results_file,
        generation_file=files.generation_file,
        generation_jsonl_file=files.generation_jsonl_file,
        queries_file=files.queries_file,
        retrieval_file=files.retrieval_file,
    )
    if not logged_results:
        raise SystemExit("No logged query or generation results were found.")

    dataset = json.loads(files.dataset_file.read_text(encoding="utf-8"))
    examples_by_id, examples_by_question = _example_indexes(dataset)
    session_text_by_uuid = build_session_text_by_uuid(dataset)
    runtime = resolve_runtime_config(args)
    if not runtime.api_key:
        raise SystemExit("Missing API key for MemRec/MemHelp judge.")
    client = OpenAICompatibleLLMClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        model=runtime.model,
        timeout=runtime.timeout,
    )

    selected = _select_logged_items(logged_results, query_ids=args.query_id, max_queries=args.max_queries)
    detailed_results: list[dict[str, Any]] = []

    def evaluate_one(qid: str, logged: dict[str, Any]) -> dict[str, Any]:
        question = str(logged.get("question") or logged.get("query") or "").strip()
        example = examples_by_id.get(qid) or examples_by_question.get(question)
        if example is None:
            raise ValueError(f"ground truth not found for {qid}")
        groundtruth_memory = build_groundtruth_memory(example)
        retrieved_memory = build_retrieved_memory(logged, session_text_by_uuid=session_text_by_uuid, top_k=args.top_k)
        if not groundtruth_memory:
            raise ValueError(f"groundtruth memory is empty for {qid}")
        if not retrieved_memory:
            raise ValueError(f"retrieved memory is empty for {qid}")
        judged = judge_mem_metrics(
            client,
            question=question or example.question,
            groundtruth_memory=groundtruth_memory,
            retrieved_memory=retrieved_memory,
            query_id=qid,
        )
        generation_result = logged.get("generation_result") if isinstance(logged.get("generation_result"), dict) else {}
        return {
            "query_id": qid,
            "question": question or example.question,
            "gold_session_uuids": list(example.gold_session_uuids),
            "evidence_session_uuids": _evidence_session_uuids(logged, top_k=args.top_k),
            "groundtruth_memory": groundtruth_memory,
            "retrieved_memory": retrieved_memory,
            "generated_answer": generation_result.get("generated_answer", logged.get("generated_answer", "")),
            **judged,
        }

    if args.max_workers <= 1:
        for index, (qid, logged) in enumerate(selected, start=1):
            detailed_results.append(_safe_evaluate_one(evaluate_one, qid, logged, fail_fast=args.fail_fast))
            if args.verbose:
                print(f"[mem] {index}/{len(selected)} {qid}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(_safe_evaluate_one, evaluate_one, qid, logged, fail_fast=args.fail_fast): qid
                for qid, logged in selected
            }
            completed = 0
            for future in as_completed(futures):
                detailed_results.append(future.result())
                completed += 1
                if args.verbose:
                    print(f"[mem] {completed}/{len(selected)} {futures[future]}", flush=True)

    detailed_results.sort(key=lambda item: item.get("query_id", ""))
    summary = summarize_mem_metrics(detailed_results)
    summary.update(
        {
            "judge_model": runtime.model,
            "top_k": args.top_k,
            "dataset": str(files.dataset_file),
            "source_run_dir": str(files.run_dir),
        }
    )
    output = {
        "summary": summary,
        "artifacts": {
            "dataset": str(files.dataset_file),
            "query_results_file": str(files.query_results_file) if files.query_results_file else None,
            "generation_file": str(files.generation_file) if files.generation_file else None,
            "generation_jsonl_file": str(files.generation_jsonl_file) if files.generation_jsonl_file else None,
            "queries_file": str(files.queries_file) if files.queries_file else None,
            "retrieval_file": str(files.retrieval_file) if files.retrieval_file else None,
        },
        "detailed_results": detailed_results,
    }
    out_file = Path(args.out_file) if args.out_file else _default_output_file(files, top_k=args.top_k)
    dump_json(out_file, output)
    return {"out_file": str(out_file), "summary": summary}


def _safe_evaluate_one(func: Any, qid: str, logged: dict[str, Any], *, fail_fast: bool) -> dict[str, Any]:
    try:
        return func(qid, logged)
    except Exception as exc:
        if fail_fast:
            raise
        return {
            "query_id": qid,
            "question": str(logged.get("question") or logged.get("query") or ""),
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


def _evidence_session_uuids(logged: dict[str, Any], *, top_k: int) -> list[str]:
    generation_result = logged.get("generation_result") if isinstance(logged.get("generation_result"), dict) else {}
    session_uuids = (
        logged.get("ranked_session_uuids")
        or logged.get("retrieved_session_uuids")
        or generation_result.get("evidence_session_uuids")
        or []
    )
    if not session_uuids:
        ranked_sessions = logged.get("ranked_sessions") or generation_result.get("ranked_sessions") or []
        session_uuids = [item.get("session_uuid") for item in ranked_sessions if isinstance(item, dict)]
    result: list[str] = []
    seen: set[str] = set()
    for session_uuid in session_uuids:
        value = str(session_uuid or "").strip()
        if value and value not in seen:
            result.append(value)
            seen.add(value)
        if len(result) >= top_k:
            break
    return result


def _default_output_file(files: RunFiles, *, top_k: int) -> Path:
    base_dir = files.result_dir or files.run_dir
    return base_dir / f"memrec_memhelp_top{top_k}_metrics.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RealMemBench MemRec/MemHelp from logged TCMem results.")
    parser.add_argument("--run-dir", default=None, help="A result/results/<run> or result/logs/<run> directory.")
    parser.add_argument("--dataset", default=None, help="Original RealMemBench dialogues JSON. Inferred from logs when omitted.")
    parser.add_argument("--query-results-file", default=None)
    parser.add_argument("--generation-file", default=None)
    parser.add_argument("--generation-jsonl-file", default=None)
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top original dialogue sessions used as retrieved memory.")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=[], help="Evaluate only this query id; may be repeated.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--config", default="zhuo/runtime_config.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = evaluate_mem_metrics_from_logs(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
