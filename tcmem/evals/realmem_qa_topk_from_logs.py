from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..logging_utils import ModuleLogStore
from ..prompts import PromptRegistry
from ..serialization import dump_json, to_primitive
from ..utils.llm_client import OpenAICompatibleLLMClient
from .realmem_mem_metrics_from_logs import (
    _evidence_session_uuids,
    _example_indexes,
    _iter_jsonl_payloads,
    _safe_evaluate_one,
    _select_logged_items,
    build_groundtruth_memory,
    build_retrieved_memory,
    discover_run_files,
    load_logged_results,
)
from .realmem_top_session import RuntimeConfig, build_session_text_by_uuid, generate_answer, judge_qa_score, resolve_runtime_config


def build_qa_topk_result(
    *,
    qid: str,
    logged: dict[str, Any],
    example: Any,
    session_text_by_uuid: dict[str, str],
    top_k: int,
    client: OpenAICompatibleLLMClient,
    qa_model_name: str,
    prompt_registry: PromptRegistry | None = None,
    log_store: ModuleLogStore | None = None,
) -> dict[str, Any]:
    question = str(logged.get("question") or logged.get("query") or example.question).strip()
    evidence_text = build_retrieved_memory(logged, session_text_by_uuid=session_text_by_uuid, top_k=top_k)
    if not evidence_text:
        raise ValueError(f"top-{top_k} evidence is empty for {qid}")
    gold_memory_text = build_groundtruth_memory(example)
    if not gold_memory_text:
        raise ValueError(f"groundtruth memory is empty for {qid}")

    generated_answer = generate_answer(
        client,
        question or example.question,
        evidence_text,
        prompt_registry=prompt_registry,
    )
    judged = judge_qa_score(
        client,
        question=question or example.question,
        gold_memory_text=gold_memory_text,
        reference_answer=str(getattr(example, "reference_answer", "") or ""),
        candidate_answer=generated_answer,
        log_store=log_store,
        query_id=qid,
        prompt_registry=prompt_registry,
    )
    generation_result = logged.get("generation_result") if isinstance(logged.get("generation_result"), dict) else {}
    evidence_session_uuids = _evidence_session_uuids(logged, top_k=top_k)
    return {
        "query_id": qid,
        "question": question or example.question,
        "top_k": top_k,
        "gold_session_uuids": list(getattr(example, "gold_session_uuids", []) or []),
        "evidence_session_uuids": evidence_session_uuids,
        "evidence_used": evidence_text,
        "generated_answer": generated_answer,
        "qa_score": judged["score"],
        "qa_reason": judged["reason"],
        "model": qa_model_name,
        "generation_result": {
            "query_id": qid,
            "question": question or example.question,
            "generated_answer": generated_answer,
            "evidence_used": evidence_text,
            "evidence_session_uuids": evidence_session_uuids,
            "model": qa_model_name,
        },
        "source_qa_score": logged.get("qa_score"),
        "source_qa_reason": logged.get("qa_reason", ""),
        "source_evidence_session_uuids": generation_result.get("evidence_session_uuids", logged.get("evidence_session_uuids", [])),
    }


def summarize_qa_scores(detailed_results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(item["qa_score"]) for item in detailed_results if isinstance(item.get("qa_score"), int)]
    failed_count = sum(1 for item in detailed_results if item.get("error"))
    return {
        "query_count": len(detailed_results),
        "evaluated_query_count": len(scores),
        "failed_query_count": failed_count,
        "average_qa_score": round(sum(scores) / len(scores), 4) if scores else None,
        "qa_score_distribution": {str(score): scores.count(score) for score in range(4)},
    }


def evaluate_qa_topk_from_logs(args: argparse.Namespace) -> dict[str, Any]:
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
    runtime = _resolve_runtime_config(args, files)
    if not runtime.api_key:
        raise SystemExit("Missing API key for top-k QA supplemental evaluation.")
    client = OpenAICompatibleLLMClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        model=runtime.model,
        timeout=runtime.timeout,
    )
    prompt_registry = _load_prompt_registry(args, files)
    log_store = _build_log_store(args)
    selected = _select_logged_items(logged_results, query_ids=args.query_id, max_queries=args.max_queries)
    detailed_results: list[dict[str, Any]] = []
    _progress(args, f"[qa-top{args.top_k}] selected={len(selected)} model={runtime.model} out={_default_output_file(files, top_k=args.top_k) if not args.out_file else args.out_file}")

    def evaluate_one(qid: str, logged: dict[str, Any]) -> dict[str, Any]:
        question = str(logged.get("question") or logged.get("query") or "").strip()
        example = examples_by_id.get(qid) or examples_by_question.get(question)
        if example is None:
            raise ValueError(f"ground truth not found for {qid}")
        return build_qa_topk_result(
            qid=qid,
            logged=logged,
            example=example,
            session_text_by_uuid=session_text_by_uuid,
            top_k=args.top_k,
            client=client,
            qa_model_name=runtime.model,
            prompt_registry=prompt_registry,
            log_store=log_store,
        )

    if args.max_workers <= 1:
        for index, (qid, logged) in enumerate(selected, start=1):
            _progress(args, f"[qa-top{args.top_k}] start {index}/{len(selected)} {qid}")
            result = _safe_evaluate_one(evaluate_one, qid, logged, fail_fast=args.fail_fast)
            detailed_results.append(result)
            if result.get("error"):
                _progress(args, f"[qa-top{args.top_k}] error {index}/{len(selected)} {qid} {result.get('error_type')}: {result.get('error')}")
            else:
                _progress(args, f"[qa-top{args.top_k}] done {index}/{len(selected)} {qid} score={result.get('qa_score')}")
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for index, (qid, _logged) in enumerate(selected, start=1):
                _progress(args, f"[qa-top{args.top_k}] queued {index}/{len(selected)} {qid}")
            futures = {
                executor.submit(_safe_evaluate_one, evaluate_one, qid, logged, fail_fast=args.fail_fast): qid
                for qid, logged in selected
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                detailed_results.append(result)
                completed += 1
                qid = futures[future]
                if result.get("error"):
                    _progress(args, f"[qa-top{args.top_k}] error {completed}/{len(selected)} {qid} {result.get('error_type')}: {result.get('error')}")
                else:
                    _progress(args, f"[qa-top{args.top_k}] done {completed}/{len(selected)} {qid} score={result.get('qa_score')}")

    detailed_results.sort(key=lambda item: item.get("query_id", ""))
    summary = summarize_qa_scores(detailed_results)
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
    _progress(args, f"[qa-top{args.top_k}] wrote {out_file}")
    return {"out_file": str(out_file), "summary": summary}


def _resolve_runtime_config(args: argparse.Namespace, files: Any) -> RuntimeConfig:
    runtime = resolve_runtime_config(args)
    run_config = _source_run_config(files)
    if not args.base_url and run_config.get("base_url"):
        runtime.base_url = str(run_config["base_url"])
    if not args.model and run_config.get("model"):
        runtime.model = str(run_config["model"])
    tcmem_config = run_config.get("tcmem_config") if isinstance(run_config.get("tcmem_config"), dict) else {}
    if args.timeout is None and tcmem_config.get("llm_timeout"):
        runtime.timeout = int(tcmem_config["llm_timeout"])
    return runtime


def _source_run_config(files: Any) -> dict[str, Any]:
    if files.manifest_file is not None and files.manifest_file.exists():
        manifest = json.loads(files.manifest_file.read_text(encoding="utf-8"))
        config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
        if config:
            return config
    if files.log_dir is not None:
        for payload in _iter_jsonl_payloads(files.log_dir / "dataset.jsonl"):
            if isinstance(payload.get("tcmem_config"), dict):
                return payload
            if payload.get("model") or payload.get("base_url"):
                return payload
    return {}


def _load_prompt_registry(args: argparse.Namespace, files: Any) -> PromptRegistry:
    prompt_path = str(args.prompt_path or "").strip()
    if not prompt_path:
        config = _source_run_config(files)
        tcmem_config = config.get("tcmem_config") if isinstance(config.get("tcmem_config"), dict) else {}
        prompt_path = str(tcmem_config.get("prompt_path") or "").strip()
    return PromptRegistry.load(prompt_path)


def _build_log_store(args: argparse.Namespace) -> ModuleLogStore | None:
    if not args.log_dir:
        return None
    return ModuleLogStore(base_dir=args.log_dir, run_name=args.log_run_name or f"qa_top{args.top_k}_supplement")


def _default_output_file(files: Any, *, top_k: int) -> Path:
    base_dir = files.result_dir or files.run_dir
    return base_dir / f"qa_top{top_k}_results.json"


def _progress(args: argparse.Namespace, message: str) -> None:
    if args.verbose:
        print(message, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplemental RealMem QA scoring from logged TCMem ranked sessions.")
    parser.add_argument("--run-dir", default=None, help="A result/results/<run> or result/logs/<run> directory.")
    parser.add_argument("--dataset", default=None, help="Original RealMemBench dialogues JSON. Inferred from logs when omitted.")
    parser.add_argument("--query-results-file", default=None)
    parser.add_argument("--generation-file", default=None)
    parser.add_argument("--generation-jsonl-file", default=None)
    parser.add_argument("--queries-file", default=None)
    parser.add_argument("--retrieval-file", default=None)
    parser.add_argument("--out-file", default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top ranked sessions used for answer generation.")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-id", action="append", default=[], help="Evaluate only this query id; may be repeated.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--config", default="zhuo/runtime_config.json")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--prompt-path", default=None)
    parser.add_argument("--log-dir", default=None, help="Optional directory for supplemental judge JSON error logs.")
    parser.add_argument("--log-run-name", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    result = evaluate_qa_topk_from_logs(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
