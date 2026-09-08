#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tcmem.config import TCMemConfig
from tcmem.core.memory_system import MemorySystem
from tcmem.infrastructure.indices import NumpyVectorIndex
from tcmem.models import DialogueRecord
from tcmem.serialization import to_primitive


DEFAULT_DATASET = Path("/data/wz/agent_memory/iconip2026/locomo10.json")
DEFAULT_OUTPUT_DIR = Path("result/locomo10_smoke")
VECTOR_DIMENSION = 256

STOP_WORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "but",
    "can",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "its",
    "just",
    "like",
    "not",
    "now",
    "our",
    "out",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


class HashingEmbeddingClient:
    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def score(self, query: str, text: str) -> float:
        return _cosine(self.embed_query(query), self._vector(text))

    def rank(self, query: str, candidates: list[tuple[str, str]], limit: int) -> list[tuple[str, float]]:
        ranked = [(item_id, self.score(query, text)) for item_id, text in candidates]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros((VECTOR_DIMENSION,), dtype="float32")
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            vector[int.from_bytes(digest, "big") % VECTOR_DIMENSION] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector


class EntityOnlyLLMClient:
    json_max_attempts = 1
    json_retry_delay = 0.0

    def generate(self, prompt: str, **_kwargs: Any) -> str:
        payload = _prompt_payload(prompt)
        record = payload.get("record", {}) if isinstance(payload, dict) else {}
        if not isinstance(record, dict):
            record = {}
        text = "\n".join(
            str(record.get(key) or "")
            for key in ("user_content", "assistant_content")
            if record.get(key)
        )
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        speakers = [str(value).strip() for value in metadata.get("speakers", []) if str(value).strip()]
        entities = _unique([*speakers, *_top_terms(text, limit=3)])[:3]
        return json.dumps({"entities": entities})


def build_records_from_locomo_sample(sample: dict[str, Any], *, session_limit: int | None = None) -> list[DialogueRecord]:
    sample_id = str(sample.get("sample_id") or "sample").strip() or "sample"
    conversation = sample.get("conversation") if isinstance(sample.get("conversation"), dict) else {}
    speakers = [
        str(conversation.get("speaker_a") or "").strip(),
        str(conversation.get("speaker_b") or "").strip(),
    ]
    records: list[DialogueRecord] = []
    for session_number, session_key in _session_keys(conversation):
        if session_limit is not None and session_number > session_limit:
            continue
        turns = conversation.get(session_key) or []
        if not isinstance(turns, list):
            continue
        current_time = _parse_locomo_datetime(str(conversation.get(f"{session_key}_date_time") or ""))
        session_uuid = _session_uuid(sample_id, session_number)
        session_identifier = f"{sample_id}/session_{session_number}"
        base_time = _record_base_time(current_time)
        record_index = 0
        turn_index = 0
        while turn_index < len(turns):
            first = turns[turn_index] if isinstance(turns[turn_index], dict) else {}
            second = turns[turn_index + 1] if turn_index + 1 < len(turns) and isinstance(turns[turn_index + 1], dict) else None
            source_turn_ids = [str(first.get("dia_id") or f"D{session_number}:{turn_index + 1}")]
            source_turn_indexes = [turn_index]
            assistant_content = None
            if second is not None:
                source_turn_ids.append(str(second.get("dia_id") or f"D{session_number}:{turn_index + 2}"))
                source_turn_indexes.append(turn_index + 1)
                assistant_content = _format_turn(second)
            record_time = (base_time + timedelta(minutes=record_index)).strftime("%Y-%m-%d %H:%M:%S")
            records.append(
                DialogueRecord(
                    record_id=f"rec_{_slug(sample_id)}_s{session_number:02d}_{record_index + 1:04d}",
                    session_identifier=session_identifier,
                    session_uuid=session_uuid,
                    current_time=current_time,
                    record_time=record_time,
                    user_content=_format_turn(first),
                    assistant_content=assistant_content,
                    source_turn_indexes=source_turn_indexes,
                    source_turn_ids=source_turn_ids,
                    metadata={
                        "sample_id": sample_id,
                        "session_number": session_number,
                        "session_key": session_key,
                        "raw_date_time": str(conversation.get(f"{session_key}_date_time") or ""),
                        "speakers": [speaker for speaker in speakers if speaker],
                    },
                )
            )
            record_index += 1
            turn_index += 2 if second is not None else 1
    return records


def evidence_session_uuids(sample: dict[str, Any], evidence: list[Any]) -> list[str]:
    sample_id = str(sample.get("sample_id") or "sample").strip() or "sample"
    result: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        match = re.search(r"\bD(\d+):\d+\b", str(item))
        if not match:
            continue
        session_uuid = _session_uuid(sample_id, int(match.group(1)))
        if session_uuid not in seen:
            result.append(session_uuid)
            seen.add(session_uuid)
    return result


def run_smoke_test(
    *,
    dataset_path: Path,
    output_dir: Path,
    sample_limit: int,
    session_limit: int,
    qa_limit: int,
    top_k: int,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise ValueError("LOCOMO dataset must be a list of samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TCMemConfig(
        owner_id="locomo10_smoke",
        storage_path=str(output_dir / "state"),
        log_path=str(output_dir / "logs"),
        vector_index_backend="numpy",
        vector_index_path=str(output_dir / "vector_index"),
        task_chain_enabled=False,
        graph_seed_limit=max(1, top_k),
        graph_bm25_seed_limit=max(1, top_k),
        graph_walk_depth=1,
    )
    system = MemorySystem(
        config=config,
        llm_client=EntityOnlyLLMClient(),
        embedding_client=HashingEmbeddingClient(),
        record_index=NumpyVectorIndex(output_dir / "vector_index" / "records", index_name="records"),
    )

    selected_samples = dataset[:sample_limit]
    all_records: list[DialogueRecord] = []
    for sample in selected_samples:
        if isinstance(sample, dict):
            all_records.extend(build_records_from_locomo_sample(sample, session_limit=session_limit))
    for record in all_records:
        system.ingest_record(record)

    query_results = _evaluate_locomo_questions(
        system=system,
        samples=[sample for sample in selected_samples if isinstance(sample, dict)],
        session_limit=session_limit,
        qa_limit=qa_limit,
        top_k=top_k,
    )
    state_path = system.save(output_dir / "memory_state.json")
    queries_with_hits = sum(1 for item in query_results if item["retrieved_session_uuids"])
    recall_any_count = sum(1 for item in query_results if item["recall_any"])
    summary = {
        "dataset": str(dataset_path),
        "output_dir": str(output_dir),
        "sample_count": len(selected_samples),
        "session_limit": session_limit,
        "qa_limit": qa_limit,
        "top_k": top_k,
        "record_count": len(all_records),
        "state_summary": system.state_summary(),
        "evaluated_questions": len(query_results),
        "queries_with_hits": queries_with_hits,
        "recall_any_count": recall_any_count,
        "recall_any_rate": round(recall_any_count / len(query_results), 4) if query_results else 0.0,
        "memory_state_path": str(state_path),
        "query_results": query_results,
    }
    (output_dir / "locomo10_smoke_summary.json").write_text(
        json.dumps(to_primitive(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not all_records:
        raise RuntimeError("No LOCOMO records were parsed")
    if not query_results:
        raise RuntimeError("No LOCOMO QA examples were evaluated")
    if queries_with_hits != len(query_results):
        raise RuntimeError("At least one LOCOMO smoke query returned no retrieval hits")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an offline TCMem smoke test on locomo10.json.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-limit", type=int, default=1)
    parser.add_argument("--session-limit", type=int, default=3)
    parser.add_argument("--qa-limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_smoke_test(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sample_limit=max(1, args.sample_limit),
        session_limit=max(1, args.session_limit),
        qa_limit=max(1, args.qa_limit),
        top_k=max(1, args.top_k),
    )
    printable = dict(summary)
    printable["query_results"] = summary["query_results"][:3]
    print(json.dumps(to_primitive(printable), ensure_ascii=False, indent=2))


def _evaluate_locomo_questions(
    *,
    system: MemorySystem,
    samples: list[dict[str, Any]],
    session_limit: int,
    qa_limit: int,
    top_k: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample in samples:
        for qa in sample.get("qa", []) or []:
            if len(results) >= qa_limit:
                return results
            if not isinstance(qa, dict):
                continue
            gold = evidence_session_uuids(sample, list(qa.get("evidence") or []))
            if not gold or any(_session_number_from_uuid(session_uuid) > session_limit for session_uuid in gold):
                continue
            question = str(qa.get("question") or "").strip()
            if not question:
                continue
            retrieval = system.retrieve(question, top_k=top_k)
            retrieved: list[str] = []
            for hit in retrieval.hits:
                record = system.graph.get_record(hit.source_record_id or hit.item_id)
                if record is not None and record.session_uuid not in retrieved:
                    retrieved.append(record.session_uuid)
            results.append(
                {
                    "question": question,
                    "answer": str(qa.get("answer") or ""),
                    "category": qa.get("category"),
                    "evidence": list(qa.get("evidence") or []),
                    "gold_session_uuids": gold,
                    "retrieved_session_uuids": retrieved,
                    "recall_any": bool(set(gold) & set(retrieved)),
                    "top_hit_record_id": retrieval.hits[0].source_record_id if retrieval.hits else "",
                    "top_hit_score": retrieval.hits[0].score if retrieval.hits else 0.0,
                }
            )
    return results


def _prompt_payload(prompt: str) -> dict[str, Any]:
    match = re.search(r"## Input\s*(\{.*?\}|\[.*?\])\s*## Output format", prompt, flags=re.S | re.I)
    if match is None:
        match = re.search(r"<[A-Za-z0-9_]*Input>\s*(\{.*?\}|\[.*?\])\s*</[A-Za-z0-9_]*Input>", prompt, flags=re.S)
    if match is None:
        return {}
    parsed = json.loads(match.group(1))
    return parsed if isinstance(parsed, dict) else {}


def _session_keys(conversation: dict[str, Any]) -> list[tuple[int, str]]:
    keys: list[tuple[int, str]] = []
    for key, value in conversation.items():
        match = re.fullmatch(r"session_(\d+)", str(key))
        if match and isinstance(value, list):
            keys.append((int(match.group(1)), str(key)))
    keys.sort(key=lambda item: item[0])
    return keys


def _format_turn(turn: dict[str, Any]) -> str:
    speaker = str(turn.get("speaker") or "Unknown").strip() or "Unknown"
    text = str(turn.get("text") or turn.get("content") or "").strip()
    return f"{speaker}: {text}" if text else f"{speaker}:"


def _parse_locomo_datetime(value: str) -> str:
    text = " ".join(str(value or "").replace("\u00a0", " ").split())
    for pattern in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(text.upper(), pattern).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return "1970-01-01 00:00:00"


def _record_base_time(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime(1970, 1, 1)


def _session_uuid(sample_id: str, session_number: int) -> str:
    return f"{sample_id}::session_{session_number}"


def _session_number_from_uuid(session_uuid: str) -> int:
    match = re.search(r"::session_(\d+)$", str(session_uuid))
    return int(match.group(1)) if match else 0


def _slug(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "sample"


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in STOP_WORDS
    ]


def _top_terms(text: str, *, limit: int) -> list[str]:
    counts = Counter(_tokens(text))
    return [term for term, _count in counts.most_common(limit)]


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if item and key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0 or math.isnan(denominator):
        return 0.0
    return round(max(0.0, float(np.dot(left, right) / denominator)), 6)


if __name__ == "__main__":
    main()
