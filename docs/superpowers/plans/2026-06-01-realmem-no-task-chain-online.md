# RealMem No-Task-Chain Online Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tcmem/evals/realmem_no_task_chain.py` as an online RealMemBench ablation entry point that disables task-chain construction while preserving entity extraction, graph retrieval, vector retrieval, BM25 retrieval, and optional QA judging.

**Architecture:** Reuse the existing `realmem_top_session` evaluator by adding small hooks for evaluation mode and task-chain enablement. Keep the new ablation module as a thin entry point, and adjust disabled-chain ingestion so entity extraction still feeds graph edges.

**Tech Stack:** Python 3.10+, `unittest`/`pytest`, existing TCMem `MemorySystem`, `TCMemConfig`, RealMem evaluator utilities.

---

### Task 1: Disabled-Chain Ingestion Keeps Entity Graph

**Files:**
- Test: `tests/test_realmem_no_task_chain_eval.py`
- Modify: `tcmem/core/memory_system.py`

- [ ] **Step 1: Write the failing test**

Add this test file:

```python
import json
import tempfile
import unittest
from pathlib import Path

from tcmem.config import TCMemConfig
from tcmem.core.memory_system import MemorySystem
from tcmem.infrastructure.indices import NumpyVectorIndex
from tcmem.models import DialogueRecord


class FakeEmbeddingClient:
    def embed_texts(self, texts):
        return [[1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0] for text in texts]

    def score(self, query, text):
        query_has_alpha = "alpha" in query.lower()
        text_has_alpha = "alpha" in text.lower()
        return 1.0 if query_has_alpha == text_has_alpha else 0.1


class EntityOnlyLLMClient:
    def __init__(self):
        self.payloads = []

    def generate(self, prompt, **_kwargs):
        payload = json.loads(prompt[prompt.index("{") : prompt.rindex("}") + 1])
        self.payloads.append(payload)
        if "record" in payload:
            return json.dumps({"entities": ["alpha"]})
        raise AssertionError(f"Unexpected task-chain prompt: {prompt}")


class RealMemNoTaskChainEvalTest(unittest.TestCase):
    def _system(self, tmpdir, llm):
        config = TCMemConfig(
            storage_path=str(Path(tmpdir) / "state"),
            log_path=str(Path(tmpdir) / "logs"),
            vector_index_backend="numpy",
            vector_index_path=str(Path(tmpdir) / "vector_index"),
            task_chain_enabled=False,
        )
        return MemorySystem(
            config=config,
            llm_client=llm,
            embedding_client=FakeEmbeddingClient(),
            record_index=NumpyVectorIndex(Path(tmpdir) / "vector_index", index_name="records"),
        )

    def test_disabled_task_chain_ingestion_keeps_entities_graph_edges_and_no_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            llm = EntityOnlyLLMClient()
            system = self._system(tmpdir, llm)
            first = DialogueRecord(
                record_id="rec_1",
                session_identifier="s1",
                session_uuid="uuid_1",
                current_time="2026-06-01",
                user_content="alpha planning",
            )
            second = DialogueRecord(
                record_id="rec_2",
                session_identifier="s2",
                session_uuid="uuid_2",
                current_time="2026-06-01",
                user_content="alpha follow up",
            )

            system.ingest_record(first)
            system.ingest_record(second)

        self.assertEqual(first.entities, ["alpha"])
        self.assertEqual(second.entities, ["alpha"])
        self.assertGreater(system.state_summary()["edge_count"], 0)
        self.assertEqual(system.state_summary()["task_count"], 0)
        self.assertEqual(len(llm.payloads), 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_disabled_task_chain_ingestion_keeps_entities_graph_edges_and_no_tasks -q
```

Expected: FAIL because disabled-chain ingestion currently skips entity extraction and graph edges do not appear.

- [ ] **Step 3: Write minimal implementation**

In `MemorySystem.ingest_record()`, change the disabled-chain branch to call:

```python
self.task_manager.extract_record_entities(record)
```

before `self.graph.add_record(record)`.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_disabled_task_chain_ingestion_keeps_entities_graph_edges_and_no_tasks -q
```

Expected: PASS.

### Task 2: Shared Top-Session Evaluator Hooks

**Files:**
- Test: `tests/test_realmem_no_task_chain_eval.py`
- Modify: `tcmem/evals/realmem_top_session.py`

- [ ] **Step 1: Write the failing tests**

Append these tests:

```python
from tcmem.evals import realmem_top_session


class RealMemNoTaskChainEvalTest(unittest.TestCase):
    def test_standard_eval_options_keep_task_chain_enabled(self):
        args = realmem_top_session.parse_args([])

        options = realmem_top_session.evaluation_options_from_args(args)

        self.assertEqual(options.mode, "tcmem_top_session")
        self.assertTrue(options.task_chain_enabled)

    def test_config_builder_can_disable_task_chains_for_ablation(self):
        args = realmem_top_session.parse_args(["--vector-index-backend", "numpy"])

        config = realmem_top_session.build_tcmem_config(
            args,
            output_dir=Path("out"),
            log_store_base_dir="logs",
            runtime=realmem_top_session.RuntimeConfig(api_key="key"),
            task_chain_enabled=False,
        )

        self.assertFalse(config.task_chain_enabled)
        self.assertEqual(config.vector_index_backend, "numpy")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_standard_eval_options_keep_task_chain_enabled tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_config_builder_can_disable_task_chains_for_ablation -q
```

Expected: FAIL because `evaluation_options_from_args` and `build_tcmem_config` do not exist.

- [ ] **Step 3: Write minimal implementation**

Add an `EvaluationOptions` dataclass to `realmem_top_session.py` with:

```python
@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    mode: str = "tcmem_top_session"
    task_chain_enabled: bool = True
```

Add:

```python
def evaluation_options_from_args(_args: argparse.Namespace) -> EvaluationOptions:
    return EvaluationOptions()
```

Extract the existing `TCMemConfig(...)` construction into:

```python
def build_tcmem_config(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    log_store_base_dir: str | Path,
    runtime: RuntimeConfig,
    task_chain_enabled: bool,
) -> TCMemConfig:
    return TCMemConfig(..., task_chain_enabled=task_chain_enabled, ...)
```

Use this helper from `run_evaluation()`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_standard_eval_options_keep_task_chain_enabled tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_config_builder_can_disable_task_chains_for_ablation -q
```

Expected: PASS.

### Task 3: New No-Task-Chain Evaluation Module

**Files:**
- Create: `tcmem/evals/realmem_no_task_chain.py`
- Test: `tests/test_realmem_no_task_chain_eval.py`
- Modify: `tcmem/evals/realmem_top_session.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from tcmem.evals import realmem_no_task_chain


class RealMemNoTaskChainEvalTest(unittest.TestCase):
    def test_no_task_chain_eval_options_force_ablation_mode(self):
        args = realmem_no_task_chain.parse_args([])

        options = realmem_no_task_chain.evaluation_options_from_args(args)

        self.assertEqual(options.mode, "tcmem_no_task_chain_online")
        self.assertFalse(options.task_chain_enabled)

    def test_render_report_includes_mode(self):
        report = realmem_top_session.render_report(
            run_config={"mode": "tcmem_no_task_chain_online", "model": "m", "retrieval_record_k": 1, "session_ks": [1], "with_qa": False, "tcmem_config": {}},
            dataset_summary={"person_name": "p", "session_count": 1, "query_count": 1, "record_count": 1},
            metrics_summary={"query_count": 1},
        )

        self.assertIn("- mode: tcmem_no_task_chain_online", report)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_no_task_chain_eval_options_force_ablation_mode tests/test_realmem_no_task_chain_eval.py::RealMemNoTaskChainEvalTest::test_render_report_includes_mode -q
```

Expected: FAIL because `realmem_no_task_chain.py` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `tcmem/evals/realmem_no_task_chain.py`:

```python
from __future__ import annotations

import argparse
import json

from ..serialization import to_primitive
from . import realmem_top_session
from .realmem_top_session import EvaluationOptions


def evaluation_options_from_args(_args: argparse.Namespace) -> EvaluationOptions:
    return EvaluationOptions(mode="tcmem_no_task_chain_online", task_chain_enabled=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return realmem_top_session.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict:
    return realmem_top_session.run_evaluation(args, options=evaluation_options_from_args(args))


def main() -> None:
    result = run_evaluation(parse_args())
    print(json.dumps(to_primitive(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

Update `realmem_top_session.run_evaluation()` to accept `options: EvaluationOptions | None = None`, default to `evaluation_options_from_args(args)`, write `options.mode` to `run_config["mode"]`, and pass `options.task_chain_enabled` into `build_tcmem_config()`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py -q
```

Expected: PASS.

### Task 4: Regression Verification

**Files:**
- Verify: `tests/test_realmem_no_task_chain_eval.py`
- Verify: `tests/test_realmem_top_session_eval.py`
- Verify: `tests/test_tcmem_contract.py`

- [ ] **Step 1: Run targeted evaluator tests**

Run:

```bash
pytest tests/test_realmem_no_task_chain_eval.py tests/test_realmem_top_session_eval.py -q
```

Expected: PASS.

- [ ] **Step 2: Run relevant memory-system contract tests**

Run:

```bash
pytest tests/test_tcmem_contract.py -q
```

Expected: PASS.

- [ ] **Step 3: Check diff**

Run:

```bash
git diff -- tcmem/evals/realmem_no_task_chain.py tcmem/evals/realmem_top_session.py tcmem/core/memory_system.py tests/test_realmem_no_task_chain_eval.py
```

Expected: Diff contains only the ablation entry point, evaluator hooks, disabled-chain entity extraction, and tests.

