# RealMem Online No-Task-Chain Ablation Design

## Goal

Add an independent RealMemBench evaluation entry point:

```text
tcmem/evals/realmem_no_task_chain.py
```

The entry point runs a fresh online evaluation while disabling task-chain
construction and task-chain retrieval. It preserves the remaining TCMem
retrieval components so the experiment measures the contribution of task
chains rather than replacing the full retrieval system.

## Ablation Boundary

The online ablation keeps:

- entity extraction for each ingested dialogue record;
- entity graph insertion and graph-edge construction;
- vector-index synchronization;
- BM25 indexing;
- Path B retrieval through vector seeds, BM25 seeds, and graph expansion;
- top-session Recall and NDCG metrics;
- optional answer generation and QA judging through `--with-qa`.

The online ablation removes:

- record-to-task routing;
- task-chain creation and task-chain node writes;
- task metadata refresh;
- query-to-task routing;
- Path A retrieval;
- task-chain context scores, route scores, and chain-status penalties in Path B.

## Architecture

### Independent Evaluation Entry Point

Add `tcmem/evals/realmem_no_task_chain.py` as the explicit command-line entry
point for the ablation. It should reuse the existing RealMem top-session
evaluation flow instead of copying the full evaluator.

The ablation entry point must force:

```text
task_chain_enabled=False
mode=tcmem_no_task_chain_online
```

The mode must appear in the run manifest and generated report so artifacts are
unambiguous.

### Shared Evaluator Extension

Extend `tcmem/evals/realmem_top_session.py` with a small reusable hook for
constructing the `TCMemConfig` and selecting the evaluation mode. The default
entry point must keep its current behavior:

```text
task_chain_enabled=True
mode=tcmem_top_session
```

The new ablation entry point uses the same hook with task chains disabled and
the ablation mode label.

### No-Task-Chain Ingestion

Update `MemorySystem.ingest_record()` for `task_chain_enabled=False` so it still
extracts entities before graph insertion. The flow is:

1. extract record entities;
2. add the record to the entity graph;
3. synchronize the vector and BM25 indexes;
4. log ingestion with `task_chain_enabled=False`;
5. return without record routing or chain writes.

No pending task-route buffer is populated in this mode.

### No-Task-Chain Retrieval

The existing retrieval engine already supports the required retrieval
behavior when `task_chain_enabled=False`:

- it creates a local disabled-chain query decision without calling query task
  routing;
- it skips Path A;
- it executes Path B;
- it skips task-chain context scoring;
- it returns `path_b_vector_bm25_graph_no_task_chain`.

The implementation should preserve this behavior and verify it through tests.

## Artifacts

The online ablation writes the same artifact structure as the standard
top-session evaluator:

- `manifest.json`
- `memory_state.json`
- `memory_state_latest.json`
- `retrieval_results.json`
- `generation_results.json`
- `metrics_results.json`
- `realmem_top_session_report.md`

The manifest and report identify the run as `tcmem_no_task_chain_online`.
The saved memory state contains graph records and edges but no constructed
tasks.

## CLI

The new module supports the same CLI arguments as
`tcmem.evals.realmem_top_session`, including `--with-qa`. Its defaults remain
aligned with the standard evaluator so paired experiments differ only in the
task-chain ablation.

Example:

```bash
python -m tcmem.evals.realmem_no_task_chain \
  --dataset ../RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
  --embedding-model /data/wz/models/bgem3 \
  --embedding-device cuda \
  --verbose
```

## Error Handling

The no-task-chain run still requires the configured LLM API key because entity
extraction remains enabled. Query-level failures retain the existing evaluator
behavior: log the failure, emit empty retrieval metrics for the failed query,
save current state, and continue unless `--fail-fast` is set.

## Testing

Add focused contract tests:

1. Disabled-chain ingestion extracts entities, creates entity graph edges, and
   constructs no tasks.
2. Disabled-chain retrieval skips query task routing and returns only Path B
   hits without chain context fields.
3. The new ablation CLI delegates to the shared evaluator with
   `task_chain_enabled=False` and mode `tcmem_no_task_chain_online`.
4. The standard evaluator retains `task_chain_enabled=True` and mode
   `tcmem_top_session`.
5. The report and manifest expose the selected mode.

