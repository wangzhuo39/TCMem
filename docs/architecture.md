# TCMem Architecture and Ablation Contract

TCMem separates evidence storage, task-chain state, and retrieval ranking.
The separation is required for a valid online ablation: both variants ingest
the same dialogue prefix and are evaluated before the query turn is ingested.

## State ownership

- `DialogueGraphStore` owns dialogue records and entity-co-occurrence edges.
- `TaskChainManager` owns task chains, branches, nodes, conflict decisions,
  and task metadata refreshes. A task node always keeps its source record id.
- `RetrievalEngine` owns query routing, Path A chain candidates, Path B
  vector/BM25 graph candidates, and score fusion.
- `MemorySystem` owns the online ordering and persistence boundary.

## Intent-aware flow

For ingestion, the task-chain-enabled path extracts entities, adds the record
to the graph, batches records in a routing window, runs
`record_intent_understanding`, then performs task routing and conflict
resolution. New tasks and nodes are materialized only after the routed result
passes the review gate. Metadata refresh updates focus and entities without
changing the canonical root-task identity.

For retrieval, the query first goes through
`query_intent_understanding` and `query_routing`. The selected tasks are
expanded through deterministic parent/child/branch edges. Path A scores chain
nodes; Path B scores vector and BM25 seeds followed by graph expansion, with
task context used only in the enabled variant. Both paths are fused at the
record level before session aggregation.

## Fixed online ablation

The Full baseline sets `task_chain_enabled=true` and retains entity
extraction, graph, vector, BM25, record routing, task-chain updates, query
routing, Path A, and Path B. The `w/o task chain` variant sets
`task_chain_enabled=false`; it retains entity extraction, graph, vector, BM25,
and Path B, while disabling record/query routing, task creation and updates,
Path A, and task-chain context penalties. Its hit reason is
`path_b_vector_bm25_graph_no_task_chain`, and its exported state has zero task
chains.

Each query is evaluated against the state built from records before that query
turn. The current query record is ingested only after evaluation. Therefore a
reported gain measures memory retrieval rather than leakage from the answer
turn itself.
