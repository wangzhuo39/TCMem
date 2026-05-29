# BM25 Path B Retrieval Design

## Goal

Integrate BM25 into TCMem Path B retrieval so that BM25 participates in both seed selection and final ranking, while keeping the existing entity graph responsible only for graph construction and expansion.

## Current Behavior

Path B currently works as:

1. Sync all records into the vector index.
2. Retrieve vector seeds from `record_index.search(...)`.
3. Expand each seed over the entity co-occurrence graph with BFS.
4. Score expanded records with semantic score and graph score.
5. Apply task-chain penalties and merge with Path A.

The current implementation does not use `graph_bm25_seed_limit`, `graph_vector_seed_weight`, or `graph_bm25_seed_weight`, even though those fields already exist in config. Tests also already expect a BM25-aware Path B reason string.

## Scope

This design changes only record retrieval in Path B.

Included:

- Record-level BM25 indexing over `DialogueRecord.combined_content`
- BM25 seed retrieval in parallel with vector seed retrieval
- Explicit BM25 contribution in final Path B ranking
- Config support for BM25 ranking weight
- Tests covering BM25 recall and ranking behavior

Excluded:

- Changing entity extraction prompts or behavior
- Replacing graph construction with lexical links
- Changing graph walk from unweighted BFS to weighted traversal
- Refactoring Path A logic

## Design Summary

Path B will become a hybrid retrieval path:

1. Retrieve vector seeds from the vector index.
2. Retrieve BM25 seeds from a record-text BM25 index.
3. Merge the two seed sets by `record_id`.
4. Compute a blended seed score from vector and BM25 seed scores.
5. Expand merged seeds over the existing entity graph.
6. Score each expanded record with semantic score, BM25 score, and graph score.
7. Apply the existing task-chain penalty logic.
8. Merge Path B results with Path A as before.

Entities remain graph-only. BM25 reads raw `record.combined_content` and does not depend on LLM-extracted entities.

## Architecture

### 1. BM25 Index Layer

Add a small in-memory BM25 index for records in `tcmem/infrastructure/indices.py`.

Responsibilities:

- Store `record_id -> text`
- Tokenize `combined_content`
- Build a BM25 model over all current records
- Return ranked `record_id` hits with normalized BM25 scores

The BM25 index should expose a minimal interface parallel to the existing vector index behavior:

- `sync_items(items)`
- `search(query, top_k)`

The BM25 index does not need persistence in this iteration. It can be rebuilt from in-memory records during retrieval sync, matching the current lightweight Path B flow.

### 2. Tokenization Policy

BM25 operates only on `DialogueRecord.combined_content`.

Tokenization should be lightweight and deterministic:

- Lowercase the text
- Split into lexical tokens with regex-based tokenization
- Ignore empty tokens

This iteration does not use entities as BM25 tokens and does not attempt language-specific segmentation beyond a lightweight tokenizer. The goal is to introduce a stable lexical retrieval path without entangling it with the graph pipeline.

### 3. Seed Retrieval

Path B seed retrieval becomes parallel:

- Vector seeds: top `graph_seed_limit`
- BM25 seeds: top `graph_bm25_seed_limit`

Merged seed records keep both seed scores when available:

- `semantic_seed_score`
- `bm25_seed_score`

If a record appears in only one seed source, the missing score is treated as `0.0`.

The merged seed score is:

```text
seed_score =
  graph_vector_seed_weight * semantic_seed_score
+ graph_bm25_seed_weight   * bm25_seed_score
```

This score is used only as the graph expansion source strength.

### 4. Graph Expansion

Graph expansion remains unchanged structurally:

- Start from merged seed record ids
- Walk the existing entity graph with `graph.walk(...)`
- Use the existing `graph_walk_depth`

Expansion scoring remains depth-decayed:

```text
graph_score = seed_score / (depth + 1)
```

If the same expanded record is reached from multiple seeds, keep the highest `graph_score`, as the current implementation already does.

This design intentionally does not use edge weights in traversal or scoring. That keeps the change isolated to lexical retrieval and avoids mixing two ranking changes in one feature.

### 5. Final Path B Ranking

Each expanded record receives three retrieval signals:

- `semantic_score`
- `bm25_score`
- `graph_score`

`semantic_score` is the record-level vector similarity to the query.

`bm25_score` is the record-level BM25 similarity to the query text, whether or not the record was itself a seed.

The base Path B score becomes:

```text
path_b_base =
  path_b_semantic_weight * semantic_score
+ path_b_bm25_weight     * bm25_score
+ path_b_graph_weight    * graph_score
```

The existing task-chain penalty logic still applies after this base score is computed:

```text
path_b_score = path_b_base * penalty
```

### 6. Score Normalization

Vector scores are already returned in a bounded similarity-like range.

BM25 scores are not naturally on the same scale, so the BM25 search results for a query must be normalized before use in either seed blending or final Path B ranking.

Normalization rule for this iteration:

- Use per-query min-max normalization over all BM25 hit scores returned for the current query
- If all raw BM25 scores are equal and positive, normalize them to `1.0`
- If no BM25 hit exists for a record, use `0.0`
- If there are no BM25 hits at all, all BM25 scores are `0.0`

This keeps BM25 contribution bounded without introducing global calibration complexity.

## Data Model Changes

### `SearchHit`

Add:

- `bm25_score: float = 0.0`

This allows retrieval logs, debugging, and evaluation outputs to inspect BM25 contribution directly.

### Config

Use existing fields:

- `graph_bm25_seed_limit`
- `graph_vector_seed_weight`
- `graph_bm25_seed_weight`

Add:

- `path_b_bm25_weight`

Config validation should clamp numeric limits as already done for other retrieval settings.

## Behavior Notes

- BM25 does not replace semantic scoring; it complements it.
- Records can win through lexical match even when vector seeds miss them.
- Graph expansion can still surface neighboring records that are not direct lexical matches.
- Final Path B ranking should reflect all three signals explicitly and transparently.

## Error Handling

- Empty record collection: BM25 search returns no hits.
- Empty query tokenization result: BM25 search returns no hits.
- Missing record during graph expansion: skip it, matching current behavior.
- Missing BM25 score for a non-seed expanded record: use `0.0`.
- Missing semantic seed score for a seed or expanded record: compute semantic score from the record text, matching current fallback behavior.

## Testing Strategy

Add or update tests to cover:

1. BM25 can retrieve a keyword-heavy record when vector seed retrieval misses it.
2. Final Path B ranking changes when BM25 score differs between otherwise similar records.
3. `SearchHit.reason` includes `path_b_vector_bm25_graph`.
4. `SearchHit.bm25_score` is populated for BM25-matched results.
5. Records reached only by graph expansion still receive BM25 score `0.0` when they do not match lexically.
6. Empty BM25 result sets do not break retrieval.

## File Changes

- Modify `tcmem/infrastructure/indices.py`
  - Add BM25 index implementation and score normalization helpers
- Modify `tcmem/core/retrieval.py`
  - Retrieve BM25 seeds
  - Blend seed scores
  - Include BM25 in final Path B scoring
  - Propagate `bm25_score` and new reason string
- Modify `tcmem/config.py`
  - Add `path_b_bm25_weight`
  - Validate the new weight
- Modify `tcmem/models.py`
  - Extend `SearchHit` with `bm25_score`
- Modify `tests/test_tcmem_contract.py`
  - Add red-green tests for BM25 seed recall and final ranking behavior
- Modify `README.md`
  - Update Path B documentation and config reference after implementation

## Success Criteria

- Path B uses both vector and BM25 seeds.
- BM25 score is visible in returned search hits.
- Final Path B ranking explicitly includes BM25 contribution.
- Existing graph behavior remains intact.
- The BM25-aware contract tests pass.
