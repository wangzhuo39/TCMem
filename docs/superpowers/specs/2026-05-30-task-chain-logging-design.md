# Task Chain Logging Design

## Goal

Add a per-task-chain audit log for TCMem task-chain updates. Each task chain writes to its own JSONL file so creation, node append/update, and metadata refresh history can be inspected without scanning the shared module logs.

## Scope

This change covers task-chain lifecycle events inside `TaskChainManager`:

- `task_chain_created` when `create_task_from_record()` creates a new task.
- `node_added` when `apply_record()` appends a record as a chain node.
- `metadata_refreshed` when `refresh_task_metadata()` updates task-level description or entities.

It does not change routing behavior, retrieval scoring, saved memory state, or the existing shared module logs.

## Storage

Reuse the existing `ModuleLogStore` run directory. For a run named `20260530_120000`, task `task_abc12345` writes:

```text
<log_path>/20260530_120000/task_chains/task_abc12345.jsonl
```

Each line is one JSON object with the same outer structure as module logs:

```json
{
  "timestamp": "2026-05-30T12:00:00",
  "module": "task_chain",
  "event": "node_added",
  "payload": {
    "task_id": "task_abc12345"
  }
}
```

The file name uses a sanitized task id to prevent path traversal or invalid path separators.

## Interface

Extend `ModuleLogStore` with a focused helper:

```python
log_task_chain(task_id: str, event: str, **payload: Any) -> None
```

`TaskChainManager` calls this helper when available. If a custom `log_store` only exposes the older `log()` method, task-chain logging is skipped instead of failing ingestion.

## Event Payloads

`task_chain_created` records:

- `task_id`
- `owner_id`
- `task_description`
- `created_at`
- `source_record_id`
- `entities`

`node_added` records:

- `task_id`
- `node_id`
- `branch_id`
- `position`
- `prev_node_ids`
- `source_record_id`
- `source_turn_ids`
- `record_count`
- `entity_count`
- `updated_at`

`metadata_refreshed` records:

- `task_id`
- `old_task_description`
- `new_task_description`
- `old_entities`
- `new_entities`
- `changed`

The refresh event is logged after parsing and applying the LLM result, even if the returned metadata is identical, so the refresh attempt remains visible.

## Error Handling

Task-chain logging is best-effort but should use the same file-writing path as `ModuleLogStore.log()`. If the filesystem cannot be written, the behavior remains consistent with existing module logging and the exception surfaces.

## Testing

Add contract tests that:

- Create a task with a temporary `ModuleLogStore` and verify a per-task JSONL file exists with a creation event.
- Append a record and verify the same task file receives a node event.
- Force metadata refresh and verify the same task file receives a refresh event with old and new metadata.
- Verify unsafe task ids are sanitized when building the task-chain log path.

Run the focused TCMem contract tests and then the full TCMem test suite.
