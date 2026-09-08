# TCMem

TCMem is a task-chain memory system for long-context dialogue memory. It stores
dialogue records, routes records into task chains, builds entity-linked dialogue
graphs, and retrieves evidence through a two-path retrieval engine.

The project is designed for RealMemBench-style evaluation, especially
top-session recall: retrieve many records, aggregate them into ranked sessions,
then optionally generate an answer from the retrieved session text and judge QA
quality.

## What TCMem Contains

- **Dialogue records**: user/assistant turn pairs parsed from sessions.
- **Task chains**: LLM-routed workstreams that connect related records.
- **Dialogue graph**: entity co-occurrence graph over records.
- **Vector index**: persistent record-level semantic index, usually Chroma.
- **Two-path retrieval**:
  - Path A: query intent router -> task chains -> task-chain nodes.
  - Path B: vector retrieval -> graph expansion -> task-chain context scoring.
- **External prompts**: TCMem core prompts live in editable `.txt` files.
- **RealMemBench top-session evaluator**: evaluates recall over ranked sessions,
  not only raw records.

## Repository Layout

```text
TCMem/
  tcmem/
    api/                    Public facade
    core/
      memory_system.py      End-to-end ingestion/retrieval orchestration
      task_chain.py         Entity extraction, record routing, query routing
      graph_store.py        Entity co-occurrence dialogue graph
      retrieval.py          Path A / Path B retrieval fusion
    evals/
      realmem_top_session.py
    infrastructure/
      indices.py            Chroma and numpy vector index backends
    prompts/
      entity_extraction.txt
      task_routing.txt
      query_routing.txt
      task_metadata_refresh.txt
      default_prompts.yaml  RealMem answer/judge prompt defaults
      registry.py
    utils/
      embedding_client.py   sentence-transformers embedding client
      llm_client.py         OpenAI-compatible chat client
    config.py
    models.py
  tests/
  environment.yml
  pyproject.toml
```

## Environment

Create the conda environment:

```bash
cd /data/wz/agent_memory/iconip2026/TCMem
conda env create -f environment.yml
conda activate tcmem
```

If the environment already exists:

```bash
conda activate tcmem
pip install -e .[dev]
```

On this machine, the intended embedding model is:

```text
/data/wz/models/bgem3
```

Use it with:

```bash
--embedding-model /data/wz/models/bgem3
--embedding-device cuda
```

## Minimal Usage

```python
from tcmem import DialogueRecord, MemorySystem, TCMemConfig

config = TCMemConfig(
    owner_id="demo",
    storage_path="data/demo/state",
    log_path="logs",
    embedding_model="/data/wz/models/bgem3",
    embedding_device="cuda",
    vector_index_backend="chroma",
    vector_index_path="data/demo/vector_index",
    prompt_path="tcmem/prompts",
)

system = MemorySystem(config=config)

record = DialogueRecord(
    record_id="rec_demo_0001",
    session_identifier="demo_session",
    session_uuid="demo_uuid",
    current_time="2026-05-30",
    user_content="I want to build a wealth management plan.",
    assistant_content="A good next step is to list liabilities and debts.",
)

system.ingest_record(record)
result = system.retrieve("What should I do before investing?", top_k=20)
print(result.routed_task_ids)
print(result.hits[:3])
```

An LLM client is required for entity extraction, record routing, query routing,
and task metadata refresh. If no API key/client is available, routing raises an
error instead of silently falling back.

## Ingestion Flow

For each `DialogueRecord`, `MemorySystem.ingest_record()` runs:

1. **Entity extraction**
   - Prompt: `tcmem/prompts/entity_extraction.txt`
   - Output: `{"entities": [...]}`
   - Invalid outputs are retried. If entity extraction remains invalid after
     retries, the record can continue with empty entities.

2. **Graph insertion**
   - The record is added to `DialogueGraphStore`.
   - If it shares entities with previous records, entity co-occurrence edges are
     added.

3. **Record intent understanding**
   - Prompt: `tcmem/prompts/record_intent_understanding.txt`
   - Output: one intent object used by downstream routing, including
     `main_intent`, `topic_hint`, `key_entities`, `is_substantive`, and
     `reason`.

4. **Record task routing**
   - Prompt: `tcmem/prompts/task_routing.txt`
   - Output:
     ```json
     {
       "linked_task_ids": ["..."],
       "new_tasks": [{"task_description": "..."}],
       "confidence": 0.8,
       "reason": "..."
     }
     ```
   - Input includes the upstream `record_intent_understanding` result.

5. **Record route review**
   - Prompt: `tcmem/prompts/task_routing_review.txt`
   - Reviews the draft route against the upstream intent before any new task is
     materialized.
   - Every substantive record must connect to at least one task.
   - If the LLM returns an empty route, TCMem retries.
   - If all retries still return no task, ingestion fails loudly.

6. **Record conflict resolution and task-chain write**
   - Prompt: `tcmem/prompts/record_conflict_resolution.txt`
   - Routed records are written with `create`, `override`, `branch`, or
     `duplicate` actions.
   - Nodes preserve source record ids, branch ids, status, and chain position.

7. **Task metadata refresh**
   - Prompt: `tcmem/prompts/task_metadata_refresh.txt`
   - Controlled by `task_metadata_refresh_interval`.
   - Default: refresh every 5 records per task.
   - `task_description`/`canonical_description` is the stable root-task identity.
   - Ordinary refresh updates `current_focus` and merges new entities; it does
     not overwrite the root-task identity.
   - Branches have independent goals, focus, heads, lifecycle state, and
     branch-local child-task links.
   - Child tasks link to their parent through `parent_task_id` and
     `child_task_ids`; when the work belongs to a specific route they also
     carry `parent_branch_id` and appear in that branch's `child_task_ids`.

8. **Vector index sync**
   - `sync_record_index()` persists record vectors for retrieval.

## Retrieval Flow

`MemorySystem.retrieve(query, top_k)` calls `RetrievalEngine.retrieve()`.

### Step 1: Query Intent Understanding

Prompt:

```text
tcmem/prompts/query_intent_understanding.txt
```

Output:

```json
{
  "main_intent": "...",
  "topic_hint": "...",
  "key_entities": ["..."],
  "is_substantive": true,
  "reason": "..."
}
```

### Step 2: Query Routing

Prompt:

```text
tcmem/prompts/query_routing.txt
```

Input:

```json
{
  "query": "...",
  "intent": {
    "main_intent": "...",
    "topic_hint": "...",
    "key_entities": ["..."],
    "is_substantive": true,
    "reason": "..."
  },
  "candidate_count": 5,
  "task_catalog": [
    {
      "task_id": "...",
      "canonical_description": "...",
      "current_focus": "...",
      "status": "active",
      "entities": ["..."],
      "parent_task_id": null,
      "parent_branch_id": null,
      "child_task_ids": [],
      "branches": []
    }
  ]
}
```

Output:

```json
{
  "routed_task_ids": ["..."],
  "reason": "..."
}
```

The final `routed_task_ids`, deterministic `expanded_task_ids`, `query_intent`,
and `query_route_reason` are saved in retrieval logs and evaluation outputs.

### Step 3: Path A, Task-Chain Retrieval

Path A starts from LLM-selected `routed_task_ids`, expands explicit parent,
child, and branch-local child links into `expanded_task_ids`, then scores the
corresponding task-chain nodes. The primary route remains separately visible
for diagnosing whether recall came from the LLM router or hierarchy expansion:

```text
path_a_score =
  path_a_semantic_weight * semantic_score
+ path_a_status_weight   * status_score
+ path_a_chain_weight    * chain_score
```

Default weights:

```text
path_a_semantic_weight = 0.7
path_a_status_weight   = 0.2
path_a_chain_weight    = 0.1
```

### Step 4: Path B, Vector + Graph Retrieval

Path B first retrieves both vector seeds and BM25 seeds:

```text
graph_seed_limit      = 12
graph_bm25_seed_limit = 12
```

The union of vector and BM25 seeds is merged, and each merged seed uses this blended score, with any missing signal contributing `0`:

```text
seed_score =
  graph_vector_seed_weight * semantic_seed_score
+ graph_bm25_seed_weight   * bm25_seed_score
```

Then it walks the dialogue graph:

```text
graph_walk_depth = 2
```

Each graph-expanded record receives:

```text
graph_score = seed_score / (depth + 1)
```

Then Path B combines semantic, BM25, and graph scores:

```text
base_score =
  path_b_semantic_weight * semantic_score
+ path_b_bm25_weight     * bm25_score
+ path_b_graph_weight    * graph_score
```

Default weights:

```text
path_b_semantic_weight   = 0.45
path_b_bm25_weight       = 0.2
path_b_graph_weight      = 0.55
graph_vector_seed_weight = 0.7
graph_bm25_seed_weight   = 0.3
```

Task-chain context may apply a status-based penalty after `base_score`, while route context is used only when selecting the best matching chain context.

### Step 5: Path Fusion

Path A and Path B are merged by record id:

```text
final_score =
  path_a_weight * path_a_score
+ path_b_weight * path_b_score
```

Default:

```text
path_a_weight = 0.6
path_b_weight = 0.4
```

The output is a ranked list of `SearchHit` objects.

## Prompt System

Core TCMem prompts are plain text files:

```text
tcmem/prompts/entity_extraction.txt
tcmem/prompts/record_intent_understanding.txt
tcmem/prompts/task_routing.txt
tcmem/prompts/task_routing_review.txt
tcmem/prompts/record_conflict_resolution.txt
tcmem/prompts/query_intent_understanding.txt
tcmem/prompts/query_routing.txt
tcmem/prompts/task_metadata_refresh.txt
```

These are treated as complete user prompts. The LLM `system_prompt` is empty for
these stages. Each file contains `{{payload_json}}`, which is replaced with
the stage input.

RealMem evaluation prompts live in:

```text
tcmem/prompts/default_prompts.yaml
```

They are aligned with RealMemBench's answer-generation and QA-judge prompts.

Use a custom prompt directory:

```bash
--prompt-path /data/wz/agent_memory/iconip2026/TCMem/tcmem/prompts
```

Or set it in config:

```python
TCMemConfig(prompt_path="tcmem/prompts")
```

## Configuration

Important `TCMemConfig` fields:

| Field | Default | Meaning |
|---|---:|---|
| `embedding_model` | `BAAI/bge-m3` | SentenceTransformer model path/name |
| `embedding_device` | `auto` | `auto`, `cuda`, or `cpu` |
| `embedding_batch_size` | `32` | Embedding batch size |
| `vector_index_backend` | `chroma` | `chroma` or `numpy`; `faiss` is declared but not implemented |
| `vector_index_path` | `TCMem/data/default/vector_db` | Persistent vector index path |
| `vector_index_rebuild` | `False` | Rebuild vector index on sync |
| `path_a_weight` | `0.6` | Final fusion weight for task-chain path |
| `path_b_weight` | `0.4` | Final fusion weight for vector/graph path |
| `graph_seed_limit` | `12` | Number of vector seeds for graph expansion |
| `graph_bm25_seed_limit` | `12` | Number of BM25 seeds for graph expansion |
| `graph_vector_seed_weight` | `0.7` | Weight of vector seed score in blended graph seed strength |
| `graph_bm25_seed_weight` | `0.3` | Weight of BM25 seed score in blended graph seed strength |
| `graph_walk_depth` | `2` | Max graph expansion depth |
| `path_b_bm25_weight` | `0.2` | BM25 contribution in Path B record ranking |
| `routed_task_score` | `1.0` | Route score for records in routed tasks |
| `unrouted_task_score` | `0.3` | Route score for records outside routed tasks |
| `task_metadata_refresh_interval` | `5` | Refresh task metadata every N records per task; `0` disables |
| `task_router_entity_limit` | `20` | Max entities shown per task in router inputs |
| `query_router_candidate_count` | `5` | Maximum number of tasks returned by the LLM query router |
| `query_router_pool_size` | `48` | Maximum number of locally prefiltered tasks sent to the LLM query router |
| `query_router_summary_limit` | `512` | Maximum characters for each compact task routing summary |
| `prompt_path` | `""` | Prompt file or directory override |
| `llm_api_key` | env | `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` |
| `llm_base_url` | `https://api.deepseek.com` | OpenAI-compatible endpoint |
| `llm_model` | `deepseek-v4-flash` | Chat model name |

## RealMemBench Evaluation

The evaluator is:

```bash
python -m tcmem.evals.realmem_top_session
```

It performs top-session evaluation:

1. Ingest all dialogue records before a query.
2. Retrieve many records for the query.
3. Aggregate records into ranked sessions.
4. Compute `recall_any@k`, `recall_all@k`, and `ndcg@k`.
5. Optionally generate an answer from top session text and judge QA.

### Small Prefix Test

```bash
cd /data/wz/agent_memory/iconip2026/TCMem

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n tcmem python -m tcmem.evals.realmem_top_session \
  --dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_prefix_for_first5_queries.json \
  --config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
  --run-name tcmem_realmem_adeleke_prefix5_prompt_txt_20260530 \
  --output-dir /data/wz/agent_memory/iconip2026/result/results/tcmem_realmem_adeleke_prefix5_prompt_txt_20260530 \
  --log-dir /data/wz/agent_memory/iconip2026/result/logs \
  --embedding-model /data/wz/models/bgem3 \
  --embedding-device cuda \
  --vector-index-backend chroma \
  --session-ks 5,10,20,30 \
  --retrieval-record-k 200 \
  --evidence-top-k 20 \
  --build-model gpt-4o-mini \
  --eval-model gpt-4o \
  --state-save-every-records 5 \
  --prompt-path /data/wz/agent_memory/iconip2026/TCMem/tcmem/prompts \
  --with-qa \
  --verbose
```

### Full Adeleke Test

```bash
cd /data/wz/agent_memory/iconip2026/TCMem

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run -n tcmem python -m tcmem.evals.realmem_top_session \
  --dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
  --config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
  --run-name tcmem_realmem_adeleke_full_prompt_txt_20260530 \
  --output-dir /data/wz/agent_memory/iconip2026/result/results/tcmem_realmem_adeleke_full_prompt_txt_20260530 \
  --log-dir /data/wz/agent_memory/iconip2026/result/logs \
  --embedding-model /data/wz/models/bgem3 \
  --embedding-device cuda \
  --vector-index-backend chroma \
  --session-ks 5,10,20,30 \
  --retrieval-record-k 200 \
  --evidence-top-k 20 \
  --build-model gpt-4o-mini \
  --eval-model gpt-4o \
  --state-save-every-records 5 \
  --prompt-path /data/wz/agent_memory/iconip2026/TCMem/tcmem/prompts \
  --with-qa \
  --verbose
```

To run retrieval only, remove:

```text
--with-qa
```

## Evaluation Outputs

Given:

```text
--run-name RUN
--output-dir result/results/RUN
--log-dir result/logs
```

Logs are written under:

```text
result/logs/RUN/
```

Important files:

| File | Meaning |
|---|---|
| `dataset.jsonl` | Dataset and run configuration events |
| `progress.jsonl` | Live progress across records, sessions, and queries |
| `ingestion.jsonl` | Record ingestion events and final task routing ids |
| `retrieval.jsonl` | Query retrieval events, routed task ids, `query_intent`, `query_route_reason`, and hits |
| `query_results.jsonl` | Completed/failed query payloads |
| `metrics.jsonl` | Cumulative metric snapshots |
| `llm_errors.jsonl` | JSON decode errors, empty routes, invalid payloads |

Final results are written under:

```text
result/results/RUN/
```

Important files:

| File | Meaning |
|---|---|
| `memory_state_latest.json` | Incremental graph/task-chain snapshot |
| `memory_state.json` | Final graph/task-chain snapshot |
| `retrieval_results.json` | Query retrieval outputs |
| `generation_results.json` | QA generated answers and evidence text |
| `metrics_results.json` | Summary and per-query metrics |
| `realmem_top_session_report.md` | Human-readable report |
| `progress.jsonl` | Mirrored live progress for tailing from the result directory |
| `progress_latest.json` | Latest progress event for quick status checks |
| `manifest.json` | Run metadata and artifact paths |

Current successful router results are persisted as filtered ids:

- Task router: `ingestion.jsonl` stores `routed_task_ids`.
- Query router: `retrieval.jsonl`, `query_results.jsonl`, and
  `retrieval_results.json` store `routed_task_ids`, `query_intent`, and
  `query_route_reason`.

Successful router `query_route_reason` is now persisted for retrieval.
Full raw LLM outputs are still logged only on error paths.

## Retrieval Path Ablation

To test Path A / Path B contributions, keep all other parameters fixed and vary:

```bash
--path-a-weight A
--path-b-weight B
```

Recommended first sweep:

| Run | `path_a_weight` | `path_b_weight` |
|---|---:|---:|
| Path A only | 1.0 | 0.0 |
| Path B only | 0.0 | 1.0 |
| A dominant | 0.8 | 0.2 |
| Current | 0.6 | 0.4 |
| Balanced | 0.5 | 0.5 |
| B dominant | 0.3 | 0.7 |
| Strong B | 0.2 | 0.8 |

For fast first-pass experiments, remove `--with-qa` and compare:

```text
recall_all@20
recall_all@30
ndcg@20
failed_query_count
```

Then rerun the best 2-3 configurations with `--with-qa`.

## Retrieval Parameter Sweep (deferred for this experiment)

The parameter sweep is retained as legacy tooling/documentation only. Per the
current experiment plan, do not run it; use the fixed Full-baseline versus
w/o-task-chain commands and `scripts/run_fixed_ablation_resume.sh` instead.

The historical sweep command below is not part of the current ablation result.

After an evaluation has produced `memory_state_latest.json`, run retrieval-only
sweeps without re-ingesting records:

```bash
cd /data/wz/agent_memory/iconip2026/TCMem

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. conda run --no-capture-output -n tcmem python -m tcmem.evals.realmem_retrieval_sweep \
  --dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
  --state-path /data/wz/agent_memory/iconip2026/result/results/tcmem_realmem_adeleke_full_prompt_txt_20260530/memory_state_latest.json \
  --config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
  --session-ks 10,20,30 \
  --objective recall_all@20 \
  --retrieval-record-k 300 \
  --preset small \
  --output-dir /data/wz/agent_memory/iconip2026/result/sweeps/tcmem_realmem_adeleke_retrieval_sweep_20260530 \
  --checkpoint-every 1 \
  --verbose
```

`conda run` captures `stdout`/`stderr` by default, so live progress can look
stuck even when the sweep is running. Use `--no-capture-output` (or
`--live-stream`) if you want to see `[trial-start]`, `[query-start]`, and
`[query]` lines as they happen.

Useful presets:

| Preset | Trials | Parameters covered |
|---|---:|---|
| `small` | 32 | path A/B fusion, graph seed/depth, path B semantic/graph weights |
| `medium` | 144 | wider graph seed/depth plus path B weights |
| `large` | 432 | `medium` plus status penalties |

For a focused custom grid:

```bash
--grid-json '{
  "path_a_weight": [0.2, 0.4, 0.6],
  "path_b_weight": [0.8, 0.6, 0.4],
  "graph_seed_limit": [12, 24, 48],
  "graph_walk_depth": [2, 3],
  "path_b_semantic_weight": [0.35, 0.45, 0.55],
  "path_b_graph_weight": [0.65, 0.55, 0.45],
  "unrouted_task_score": [0.3, 0.5, 0.7]
}'
```

The sweep writes incremental files after each trial:

| File | Meaning |
|---|---|
| `sweep_events.jsonl` | Fine-grained run, trial, and per-query progress events |
| `sweep_progress.jsonl` | One JSON line per completed trial |
| `sweep_results_partial.json` | Best-so-far and all completed trials |
| `sweep_results_partial.csv` | Spreadsheet-friendly partial results |
| `best_retrieval_config_partial.json` | Best-so-far full config |
| `query_route_cache.json` | Cached query router output |

Final files are `sweep_results.json`, `sweep_results.csv`, and
`best_retrieval_config.json`.

Watch live progress inside a long trial:

```bash
tail -f /data/wz/agent_memory/iconip2026/result/sweeps/tcmem_realmem_adeleke_retrieval_sweep_20260530/sweep_events.jsonl
```

## Useful Commands

Run tests:

```bash
cd /data/wz/agent_memory/iconip2026/TCMem
conda run -n tcmem python -m pytest -q tests
```

Check packaging/config syntax:

```bash
git diff --check
conda run -n tcmem python -m tcmem.evals.realmem_top_session --help
```

Inspect current prompt rendering:

```bash
conda run -n tcmem python - <<'PY'
from tcmem.prompts import PromptRegistry

prompt = PromptRegistry.default().render(
    "task_routing",
    payload_json='{"current_record": {}, "task_catalog": []}',
)
print("system:", repr(prompt.system_prompt))
print(prompt.user_prompt[:1000])
PY
```

## Current Limitations

- `faiss` appears as a config option but is not implemented; use `chroma` or
  `numpy`.
- Successful LLM call prompts and raw responses are not saved by default.
- Successful router reasons are not saved by default.
- The task-level vector/BM25 routing index is not yet implemented; the current
  fixed candidate pool still uses local lexical/entity prefiltering.
- Branch and child-task expansion is deterministic. LLM routing chooses the
  root task and can declare `parent_branch_id` for a newly decomposed child;
  branch-level query scope is still represented inside the task profile rather
  than exposed as a separate query result field.
- Entity extraction can continue with empty entities after repeated invalid
  payloads; this protects long evaluations from stopping but may reduce graph
  quality.

## Development Notes

- Use `rg` for searching code.
- Keep prompts editable in `tcmem/prompts`.
- Do not put API keys in committed config files.
- `TCMemConfig.to_dict()` redacts `llm_api_key` unless
  `include_secrets=True`.
- State is JSON so it is easy to inspect and debug during long runs.
