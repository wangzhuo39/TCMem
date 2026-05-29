# TCMem

TCMem is a task-chain memory project organized like a normal memory system:
configuration, embedding client, vector index, storage repository, retrieval
engine, and public facade are separated.

The special structures are still project-specific:

- dialogue graph: stores records and entity co-occurrence edges
- task chain: stores routed task nodes and branch/status metadata

Everything else follows the same engineering shape as `xMemory`: persistent
vector storage, explicit embedding client, filesystem state, and one facade for
ingestion/retrieval.

## Environment

```bash
conda env create -f environment.yml
conda activate tcmem
```

If the environment already exists:

```bash
conda activate tcmem
pip install -e .[dev]
```

For this machine, CUDA 12.8 is available, so the environment file installs
`torch==2.6.0+cu124` from the PyTorch CUDA 12.4 wheel index. That is compatible
with the current driver and can load the local BGE-M3 `.bin` checkpoint while
keeping embedding inference on GPU when `embedding_device="auto"`.

## Smoke Test

```bash
pytest -q
```

## RealMemBench Top-Session Evaluation

From `/data/wz/agent_memory/iconip2026`:

```bash
conda run -n tcmem python -m tcmem.evals.realmem_top_session \
  --dataset RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
  --config zhuo/runtime_config.json \
  --run-name tcmem_realmem_adeleke_top_session \
  --output-dir result/results/tcmem_realmem_adeleke_top_session \
  --log-dir result/logs \
  --embedding-model /data/wz/models/bgem3 \
  --retrieval-record-k 200 \
  --session-ks 5,10,20 \
  --verbose
```

Add `--with-qa` to generate answers from the top sessions and judge QA.
During evaluation, `dataset.jsonl` records dataset-level events only. Use
`progress.jsonl` for live progress across sessions, records, and queries.

## Minimal Usage

```python
from tcmem import TCMemConfig, MemorySystem

config = TCMemConfig(
    owner_id="adeleke",
    storage_path="data/adeleke",
    embedding_model="BAAI/bge-m3",
    vector_index_backend="chroma",
    vector_index_path="data/adeleke/vector_db",
)
system = MemorySystem(config=config, llm_client=client)
```

Query routing is intentionally strict: when no LLM client is available, routing
raises an error instead of silently falling back.
