# TCMem 工程结构

TCMem 参考 `xMemory` 的工程分层，但保留当前项目自己的特殊结构：图结构和任务链。

## 分层对应

| xMemory | TCMem | 作用 |
| --- | --- | --- |
| `src/config.py` | `tcmem/config.py` | 统一配置检索、embedding、向量库、LLM |
| `src/utils/embedding_client.py` | `tcmem/utils/embedding_client.py` | 开源 embedding 模型封装 |
| `src/infrastructure/indices.py` | `tcmem/infrastructure/indices.py` | 向量索引抽象与 Chroma/Numpy 实现 |
| `src/storage/*` | `tcmem/storage/repositories.py` | 文件系统状态保存与加载 |
| `src/core/memory_system.py` | `tcmem/core/memory_system.py` | 统一编排 ingestion、routing、retrieval |
| `src/search/*` | `tcmem/search/unified_search.py` | 对外统一搜索服务 |
| `src/api/facade.py` | `tcmem/api/facade.py` | 用户侧 facade |

## 关键设计

1. Record 有持久化向量库。
   `RetrievalEngine.sync_record_index()` 会把 `DialogueGraphStore.records` 同步到 `VectorIndex`。默认 backend 是 `chroma`，测试和轻量场景可用 `numpy`。

2. Embedding 是工程对象。
   `EmbeddingClient` 使用 `sentence-transformers` 加载开源模型，默认 `BAAI/bge-m3`。`embedding_device="auto"` 时会优先使用 CUDA。

3. Query routing 不做静默降级。
   `TaskChainManager.route_for_query()` 没有 LLM client 时直接报错：`LLM client missing for stage query_routing`。

4. Path A 和 Path B 保持简单加权。
   Path A 来自任务链节点，Path B 来自 record vector seeds + graph walk。最后按 `path_a_weight` 和 `path_b_weight` 融合。

## 主要可调参数

- embedding: `embedding_model`, `embedding_device`, `embedding_batch_size`, `embedding_normalize`
- vector index: `vector_index_backend`, `vector_index_path`, `chroma_collection_prefix`, `vector_index_rebuild`
- Path A/B: `path_a_weight`, `path_b_weight`
- Path A 内部: `path_a_semantic_weight`, `path_a_status_weight`, `path_a_chain_weight`
- Path B 内部: `path_b_semantic_weight`, `path_b_graph_weight`, `graph_seed_limit`, `graph_walk_depth`
- 状态/惩罚: `active_status_score`, `branched_status_score`, `deprecated_status_score`, `superseded_status_score`, `active_penalty`, `branched_penalty`, `deprecated_penalty`, `superseded_penalty`

## 环境

当前已创建 conda 环境：

```bash
conda activate tcmem
```

验证结果：

- `torch 2.5.1+cu121`
- `cuda_available=True`
- `device_count=8`
- `chromadb 1.5.9`
- `faiss 1.14.2`
- `sentence_transformers 5.5.1`
- `transformers 4.57.6`

测试命令：

```bash
conda run -n tcmem python -m pytest -q TCMem/tests
```
