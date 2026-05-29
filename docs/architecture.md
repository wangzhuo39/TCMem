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
   LLM 返回非法 JSON 时会重试；最终失败会写入 `llm_errors.jsonl`，评估层记录该 query 失败并继续后续 query。
   Router 只接收压缩后的 task summary：`task_id`、`task_description`、最多 20 个 `entities`。Task 的 `topic` 和 task-level `status` 不再参与存储或路由。

4. Record routing 必须入链。
   `TaskChainManager.route_record()` 要求每条 record 至少连接到一个已有 task 或创建一个新 task。LLM 返回空路由时会重试，默认最多 5 次；最终仍为空则报错，避免桥接 record 静默丢失。

5. Task 元信息会定期刷新。
   每个 task 默认每累计 5 条入链 record 触发一次 `task_metadata_refresh`，用最近节点刷新 `task_description` 和 `entities`，避免早期 task 描述长期偏离。

6. Path A 和 Path B 保持简单加权。
   Path A 来自任务链节点，Path B 来自 record vector seeds + graph walk。最后按 `path_a_weight` 和 `path_b_weight` 融合。

7. 评估中间结果实时保存。
   每个 query 的完整结果追加到 `query_results.jsonl`。Graph/task-chain 的最新 JSON 快照保存为 `memory_state_latest.json`，默认每 10 条 record 和每个 query 后更新一次。

## 主要可调参数

- embedding: `embedding_model`, `embedding_device`, `embedding_batch_size`, `embedding_normalize`
- vector index: `vector_index_backend`, `vector_index_path`, `chroma_collection_prefix`, `vector_index_rebuild`
- Path A/B: `path_a_weight`, `path_b_weight`
- Path A 内部: `path_a_semantic_weight`, `path_a_status_weight`, `path_a_chain_weight`
- Path B 内部: `path_b_semantic_weight`, `path_b_graph_weight`, `graph_seed_limit`, `graph_walk_depth`
- task routing: `task_metadata_refresh_interval`, `task_router_entity_limit`
- 状态/惩罚: `active_status_score`, `branched_status_score`, `deprecated_status_score`, `superseded_status_score`, `active_penalty`, `branched_penalty`, `deprecated_penalty`, `superseded_penalty`
- 评估保存: `--state-save-every-records`

## 环境

当前已创建 conda 环境：

```bash
conda activate tcmem
```

验证结果：

- `torch 2.6.0+cu124`
- `cuda_available=True`
- `device_count=8`
- `chromadb 1.5.9`
- `faiss 1.14.2`
- `sentence_transformers 5.5.1`
- `transformers 4.56.2`

测试命令：

```bash
conda run -n tcmem python -m pytest -q TCMem/tests
```
