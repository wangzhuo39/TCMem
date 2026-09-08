# TCMem RealMemBench 任务链消融实验参考

更新时间：2026-09-07

目标仓库：`/data/wz/agent_memory/iconip2026/TCMem`

实验平台：A100 服务器

实验产物根目录：`/data/wz/agent_memory/iconip2026/work`

## 1. 实验目标

本实验比较同一份 RealMemBench 数据上启用任务链与禁用任务链的差异，回答：

> 在实体抽取、实体图、向量检索、BM25 和图扩展都保持不变时，任务路由、任务链构建、任务链上下文和 Path A 是否带来可测量的检索与回答收益？

实验结论必须来自同一数据、同一模型、同一 embedding、同一检索参数和同一 QA 设置的成对运行。不能把旧版本结果和当前版本结果直接做差。

## 2. 两个实验分支

| 分支 | CLI 入口 | `task_chain_enabled` | 保留组件 | 禁用组件 |
| --- | --- | ---: | --- | --- |
| Full baseline | `tcmem.evals.realmem_top_session` | `true` | 实体抽取、实体图、向量、BM25、任务路由、任务链、Path A、Path B、可选 QA | 无 |
| w/o task chain | `tcmem.evals.realmem_no_task_chain` | `false` | 实体抽取、实体图、向量、BM25、Path B、可选 QA | 记录路由、任务链创建/更新、任务元数据刷新、查询路由、Path A、任务链上下文惩罚 |

无任务链分支的正式模式标识为 `tcmem_no_task_chain_online`。其检索命中原因应为 `path_b_vector_bm25_graph_no_task_chain`，且不会产生任务节点。

### 2.1 不是正式结果的近似消融

`tcmem.evals.realmem_task_chain_ablation_from_logs` 可以读取已有日志，把已经记录的候选重新按 Path B 公式排序。这种方法：

- 不调用新的 LLM 或 embedding；
- 适合快速检查日志、验证指标计算和估计排序变化；
- 不会重新生成无任务链分支本应出现的候选；
- 不能反映任务链对候选生成、实体抽取时序和在线状态的影响。

因此它只能标为 `log-only approximation`，不能替代完整在线 `w/o task chain` 实验。

### 2.2 中断恢复契约

正式运行支持 `--resume-from`。恢复源必须是此前运行的 `results/` 目录（或其中的
`memory_state_latest.json`），并且数据集、模式、session-k、record-k、evidence-k
和 QA 开关必须一致。程序会：

- 从 state 恢复 graph/task-chain，并重建独立输出目录中的向量/BM25 索引；
- 校验保存的 record_id 是数据集 record 的严格前缀，再跳过已处理前缀；
- 从 `metrics_results.json` 和 append-only `query_results.jsonl` 恢复已完成 query，
  不重复调用 query LLM；
- 每个 query 完成后立即写入结果 checkpoint；中断不会清空已有结果。

启动前的 tiny chat preflight 默认开启；额度不足时会在处理数据前失败。离线单元
测试可传 `--skip-llm-preflight`，正式实验不要跳过。

## 3. 固定实验条件

建议在实验记录中固定下列值，baseline 与 ablation 必须完全一致：

```text
dataset: /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json
embedding_model: /data/wz/models/bgem3
embedding_device: cuda
vector_index_backend: chroma
session_ks: 5,10,20,30
retrieval_record_k: 200
evidence_top_k: 20
model/build_model/eval_model: gpt-4o-mini
```

当前数据集检查结果：197 个 dialogue/session、688 条 record、124 个 query。

使用 A100 的 `tcmem` 环境：

```text
/data/wz/anaconda3/envs/tcmem/bin/python
```

不要使用 `/data/wz/anaconda3/bin/python` 作为正式实验解释器；base 环境缺少本项目所需的部分运行依赖。

## 4. 本次运行时配置

本次用户指定的 LLM 配置是 `/data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_qwen/gemini.env`；正式运行只使用该文件中的 key、base URL 和 model。下面旧的 OpenLux 示例仅保留为历史记录，不作为本轮命令配置。

## 4a. OpenLux 历史配置

项目内的私有配置文件为：

```text
/data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_openlux/openlux.env
```

该文件应保持 `chmod 600`，只存放在 A100 的 data 盘，不提交 Git，不复制到日志、manifest、prompt 或结果 JSON。SDK 使用的 URL 是：

```text
https://api.openlux.ai/v1
```

当前 evaluator 通过 CLI 参数或 JSON config 读取运行时配置，并不会自动读取 `OPENAI_API_KEY` 环境变量。推荐在当前 shell 中加载私有环境文件，再通过 shell 变量传入；不要使用 `set -x`，不要在脚本中打印 key：

```bash
set -a
source /data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_openlux/openlux.env
set +a
```

如果需要长期批量运行，建议生成一个只在 data 盘保存、权限为 `600` 的临时 runtime JSON，并在 manifest 中记录 URL/model，不记录 key。不要把真实 key 写进本文档。

## 5. 实验目录约定

每一次运行都使用唯一的 run name 和独立目录，不覆盖旧结果。例如：

```text
/data/wz/agent_memory/iconip2026/work/realmem_ablation_20260907/
  baseline/results/
  baseline/logs/
  no_task_chain/results/
  no_task_chain/logs/
  reports/
```

建议不要把完整实验产物放到项目 `result/` 目录中，以免和旧实验混淆。所有新实验文件放到 `/data/wz/agent_memory/iconip2026/work/...`。

## 6. 执行步骤

### Step 0：代码和依赖门禁

在 A100 上执行：

```bash
cd /data/wz/agent_memory/iconip2026/TCMem

/data/wz/anaconda3/envs/tcmem/bin/python - <<'PY'
import importlib.util
import torch

for name in ("sentence_transformers", "chromadb", "openai", "rank_bm25"):
    print(name, bool(importlib.util.find_spec(name)))
print("cuda", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_2", torch.cuda.get_device_name(2))
PY

/data/wz/anaconda3/envs/tcmem/bin/python -m tcmem.evals.realmem_no_task_chain --help
/data/wz/anaconda3/envs/tcmem/bin/python -m pytest -q \
  tests/test_realmem_no_task_chain_eval.py \
  tests/test_realmem_task_chain_ablation_from_logs.py \
  tests/test_realmem_top_session_eval.py
```

验收：依赖全部为 `True`，CUDA 可用，专用测试全部通过。当前已验证结果为 `28 passed`。

### Step 1：OpenLux 连通性 smoke

不要用真实 prompt 做连通性测试，先发最小请求：

```bash
set -a
source /data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_openlux/openlux.env
set +a
cd /data/wz/agent_memory/iconip2026/TCMem

/data/wz/anaconda3/envs/tcmem/bin/python - <<'PY'
import os
from tcmem.utils.llm_client import OpenAICompatibleLLMClient

client = OpenAICompatibleLLMClient(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["OPENAI_MODEL"],
    timeout=int(os.environ.get("OPENAI_TIMEOUT", "120")),
)
reply = client.generate(
    "Reply with exactly: TCMem LLM connectivity OK",
    temperature=0.0,
    max_tokens=20,
)
print("llm_call_ok", reply.strip() == "TCMem LLM connectivity OK")
PY
```

若出现 `Network is unreachable`，先检查 endpoint；A100 曾经可以访问 OpenLux，但不能访问旧的 `yunwu.ai` 配置。

### Step 2：单查询端到端 smoke

先不加 `--with-qa` 验证 ingestion、GPU embedding、Chroma、Path B 和结果落盘。由于第一个 query 位于第 26 条 record，`--max-queries 1` 仍会先处理前 25 条 record，这是正常的 query-before-ingest 行为。

```bash
RUN_ROOT=/data/wz/agent_memory/iconip2026/work/realmem_ablation_smoke_20260907
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/logs"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
/data/wz/anaconda3/envs/tcmem/bin/python \
-m tcmem.evals.realmem_no_task_chain \
--dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
--config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
--api-key "$OPENAI_API_KEY" \
--base-url "$OPENAI_BASE_URL" \
--model "$OPENAI_MODEL" \
--run-name realmem_no_task_chain_smoke_20260907 \
--output-dir "$RUN_ROOT/results" \
--log-dir "$RUN_ROOT/logs" \
--embedding-model /data/wz/models/bgem3 \
--embedding-device cuda \
--vector-index-backend chroma \
--session-ks 5,10 \
--retrieval-record-k 50 \
--evidence-top-k 5 \
--max-queries 1 \
--state-save-every-records 20 \
--progress-every-records 20 \
--verbose
```

然后用同样的命令加 `--with-qa`，只验证 answer generation、QA judge 和 memory judge。单 query smoke 的指标不能用于论文结论。

Smoke 必须检查：

```text
manifest.json 存在
memory_state.json 存在
retrieval_results.json 存在
metrics_results.json 存在
realmem_top_session_report.md 存在
task_chain_enabled=false
task_count=0
routed_task_ids=[]
expanded_task_ids=[]
hit reason 包含 path_b_vector_bm25_graph_no_task_chain
```

### Step 3：运行完整 baseline

为 baseline 建立独立目录。baseline 使用 `realmem_top_session`，不要复用 ablation 的 state 或 vector index：

```bash
BASE=/data/wz/agent_memory/iconip2026/work/realmem_ablation_20260907/baseline
mkdir -p "$BASE/results" "$BASE/logs"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
/data/wz/anaconda3/envs/tcmem/bin/python \
-m tcmem.evals.realmem_top_session \
--dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
--config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
--api-key "$OPENAI_API_KEY" \
--base-url "$OPENAI_BASE_URL" \
--model "$OPENAI_MODEL" \
--build-model "$OPENAI_MODEL" \
--eval-model "$OPENAI_MODEL" \
--run-name realmem_baseline_task_chain_20260907 \
--output-dir "$BASE/results" \
--log-dir "$BASE/logs" \
--embedding-model /data/wz/models/bgem3 \
--embedding-device cuda \
--vector-index-backend chroma \
--session-ks 5,10,20,30 \
--retrieval-record-k 200 \
--evidence-top-k 20 \
--state-save-every-records 5 \
--progress-every-records 1 \
--with-qa \
--verbose
```

### Step 4：运行完整 w/o task chain

除入口和 run name 外，参数必须与 baseline 相同：

```bash
ABL=/data/wz/agent_memory/iconip2026/work/realmem_ablation_20260907/no_task_chain
mkdir -p "$ABL/results" "$ABL/logs"

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
/data/wz/anaconda3/envs/tcmem/bin/python \
-m tcmem.evals.realmem_no_task_chain \
--dataset /data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json \
--config /data/wz/agent_memory/iconip2026/zhuo/runtime_config.json \
--api-key "$OPENAI_API_KEY" \
--base-url "$OPENAI_BASE_URL" \
--model "$OPENAI_MODEL" \
--build-model "$OPENAI_MODEL" \
--eval-model "$OPENAI_MODEL" \
--run-name realmem_no_task_chain_20260907 \
--output-dir "$ABL/results" \
--log-dir "$ABL/logs" \
--embedding-model /data/wz/models/bgem3 \
--embedding-device cuda \
--vector-index-backend chroma \
--session-ks 5,10,20,30 \
--retrieval-record-k 200 \
--evidence-top-k 20 \
--state-save-every-records 5 \
--progress-every-records 1 \
--with-qa \
--verbose
```

正式运行预计是小时级任务，主要耗时来自每条 record 的实体抽取 LLM 调用和 BGE embedding。不要在 GPU 资源不足时杀掉其他用户进程；先选择空闲 GPU 并设置 `CUDA_VISIBLE_DEVICES`。

## 7. 完成判定和结果检查

### 7.1 Manifest 检查

两个结果目录都必须有 `manifest.json`，并检查：

| 字段 | baseline | ablation |
| --- | --- | --- |
| `config.mode` | `tcmem_top_session` | `tcmem_no_task_chain_online` |
| `config.tcmem_config.task_chain_enabled` | `true` | `false` |
| `config.base_url` | OpenLux `/v1` | OpenLux `/v1` |
| `config.model/build_model/eval_model` | 相同 | 相同 |
| `config.tcmem_config.embedding_device` | `cuda` | `cuda` |
| `config.tcmem_config.vector_index_backend` | `chroma` | `chroma` |
| `config.with_qa` | `true` | `true` |
| `summary.query_count` | 124 | 124 |
| `summary.failed_query_count` | 0 | 0 |

`config.tcmem_config.llm_api_key` 应为空。项目的 `to_dict()` 默认会脱敏该字段；若发现 key 出现在任何 JSON、JSONL 或日志中，应立即停止使用该结果并隔离文件。

### 7.2 状态和检索检查

- baseline 的 `memory_state.json` 应包含任务链；ablation 的 `task_chains.tasks` 应为空。
- ablation 的每个检索结果都应有空的 `routed_task_ids` 和 `expanded_task_ids`。
- ablation 的 hit reason 应为 `path_b_vector_bm25_graph_no_task_chain`。
- baseline 才允许出现 `path_a_chain`、任务路由和层级扩展字段。
- 两个分支的 query id、gold session uuid 和 query 数必须一致。

### 7.3 指标对比

至少比较：

```text
recall_any@5/10/20/30
recall_all@5/10/20/30
ndcg@5/10/20/30
average_qa_score
average_mem_recall
average_mem_helpful_score
qa_failed_count
mem_failed_count
failed_query_count
```

主差值定义为：

```text
delta = no_task_chain - baseline
```

同时报告 query-level 配对差值，而不是只报告两个全局平均值。建议按 query id 对齐后统计均值、正负变化数量，并对关键指标做 bootstrap 置信区间。若只做工程快速检查，至少输出每个 `k` 的全局指标和 query-level delta 分布。

### 7.4 Query 级 session/task-chain 追踪

评测器从每个 `is_query=true` 的 User turn 读取其紧邻 Assistant turn 的
`memory_used`。原始 evidence 顺序保存在 `gold_memory_used`，其中每项至少包含：

```json
{"memory_index": 0, "session_uuid": "...", "content_excerpt": "..."}
```

每个 query 还会生成 `session_task_chain_trace`。它覆盖 gold session 与实际检索到的
全部 session，并记录：

```text
session_uuid
is_gold / gold_memory_used_count
retrieved_rank / retrieved_score / retrieved_record_ids
retrieval_hit_count
task_chain_memberships[]
chain_root_task_ids
in_any_task_chain
```

`task_chain_memberships` 中会给出 task id、根任务 id、父 task/父 branch、任务描述和
current focus、branch goal/status、node/record id，以及该 task 在本次 query 中的
`route_evidence`（hit count、route role/relation/depth、reason）。因此可以区分：

1. gold session 没有进入检索结果；
2. session 检索到了，但没有任务链归属；
3. session 属于任务链，但路由没有选中该 task；
4. task 被选中或层级扩展到了，但具体 node/record 没有命中；
5. 多个 gold session 是否共享同一个根任务、分别落在哪些 branch。

汇总字段 `gold_session_task_chain_summary` 包含 `gold_chain_root_task_ids`、
`gold_sessions_common_root_task_ids`、`gold_chain_groups`、未归属/无有效根的 session，
以及 `gold_sessions_share_common_root_chain` 和 `gold_sessions_share_one_root_chain`。
不要只看最终 recall；先按这些字段定位任务路由、任务链覆盖和 session 检索的具体断点。

输出位置：

```text
<run-output>/session_task_chain_trace.json
<run-output>/metrics_results.json              # detailed_results 内同步保存
<run-log>/.../session_task_chain.jsonl          # 每个 query 一条事件
<run-log>/.../query_results.jsonl               # query_completed/query_failed 内同步保存
```

快速检查示例：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("<run-output>/session_task_chain_trace.json")
data = json.loads(path.read_text(encoding="utf-8"))
for query_id, item in list(data.items())[:5]:
    summary = item["gold_session_task_chain_summary"]
    print(query_id, {
        "gold": summary["gold_session_count"],
        "retrieved": summary["gold_sessions_retrieved_count"],
        "roots": summary["gold_chain_root_task_ids"],
        "same_root": summary["gold_sessions_share_one_root_chain"],
        "without_chain": summary["gold_sessions_without_task_chain"],
    })
PY
```

## 8. 日志和故障排查

每个 run 的 `logs/` 至少应包含：

```text
dataset.jsonl
manifest.jsonl
ingestion.jsonl
retrieval.jsonl
query_results.jsonl
metrics.jsonl
progress.jsonl
errors.jsonl
llm_errors.jsonl
```

常见故障：

| 症状 | 处理 |
| --- | --- |
| `sentence-transformers` 缺失 | 使用 `/data/wz/anaconda3/envs/tcmem/bin/python`，不要使用 base Python |
| `Network is unreachable` 或 endpoint connection error | 检查 OpenLux URL 和 A100 网络；不要继续使用旧 `yunwu.ai` URL |
| Chroma import error | 先用 `--vector-index-backend numpy` 做依赖诊断；正式结果建议使用 Chroma 并记录 backend |
| LLM JSON decode error | 查看 `llm_errors.jsonl`，确认模型返回格式；不要删除错误日志后继续汇报结果 |
| 进程中断 | 保留已有 work 目录，用新的 run name 重跑；通过 `progress_latest.json` 判断中断位置，当前 evaluator 不保证自动 resume |
| `realmem_retrieval_sweep` 无法导入 | 当前工作树缺少 `tcmem/evals/realmem_retrieval_sweep.py`，先不要执行 retrieval sweep；这不影响单次 baseline/ablation |

注意：当不使用 `--with-qa` 时，当前汇总逻辑会把 `qa_score=None` 计入 `qa_failed_count`。这种运行只能用于检索 smoke 或检索主实验，解释 QA 字段时必须标注“未执行 QA”，不要将其当作真实 QA 失败。

## 9. 现有历史结果的使用边界

历史完整无任务链结果位于：

```text
/data/wz/agent_memory/iconip2026/result/results/tcmem_realmem_no_task_chain_adeleke_20260601_164851
```

该结果可用于检查输出格式和粗略基线，已记录 124 个 query、CUDA embedding 和 Chroma；但它生成于当前层级检索日志增强之前，缺少部分新的 `expanded_task_ids`/`expansion_edges` 观测字段。因此不能直接作为本轮修补后 baseline 的配对对照。正式结论应使用本轮重新运行的两个分支。

## 10. 最终验收清单

- [ ] baseline 和 ablation 使用相同数据集、模型、embedding、backend、`session_ks`、`retrieval_record_k`、`evidence_top_k`。
- [ ] 两个 run 均在 A100 的 `tcmem` 环境完成，GPU embedding 确认启用。
- [ ] 两个 run 均有 124 个 query，`failed_query_count=0`。
- [ ] ablation manifest 的 mode 为 `tcmem_no_task_chain_online`。
- [ ] ablation state 的任务数为 0，baseline state 有任务链。
- [ ] ablation 没有 query task routing、Path A 或 chain context hit。
- [ ] QA 实验的 `qa_failed_count=0`、`mem_failed_count=0`；检索-only 实验明确标记未执行 QA。
- [ ] key 未出现在代码、manifest、JSON、JSONL、prompt 或日志中。
- [ ] 所有新产物位于 `/data/wz/agent_memory/iconip2026/work/...`。
- [ ] 保存 baseline/ablation 的 manifest、report、metrics、retrieval、progress、错误日志和命令参数。
- [ ] 保存并检查每个 query 的 `session_task_chain_trace`，确认 gold session 的 task/branch/root 归属可回放。
- [ ] 汇报全局指标、query-level 配对 delta，以及失败/缺失数据说明。

## 11. 本轮已验证的参考结果

本轮已完成以下可复现性验证：

- 专用评测测试：`28 passed`。
- A100 + OpenLux + BGE-M3 + Chroma + no-task-chain，单 query 检索 smoke：成功。
- 同配置带 QA 单 query smoke：成功，`recall_all@10=1.0`、QA score `2`、`qa_failed_count=0`。
- smoke 结果目录示例：
  - `/data/wz/agent_memory/iconip2026/work/tcmem_readiness_no_task_chain_smoke_openlux_20260907/results`
  - `/data/wz/agent_memory/iconip2026/work/tcmem_readiness_no_task_chain_smoke_openlux_qa_20260907/results`

这些 smoke 只证明工程链路可运行，不替代 124-query 的正式 paired ablation。
