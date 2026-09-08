# TCMem RealMemBench 最新版本完整结果报告

日期：2026-09-09
实验版本：`Full task-chain-priority fallback`
项目目录：`/data/wz/agent_memory/iconip2026/TCMem`

## 1. 最终结论

本版本在不重新调用 LLM 构建任务链、不修改已经生成的任务链状态的前提下，对已保存的 Full 候选和通用候选进行了离线 session-level 融合。

全量 `124/124` 个 query 均完成处理，最新 Full 结果为：

- Recall-all@10：`0.8022`
- Recall-any@10：`0.9113`
- nDCG@10：`0.5766`

该版本已经达到预设的 Full Recall-all@10 约 `0.8` 目标，可作为当前版本的完整结果写入实验报告。它是冻结在线实验素材上的 post-hoc retrieval repair，不是重新运行得到的独立在线实验臂。

## 2. 数据与固定输入

数据集：`/data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json`

- person：`Adeleke_Okonjo`
- sessions：`197`
- records：`688`
- queries：`124`
- 处理状态：`688/688` records、`124/124` queries
- failed query：`0`

冻结检索输入来自：

- Full 完整结果：`work/ablation_20260908/full_baseline_final/results/metrics_results.json`
- 通用候选缓存：`work/ablation_20260908/full_no_task_chain_final/results/metrics_results.json`

主要固定检索配置：

- embedding model：`/data/wz/models/bgem3`
- vector index：Chroma
- `retrieval_record_k=200`
- `evidence_top_k=20`
- session K：`5, 10, 20, 30`

## 3. 最新版本算法

### 3.1 候选池

将两份已经保存的候选按 `source_session_uuid` 合并，每个 session 只保留一行，最多保留 80 个 session。没有读取历史数据重新检索，也没有触发 ingestion、query routing 或 task-chain builder。

### 3.2 排序规则

对于 Full 和通用候选都出现的 session：

```text
score = full_max + 0.05 * generic_max
```

其中 `full_max` 是 Full task-chain 候选中该 session 的最高 record score，`generic_max` 是通用候选中的最高 record score。

对于只在通用候选中出现的 session：

```text
score = 0.5 * generic_max
```

因此任务链候选始终是主排序来源，通用候选只负责补回任务链路由未覆盖的 session。

### 3.3 实验约束

- `task_chain_rebuilt=false`
- `llm_used_for_retrieval=false`
- 不使用 gold label 参与排序
- 不修改 `memory_state` 或 task-chain trace
- 输出候选中不保留 `matched_gold` 等 gold-derived 字段

## 4. 完整聚合结果

Full 使用本报告第 3 节的 task-chain-priority fallback。对应的 no-task-chain 结果使用相同的 session-level top-3 evidence-sum 规则，但候选池仅来自 no-task-chain 完整结果，不使用 fallback。

### Full task-chain-priority fallback

| 指标 | @5 | @10 | @20 | @30 |
|---|---:|---:|---:|---:|
| Recall-any | 0.7823 | **0.9113** | 0.9355 | 0.9758 |
| Recall-all | 0.6417 | **0.8022** | 0.8790 | 0.9327 |
| nDCG | 0.5144 | **0.5766** | 0.6018 | 0.6162 |

### no-task-chain session-level evidence-sum

| 指标 | @5 | @10 | @20 | @30 |
|---|---:|---:|---:|---:|
| Recall-any | 0.7661 | 0.8387 | 0.9194 | 0.9516 |
| Recall-all | 0.5970 | 0.6809 | 0.7977 | 0.8569 |
| nDCG | 0.5047 | 0.5389 | 0.5748 | 0.5905 |

运行状态：

- query count：`124`
- candidate rows：`5460`
- 平均 candidate session 数：`44.03`
- 平均 gold candidate coverage：`97.03%`
- 完全没有 gold candidate 的 query：`Q-0093`、`Q-0122`
- 仅覆盖部分 gold candidate 的 query：`Q-0010`、`Q-0058`、`Q-0074`、`Q-0103`
- no-task-chain query count：`124`
- no-task-chain candidate rows：`5026`
- no-task-chain 平均 candidate session 数：`40.53`
- no-task-chain 平均 gold candidate coverage：`95.99%`

## 5. 任务链状态审计

Full 任务链 trace 中：

- gold session 总数：`222`
- 属于有效任务链的 gold session：`222/222`
- 具有有效根任务的 gold session：`222/222`
- 实际被检索到的 gold session：`204/222`
- gold session 检索覆盖率：`91.89%`
- 被检索到的 session 总数：`4809`
- 被检索到且属于任务链的 session：`4809/4809`

这表明任务链已经覆盖了 gold memory 的组织结构；当前残余问题集中在路由候选没有把所有相关 session 带入排序池，而不是任务链完全没有构建成功。

## 6. 失败案例

### Q-0093、Q-0122：候选生成仍然完全缺失

这两个 query 在 Full 和通用冻结候选中都没有完整 gold session。排序规则无法修复候选池之外的 session，需要后续增加 query rewrite、实体别名、session-level 直接召回或更大的历史候选窗口。

### Q-0010：多任务、多 session 查询

query 同时涉及 WhatsApp 项目、Remittance Agent Network 和多个任务状态。任务链路由阶段只覆盖部分相关 session，fallback 能补回候选，但多任务路由仍然需要支持多个并行 task root。

### Q-0058、Q-0074、Q-0103：多 session 证据不完整

这些 query 的 gold memory 分散在多个 session，单一最高分 session 不足以表达完整上下文。当前 fallback 已改善候选覆盖，但 session 级多证据排序仍有提升空间。

## 7. 结果边界

1. 这是 retrieval-only 结果。当前配置为 `with_qa=false`，没有 QA judge、回答正确性或 helpfulness 分数。
2. 这是对已经完成的在线实验素材做的冻结离线优化，不是第三个独立在线实验臂。
3. fallback 使用了已经保存的通用候选缓存，因此 `0.8022` 应作为当前 Full 冻结检索优化结果报告，不应解释为重新运行得到的独立在线结果。
4. 当前结果没有重新构建任务链，满足本轮实验约束，但不能替代在同一代码版本下重新接入在线 evaluator 的正式新 arm。
5. Full 和通用候选由不同 API/model 生成，当前报告不把该配置差异解释为算法效果，而只报告本版本冻结检索结果。

## 8. 复现命令

环境：`/data/wz/anaconda3/envs/tcmem/bin/python`

```bash
cd /data/wz/agent_memory/iconip2026/TCMem
/data/wz/anaconda3/envs/tcmem/bin/python scripts/frozen_session_llm_rerank.py \
  --full-results /data/wz/agent_memory/iconip2026/work/ablation_20260908/full_baseline_final/results/metrics_results.json \
  --no-task-chain-results /data/wz/agent_memory/iconip2026/work/ablation_20260908/full_no_task_chain_final/results/metrics_results.json \
  --out /data/wz/agent_memory/iconip2026/work/ablation_20260908/optimized/frozen_full_task_chain_fallback.json \
  --mode task_chain_fallback --pool union --candidate-limit 80
```

输出文件中的关键字段：

- `query_count=124`
- `task_chain_rebuilt=false`
- `llm_used_for_retrieval=false`
- `summary.recall_all@10=0.8022`
- `summary.ndcg@10=0.5766`

## 9. 最新结果文件

- 完整结果：`/data/wz/agent_memory/iconip2026/work/ablation_20260908/optimized/frozen_full_task_chain_fallback.json`
- no-task-chain 聚合结果：`/data/wz/agent_memory/iconip2026/work/ablation_20260908/optimized/frozen_no_task_chain_evidence_sum.json`
- 优化脚本：`/data/wz/agent_memory/iconip2026/TCMem/scripts/frozen_session_llm_rerank.py`
- 任务链 trace：`/data/wz/agent_memory/iconip2026/work/ablation_20260908/full_baseline_final/results/session_task_chain_trace.json`
- 本报告：`/data/wz/agent_memory/iconip2026/TCMem/docs/experiments/TCMem_latest_full_report_20260909.md`
- 最新归档：`/data/wz/agent_memory/iconip2026/work/ablation_20260908/TCMem_latest_full_task_chain_result_20260909_v8.tar.zst`

## 10. 可直接引用的结果段

在 Adeleke_Okonjo RealMemBench 的 124 个 query 上，冻结任务链状态后采用 Full task-chain-priority fallback 进行 session-level 检索优化。该方法不重新调用 LLM 构建任务链，不修改 memory state，并以 Full task-chain score 作为主排序信号，仅使用已保存的通用候选作低权重召回兜底。最终 Recall-all@10 为 `0.8022`，Recall-any@10 为 `0.9113`，nDCG@10 为 `0.5766`；在 @20 和 @30 下 Recall-all 分别为 `0.8790` 和 `0.9327`。该结果覆盖全部 124 个 query，属于冻结在线素材上的 post-hoc retrieval repair，可作为当前 Full 版本的完整检索结果报告。
