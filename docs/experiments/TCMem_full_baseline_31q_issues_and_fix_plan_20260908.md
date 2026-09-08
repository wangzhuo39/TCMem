# TCMem Full Baseline 前 31 个 Query：问题证据与实验后修复方案

**文档日期**：2026-09-08
**适用项目**：`/data/wz/agent_memory/iconip2026/TCMem`
**实验**：RealMemBench Adeleke_Okonjo，Full baseline vs online no-task-chain
**文档目的**：冻结当前实验期间发现的问题和修复设计。当前实验完成前不修改检索、路由或任务链逻辑。

## 1. 执行结论

前 31 个 query 的结果说明任务链不是完全无效：它在部分连续财务/业务记忆场景中能提高较宽候选范围的完整召回，也能构建跨 session 的任务节点和分支。但 Full baseline 的 Top-5 总体没有稳定优势，且存在一个明确的失败模式：查询路由一旦选错或选得过窄，Path A 的任务链得分会压低原本正确的 Path B 候选。

因此需要区分三类问题：

1. **确定的实现缺陷**：Path A/Path B 融合、route score 没有进入 Path A 分数、多个 task membership 的选择顺序。
2. **算法设计不足**：router 没有置信度和 abstain 机制，候选任务预筛依赖词/entity overlap，任务图缺少恢复/来源/实现关系。
3. **实验与数据限制**：当前续跑使用了不同阶段的模型；前 31 个 query 的检索指标可以用于诊断，不能单独作为最终因果结论。

## 2. 当前实验状态与冻结规则

### 2.1 数据集规模

| 项目 | 数量 |
|---|---:|
| sessions | 197 |
| dialogue records | 688 |
| evaluation queries | 124 |

### 2.2 运行状态（动态记录）

文档生成时最近一次远端检查：

- Full baseline：`246 / 688 records`，`40 / 124 queries`，进程正常，manifest 尚未 finished。
- no-task-chain：`688 / 688 records`，`124 / 124 queries`，manifest 为 `finished`。
- runner 使用 `/data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_openlux/openlux.env`。

上述状态只用于说明文档生成时的运行状态；最终状态以两个 `manifest.json` 为准。

### 2.3 实验期间不得做的事

- 不修改 `tcmem/core/retrieval.py`、`task_chain.py`、router prompt、embedding/BM25 配置。
- 不删除或覆盖 `full_baseline_final`、`full_no_task_chain_final` 和原 partial checkpoint。
- 不把前 31 个 query 的结果宣称为完整消融结论。
- 不把旧 Gemini 阶段和 OpenLux 阶段的结果当作严格同模型 paired causal evidence。

实验完成后先冻结结果、提交代码快照和运行 manifest，再按本文档实施修复和重放。

## 3. 前 31 个 Query 的固定证据

下面的 paired 比较只使用两个实验在相同 query ID `Q-0001`～`Q-0031` 上的结果。no-task-chain 的全量结果已经完成，但表中只取相同的前 31 条，避免把不同 query 集合直接比较。

| 指标 | Full baseline | no-task-chain | Full - no-task-chain |
|---|---:|---:|---:|
| Recall-any@5 | 0.8387 | 0.9032 | -0.0645 |
| Recall-all@5 | 0.7097 | 0.7097 | 0.0000 |
| nDCG@5 | 0.6205 | 0.6387 | -0.0182 |
| Recall-any@10 | 0.9677 | 0.9677 | 0.0000 |
| Recall-all@10 | 0.8387 | 0.8065 | +0.0323 |
| nDCG@10 | 0.6663 | 0.6692 | -0.0029 |
| Recall-all@20 | 0.9355 | 0.9032 | +0.0323 |
| nDCG@20 | 0.6799 | 0.6908 | -0.0109 |
| Recall-all@30 | 0.9355 | 0.9355 | 0.0000 |
| nDCG@30 | 0.6799 | 0.6941 | -0.0142 |

### 3.1 退化 query

| Query | Full 主要指标 | no-task-chain 对照 | 语义/路由证据 | 初步归因 |
|---|---|---|---|---|
| Q-0010 | Recall-all@5=0，nDCG@5=0.2346 | 相同 | 查询确认 Asanify 中的 `Remittance Agent Network`；Full 只主路由到 Asanify setup task，gold session 被排到第 8 左右 | 路由过窄，未连接工具设置、remittance 规划和前置需求 |
| Q-0011 | Recall-any/all@5=0，nDCG@5=0 | any/all@5=1，nDCG@5=0.6509 | “恢复暂停的 financial analysis”；Full 路由到交易抽取、wealth management、E-Myth 等相邻 task，gold 排第 9/10 | 错误的 task context 使 Path B 正确候选降权 |
| Q-0012 | Recall-all@5=0，nDCG@5=0.3869 | 相同 | “查看 comprehensive draft analysis”；只路由到 transaction extraction，两个 gold session 分散在第 2 和第 14 | 同一财务主任务下缺少 draft/source 关系，Path A 不能覆盖所有证据 |
| Q-0017 | Recall-any/all@5=0，nDCG@5=0 | 相同 | “刚生成的 voice notes、customer KYC”；路由到 hire/onboard/train task，gold session 约第 7 | 基础语义排序和任务粒度问题，不是单纯 router 失败 |
| Q-0018 | Recall-any/all@5=0，nDCG@5=0 | any/all@5=1，nDCG@5=0.6309 | “approved remittance roadmap 的第一步”；混入 transaction、Asanify、wealth-app tasks，gold 约第 8 | router 把相邻主题当作当前执行计划，Path A 压低正确 Path B |
| Q-0020 | Recall-any/all@5=0，nDCG@5=0 | any@5=0，gold 约第 15 | “组织已有 voice notes”，同时提到 WhatsApp Live Demo；Full 候选被 playbook/KYC/project-management task 占满 | 数据本身较难；需要来源/承接关系和更好的 session 聚合 |
| Q-0030 | Recall-any/all@5=0，nDCG@5=0 | any@5=1，nDCG@5=0.6131 | “apprentice idea”；Full 路由到 first-agent hiring、Asanify、KYC training，gold session 只到第 7 | 下游执行任务覆盖了上游 E-Myth 委派语义 |

### 3.2 任务链有正向效果的 query

- **Q-0007**：将 E-Myth 学习、业务流程和停电/恢复上下文串联，Full 的完整召回明显提高。
- **Q-0022**：关于新币/储蓄的连续决策被 wealth-management task chain 捕捉；no-task-chain Top-5 未命中而 Full 命中。
- **Q-0028/Q-0029**：应急基金、Pay Yourself First、储蓄配置之间的连续关系被任务链保留，Full 的 nDCG@5 高于对照。
- 运行中的 state 还观察到真实分支：原先高风险 NaijaCoin 投资节点被 deprecated/branched，后续 `90% core seed fund + 10% experimental fund` 作为新分支写入。这说明 branch reducer 的基本能力存在，但其触发边界仍需收紧。

## 4. 确定的实现缺陷

### 4.1 Path A/Path B 融合把 Path B 变成弱门控

**代码位置**：`tcmem/core/retrieval.py:276-280`，默认配置在 `tcmem/config.py:27-35`。

Full 的最终记录分数近似为：

```text
final = path_a_weight * max(Path-A record score)
      + path_b_weight * max(Path-B record score)
```

默认 `path_a_weight=0.6`、`path_b_weight=0.4`。no-task-chain 则直接使用 Path B 分数。于是一个没有被 router 选中的正确记录，即使 Path B 分数很高，也最多只贡献 40%；一个被错误 router 选中的相邻记录则可能获得完整 Path A 贡献。

这正好解释 Q-0011、Q-0018、Q-0030：no-task-chain 能把 gold 候选排在前几名，Full 路由后反而把相邻 task 推到前面。

**修复方案**：不能简单把 `0.6/0.4` 调成另一个固定比例。应先保持 Path B 的可比性，再把任务链作为增量证据。推荐方案：

```text
base = calibrated_path_b_score
chain_bonus = alpha * route_confidence * calibrated_path_a_score
final = base + chain_bonus
```

工程上可用 rank-based fusion/RRF，避免 semantic、BM25、graph 和 chain 分数不在同一量纲。必须保留 Path B floor：低置信或错误路由不能把 base score 乘小。

**验收**：在固定 snapshot 上，错误路由模拟不能使一个高 Path-B gold 候选从 Top-5 被压到 Top-20 之外；正确路由 query 仍可获得正向 bonus。

### 4.2 Path A 计算了 route metadata，但没有使用真实 route strength

**代码位置**：`tcmem/core/retrieval.py:144-169`。

`_route_metadata()` 返回 `route_score`，但 Path A 构造 `SearchHit` 时把 `route_score` 写死为 `1.0`，并且 node score 没有乘 primary/child/parent/branch-child 的 route 分级。当前 route score 只进入 trace，不改变排序。

**修复方案**：

```text
path_a_node = semantic/status/recency score
             * calibrated_route_weight(route_role, route_depth)
```

primary、parent、child、branch-child 应有可记录、可测试的不同权重；depth 需要衰减；不能让扩展 task 和 LLM primary task 获得同等 Path A 影响。

**验收**：构造 primary/parent/child/branch-child 四种 fixture，检查同一语义分数下排序顺序与配置一致；trace 中的 route score 必须等于排序使用的值。

### 4.3 多任务 membership 的最佳 context 选择顺序不合理

**代码位置**：`tcmem/core/retrieval.py:340-376`。

`_best_chain_context()` 的比较 key 是：

```text
(penalty, status_score, route_score, chain_score)
```

也就是说，节点是否 active/branched 的 penalty 和 status 优先于当前 query 是否真正路由到该 task。一个 active 但无关的 membership 可能压过一个已路由但 branched 的相关 membership。Q-0018、Q-0020 这类跨 task 记录会因此产生错误的 Path B context。

**修复方案**：

1. 先以 query relevance 和 route confidence 选择 membership。
2. 再用 branch status、superseded/deprecated penalty 做状态修正。
3. 不要只保留一个 context；内部保留每个 membership 的 evidence，最终对记录做 max/softmax 聚合。

**验收**：同一 record 属于两个 task 时，primary route 必须优先于 unrouted active task；deprecated 节点不得因 active 状态的另一 task membership 被重新当作主证据。

### 4.4 Path B 分量仍存在量纲和校准问题

**代码位置**：`tcmem/core/retrieval.py:187-239`、`tcmem/infrastructure/indices.py:332-357`、`:405-419`。

Path B 将 embedding semantic、query-local min-max BM25、graph walk score 线性相加。BM25 是每个 query 独立归一化，graph score 又来自 seed score/(depth+1)，不同 query 之间不可直接比较。当前默认 semantic/BM25/graph 权重也不代表经过校准的概率。

**修复方案**：

- 在 query 内对所有候选统一做 rank fusion，或使用固定 corpus-level calibration；不要直接把 query-local min-max 值当作跨 query 可比分数。
- 记录每个 query 的 component 分布、候选数、seed 来源和 graph depth。
- 把 BM25 tokenizer（目前 `tcmem/infrastructure/indices.py:405-406` 只匹配 `[a-z0-9]+`）改为可配置，并明确记录本轮实验的语言覆盖范围。

**验收**：同一 query 的 component score 能复现；不同 query 的 final score 不再依赖某个 query 的 min/max 极端值；中文/非 ASCII 文本测试不应静默变成空 token。

## 5. Router 和任务链设计问题

### 5.1 Router 没有 confidence/abstain

**代码位置**：`tcmem/core/task_chain.py:261-320`、`tcmem/prompts/query_routing.txt`。

LLM 只返回 `routed_task_ids` 和 reason，没有每个候选的 relevance score、margin 或整体 confidence。即使候选都只是相邻主题，系统仍会把前 N 个 task 当作 primary route。

**修复方案**：输出结构化 route candidates：`task_id`、`score`、`evidence`、`confidence`。当最高分低于阈值或 top-1/top-2 margin 太小：

- routed task 设为空或只保留 top-1；
- 让 Path B 保持主导；
- 把低置信事件写入 trace，便于后验分析。

### 5.2 候选池在 48 个 task 后被 lexical/entity 预筛截断

**代码位置**：`tcmem/core/task_chain.py:1766-1825`，配置 `query_router_pool_size=48`。

候选池超过 48 后，先按 entity overlap、token overlap、active bonus 和更新时间排序，再把剩余 task 丢弃。`resume paused plan`、`the one we just made`、`first step from approved roadmap` 等查询依赖上下文和语义承接，可能在 LLM router 看到候选前就丢失正确 task。

**修复方案**：用三路候选并集：

1. embedding top-M task summaries；
2. entity/exact-term top-M；
3. recent/parent-child/branch context top-M；

再去重并保留候选来源和分数。候选池大小必须写入 manifest，不得只记录最终 routed IDs。

### 5.3 任务图缺少跨任务语义关系

当前图主要表达 parent/child/branch，不能明确表达：

- `paused_by`：任务被哪个事件暂停；
- `resumes`：当前 query 恢复哪个任务；
- `derived_from`：当前 roadmap/方案来自哪个讨论；
- `implements`：当前操作步骤实现哪个计划；
- `supports`：辅助任务对主任务的支持关系。

Q-0011、Q-0018、Q-0030 都是这种“上游来源/中断恢复/下游执行”关系，而不是简单的同 task continuation。

**修复方案**：扩展 task graph edge schema，边上保存 relation、confidence、source record/query 和 timestamp；检索时根据 query intent 选择关系类型，而不是把所有邻居统一展开。

### 5.4 生命周期状态没有形成在线闭环

当前 checkpoint 中任务大多仍为 `active`。代码虽有 `update_branch_status()` 和 `_sync_task_status_from_branches()`，但必须确认 evaluator 的 ingest 流程实际消费了 LLM 的 status transition；不能仅依赖节点的 `active/branched/deprecated`。

**修复方案**：

- 将 `completed/blocked/cancelled/merged` 纳入 reducer 的明确输入和事件日志；
- branch status 变化后同步 task status；
- query routing 对 completed/blocked task 默认降低权重，但允许明确历史查询访问；
- 加入状态转移合法性测试和 checkpoint invariant。

### 5.5 Branch action 触发边界需要 deterministic guard

分支对 NaijaCoin 风险路线是合理的，但普通“换一个类比”“估算价格”“补充一个例子”不应自动成为独立 branch。LLM action 必须经过 reducer 规则校验：只有目标/约束仍独立有效且与新路线并存时才允许 branch；纯细化应是 create，明确废弃旧约束才是 override。

## 6. 实验有效性与重放方案

### 6.1 当前结果的限制

本轮从旧 Gemini checkpoint 切换到 OpenLux `gpt-5.6-luna` 继续运行，且两组切换时刻不同。因而：

- Full/no-task-chain 仍是在线、同数据集、同代码版本的两臂运行；
- 但前缀 query 可能由不同模型阶段生成，不能把混合模型结果解释为严格单模型 causal ablation；
- 当前未启用 `--with-qa`，指标只反映 session retrieval，`average_qa_score=None` 不代表检索故障。

### 6.2 实验完成后的顺序

1. 检查两个 manifest 都为 `status=finished`，确认 `688/688 records`、`124/124 queries`。
2. 复制结果目录为只读归档，记录 git diff、dataset hash、prompt hash、embedding model、LLM model、API base URL、config 和 fallback count。
3. 保存当前版本的 Full/no-task-chain paired report，作为 baseline，不覆盖。
4. 给 retrieval、router、task reducer 增加单元/契约测试。
5. 在已保存的每个 query snapshot 上先离线重放“只改排序/融合”的版本；这样不需要重新调用 ingest LLM。
6. 若修改 task routing、task creation、branch/lifecycle，则必须从数据集前缀重新构建 state，并用同一个 API/model 成对运行。
7. 修复版必须使用独立的 output、state、vector index、log 目录，不能复用旧 Chroma collection。

### 6.3 推荐修复批次

**Batch A：纯检索排序（低风险）**

- Path B floor + rank fusion；
- route score 真正进入 Path A；
- multi-membership context 优先级；
- component/fallback trace；
- 不改变 ingest state。

**Batch B：router 候选和置信度（中风险）**

- 三路候选池；
- confidence/margin/abstain；
- 低置信时 Path B 主导；
- 增加 Q-0011/Q-0018/Q-0030 回归测试。

**Batch C：任务图和生命周期（高风险）**

- 跨任务语义边；
- branch reducer guard；
- task/branch status reducer；
- 需要从同一数据集、同一模型、独立 state 成对重跑。

## 7. 修复后验收标准

### 7.1 正确性门禁

- 所有既有测试通过，新增 retrieval/router/task-chain contract tests 全部通过。
- `py_compile`、`git diff --check`、runner `bash -n` 通过。
- state invariant：record 前缀严格有序；task node source record 唯一或显式允许多 membership；branch/head/parent 不悬空；无 cycle。
- 失败或 API 中断后可从 checkpoint 恢复，不重复计费已完成 query。

### 7.2 检索门禁

- Q-0011、Q-0018、Q-0030 不得因错误 route 将原本 Top-5 的 gold 全部压出 Top-10。
- Q-0007、Q-0022、Q-0028、Q-0029 的任务链正向增益不能消失。
- Full 的 Path B floor 不低于 no-task-chain 的对应基础排序；任务链只能增加可解释 bonus。
- 报告同时给出 `Recall-any/all@5/10/20/30`、nDCG、gold rank、router confidence、Path A/B contribution、fallback count。

### 7.3 实验门禁

- Full 和 no-task-chain 使用同一 commit、dataset、prompt、embedding、LLM model 和固定配置。
- 每个 trial 有独立 state/index/log/output。
- 不把不同模型阶段或不匹配 query 前缀的结果合并为单一论文数字。
- 最终报告至少包含完整 paired table、按 query 的 delta、任务链语义审计和失败原因分类。

## 8. 需要保留的当前证据文件

- Full 当前结果：`work/ablation_20260908/full_baseline_final/results/`
- no-task-chain 完整结果：`work/ablation_20260908/full_no_task_chain_final/results/`
- Full query metrics：`full_baseline_final/results/metrics_results.json`
- Full task-chain trace：`full_baseline_final/results/session_task_chain_trace.json`
- Full checkpoint：`full_baseline_final/results/memory_state_latest.json`
- 运行日志：`full_baseline_final/run.stdout.log` 和 orchestrator runner log

这些文件是诊断和重放输入，实验未结束前不得清理。

## 9. 最终判断

前 31 个 query 的问题不是“任务链完全无效”，而是任务链目前承担了过强的排序门控责任，却没有置信度、关系类型和 Path B 保底。任务链构建对长期主题和决策分支有真实价值；需要先修复融合与路由边界，再评估其是否稳定提高 memory 场景效果。

修复可以完成，但应在本轮实验完成并归档后进行。最先实施 Batch A，因为它能修复当前已证实的排序退化，同时不改变任务链 state；Batch B/C 再分别验证 router 和任务图设计的因果贡献。
