# RealMemBench 固定消融实验状态（2026-09-08）

## 代码门禁

- 固定配置（Full baseline 与 w/o task-chain）测试：`121 passed`。
- sweep 不属于本轮实验；相关测试/模块未纳入门禁。
- evaluator 已支持 `--resume-from`，恢复时校验 record 前缀，复用 append-only query 日志，并在每个 query 后写 checkpoint。
- 启动前 tiny chat preflight 已开启；额度不足时不会先处理数据。

## 当前外部状态

`/data/wz/agent_memory/iconip2026/TCMem/.llm_runtime_qwen/gemini.env` 的 chat 预检持续返回 HTTP 403 `insufficient_user_quota`，服务端报告剩余额度约 `-0.074254`。A100 上的 `scripts/run_fixed_ablation_resume.sh` 已后台运行，每 10 分钟重读配置并重试；额度恢复后会自动启动两个固定分支并从 checkpoint 续跑。

## 已保存的 partial checkpoint（不可作为完整结果）

| 分支 | checkpoint records | completed queries | 状态 |
|---|---:|---:|---|
| Full baseline | 128 / 688 | 18 / 124 | invalid（API 配额中断） |
| w/o task-chain | 485 / 688 | 87 / 124 | invalid（API 配额中断） |

两者处理的 record 前缀不同，不能直接比较 partial 指标或宣称任务链优于消融。恢复校验已确认 baseline 下一条为 `rec_a9706c77_0001`，no-task-chain 下一条为 `rec_48aff13e_0001`。

## Full baseline 语义审计

当前 checkpoint 含 35 个 active task、35 个 main branch、128 个 records；抽样显示 WhatsApp 工具选择、Whisper 语音转录、CEO 职位目标、E-Myth audiobook 等连续语义被合并到相应任务，明显主题切换时建立新任务。当前 18 个已完成 query 没有 any-recall miss，且未发现低词汇重合节点；branch/override 冲突场景尚未被覆盖，因此不能据此断言完整任务链质量。

审计文件：`full_baseline/results/semantic_audit.json` 与 `.md`。额度恢复并形成 finished manifest 后，需重新运行 `scripts/audit_full_baseline_semantics.py`，再按完整 baseline 与 no-task-chain 的成对指标分析。

### 当前可疑低排序样本（仅诊断）

审计把 10/18 个 partial query 标为低 `nDCG@5` 或低 `recall_all@5`。其中 Q-0010（用户明确询问 Remittance Agent Network 的后续）、Q-0011/Q-0012（财务分析/净资产连续追问）、Q-0017（明确指向刚制作的 voice notes）被路由到相邻但不完全匹配的 task，说明 query router 的候选任务语义相似时可能过度依赖最近/词汇重合任务；Path B 在更大 k 有时补回 gold，但 Path A 的错误路由会拉低前排排序。这个问题应在完整成对结果中按 query route、Path A/B 命中原因和 gold session 再确认，不能用 partial 数字作最终结论。
