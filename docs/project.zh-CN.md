# ResolveDesk 项目介绍

面向英文零售客服的 Qwen3 Agent 原型，支持订单查询、退货、换货、缺货回退、补充联系信息和转人工。订单、支付、客户与交易均为合成模拟数据。

## 个人实现与改进

- 使用 FastAPI、SQLite 和本地 Ollama 构建客服工作流，实现工具参数校验、澄清状态管理、独立客户确认、事务提交和幂等回执。
- 保留原始模型输出、错误原因、全部纠正调用、授权审计和失败案例，将初始意图解析与业务最终成功分开评测。
- 对 Qwen3-4B 已有适配器执行实际 LoRA 续训：824 条去重 SFT 样本来自 96 个来源场景，完成 1,344 次优化更新，更新 112 个张量、约 367 万参数。
- 在 48 条新编写的合成测试上完成两组共 96 条轨迹、509 次模型调用的匹配对照，并以 CPU 独立重放核验，整理六组 PNG/SVG 图、CSV 和中英文报告。

## 实际结果

| 指标 | 微调前已有适配器 | CUDA 续训后 |
| --- | ---: | ---: |
| 首轮七字段全部正确 | 64.6%（31/48） | 81.3%（39/48） |
| 无依据的初始回退授权 YES | 12/36 | 3/36，减少 75% |
| 整体业务任务成功 | 95.8%（46/48） | 93.8%（45/48），下降 2.1 个百分点 |
| 可用库存换货 | 14/14 | 14/14 |
| 必要转人工 | 16/16 | 16/16 |
| 补邮箱任务 | 8/8 | 8/8 |
| 缺货回退任务 | 3/4 | 2/4 |

该结果表明意图解析与初始授权判断改善，但业务整体成功率未提升。75% 指无依据的初始 YES 预测减少，不能表述为实际未授权交易减少。对照基线本身已经微调；CUDA 路径是 FP16 冻结底座上的 LoRA，并非 NF4 QLoRA。测试为小规模自编合成场景，不是官方 τ-bench 或真实客户生产指标。

[完整中文评测与六组图表](cuda-evaluation2.zh-CN.md) · [逐案例 CSV](figures/cuda-eval2/per-case.csv)

## 当前进度

已完成的应用、CUDA 训练和 run002 对照结果可以作为个人项目展示。V8 修复方案和数据已准备，4 步 GPU 预检完成；正式训练在模型加载前因相对路径错误退出，实际正式更新为 0 步，尚无 V8 三组评测或提升结论。后续结果应在真实完成、独立核验后单独补充。

## 简历参考

> Developed a Qwen3-4B retail service agent with validated tools, customer-confirmed transactions and idempotent SQLite commits. Continued a 3.67M-parameter LoRA adapter on 824 deduplicated SFT pairs for 1,344 updates; evaluated 96 matched synthetic trajectories across 48 new tasks with 509 captured model calls and six visual report groups. Improved joint first-output field accuracy from 64.6% to 81.3% and reduced unsupported initial fallback-consent predictions by 75%, while documenting the overall task-success regression from 95.8% to 93.8%.

保留对照对象、合成测试范围和整体回归结果，不能将 V8 计划写为已完成成果。
