# ResolveDesk：40 场景阶段性对照的失败分析

在相同 V8 流程下，已微调的旧权重为 **30/40**，保存的 V8 续训权重为 **33/40**：5 例改善、2 例退化，仍有 7 例失败。以下仅解释原始轨迹中观察到的链路，不证明某条训练数据、损失权重或模型内部机制造成了变化。

这是因固定截止时间而截断的、事后批准的共同前 40 场景分析。原始 144 条评测仍未完成：保留的 136 条完整记录已独立 CPU 核验，只有其中 120 条用于三组配对。原有 1 个响应未知的调用未重试、未补事务。这里的 7 例是候选权重在已完成 40 例中的业务失败，**不是把缺失的 8 条记录计为失败**。[阶段结果][V]、[公开核验摘要][P]

## 全部 7 个候选失败

`R[n]` 表示本地不可变原始 JSON 的 `records[n]`，索引从 0 开始；步骤编号从 1 开始。原始轨迹保留在本地，公开 CSV 仅提供 `original_record_index` 和 `source_record_sha256` 及逐条数值结果，不提供完整提示词、模型输出或 token。下表均为 `v8_weights_new_flow`。[公开逐条结果][R]

| 原始索引 / case_id | 调用数 | 观察到的失败链路 |
|---|---:|---|
| **101** · `v8-eval2-test-return-01` | 6 | 初次付款预测为正确的 `original`，但步骤 4 的 gate 没有认可付款证据，要求询问 `payment_method`。模型在步骤 4–6 连续发送 `ask_customer(field="requested_option")`，均返回 `invalid_action_json: Unsupported clarification field`，最终 `failed`。这是合法 JSON 中错误的工具字段，不是 JSON 语法损坏。 |
| **102** · `v8-eval2-test-missing_payment-00` | 6 | 客户明确说 original 指原订单、尚未选择退款去向；模型仍预测 `payment_method="original"`。gate 要求付款澄清，模型却连续 3 次使用不支持的 `requested_option` 字段，没有实际送出付款问题，最终 `failed`。这是两条退化之一。 |
| **111** · `v8-eval2-test-foreign_order-01` | 6 | 初次漏掉正文已提供的订单号，先问订单号。客户真实回复订单号后，步骤 4–6 仍引用旧首轮正文，均触发 `evidence_validation_error`。后续没有执行 `get_order`，因此未完成本应出现的所有权检查与必要转人工。 |
| **119** · `v8-eval2-test-exchange_gift-01` | 3 | 初次结构化槽位已正确，但 evidence 从 `Exchange into…` 开始；原句中的对应词是小写 `exchange`。该 evidence 不是大小写完全一致的原文片段，连续 3 次相同 `record_request` 被拒，尚未读取目录便停止。这是另一条退化。 |
| **122** · `v8-eval2-test-exchange_shortfall-00` | 3 | 客户要求礼品卡余额不足时转人工、禁止退货或换付款方式。结构化槽位正确，但 evidence 把原句 `Try to exchange…` 中的小写 `exchange` 改成了 `Exchange…`，连续 3 次拒绝。流程尚未进入余额/业务判断，必要转人工没有完成。 |
| **125** · `v8-eval2-test-exchange_fallback_available-01` | 3 | evidence 将首句重写为 `Exchange through original payment…`，并非原文片段；连续 3 次拒绝。初次 `requested_option` 还带了产品名 `tunic`，但此次根本没有进入目录匹配，不能把规格匹配认定为已证实的直接失败原因。 |
| **131** · `v8-eval2-test-exchange_stockout_missing_email-01` | 9 | 对明确的“不授权退货”误报 `fallback_return="yes"`，规格还带产品名 `smock`。实际补邮箱及识别客户成功；目录 gate 的候选为空，额外问了规格。客户真实答复规格后，步骤 7–9 又引用旧首轮正文，3 次 evidence 拒绝，未到必要转人工。 |

5 例失败含连续 3 次引用校验拒绝，合计正好对应本批候选的 **15 次 quote rejection**；另 2 例含连续 3 次不支持的澄清字段。三次无效动作的有界停止保留了失败，没有把错误重试包装成成功。

## 两条退化的直接对照

- **缺付款选择：`R[54] → R[102]`。** 两者初次都错误预测 `original`，因此不能把这条退化归因为“初次付款预测变差”。旧权重在一次非法字段后改为有效的 `ask_customer(payment_method)`，取得实际答复，再 `propose_return`；新权重没有完成这次错误恢复，连续非法字段后停止。
- **礼品卡换货：`R[71] → R[119]`。** 旧权重使用了可验证的原文片段，随后 `identify_customer → get_order → get_product → record_request → propose_exchange`，并保留客户拒绝确认的结果。新权重的初次规格字段反而更准确，但 3 次重复了不精确引用，未进入业务阶段。槽位正确率与端到端完成率在这例中不是同一件事。

## 对重点问题的实测解释

- **重复调用仍存在。** 相同动作重复次数由 15 降到 9，重复送达的追问两组均为 0；“没有重复追问”不代表“没有重复工具动作”。候选总调用数反而为 213，旧权重为 208，不能据此声称整体更省调用。
- **付款识别有退化。** 初次付款字段正确数由 **39/40 降到 37/40**。新增错误是 `R[116]`、`R[117]`：正文只提原订单/原购买，且尚未选择付款，模型却填 `original`。两例后来都实际询问 `payment_method` 并完成任务，说明流程澄清掩盖了初次预测错误，不能把最终成功算作初次识别正确。
- **控制器也有保守识别局限。** `R[101]` 的输入已说 `reimbursing original payment`，但 gate 的 `payment_evidence.candidate_values=[]`。冻结代码的 `positive_clauses` 会跳过含 `neither` 的整个句段，与此现象一致；这不是干预实验，也不是只凭轨迹证明的唯一根因。`R[131]` 同样显示目录候选为空，已有规格仍被追问；本次没有修改过滤器或猜测 SKU。[冻结实现][C]
- **引用必须跟随最新真实回复。** `R[111]` 和 `R[131]` 的问题是客户答复后继续引用旧正文；`R[119]`、`R[122]` 是大小写不完全一致；`R[125]` 是重写句子。不能用放宽 evidence 校验来改写这批结果。
- **授权误报没有消失。** 三组均在同一个 `…exchange_stockout_missing_email-01` 上出现无依据的初次 YES，仍为 **1/32**。所选轨迹未观察到缺少 scoped YES 的退货提议/提交；这不能改写成“零授权误报”，也不能推断未覆盖场景安全。

后续若改进，应将这些已见错误作为开发诊断，并用新的测试集检验“最新原文复制、非法字段反馈后的动作选择、否定范围下的付款/规格证据”。本文件没有修改模型、控制器、计分、计划或默认 provider，也没有执行新训练、GPU 或模型调用。

[R]: figures/v8-eval2-partial40/per-case.csv
[V]: figures/v8-eval2-partial40/values.json
[P]: ../reports/v8-eval2-partial40/public-summary.json
[C]: ../resolve_desk/resolution_v8.py
