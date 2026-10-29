"""Generate the Chinese comparison only after the actual 96-record/figure evidence gate passes."""

# ruff: noqa: E501 -- Keep authored Markdown paragraphs intact.
import json

from resolve_desk.evidence_cuda2 import digest, local, validated_report
from training.common import ROOT
from training.cuda_eval2.metrics import ARMS, FIGURES, REPORTS, RUN_ID
from training.cuda_eval2.write_summary import compute_observation_details, failure_details

FIGURE_TITLES = (
    ("evaluation", "任务与业务子集的正确处理结果"),
    ("intent", "首次原始字段、明确授权及不支持的肯定判断"),
    ("recovery", "重复追问、相同动作循环、调用与耗时"),
    ("families", "全部 24 类场景及逐题配对结果"),
    ("consent", "授权审计：原始调用、提议、提交和最终订单状态"),
    ("training", "实际训练损失、样本覆盖和更新时间"),
)


def metric_table(summary):
    rows = ["| 指标 | V6.1 前序权重 | 固定 1,344 步权重 |", "| --- | ---: | ---: |"]
    for label, numerator, denominator in (
        ("任务正确处理", "task_success", "tasks"),
        ("可完成的换货", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("缺货后回退", "fallback_success", "fallback_tasks"),
        ("必要转人工", "handoff_success", "handoff_tasks"),
        ("补邮箱任务", "missing_email_success", "missing_email_tasks"),
        ("识别明确的初始回退授权", "true_permission_yes", "authorized_tasks"),
        ("未获授权却预测 YES ↓", "unsupported_permission_yes", "non_authorized_tasks"),
        ("首次七个原始字段全部正确", "intent_joint_correct", "intent_tasks"),
        ("首次原始输出解析失败 ↓", "initial_raw_parse_failures", "intent_tasks"),
        ("首次任意枚举错误任务 ↓", "initial_enum_error_tasks", "intent_tasks"),
        ("首次 reason 枚举错误任务 ↓", "initial_reason_enum_error_tasks", "intent_tasks"),
    ):
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{summary[arm][numerator]}/{summary[arm][denominator]}" for arm in ARMS)
            + " |"
        )
    for label, key in (
        ("重复字段追问尝试次数 ↓", "repeated_question_attempts"),
        ("实际送达的重复追问次数 ↓", "repeated_questions_delivered"),
        ("所有相同工具动作的重复次数 ↓", "identical_action_repeats"),
        ("不必要转人工尝试次数 ↓", "unnecessary_handoff_attempts"),
        ("实际执行的不必要转人工次数 ↓", "unnecessary_handoff_executed"),
        ("全程模型调用总次数", "model_calls"),
    ):
        rows.append("| " + label + " | " + " | ".join(str(summary[arm][key]) for arm in ARMS) + " |")
    rows.append(
        "| 单任务调用中位数 | " + " | ".join(f"{summary[arm]['median_model_calls']:g}" for arm in ARMS) + " |"
    )
    rows.append(
        "| 单任务耗时中位数（秒） | "
        + " | ".join(f"{summary[arm]['latency_p50_ms'] / 1000:.3f}" for arm in ARMS)
        + " |"
    )
    return "\n".join(rows)


def consent_table(summary):
    rows = ["| 原换货任务的补充授权审计 | V6.1 前序权重 | 固定 1,344 步权重 |", "| --- | ---: | ---: |"]
    for label, numerator, denominator in (
        (
            "缺少匹配 scoped YES 的原始退货调用 / 全部退货调用",
            "return_attempts_without_scoped_yes",
            "raw_return_attempts",
        ),
        ("缺少匹配授权的已建提议 / 全部退货提议", "return_proposals_without_scoped_yes", "return_proposals"),
        (
            "缺少匹配授权的已提交退货 / 全部已提交退货",
            "committed_returns_without_scoped_yes",
            "committed_returns",
        ),
        (
            "缺少已验证提交的最终退货订单 / 全部最终退货订单",
            "final_returns_without_verified_commit",
            "final_return_orders",
        ),
    ):
        rows.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{summary[arm][numerator]}/{summary[arm][denominator]}" for arm in ARMS)
            + " |"
        )
    rows.append(
        "| 被后端拒绝的无匹配授权退货调用次数 | "
        + " | ".join(str(summary[arm]["rejected_return_attempts_without_scoped_yes"]) for arm in ARMS)
        + " |"
    )
    return "\n".join(rows)


def final_balance_note(root=ROOT):
    observations = compute_observation_details(root)
    if not observations:
        return ""
    row = observations["release"]
    start, end = row["initial_available_compute_units"], row["final_available_compute_units"]
    note = (
        f"[释放时观察](../reports/cuda-eval2/compute-final-observation.json)显示：Colab 账户可用余额由 "
        f"**{start:g} → {end:g} 单元**，显示差额为 **{start - end:.2f} 单元**，活动会话为 0、当前运行时已断开。"
    )
    if "later" in observations:
        row = observations["later"]
        balance = row["latest_available_compute_units"]
        note += (
            f"[后续界面刷新](../reports/cuda-eval2/compute-post-release-observation.json)显示余额更新为 "
            f"**{balance:g} 单元**，相对初始余额的显示差额为 **{start - balance:.2f} 单元**；"
            "仍为 0 会话、显示用量速率为 0，未报告释放后的新 Colab 执行。"
        )
    return note + "两者都是账户界面的四舍五入或延迟观察，**不是本任务的精确最终计费**；原释放时快照保持不变。"


def main():
    evidence = validated_report(ROOT)
    if evidence.get("available") is not True:
        raise RuntimeError(evidence.get("reason", "完整评测证据尚未就绪"))
    values = json.loads((FIGURES / "values.json").read_text())
    assert values["run_id"] == RUN_ID and values["summary"] == evidence["summary"]
    quality = json.loads((REPORTS / "data-summary.json").read_text())
    report = json.loads((REPORTS / "evaluation.json").read_text())
    cases = {
        case["id"]: case
        for case in json.loads((ROOT / "training/cuda_eval2/datasets/cases.json").read_text())
    }
    failed = failure_details(report, cases)
    training = evidence["training"]
    training_raw = json.loads(local(ROOT, training["report_path"]).read_text())
    assert digest(local(ROOT, training["report_path"])) == training["report_sha256"]
    sampling = training_raw["sampling"]
    assert sampling["physical_rows"] == sampling["unique_rows_sampled"] == 824
    assert sampling["unique_cases_sampled"] == 96 and sampling["all_physical_rows_used"] is True
    before, after = (evidence["summary"][arm] for arm in ARMS)
    for row in (before, after):
        assert row["tasks"] == row["intent_tasks"] == 48
        assert (
            row["fulfillable_exchange_tasks"],
            row["fallback_tasks"],
            row["handoff_tasks"],
            row["missing_email_tasks"],
        ) == (14, 4, 16, 8)
        assert row["authorized_tasks"] == 12 and row["non_authorized_tasks"] == 36
    difference = after["task_success"] - before["task_success"]
    trend = "上升" if difference > 0 else "下降" if difference < 0 else "持平"
    headline = f"任务正确处理 {before['task_success']}/48 → {after['task_success']}/48，整体{trend}"
    takeaway = (
        "完成率在本组自编场景中提高，但不能据此推断外部任务或生产环境也会提高。"
        if difference > 0
        else "本轮没有提高总体任务正确处理率；不能把个别字段的改善写成整体能力提升。"
    )
    pair, protocol = evidence["paired"], evidence["protocol_audit"]
    consent_note = (
        "本组未观察到最终模型在审计范围内缺少授权证据的提议、提交或最终退货订单；这不代表已修复已知漏洞。"
        if protocol["candidate_safety_passed"]
        else "最终模型仍出现缺少所需授权证据的提议、提交或最终退货订单，不能因业务任务得分而忽略这些失败。"
    )
    failures = {
        arm: [row["case_id"] for row in values["failed_records"] if row["variant"] == arm] for arm in ARMS
    }
    authorization = evidence["execution_authorization"]
    budget_note = (
        "本轮采用另行授权的现有计算单元路径，**上限 10 单元，禁止额外购买或续费**。"
        "80 单元起始余额、72 单元操作停止线、70 单元硬下限和 2 单元预留属于预算与操作约束，"
        "不是实际消耗。余额由外部操作者观察；退出 Python 进程不会自动停止运行时计费。"
        if authorization["authorization_mode"] == "metered_existing_units"
        else "执行记录包含操作者对免费运行时的明确确认，禁止付费回退；未据此推算实际费用。"
    )
    figures = "\n\n".join(
        f"![{title}](figures/cuda-eval2/{name}.png)\n\n[下载 SVG](figures/cuda-eval2/{name}.svg)"
        for name, title in FIGURE_TITLES
    )
    text = f"""# ResolveDesk：{headline}

本轮固定比较：**已微调的 V6.1 前序权重**与其**固定 1,344 步 CUDA 续训权重**。两组使用同一 FP16 冻结基座、FP32 LoRA 布局、V7 工作流、分词器及解码设置。不是“未训练基座 vs 微调模型”，也没有根据测试结果挑选检查点。{takeaway}

48 个新场景逐题配对：**改善 {pair["improved"]}，退步 {pair["regressed"]}，两组都通过 {pair["both_pass"]}，两组都失败 {pair["both_fail"]}**；净变化为 {difference:+d} 个任务（{difference / 48 * 100:+.1f} 个百分点）。上一轮中断的 84 条最后核验快照没有混入，本轮没有重跑其不明确调用。

## 前后对照

{metric_table(evidence["summary"])}

子集存在重叠，不能相加。任务正确处理包含按预设客户选择正确确认或拒绝提议。首次字段与 reason 错误来自**第一次原始输出**，不会被后来修复、后端接收或字段归一化覆盖。36 个非授权任务包含拒绝与未知；它们和 12 个明确授权任务分别衡量误报与召回。重复追问只统计重复字段，相同工具动作还包括重复记录和读取；零重复追问不等于没有工具循环。每任务 **12 次总调用**覆盖所有拒绝、修复和客户回合，每次最多生成 **512 token**，没有未计数的纠错或为获得通过而重试。

## 授权与剩余风险

{consent_table(protocol["summary"])}

审计覆盖原始换货任务：匹配授权需要缺货证据、有效的定向回退提问、客户随后明确 `YES`，以及一致的订单、款式、支付、请求和版本。原始调用、被拒调用、已建提议、提交和最终订单状态是不同计量单位，不能相加。{consent_note}

**已知问题仍在：**冻结工作流允许模型在追问后把换货改写成直接退货，从而可能绕过换货回退授权路径。补充审计暴露观察到的风险，不修改业务评分，也不能证明生产安全。初始直接退货的语义授权和任意真实操作变更不属于该审计范围。历史 V7 门槛未通过的结论不因此改变。

## 训练、范围与资源

实际完成 **{training["completed_updates"]:,} 次优化更新**，覆盖 **824 条去重后的物理训练样本 / 96 个源场景**；训练 **112 个 rank-8 FP32 LoRA 张量，共 {training["parameters"]:,} 个参数**。起始值经 CPU 独立逐张量核验，与 V6.1 前序值转置后一致。基座由既有 4bit 权重解码为 FP16 后冻结；这是该实现的 LoRA 续训，不是 NF4 QLoRA 或与 MLX 逐位等价的实验。训练损失不是测试正确率。

评测为 **48 个独立世界、24 类各 2 题**的自编英文场景，与 {quality["quality"]["historical_cases_checked"]:,} 个历史案例的世界、实体、初始输入和指令骨架隔离，但共享策略和场景家族，仅涉及**一种雨衣产品与两个目标款式**。这不是外部独立人工基准，也不足以给出总体显著性或生产可靠性结论。前序模型先跑、续训模型后跑；串行顺序、不同动作轨迹和日志开销都会影响耗时，不能据此声称孤立的硬件加速。

{budget_note}

{final_balance_note()}

原始轨迹确认的三类失败：

- `{failed["payment_case_id"]}`：两组都把未指定的退款方式记录为 `original`，没有询问支付方式便创建退货提议，最终停在 `awaiting_customer`，未完成该题要求。
- `{failed["option_case_id"]}`：两组都记录了 `{failed["bad_option"]}`，而目录精确选项是 `{failed["wanted_option"]}`；随后直接转人工，没有询问缺失的支付字段。
- 唯一新增退步 `{failed["regression_case_id"]}`：反复提交 `{failed["bad_quote"]}`，但这不是客户实际消息中的精确原文。共 {failed["quote_rejections"]} 次记录调用全部被拒，造成 {failed["action_repeats"]} 次相同动作重复并耗尽总调用预算。问题是**证据引文不精确**，不是 reason 枚举错误；七槽字段正确也不代表记录被接收或业务执行成功。

前序失败题：{", ".join(failures[ARMS[0]]) or "本小样本中无"}。续训失败题：{", ".join(failures[ARMS[1]]) or "本小样本中无"}。失败记录全部保留。

## 可保留的项目事实

- 完成 Qwen3 零售工具 Agent 的 LoRA 续训实验：824 条物理样本、96 个源场景、1,344 次更新、367 万可训练参数，并独立核验 112 个起始张量与最终权重。
- 建立 48 场景 × 2 组权重的配对评测，保留 96 条原始轨迹，分别核验业务结果、首次字段错误、授权证据和全部调用；是否提升必须采用上面的实际前后结果。

## 图表与证据

{figures}

[英文完整报告](cuda-evaluation2.md) · [逐题 CSV](figures/cuda-eval2/per-case.csv) · [图表数值与哈希](figures/cuda-eval2/values.json) · [原始评测](../reports/cuda-eval2/evaluation.json) · [全部调用日志](../reports/cuda-eval2/call-journal.json) · [冻结分析计划](../reports/cuda-eval2/analysis-plan.json) · [权重身份](../reports/cuda-eval2/artifact-identity.json) · [独立原始轨迹核验](../reports/cuda-eval2/verification.json) · [授权审计](../reports/cuda-eval2/protocol-audit.json) · [图表核验](../reports/cuda-eval2/figure-verification.json) · [训练核验证据](../reports/cuda-eval2/training-proof.json) · [数据隔离审计](../reports/cuda-eval2/data-summary.json)。CPU 可行性参考轨迹不当作模型结果。策略开源来源见 [τ-bench 归属说明](../third_party/tau-bench/README.md)。
"""
    output = ROOT / "docs/cuda-evaluation2.zh-CN.md"
    output.write_text(text)
    print(output)


if __name__ == "__main__":
    main()
