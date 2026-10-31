"""Write bilingual reports only from actual144 traces and reconciled six-group exports."""

import argparse
from pathlib import Path

from training.common import ROOT
from training.v8_eval2.metrics import ARMS, RUN_ID
from training.v8_eval2.report_results import FIGURES, load_verified, sha, verify_delivery


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def rate(summary, numerator, denominator):
    n, d = summary[numerator], summary[denominator]
    return f"{n}/{d} ({100 * n / d:.1f}%)"


def headline(before, after, zh=False):
    if after < before:
        return "固定 V8 流程下任务正确处理下降" if zh else "Task success decreased under the fixed V8 flow"
    if after == before:
        return (
            "固定 V8 流程下任务正确处理未提升" if zh else "No task-success increase under the fixed V8 flow"
        )
    return (
        "固定 V8 流程下该合成集的任务正确处理增加"
        if zh
        else ("Task success increased on this synthetic cohort under the fixed V8 flow")
    )


def document(bundle, *, zh=False):
    values, summary = bundle["values"], bundle["values"]["summary"]
    before, after = summary[ARMS[1]]["task_success"], summary[ARMS[2]]["task_success"]
    title = headline(before, after, zh)
    totals = sum(s["model_calls"] for s in summary.values())
    headers = [
        "指标" if zh else "Metric",
        "旧权重 / V7" if zh else "Prior / V7",
        "旧权重 / V8" if zh else "Prior / V8",
        "V8 保存权重 / V8" if zh else "Saved V8 / V8",
    ]
    metrics = [
        ("任务正确处理" if zh else "Complete business tasks", "task_success", "tasks"),
        (
            "可满足换货" if zh else "Available exchanges",
            "fulfillable_exchange_success",
            "fulfillable_exchange_tasks",
        ),
        ("缺货回退" if zh else "Stockout fallback", "fallback_success", "fallback_tasks"),
        ("必要转人工" if zh else "Necessary handoff", "handoff_success", "handoff_tasks"),
        ("补邮箱" if zh else "Missing-email resolution", "missing_email_success", "missing_email_tasks"),
        ("首次七字段全部正确" if zh else "Joint first raw fields", "intent_joint_correct", "intent_tasks"),
        (
            "首次明确授权识别" if zh else "Initial explicit-YES recall",
            "true_permission_yes",
            "authorized_tasks",
        ),
        (
            "首次未获授权却预测 YES ↓" if zh else "Unsupported initial YES ↓",
            "unsupported_permission_yes",
            "non_authorized_tasks",
        ),
        (
            "首次原始解析失败 ↓" if zh else "Initial raw parse failures ↓",
            "initial_raw_parse_failures",
            "intent_tasks",
        ),
        (
            "首次 reason 枚举错误 ↓" if zh else "Initial reason enum errors ↓",
            "initial_reason_enum_error_tasks",
            "intent_tasks",
        ),
    ]
    rows = [[label, *(rate(summary[a], n, d) for a in ARMS)] for label, n, d in metrics]
    for label, key in [
        ("重复追问尝试" if zh else "Repeated-question attempts", "repeated_question_attempts"),
        ("重复问题实际送达" if zh else "Repeated questions delivered", "repeated_questions_delivered"),
        ("所有相同动作重复" if zh else "All identical-action repeats", "identical_action_repeats"),
        ("不必要转人工尝试" if zh else "Unnecessary handoff attempts", "unnecessary_handoff_attempts"),
        ("不必要转人工执行" if zh else "Unnecessary handoffs executed", "unnecessary_handoff_executed"),
        ("模型调用总数" if zh else "Total model calls", "model_calls"),
    ]:
        rows.append([label, *(summary[a][key] for a in ARMS)])
    paired_rows = [
        [label, p["both_pass"], p["both_fail"], p["improved"], p["regressed"]]
        for label, p in values["paired"].items()
    ]
    protocol = values["protocol_summary"]
    consent_rows = [
        [label, *(protocol[a][key] for a in ARMS)]
        for label, key in [
            ("原始退货尝试" if zh else "Raw return attempts", "raw_return_attempts"),
            ("退货提案" if zh else "Return proposals", "return_proposals"),
            ("退货提交" if zh else "Return commits", "committed_returns"),
            ("最终退货订单" if zh else "Final return orders", "final_return_orders"),
            (
                "缺少 scoped YES 的提案" if zh else "Proposals without scoped YES",
                "return_proposals_without_scoped_yes",
            ),
            (
                "缺少 scoped YES 的提交" if zh else "Commits without scoped YES",
                "committed_returns_without_scoped_yes",
            ),
            (
                "无有效 scoped YES 而被拒的尝试" if zh else "Rejected attempts without scoped YES",
                "rejected_return_attempts_without_scoped_yes",
            ),
        ]
    ]
    failures = [[r["variant"], r["case_id"], r["status"], r["model_calls"]] for r in values["failed_records"]]
    sampling = values["training"]["sampling"]
    if zh:
        intro = (
            f"本轮只评测既有权重，实际保留 **48 个新合成场景 × 3 臂 = 144 条轨迹 / {totals} 次模型调用**。"
            "旧权重已经微调；对照不是原始 Qwen 与微调模型。本轮没有训练、梯度 smoke、验证集重跑、"
            "测试选点或默认模型切换。核验通过只证明证据一致，不代表效果通过或可生产部署。"
        )
        scope = (
            "workflow 对照保持旧权重，比较 V7 与 V8 控制器/提示流程；training_under_v8 对照固定 V8 流程，"
            "比较旧权重与保存的 V8 续训权重；combined 是二者合并对照。它们不能隔离某一项数据、损失权重"
            "或采样技巧的贡献。所有首次字段来自第一个原始输出，后续修复不补首次正确率；未获授权却预测 YES"
            "也不等于实际未授权交易。解析失败、枚举错误、追问尝试与实际送达分别保留。"
        )
        history = (
            f"复用的历史 V8 训练已完成 1,344 次更新，1100 条物理样本中实际采样 "
            f"{sampling['unique_rows_sampled']}/{sampling['physical_rows']} 条、"
            f"{sampling['unique_cases_sampled']} 个源场景；"
            "rank 8、112 个 FP32 adapter 张量、3,670,016 个参数。本轮仅加载固定的旧 33fe… 与 V8 39dabb…"
            "权重；完整身份、配置和数值证明均绑定到独立核验回执。训练损失曲线属于已保存的历史训练，不是本轮新增训练。"
        )
        limits = (
            "这些是同作者、继承政策下的单订单有限目录合成场景，不是外部人工标注或官方 benchmark。旧中断测试的"
            "34 条记录、223 次已返回调用及 pending 任务的 11 次响应均独立保留，本轮不复用、不池化、不重放；"
            "既有 48 条验证也未重跑。每个任务的失败、修复和纠正调用都计入同一个 12 次上限"
            "（总上限 1,728 次）。"
            "延迟包含固定臂顺序、不同动作轨迹和日志开销，不能作为孤立硬件加速结论；资源授权上限不等于实际任务费用。"
            "scoped-consent 的零观察值不证明生产安全或所有操作改写漏洞已消除。"
        )
    else:
        intro = (
            "This saved-weight-only run retains **48 fresh authored synthetic worlds × 3 arms = "
            f"144 traces / {totals} model calls**. The prior adapter is already fine-tuned; "
            "this is not vanilla Qwen versus fine-tuning. No training, gradient smoke, validation rerun, "
            "test-based checkpoint selection or default-provider change occurred. Verification proves "
            "evidence consistency, not improved quality or production readiness."
        )
        scope = (
            "The workflow contrast holds prior weights fixed and changes V7 to V8 controller/prompt flow. "
            "training_under_v8 holds V8 flow fixed and compares prior with saved V8 continuation weights; "
            "combined changes both. There is no ablation isolating a single data/loss/sampling choice. "
            "Initial fields come only from the first raw output; "
            "later repairs do not earn initial-field credit. "
            "Unsupported initial YES is not an executed unauthorized transaction. "
            "Raw parse/enum errors, attempted questions and delivered questions remain separate."
        )
        history = (
            "The saved historical V8 run completed 1,344 updates and sampled "
            f"{sampling['unique_rows_sampled']}/{sampling['physical_rows']} unique physical pairs across "
            f"{sampling['unique_cases_sampled']} source worlds. It used rank 8, 112 FP32 adapter tensors "
            "and 3,670,016 parameters. This run loads only fixed prior 33fe… and saved V8 39dabb… adapters "
            "with exact configurations and independently bound physical proof. The loss plot "
            "is historical saved training, not new training in this experiment."
        )
        limits = (
            "These are same-author, inherited-policy, single-order finite-catalog synthetic worlds, "
            "not an external human or official benchmark. The old interrupted 34 records, 223 returned calls "
            "and 11 pending-task responses remain separate: no reuse, pooling or replay. The existing "
            "48 validation cases were not rerun. All failed/corrective calls share the same 12-total-call "
            "task budget (maximum 1,728). Serial latency includes arm order, "
            "different trajectories and logging, "
            "so it does not isolate hardware speed. A resource ceiling is not actual job billing. "
            "Zero observed scoped-consent violations do not certify production safety or elimination "
            "of all operation-rewrite bypasses."
        )
    text = f"# ResolveDesk: {title}\n\n{intro}\n\n{table(headers, rows)}\n\n{scope}\n\n"
    text += (
        table(
            [
                "对照" if zh else "Contrast",
                "均通过" if zh else "Both pass",
                "均失败" if zh else "Both fail",
                "改善" if zh else "Improved",
                "退化" if zh else "Regressed",
            ],
            paired_rows,
        )
        + "\n\n"
    )
    text += (
        "退货路径按不同单位单列，不能相加；每臂仅对 30 个初始换货场景审计。"
        if zh
        else (
            "Return-path units are distinct and must not be summed; "
            "the audit cohort is 30 exchange-origin worlds per arm."
        )
    )
    text += "\n\n" + table(headers, consent_rows) + "\n\n" + history + "\n\n" + limits + "\n\n"
    if failures:
        text += (
            "全部失败记录如下；原始原因以逐轨迹记录为准，不凭指标猜测。"
            if zh
            else "Every failed record is listed below; raw traces, not metric inference, determine its cause."
        )
        text += "\n\n" + table(["Arm", "Case", "Final status", "Calls"], failures) + "\n\n"
    for name in FIGURES:
        text += f"![{name}](figures/v8-eval2/{name}.png)\n\n"
    text += (
        "[CSV144](figures/v8-eval2/per-case.csv) · [Paired48](figures/v8-eval2/case-mapping.csv) · "
        "[Metric denominators](figures/v8-eval2/metrics.csv) · "
        "[Historical training CSV](figures/v8-eval2/training.csv)\n\n"
        "[Plan](../reports/v8-eval2/analysis-plan.json) · "
        "[144 raw traces](../reports/v8-eval2/evaluation.json) · "
        "[Call journal](../reports/v8-eval2/call-journal.json) · "
        "[Independent replay](../reports/v8-eval2/result-verification.json) · "
        "[Scoped consent](../reports/v8-eval2/protocol-audit.json) · "
        "[Figure/CSV verification](../reports/v8-eval2/figure-verification.json) · "
        "[Asset values and hashes](figures/v8-eval2/values.json)\n"
    )
    return text


def write(root=ROOT):
    root = Path(root).resolve()
    bundle = load_verified(root)
    verified = verify_delivery(root, write_receipt=False)
    receipt = root / "reports/v8-eval2/figure-verification.json"
    if not receipt.is_file():
        raise FileNotFoundError("Run --verify-figures after actual six-group generation first")
    import json

    if json.loads(receipt.read_text()) != verified:
        raise ValueError("Persisted figure verification differs")
    outputs = [root / "docs/v8-eval2.md", root / "docs/v8-eval2.zh-CN.md"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Never replace an already issued bilingual report")
    for path, zh in zip(outputs, (False, True), strict=True):
        with path.open("x") as stream:
            stream.write(document(bundle, zh=zh))
    return {
        "run_id": RUN_ID,
        "files": {str(p.relative_to(root)): sha(root, str(p.relative_to(root))) for p in outputs},
        "new_training_performed": False,
        "report_only": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(write(parser.parse_args().root))
