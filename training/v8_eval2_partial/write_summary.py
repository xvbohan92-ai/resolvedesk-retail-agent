"""Bilingual, explicitly partial40 narrative after independent replay and asset verification."""

import argparse
import json
from pathlib import Path

from training.common import ROOT
from training.v8_eval2_partial.metrics import ANALYSIS_ID, ARMS, RUN_ID
from training.v8_eval2_partial.report_results import FIGURES, load_verified, sha, verify_delivery

EXCLUDED_BRANCHES = {
    "exchange_available_refusal": (
        "可满足换货中明确拒绝回退的两条分支样例未纳入，不能由其它 NO 识别率推断此分支已改善。",
        "The available-exchange explicit-refusal branch is omitted; "
        "other NO predictions cannot establish its improvement.",
    ),
    "exchange_fallback_missing_payment": (
        "缺货回退时缺支付方式的澄清与恢复未覆盖，不能声称已解决该回退失败。",
        "Stockout fallback with a missing payment method is not covered; "
        "this cannot establish that fallback fix.",
    ),
    "exchange_missing_email": (
        "可满足换货时补邮箱及后续保持换货意图的专门分支未覆盖。",
        "The dedicated available-exchange missing-email and operation-preservation branch is omitted.",
    ),
    "missing_email_payment": (
        "邮箱和支付方式双缺失的多轮恢复分支未覆盖。",
        "The combined missing-email/payment multi-turn recovery branch is omitted.",
    ),
}


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
        ]
    )


def fraction(row, numerator, denominator):
    n, d = row[numerator], row[denominator]
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else f"{n}/{d} (not sampled)"


def headline(before, after, zh=False):
    if zh:
        direction = "下降" if after < before else "未提升" if after == before else "增加"
        return "共同40样本中固定 V8 流程下任务正确处理" + direction
    direction = "decreased" if after < before else "did not increase" if after == before else "increased"
    return "Correct task handling " + direction + " on the paired40 subset under V8 flow"


def document(bundle, *, zh=False):
    values = bundle["values"]
    summaries = values["summary"]
    tasks, records = values["paired_cases"], values["paired_records"]
    calls = sum(s["model_calls"] for s in summaries.values())
    headers = [
        "指标" if zh else "Metric",
        "旧权重/V7" if zh else "Prior / V7",
        "旧权重/V8" if zh else "Prior / V8",
        "保存V8权重/V8" if zh else "Saved V8 / V8",
    ]
    specs = [
        (
            "正确处理任务（含正确拒绝）" if zh else "Correct business handling (including correct declines)",
            "task_success",
            "tasks",
        ),
        (
            "可满足换货子集" if zh else "Available-exchange subset",
            "fulfillable_exchange_success",
            "fulfillable_exchange_tasks",
        ),
        ("缺货回退子集" if zh else "Stockout-fallback subset", "fallback_success", "fallback_tasks"),
        ("必要转人工" if zh else "Necessary handoff", "handoff_success", "handoff_tasks"),
        ("补邮箱子集" if zh else "Missing-email subset", "missing_email_success", "missing_email_tasks"),
        ("首次七字段全部正确" if zh else "Joint first raw fields", "intent_joint_correct", "intent_tasks"),
        (
            "首次明确授权识别" if zh else "Initial explicit-YES recall",
            "true_permission_yes",
            "authorized_tasks",
        ),
        (
            "首次未获授权却预测YES ↓" if zh else "Unsupported first YES ↓",
            "unsupported_permission_yes",
            "non_authorized_tasks",
        ),
        (
            "首次解析失败 ↓" if zh else "First-output parse failures ↓",
            "initial_raw_parse_failures",
            "intent_tasks",
        ),
        (
            "首次 reason 枚举错误 ↓" if zh else "First reason-enum errors ↓",
            "initial_reason_enum_error_tasks",
            "intent_tasks",
        ),
    ]
    rows = [[label, *(fraction(summaries[a], n, d) for a in ARMS)] for label, n, d in specs]
    for label, key in [
        ("重复追问尝试" if zh else "Repeated-question attempts", "repeated_question_attempts"),
        ("重复问题实际送达" if zh else "Repeated questions delivered", "repeated_questions_delivered"),
        ("所有相同动作重复" if zh else "All identical-action repeats", "identical_action_repeats"),
        ("不必要转人工尝试" if zh else "Unnecessary handoff attempts", "unnecessary_handoff_attempts"),
        ("不必要转人工执行" if zh else "Unnecessary handoffs executed", "unnecessary_handoff_executed"),
        ("模型调用" if zh else "Model calls", "model_calls"),
    ]:
        rows.append([label, *(summaries[a][key] for a in ARMS)])
    title = headline(summaries[ARMS[1]]["task_success"], summaries[ARMS[2]]["task_success"], zh)
    if zh:
        intro = (
            f"这是用户明确授权后的**另行回顾性阶段分析**：固定顺序共同前 {tasks} 个场景 × 3 臂，"
            f"共 {records} 条完整轨迹、{calls} 次调用。原计划 144 条仍未完整、终止摘要和完整门槛保持不变；"
            "本报告不是完整48场景评测，也没有补跑模型。旧对照权重此前已经微调。\n\n"
            "独立CPU核验覆盖全部136条完整记录、765次已保留返回及1次未知调用的请求；"
            "下面的效果指标只使用共同子集120条轨迹，另16条完整尾部记录不参与对照。\n\n"
            "业务正确处理包含客户确认或拒绝；正确拒绝后订单应保持原状。"
            "因此换货/回退子集的成功数不等于实际换货/退款执行数。\n\n"
        )
        method = (
            "workflow 对照固定旧权重，仅比较 V7→V8 控制器和提示流程；training_under_v8 固定 V8 流程，"
            "比较旧权重与已保存的 V8 续训权重。combined 同时改变两者。"
            "这不能隔离某一项训练数据、采样或损失设计的贡献，也不是原始 Qwen 对比微调模型。\n\n"
        )
        limits = (
            "共同前40由固定臂块顺序和截止时间产生，并非随机截断或新的独立 benchmark。"
            "原完整记录为48/48/40，另16条旧权重臂尾部完整记录被排除；7个新权重任务未启动，"
            "1个任务有一次已发出但返回状态不明的调用。缺失8条不计成功或失败，不自动重放或续接。"
            "不能把此子集的比例外推到原48，更不能推断未覆盖的V8改进分支有效。\n\n"
            "类别构成也已偏移：必要转人工16条全部纳入，占16/40，而原为16/48；缺失8条全属非转人工，"
            "非转人工覆盖从32条降为24条。三组40内部仍可按同场景配对描述，"
            "但不能把总体率直接与原48或历史96条评测的总体率比较。\n\n"
            "引文纠正、商品尾缀/规格匹配和 original order 与支付方式的区分，也不是单独随机化专项对照；"
            "只能结合已选轨迹讨论，不能从整体业务率声称所有这些修复均有效。\n\n"
            "首次字段只取第一个原始输出，后续修复不补首次正确率。未获授权却预测YES不是实际未授权交易。"
            "重复追问尝试、实际送达与所有动作重复分列；所有失败和纠正调用均计入每任务12次上限。"
            "同作者零售合成场景、有限目录和脚本回复限制泛化；串行延迟包含固定臂顺序、轨迹与日志开销，"
            "不构成孤立硬件速度对照。零观察的 scoped-consent 违规不证明生产安全。\n\n"
        )
    else:
        intro = (
            f"This is a **separately authorized retrospective stage analysis** of the fixed common prefix: "
            f"{tasks} cases × 3 arms, {records} complete traces and {calls} calls. The original144 remains "
            "incomplete; its terminal summary and full-result gate are unchanged. This is not the complete48 "
            "evaluation and no model call was rerun. The prior comparison adapter was already fine-tuned.\n\n"
            "The independent CPU audit covers all 136 complete records, 765 retained responses and the "
            "one unknown call's request. Effect metrics below use only the 120 paired rows, excluding "
            "the 16 complete tail records.\n\n"
            "Correct business handling includes the customer's approval or decline; a correctly declined "
            "proposal preserves the order. Exchange/fallback success counts "
            "are not executed transaction counts.\n\n"
        )
        method = (
            "workflow holds prior weights fixed and compares V7 with V8 controller/prompt flow. "
            "training_under_v8 holds V8 flow fixed and compares prior with saved V8 continuation weights. "
            "combined changes both. These do not isolate a particular data, sampling or loss choice, "
            "and do not compare vanilla Qwen against fine-tuning.\n\n"
        )
        limits = (
            "The common40 prefix results from fixed arm-block order and a deadline, not random truncation "
            "or a new independent benchmark. Of48/48/40 completed records,16 old-weight tail records are "
            "excluded. Seven new-weight tasks were unstarted; one task has an issued call with unknown "
            "response status. Eight missing records are neither successes nor failures; no automatic replay "
            "or resume is allowed. Do not extrapolate subset rates to 48 cases "
            "or claim coverage of omitted V8 fixes.\n\n"
            "Case mix shifted: all16 necessary-handoff cases remain (16/40 versus16/48 originally); "
            "all eight missing cases are non-handoff, reducing that group from32 to24. Matched contrasts "
            "within the three40-case arms remain descriptive, but their overall rates cannot be directly "
            "compared with the original48 or historical96-trace evaluation.\n\n"
            "Quote correction, product-suffix/option grounding and the distinction between original order "
            "and payment method are not separately randomized contrasts; discuss only observed selected "
            "traces, not a claim that all these fixes work from aggregate business handling alone.\n\n"
            "Only first raw outputs earn initial-field credit; later repair cannot erase them. Unsupported "
            "initial YES is not an executed unauthorized transaction. Attempted/delivered repeated questions "
            "and all repeated actions use distinct units. "
            "Failed/corrective calls share the 12-call task budget. "
            "Same-author synthetic scenarios, finite catalogs and scripted replies limit generalization. "
            "Serial timing includes fixed arm order, trajectories and logging; "
            "it is not isolated hardware speed. "
            "Zero observed scoped-consent violations do not certify production safety.\n\n"
        )
    text = f"# ResolveDesk: {title}\n\n" + intro + table(headers, rows) + "\n\n" + method
    text += (
        table(
            ["Contrast", "Both pass", "Both fail", "Improved", "Regressed"],
            [
                [name, *(p[k] for k in ("both_pass", "both_fail", "improved", "regressed"))]
                for name, p in values["paired"].items()
            ],
        )
        + "\n\n"
    )
    focus = values["coverage"].get("focus_coverage", {})
    if focus:
        text += (
            "关键条件的覆盖数量（不是正确率）：\n\n"
            if zh
            else "Coverage of key conditions (counts, not correctness rates):\n\n"
        )
        text += (
            table(
                ["Condition", "Included40", "Original48"],
                [[name, item["selected40_tasks"], item["original48_tasks"]] for name, item in focus.items()],
            )
            + "\n\n"
        )
    text += "未覆盖分支（不计零分）：\n\n" if zh else "Omitted branches (not zero-scored):\n\n"
    text += (
        table(
            ["Family", "Included / original", "Interpretation"],
            [
                [
                    name,
                    f"0/{values['coverage']['per_family'][name]['original_cases']}",
                    EXCLUDED_BRANCHES.get(name, ("该分支未覆盖。", "This branch is not covered."))[
                        0 if zh else 1
                    ],
                ]
                for name in values["coverage"]["excluded_families"]
            ],
        )
        + "\n\n"
        + limits
    )
    p = values["protocol_summary"]
    text += (
        "以下均为已选初始换货场景，单位不得相加：\n\n"
        if zh
        else "The following concern selected exchange-origin tasks; units must not be summed:\n\n"
    )
    text += (
        table(
            headers,
            [
                [label, *(p[a][key] for a in ARMS)]
                for label, key in [
                    ("Exchange-origin tasks", "gold_exchange_tasks"),
                    ("Raw return attempts", "raw_return_attempts"),
                    ("Return proposals", "return_proposals"),
                    ("Return commits", "committed_returns"),
                    ("Final return orders", "final_return_orders"),
                    ("Attempts without scoped YES", "return_attempts_without_scoped_yes"),
                    ("Proposals without scoped YES", "return_proposals_without_scoped_yes"),
                    ("Commits without scoped YES", "committed_returns_without_scoped_yes"),
                    ("Final orders without verified commit", "final_returns_without_verified_commit"),
                    ("Rejected attempts without scoped YES", "rejected_return_attempts_without_scoped_yes"),
                ]
            ],
        )
        + "\n\n"
    )
    sampling = values["training"]["sampling"]
    text += (
        f"历史保存的 V8 训练完成1344更新，实际采样{sampling['unique_rows_sampled']}/"
        f"{sampling['physical_rows']}条物理训练pair、{sampling['unique_cases_sampled']}个源场景。"
        "rank8、112个FP32张量、3,670,016参数。本轮新增更新为0；历史训练损失不是本轮新增训练或准确率。\n\n"
        if zh
        else "The saved historical V8 run completed 1344 updates and sampled "
        f"{sampling['unique_rows_sampled']}/{sampling['physical_rows']} physical pairs across "
        f"{sampling['unique_cases_sampled']} source worlds. "
        "Rank 8, 112 FP32 tensors and 3,670,016 parameters; this evaluation performed 0 updates. "
        "Historical loss is not new training or accuracy.\n\n"
    )
    failures = values["failed_records"]
    text += "所选120条内全部失败记录：\n\n" if zh else "Every failed selected record:\n\n"
    text += (
        table(
            ["Arm", "Case", "Final status", "Calls"],
            [[r["variant"], r["case_id"], r["status"], r["model_calls"]] for r in failures],
        )
        + "\n\n"
    )
    if not failures:
        text += (
            "该子集未观察到失败；这不是生产可靠性证明。\n\n"
            if zh
            else "No failure observed in this subset; this is not proof of production reliability.\n\n"
        )
    for name in FIGURES:
        text += f"![Stage-only {name}](figures/v8-eval2-partial40/{name}.png)\n\n"
    text += (
        "[CSV120](figures/v8-eval2-partial40/per-case.csv) · "
        "[Paired40](figures/v8-eval2-partial40/case-mapping.csv) · "
        "[Metric denominators](figures/v8-eval2-partial40/metrics.csv) · "
        "[Historical training](figures/v8-eval2-partial40/training.csv)\n\n"
        "[New retrospective plan](../reports/v8-eval2-partial40/analysis-plan.json) · "
        "[Selection](../reports/v8-eval2-partial40/selection.json) · "
        "[Coverage boundary audit](../reports/v8-eval2-partial40/selection-audit.json) · "
        "[Partial evidence verification](../reports/v8-eval2-partial40/partial-evidence-verification.json) · "
        "[Original terminal summary unchanged](../reports/v8-eval2/terminal-summary.json) · "
        "[Asset values and hashes](figures/v8-eval2-partial40/values.json)\n"
    )
    return text


def write(root=ROOT):
    root = Path(root).resolve()
    bundle = load_verified(root)
    verified = verify_delivery(root, write_receipt=False)
    receipt = root / "reports/v8-eval2-partial40/figure-verification.json"
    if not receipt.is_file() or json.loads(receipt.read_text()) != verified:
        raise ValueError("Issue and reconcile actual partial40 figure verification first")
    outputs = [root / "docs/v8-eval2-partial40.md", root / "docs/v8-eval2-partial40.zh-CN.md"]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Never replace an issued partial40 bilingual report")
    for path, zh in zip(outputs, (False, True), strict=True):
        with path.open("x") as stream:
            stream.write(document(bundle, zh=zh))
    return {
        "analysis_id": ANALYSIS_ID,
        "run_id": RUN_ID,
        "original144_gate_complete": False,
        "files": {str(p.relative_to(root)): sha(root, str(p.relative_to(root))) for p in outputs},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(write(parser.parse_args().root))
