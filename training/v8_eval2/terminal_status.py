"""Sanitized terminal progress only; never score partial trajectories or run a model."""

import argparse
import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

from training.common import ROOT

RUN_ID = "resolvedesk-v8-eval-002"
ARMS = ("prior_weights_old_flow", "prior_weights_new_flow", "v8_weights_new_flow")
LABELS = ("Prior adapter / V7 flow", "Prior adapter / V8 flow", "Saved V8 adapter / V8 flow")
SUMMARY = "reports/v8-eval2/terminal-summary.json"
SUMMARY_SHA = "2e7f1d2454b14bd00647e3a437bde097ccd5561e692cb98152d6971b511e7e12"
PLAN_SHA = "544521fbf83c63c1658af6605331f4e7d7b334523ea60fff7177d5cf9859ddac"
IDENTITY_SHA = "5e8ea77ec9076b31e7be1b9cf0e5e19c983fa5d87200a506150b08bd72b5b36b"
OUTPUT = "docs/figures/v8-eval2-status"
STATUS_ASSETS = {
    "completion.png": "image/png",
    "completion.svg": "image/svg+xml",
    "completion.csv": "text/csv",
}


def checked(root, name):
    root, relative = Path(root).resolve(), Path(name)
    if relative.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("Expected a normalized local relative path")
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(root):
        raise ValueError("No symlink or path escape in terminal evidence")
    return path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_terminal(root=ROOT):
    raw = checked(root, SUMMARY).read_bytes()
    if digest(raw) != SUMMARY_SHA:
        raise ValueError("Terminal summary differs from the reviewed immutable sanitized bytes")
    data = json.loads(raw)
    if data["run_id"] != RUN_ID or data["schema_version"] != 1:
        raise ValueError("Different terminal run")
    if data["status"] != "terminal_incomplete_deadline" or data["process_return_code"] != 124:
        raise ValueError("Not the deadline-stopped incomplete evaluation")
    if data["analysis_plan_sha256"] != PLAN_SHA or data["artifact_identity_sha256"] != IDENTITY_SHA:
        raise ValueError("Terminal summary binds different frozen inputs")
    if digest(checked(root, "reports/v8-eval2/analysis-plan.json").read_bytes()) != PLAN_SHA:
        raise ValueError("Frozen plan changed")
    if digest(checked(root, "reports/v8-eval2/artifact-identity.json").read_bytes()) != IDENTITY_SHA:
        raise ValueError("Frozen saved-weight identity changed")
    counts = data["records_by_arm"]
    planned = data["planned_records_per_arm"]
    if set(counts) != set(ARMS) or planned != 48 or data["planned_test_records"] != 144:
        raise ValueError("Different declared cohort")
    if not all(type(counts[a]) is int and 0 <= counts[a] <= planned for a in ARMS):
        raise ValueError("Invalid completed-record count")
    if sum(counts.values()) != data["completed_test_records"] or not 0 < sum(counts.values()) < 144:
        raise ValueError("Inconsistent terminal record count")
    if data["missing_completed_records"] != 144 - data["completed_test_records"]:
        raise ValueError("Missing record count differs")
    if (
        data["unstarted_task_count"] + data["ambiguous_pending_task_count"]
        != data["missing_completed_records"]
    ):
        raise ValueError("Unfinished task categories do not reconcile")
    if data["completed_call_entries"] != data["calls_attached_to_completed_rows"]:
        raise ValueError("Completed responses include an unattached task")
    if data["model_calls_issued"] != (
        data["completed_call_entries"] + data["started_response_unknown_call_entries"]
    ):
        raise ValueError("Issued call counts differ")
    for key in (
        "full_result_verified",
        "quality_metrics_computed",
        "model_performance_charts_generated",
        "old_results_pooled",
        "resume_or_replay_allowed",
    ):
        if data[key] is not False:
            raise ValueError("Cannot present this incomplete run as quality evidence or permit replay")
    if data["training_updates_this_run"] != 0 or data["no_retraining_performed"] is not True:
        raise ValueError("Saved-weight-only scope changed")
    if data["gpu_runtime_released"] is not True or data["own_active_sessions_after_release"] != 0:
        raise ValueError("Terminal runtime release was not confirmed")
    return data


def progress_rows(summary):
    return [
        {
            "arm": arm,
            "label": label,
            "complete_task_records": summary["records_by_arm"][arm],
            "no_complete_task_record": summary["planned_records_per_arm"] - summary["records_by_arm"][arm],
            "planned_task_records": summary["planned_records_per_arm"],
            "measure": "record_completion_only_not_model_performance",
        }
        for arm, label in zip(ARMS, LABELS, strict=True)
    ]


def csv_bytes(summary):
    rows = progress_rows(summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def status_values(summary):
    return {
        "run_id": RUN_ID,
        "status": summary["status"],
        "summary_sha256": SUMMARY_SHA,
        "measure": "record_completion_only_not_model_performance",
        "rows": progress_rows(summary),
        "completed_test_records": summary["completed_test_records"],
        "planned_test_records": summary["planned_test_records"],
        "missing_records_are_model_failures": False,
        "quality_metrics_computed": False,
        "model_performance_charts_generated": False,
    }


def document(summary, zh=False):
    done, planned, missing = (
        summary[k] for k in ("completed_test_records", "planned_test_records", "missing_completed_records")
    )
    table = "| Arm | Complete records | No complete record | Planned |\n| --- | --- | --- | --- |\n"
    table += "".join(
        f"| {r['label']} | {r['complete_task_records']} | {r['no_complete_task_record']} | "
        f"{r['planned_task_records']} |\n"
        for r in progress_rows(summary)
    )
    if zh:
        title = f"# ResolveDesk V8 三臂评测已终止：保存 {done}/{planned} 条完整任务记录\n\n"
        intro = (
            f"本轮在固定截止时间 **{summary['process_stopped_at_utc']}** 因 deadline-stop 结束，"
            f"进程返回码为 {summary['process_return_code']}。三臂分别保存 **48、48、40** 条完整记录。"
            f"这只是证据完成进度，**不是模型成功率**。缺少的 {missing} 条完整记录"
            "没有被算作模型成功或失败。\n\n"
        )
        body = (
            f"尚未完成的 {missing} 个任务中，{summary['unstarted_task_count']} 个未启动，"
            f"{summary['ambiguous_pending_task_count']} 个存在一次已发出、未保留返回的调用。"
            f"调用账本共记录 {summary['model_calls_issued']} 次已发出调用："
            f"{summary['completed_call_entries']} 次保留返回，均附属于已保存的完整任务记录；"
            f"{summary['started_response_unknown_call_entries']} 次返回状态不明。"
            "原始输出、调用账本及截止证据均已归档；不自动重试、续接或重放这次状态不明的调用。\n\n"
            "没有完整的 144 条匹配三臂数据，因此未计算整体效果对照，未运行完整结果质量核验，"
            "也未生成六组最终性能图。不能据此声称微调提升、流程提升、退化幅度或生产可靠性。"
            "下图只区分已保存完整记录与尚无完整记录，不编码任务处理的对错。\n\n"
            f"本轮只加载保存权重，新增训练为 {summary['training_updates_this_run']} 次；"
            f"已保存的 V8 历史训练为 {summary['saved_v8_training_updates']} 次更新，仍保留其证明。"
            "旧对照权重此前已经微调。48 个场景是同作者零售合成场景，不是外部 benchmark 或生产数据。"
            "旧中断测试及历史验证独立保留，没有池化或重放。GPU 运行时已释放，观察到本项目活动会话为 0；"
            "本报告不公开账户、余额、资源授权或原始会话内容。\n\n"
        )
    else:
        title = f"# ResolveDesk V8 evaluation stopped: {done}/{planned} complete task records saved\n\n"
        intro = (
            f"The fixed deadline stopped this run at **{summary['process_stopped_at_utc']}** "
            f"with process return code {summary['process_return_code']}. The three arms saved "
            "**48, 48 and 40** complete task records. This is evidence-completion progress, "
            f"**not a model success rate**. The {missing} missing complete records are not scored "
            "as model successes or failures.\n\n"
        )
        body = (
            f"Of the {missing} unfinished tasks, {summary['unstarted_task_count']} were unstarted and "
            f"{summary['ambiguous_pending_task_count']} contains an issued call with no retained response. "
            f"The journal contains {summary['model_calls_issued']} issued calls: "
            f"{summary['completed_call_entries']} retained responses attached to "
            "saved complete task records, "
            f"and {summary['started_response_unknown_call_entries']} call with unknown response status. "
            "Raw outputs, the journal and deadline evidence remain archived. No automatic retry, "
            "resume or replay of the ambiguous call is allowed.\n\n"
            "There is no complete matched 144-record three-arm comparison. Overall comparative quality "
            "metrics and full-result quality verification were not produced; the six final performance "
            "chart groups were not generated. No training uplift, workflow uplift, regression magnitude "
            "or production-reliability claim follows. The figure distinguishes saved complete records "
            "from absent complete records; it does not encode correct or incorrect business handling.\n\n"
            "This saved-weight-only window performed "
            f"{summary['training_updates_this_run']} optimizer updates. "
            f"The prior saved V8 training completed {summary['saved_v8_training_updates']} updates; "
            "its proof "
            "remains retained. The comparison adapter was already fine-tuned. The 48 same-author synthetic "
            "retail scenarios are not an external benchmark or production data. Earlier interrupted test "
            "outputs and validation remain separate, without pooling or replay. The GPU runtime was "
            "released and zero own active sessions were observed. Account, balance, resource approval "
            "and raw session contents are outside this report.\n\n"
        )
    links = (
        "![Record completion only; not model performance](figures/v8-eval2-status/completion.png)\n\n"
        "[CSV](figures/v8-eval2-status/completion.csv) · "
        "[SVG](figures/v8-eval2-status/completion.svg) · "
        "[Sanitized terminal source](../reports/v8-eval2/terminal-summary.json) · "
        "[Chart values](figures/v8-eval2-status/values.json)\n"
    )
    return title + intro + table + "\n" + body + links


def verify_status_delivery(root=ROOT):
    summary = read_terminal(root)
    folder = checked(root, OUTPUT)
    receipt = json.loads((folder / "status-receipt.json").read_text())
    if receipt["run_id"] != RUN_ID or receipt["summary_sha256"] != SUMMARY_SHA:
        raise ValueError("Status export belongs to another source")
    expected_files = {"completion.png", "completion.svg", "completion.csv", "values.json"}
    if set(receipt["files"]) != expected_files or receipt["quality_metrics_computed"] is not False:
        raise ValueError("Status export must contain progress only")
    for name, expected in receipt["files"].items():
        if digest(checked(root, f"{OUTPUT}/{name}").read_bytes()) != expected:
            raise ValueError("Progress asset changed")
    if (folder / "completion.csv").read_bytes() != csv_bytes(summary):
        raise ValueError("Progress CSV differs from reviewed counts")
    if json.loads((folder / "values.json").read_text()) != status_values(summary):
        raise ValueError("Progress values differ")
    for name, expected in receipt["documents"].items():
        if digest(checked(root, name).read_bytes()) != expected:
            raise ValueError("Terminal report changed")
    return receipt


def deliver(root=ROOT):
    summary = read_terminal(root)  # Before renderer import or output creation.
    root = Path(root).resolve()
    destination = checked(root, OUTPUT)
    docs = {"docs/v8-eval2-incomplete.md": False, "docs/v8-eval2-incomplete.zh-CN.md": True}
    if destination.exists() or any(checked(root, name).exists() for name in docs):
        raise FileExistsError("Never replace a terminal progress export or report")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": RUN_ID + "-terminal-progress",
        }
    )
    rows = progress_rows(summary)
    fig, ax = plt.subplots(figsize=(11, 6))
    complete = [r["complete_task_records"] for r in rows]
    missing = [r["no_complete_task_record"] for r in rows]
    positions = list(range(3))
    ax.barh(positions, complete, color="#376786", height=0.54, label="Saved complete task records")
    ax.barh(
        positions,
        missing,
        left=complete,
        color="#e1e5e7",
        edgecolor="#82929c",
        hatch="///",
        height=0.54,
        label="No complete task record",
    )
    for y, n, remaining in zip(positions, complete, missing, strict=True):
        ax.text(n / 2, y, f"{n}/48 records", ha="center", va="center", color="white", weight="bold")
        if remaining:
            ax.text(n + remaining / 2, y, f"{remaining}\nmissing", ha="center", va="center", fontsize=11)
    ax.set(
        yticks=positions,
        yticklabels=LABELS,
        xlim=(0, 48),
        xticks=[0, 12, 24, 36, 48],
        xlabel="Task records per arm · 48 planned",
        title="Evaluation stopped: 136/144 complete records saved",
    )
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.30), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.29, right=0.96, bottom=0.34, top=0.86)
    fig.text(
        0.03,
        0.025,
        "Record completion only, NOT model performance. Missing records are not model failures.\n"
        "Fixed deadline stop; 7 tasks unstarted and 1 task has a call with unknown response status.",
        fontsize=11,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v8-eval2-status-", dir=destination.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        fig.savefig(stage / "completion.png", dpi=170)
        fig.savefig(stage / "completion.svg", metadata={"Date": None})
        plt.close(fig)
        (stage / "completion.csv").write_bytes(csv_bytes(summary))
        (stage / "values.json").write_text(json.dumps(status_values(summary), indent=2) + "\n")
        documents = {name: document(summary, zh) for name, zh in docs.items()}
        receipt = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "status": "verified_terminal_progress_only",
            "summary_sha256": SUMMARY_SHA,
            "quality_metrics_computed": False,
            "model_performance_charts_generated": False,
            "visual_qa_performed": False,
            "source_sha256": digest(Path(__file__).read_bytes()),
            "files": {p.name: digest(p.read_bytes()) for p in stage.iterdir()},
            "documents": {name: digest(text.encode()) for name, text in documents.items()},
        }
        (stage / "status-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for name, text in documents.items():
            with checked(root, name).open("x") as stream:
                stream.write(text)
        stage.rename(destination)
    return verify_status_delivery(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--deliver", action="store_true")
    args = parser.parse_args()
    print(json.dumps(deliver(args.root) if args.deliver else status_values(read_terminal(args.root))))
