"""Export actual, independently reconciled CUDA comparison figures; never create demo results."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from training.common import ROOT, sha
from training.cuda_eval2.metrics import (
    ARMS,
    FIELDS,
    FIGURE_NAMES,
    FIGURES,
    LABELS,
    REPORTS,
    RUN_ID,
    derived_values,
)
from training.cuda_eval2.verify_results import require_complete_records, verify_run_identity

COLORS = ("#94a3ae", "#315a7d")
SHORT = ("Predecessor", "After 1,344")


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    path = FIGURES / f"{name}.svg"
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def count_groups(ax, groups, summary, title):
    maximum = 1
    for j, arm in enumerate(ARMS):
        values = [summary[arm][key] for _, key in groups]
        positions = np.arange(len(groups)) + (j - 0.5) * 0.34
        ax.bar(positions, values, 0.32, color=COLORS[j], label=SHORT[j], hatch="" if j else "//")
        maximum = max(maximum, *values)
        for x, value in zip(positions, values, strict=True):
            ax.text(x, value + 0.15, str(value), ha="center", fontsize=10)
    ax.set(
        xticks=range(len(groups)),
        xticklabels=[name for name, _ in groups],
        ylim=(0, maximum * 1.25 + 1),
        title=title,
        ylabel="Count",
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)


def figures(report, values, training):
    summary = values["summary"]
    count = summary[ARMS[0]]["tasks"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#253744",
            "svg.fonttype": "none",
            "svg.hashsalt": "resolvedesk-cuda-eval-002",
        }
    )
    specs = [
        ("All authored tasks", "task_success", "tasks"),
        ("Available-exchange tasks", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
        ("Stockout-fallback tasks", "fallback_success", "fallback_tasks"),
        ("Necessary-handoff tasks", "handoff_success", "handoff_tasks"),
        ("Missing-email tasks", "missing_email_success", "missing_email_tasks"),
    ]
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    for j, arm in enumerate(ARMS):
        nums = [summary[arm][key] for _, key, _ in specs]
        dens = [summary[arm][den] for _, _, den in specs]
        positions = np.arange(len(specs)) + (j - 0.5) * 0.34
        rates = [100 * n / d for n, d in zip(nums, dens, strict=True)]
        ax.barh(positions, rates, 0.32, color=COLORS[j], label=LABELS[arm], hatch="//" if j == 0 else "")
        for y, rate, n, d in zip(positions, rates, nums, dens, strict=True):
            ax.text(rate + 1, y, f"{n}/{d}", va="center", fontsize=11)
    ax.set(
        yticks=range(len(specs)),
        yticklabels=[name for name, _, _ in specs],
        xlim=(0, 113),
        xlabel="Correctly handled tasks (%)",
        title=f"Matched FP16 comparison on {count} new authored interactions",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.subplots_adjust(left=0.24, bottom=0.23)
    fig.text(
        0.04,
        0.025,
        "Same V7 workflow, decoded base and CUDA backend. Correct handling includes the fixture's "
        "accept/decline choice.\nSmall synthetic suite; retrospective V7 results are not pooled "
        "and no population reliability is claimed.",
        fontsize=10,
    )
    save(fig, "evaluation")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    ax = axes[0, 0]
    for j, arm in enumerate(ARMS):
        nums = [summary[arm]["intent_field_correct"][key] for key in FIELDS]
        positions = np.arange(len(FIELDS)) + (j - 0.5) * 0.34
        ax.barh(positions, np.array(nums) / count * 100, 0.32, color=COLORS[j], label=SHORT[j])
        for y, n in zip(positions, nums, strict=True):
            ax.text(n / count * 100 + 1, y, f"{n}/{count}", va="center", fontsize=9)
    ax.set(
        yticks=range(len(FIELDS)),
        yticklabels=[key.replace("_", " ") for key in FIELDS],
        xlim=(0, 118),
        xlabel="Correct raw first-output fields (%)",
        title="A. First output, before any recovery",
    )
    ax.invert_yaxis()
    for ax, key, den, title in [
        (axes[0, 1], "true_permission_yes", "authorized_tasks", "B. Explicit grants recognized"),
        (axes[1, 0], "unsupported_permission_yes", "non_authorized_tasks", "C. Unsupported first-output YES"),
    ]:
        nums = [summary[arm][key] for arm in ARMS]
        ax.bar(range(2), nums, color=COLORS, width=0.55)
        for j, arm in enumerate(ARMS):
            ax.text(j, nums[j] + 0.2, f"{nums[j]}/{summary[arm][den]}", ha="center")
        ax.set(
            xticks=range(2),
            xticklabels=SHORT,
            ylim=(0, max(summary[arm][den] for arm in ARMS) * 1.15 + 1),
            ylabel="Tasks",
            title=title,
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    count_groups(
        axes[1, 1],
        [
            ("Raw request\nparse failure", "initial_raw_parse_failures"),
            ("Any first\nenum error", "initial_enum_error_tasks"),
            ("First reason\nenum error", "initial_reason_enum_error_tasks"),
        ],
        summary,
        "D. Initial errors retained",
    )
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    fig.subplots_adjust(left=0.12, bottom=0.13, wspace=0.3, hspace=0.45)
    fig.text(
        0.04,
        0.025,
        "Exactly the first model output is scored, including rejected enums and malformed requests. "
        "Repairs cannot erase its error.\nReason errors are a subset of enum errors; partial malformed "
        "records can have both parse and enum errors. These counts are not additive.",
        fontsize=10,
    )
    save(fig, "intent")

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11))
    count_groups(
        axes[0, 0],
        [
            ("Repeated field\nattempts", "repeated_question_attempts"),
            ("Repeated questions\ndelivered", "repeated_questions_delivered"),
            ("All identical\naction repeats", "identical_action_repeats"),
        ],
        summary,
        "A. Question repeats and action loops",
    )
    count_groups(
        axes[0, 1],
        [
            ("Unneeded H\nattempts", "unnecessary_handoff_attempts"),
            ("Unneeded H\nexecuted", "unnecessary_handoff_executed"),
            ("Record calls\nrejected", "record_rejections"),
            ("Quote\nrejections", "quote_rejections"),
        ],
        summary,
        "B. Observed attempts and rejections",
    )
    for ax, key, scale, title, ylabel in [
        (axes[1, 0], "model_calls", 1, "C. Total calls per task", "Calls, including every rejected action"),
        (axes[1, 1], "latency_ms", 1000, "D. Serial task elapsed time", "Seconds"),
    ]:
        arrays = [[r[key] / scale for r in report["records"] if r["variant"] == arm] for arm in ARMS]
        boxes = ax.boxplot(arrays, tick_labels=SHORT, widths=0.5, patch_artist=True, showfliers=True)
        for box, color in zip(boxes["boxes"], COLORS, strict=True):
            box.set(facecolor=color, alpha=0.7)
        maximum = max(max(array) for array in arrays)
        for j, array in enumerate(arrays, 1):
            median = float(np.median(array))
            ax.text(j, maximum * 1.1 + 0.2, f"median {median:.1f}", ha="center")
        ax.set(ylim=(0, maximum * 1.3 + 1), title=title, ylabel=ylabel)
        if key == "model_calls":
            ax.set_ylim(0, max(13.5, maximum * 1.3 + 1))
            ax.axhline(12, color="#56616c", linestyle="--", linewidth=1)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.13, wspace=0.28, hspace=0.38)
    fig.text(
        0.04,
        0.025,
        "Identical actions include repeated records/reads, not just questions. Rejection categories "
        "may overlap.\nBoxes show observed task distributions, not confidence intervals. Twelve calls "
        "are the TOTAL task budget across customer turns; serial timings do not isolate speed effects.",
        fontsize=10,
    )
    save(fig, "recovery")

    families = list(values["family_success"])
    fig, ax = plt.subplots(figsize=(14, 13))
    for j, arm in enumerate(ARMS):
        nums = [values["family_success"][family][arm] for family in families]
        positions = np.arange(len(families)) + (j - 0.5) * 0.34
        ax.barh(positions, nums, 0.32, color=COLORS[j], label=SHORT[j], hatch="//" if j == 0 else "")
        for y, n in zip(positions, nums, strict=True):
            ax.text(n + 0.025, y, f"{n}/2", va="center", fontsize=9)
    ax.set(
        yticks=range(len(families)),
        yticklabels=[name.replace("_", " ") for name in families],
        xlim=(0, 2.28),
        xticks=[0, 1, 2],
        xlabel="Correct tasks / two per authored family",
        title="Every family retained; two cases per family",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.05))
    fig.subplots_adjust(left=0.38, bottom=0.1)
    pair = values["paired"]
    fig.text(
        0.04,
        0.015,
        f"Paired case outcomes: {pair['improved']} improved, {pair['regressed']} regressed, "
        f"{pair['both_pass']} both pass, {pair['both_fail']} both fail.\nTwo cases per family are "
        "descriptive coverage, not a reliable family-level population estimate.",
        fontsize=10,
    )
    save(fig, "families")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    protocol = values["protocol_summary"]
    specs = [
        ("Created\nproposals", "return_proposals_without_scoped_yes", "return_proposals"),
        ("Committed\nreturns", "committed_returns_without_scoped_yes", "committed_returns"),
        ("Final returned\norders", "final_returns_without_verified_commit", "final_return_orders"),
    ]
    maximum = 1
    for j, arm in enumerate(ARMS):
        nums = [protocol[arm][key] for _, key, _ in specs]
        dens = [protocol[arm][den] for _, _, den in specs]
        positions = np.arange(len(specs)) + (j - 0.5) * 0.34
        axes[0].bar(positions, nums, 0.32, color=COLORS[j], label=SHORT[j])
        maximum = max(maximum, *nums)
        for x, n, d in zip(positions, nums, dens, strict=True):
            axes[0].text(x, n + 0.15, f"{n}/{d}", ha="center")
    axes[0].set(
        xticks=range(len(specs)),
        xticklabels=[name for name, _, _ in specs],
        ylim=(0, maximum * 1.3 + 1),
        ylabel="Missing evidence / all observed events or orders",
        title="A. Exchange-origin execution audit",
    )
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    count_groups(
        axes[1],
        [
            ("Raw return calls\nwithout scoped YES", "return_attempts_without_scoped_yes"),
            ("Rejected such\nreturn calls", "rejected_return_attempts_without_scoped_yes"),
            ("Declined return\nproposals", "declined_return_proposals"),
        ],
        protocol,
        "B. Attempts and declined proposals remain separate",
    )
    axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    fig.subplots_adjust(bottom=0.3, wspace=0.27)
    fig.text(
        0.04,
        0.025,
        "Original-exchange cohort only. Proposals/commits require an actual matching scoped YES; "
        "final returned orders require a verified commit.\nThese units are never summed. Zero "
        "observations do not close the known operation-rewrite bypass or certify direct-return semantics.",
        fontsize=10,
    )
    save(fig, "consent")

    assert training["status"] == "completed" and training["completed_true_optimizer_updates"] == 1344
    entries = training["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for bucket, color in [("intent_and_record_repair", "#a9844c"), ("execution_and_reply", "#94a3ae")]:
        subset = [entry for entry in entries if entry["sampling_bucket"] == bucket]
        axes[0].scatter(
            [e["iteration"] for e in subset],
            [e["loss"] for e in subset],
            s=5,
            color=color,
            alpha=0.5,
            label="Record / repair" if bucket.startswith("intent") else "Execution / reply",
        )
    axes[0].set(
        yscale="symlog",
        xlabel="True optimizer updates",
        ylabel="Weighted completion NLL (symlog)",
        title="A. Actual sampled training losses",
    )
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9)
    seen, coverage = set(), []
    for entry in entries:
        seen.add(entry["sample_index"])
        coverage.append(len(seen))
    axes[1].plot(range(1, len(entries) + 1), coverage, color=COLORS[1])
    for step in [672, 1344]:
        axes[1].scatter([step], [coverage[step - 1]], color=COLORS[1])
        axes[1].annotate(
            f"{coverage[step - 1]}/824",
            (step, coverage[step - 1]),
            xytext=(-55, 14),
            textcoords="offset points",
        )
    axes[1].set(
        xlabel="True optimizer updates",
        ylabel="Unique physical training rows seen",
        ylim=(0, 900),
        title="B. Coverage of 824 deduplicated pairs",
    )
    axes[2].plot(
        [e["iteration"] for e in entries],
        [e["elapsed_gpu_sync_seconds"] for e in entries],
        linewidth=0.7,
        color=COLORS[1],
    )
    axes[2].set(
        yscale="log",
        xlabel="True optimizer updates",
        ylabel="Synchronized update seconds (log)",
        title="C. Full observed update-time trace",
    )
    fig.subplots_adjust(bottom=0.22, wspace=0.32)
    fig.text(
        0.04,
        0.025,
        "The final 1,344-update checkpoint was fixed before this test, without test-based selection. "
        "Sampled training loss is not held-out accuracy.\nFP16 LoRA on a decoded-4bit-derived base; "
        "no NF4/MLX bitwise or hardware-only speed claim. Timing includes the recorded audit overhead.",
        fontsize=10,
    )
    save(fig, "training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-report", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/report.json"
    )
    args = parser.parse_args()
    from training.cuda_eval2.fixtures import load_cases

    report_path, protocol_path = REPORTS / "evaluation.json", REPORTS / "protocol-audit.json"
    report, protocol = json.loads(report_path.read_text()), json.loads(protocol_path.read_text())
    plan, identity = (
        json.loads((REPORTS / name).read_text()) for name in ("analysis-plan.json", "artifact-identity.json")
    )
    require_complete_records(report, plan["case_ids"])
    verify_run_identity(report, plan, identity)
    verification = json.loads((REPORTS / "verification.json").read_text())
    assert verification["evaluation_sha256"] == sha(report_path) and verification[
        "protocol_audit_sha256"
    ] == sha(protocol_path)
    assert verification["run_id"] == protocol["run_id"] == RUN_ID
    assert verification["previous_interrupted_run"] == plan["previous_interrupted_run"]
    assert verification["analysis_plan_sha256"] == sha(REPORTS / "analysis-plan.json")
    assert verification["artifact_identity_sha256"] == sha(REPORTS / "artifact-identity.json")
    assert verification["source_sha256"] == sha(ROOT / "training/cuda_eval2/verify_results.py")
    assert verification["records_reconciled"] == verification["scoped_consent_rows_recomputed"] == 96
    assert verification["prompt_and_output_tokens_replayed"] is True
    cases = {case["id"]: case for case in load_cases("test")}
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == Counter(
        {(arm, cid): 1 for arm in ARMS for cid in cases}
    )
    training = json.loads(args.training_report.read_text())
    proof = json.loads((REPORTS / "training-proof.json").read_text())
    assert sha(args.training_report) == proof["report_sha256"]
    assert verification["training_proof_sha256"] == sha(REPORTS / "training-proof.json")
    values = derived_values(report, cases, protocol)
    FIGURES.mkdir(parents=True, exist_ok=True)
    columns = [
        "variant",
        "case_id",
        "family",
        "task_success",
        "model_calls",
        "latency_ms",
        "status_final",
        "intent_initial",
        "initial_raw_parse_failure",
        "initial_enum_errors",
        "enum_error_attempts",
        "enum_recovered",
        "reason_enum_errors",
        "reason_enum_recovered",
        "repeated_question_attempts",
        "repeated_questions_delivered",
        "identical_action_repeats",
        "unnecessary_handoff_attempts",
        "unnecessary_handoff_executed",
        "quote_rejections",
        "record_rejections",
        "slot_normalizations",
    ]
    with (FIGURES / "per-case.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(r[key], sort_keys=True)
                if key in {"intent_initial", "initial_enum_errors"}
                else r[key]
                for key in columns
            }
            for r in report["records"]
        )
    figures(report, values, training)
    values.update(
        run_id=RUN_ID,
        previous_interrupted_run=plan["previous_interrupted_run"],
        evaluation_sha256=sha(report_path),
        protocol_audit_sha256=sha(protocol_path),
        analysis_plan_sha256=sha(REPORTS / "analysis-plan.json"),
        verification_sha256=sha(REPORTS / "verification.json"),
        training_report_sha256=sha(args.training_report),
        training_proof_sha256=sha(REPORTS / "training-proof.json"),
        artifact_identity_sha256=sha(REPORTS / "artifact-identity.json"),
        analysis_source_sha256=sha(__file__),
        metrics_source_sha256=sha(ROOT / "training/cuda_eval2/metrics.py"),
        csv_sha256=sha(FIGURES / "per-case.csv"),
        figure_sha256={
            f"{name}.{extension}": sha(FIGURES / f"{name}.{extension}")
            for name in FIGURE_NAMES
            for extension in ("png", "svg")
        },
    )
    (FIGURES / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    print(
        json.dumps({"figures": 6, "records": len(report["records"]), "values": str(FIGURES / "values.json")})
    )


if __name__ == "__main__":
    main()
