"""Reproduce export figures and per-case values directly from the three-arm run."""

import csv
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from training.common import ROOT, sha
from training.v2.common import REPORTS, verify_frozen
from training.v2.fixtures import FAMILIES

ARMS = ["legacy_base", "workflow_base", "workflow_qlora_v2"]
LABELS = ["Legacy\n+ base", "V2 workflow\n+ base", "V2 workflow\n+ QLoRA"]
COLORS = ["#a7afad", "#a9844c", "#335c62"]


def repeated_actions(row):
    keys = [json.dumps(o["action"], sort_keys=True) for o in row["trace"] if o["action"]]
    return len(keys) - len(set(keys))


def paired(records, left, right):
    a = {r["case_id"]: r["task_success"] for r in records if r["variant"] == left}
    b = {r["case_id"]: r["task_success"] for r in records if r["variant"] == right}
    assert a.keys() == b.keys()
    counts = Counter((a[i], b[i]) for i in a)
    return {
        "improved": counts[False, True],
        "regressed": counts[True, False],
        "both_pass": counts[True, True],
        "both_fail": counts[False, False],
    }


def save(fig, folder, name):
    for ext in ("png", "svg"):
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=160, facecolor="white", metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    folder = ROOT / "docs/figures/v2"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#344950",
            "text.color": "#263b41",
            "svg.fonttype": "none",
            "svg.hashsalt": "resolvedesk-v2",
        }
    )
    summary = report["summary"]
    pairs = {
        "engineering": paired(report["records"], "legacy_base", "workflow_base"),
        "training": paired(report["records"], "workflow_base", "workflow_qlora_v2"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, title, numerator, denominator in [
        (axes[0, 0], "A. All business tasks", "task_success", "tasks"),
        (axes[0, 1], "B. Exchange-related scenarios", "exchange_related_success", "exchange_related_tasks"),
    ]:
        vals = [summary[v][numerator] / summary[v][denominator] * 100 for v in ARMS]
        ax.bar(range(3), vals, color=COLORS, width=0.6)
        for i, v in enumerate(ARMS):
            ax.text(
                i,
                vals[i] + 2,
                f"{summary[v][numerator]}/{summary[v][denominator]}\n{vals[i]:.1f}%",
                ha="center",
                fontsize=11,
            )
        ax.set_xticks(range(3), LABELS)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Correct final state + prerequisites (%)")
        ax.set_title(title, loc="left", pad=15)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    ax = axes[1, 0]
    vals = [summary[v]["duplicate_read_executions"] for v in ARMS]
    ax.bar(range(3), vals, color=COLORS, width=0.6)
    for i, value in enumerate(vals):
        ax.text(i, value + max(0.1, max(vals) * 0.025), str(value), ha="center")
    ax.set_xticks(range(3), LABELS)
    ax.set_ylim(0, max(1.5, max(vals) * 1.2))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Repeated successful backend reads (count)")
    ax.set_title("C. Duplicate execution guard", loc="left", pad=15)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax = axes[1, 1]
    counts = pairs["training"]
    vals = [counts[k] for k in ("improved", "regressed", "both_pass", "both_fail")]
    ax.barh(range(4), vals, color=[COLORS[2], COLORS[1], COLORS[0], "#d1d5cc"], height=0.6)
    for i, value in enumerate(vals):
        ax.text(value + 0.3, i, str(value), va="center")
    ax.set_yticks(range(4), ["Improved", "Regressed", "Both pass", "Both fail"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) + 4)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(f"Paired tasks (N={summary[ARMS[0]]['tasks']})")
    ax.set_title("D. Training effect within V2 workflow", loc="left", pad=15)
    fig.suptitle("ResolveDesk | Round 2 engineering and QLoRA evaluation", x=0.07, ha="left", fontsize=18)
    fig.subplots_adjust(left=0.085, right=0.96, top=0.9, bottom=0.16, hspace=0.58, wspace=0.42)
    fig.text(
        0.07,
        0.07,
        "42 new synthetic English tasks; 14 shared families. One greedy trial per task.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.04,
        "V2 guards are code, not learned compliance. Fixed V2 workflow for training comparison.",
        fontsize=9,
    )
    save(fig, folder, "evaluation")
    fig, ax = plt.subplots(figsize=(13, 10))
    y = np.arange(len(FAMILIES))
    for j, arm in enumerate(ARMS):
        counts = [
            sum(r["task_success"] for r in report["records"] if r["variant"] == arm and r["family"] == f)
            for f in FAMILIES
        ]
        ax.barh(y + (j - 1) * 0.25, counts, 0.24, color=COLORS[j], label=LABELS[j].replace("\n", " "))
        for yi, value in zip(y + (j - 1) * 0.25, counts, strict=True):
            ax.text(value + 0.04, yi, str(value), va="center", fontsize=9)
    ax.set_yticks(y, [f.replace("_", " ") for f in FAMILIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 3.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Tasks meeting final state + prerequisites / 3 per family")
    ax.set_title("All scenario families | same new test cases in each arm", loc="left", pad=50)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.065), ncol=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.15)
    fig.subplots_adjust(left=0.26, right=0.95, top=0.86, bottom=0.1)
    fig.text(
        0.08,
        0.03,
        "Exchange-related includes fallback and handoff; different test from v1.",
        fontsize=9,
    )
    save(fig, folder, "families")
    records = [r for r in training["metrics"] if r["kind"] == "valid"]
    selected = training["best_checkpoint"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot([r["iteration"] for r in records], [r["val_loss"] for r in records], color=COLORS[2], marker="o")
    ax.scatter(
        [selected["iteration"]],
        [selected["val_loss"]],
        color=COLORS[1],
        marker="*",
        s=160,
        zorder=5,
        label=f"Selected: {selected['iteration']} completed updates",
    )
    ax.set(
        xlabel="Completed optimizer updates (MLX callback iteration)",
        ylabel="Full validation completion loss",
        title="Round 2 | checkpoint selected using validation only",
        ylim=(0, None),
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.16)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.86, bottom=0.23)
    fig.text(
        0.1,
        0.07,
        "4-bit QLoRA · rank 8 · last 8 layers · batch 2 · 280 updates · no test output used for selection.",
        fontsize=9,
    )
    save(fig, folder, "training")
    columns = [
        "variant",
        "case_id",
        "family",
        "task_success",
        "proposal_correct",
        "model_calls",
        "schema_valid",
        "duplicate_read_attempts",
        "duplicate_read_executions",
        "rejected_write_attempts",
        "latency_ms",
        "output_tokens",
        "status_final",
        "clarification_obtained",
        "catalog_inspected",
        "identical_action_repeats",
    ]
    with (folder / "per-case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in report["records"]:
            writer.writerow(
                {k: repeated_actions(row) if k == "identical_action_repeats" else row[k] for k in columns}
            )
    index = {(r["variant"], r["case_id"]): r for r in report["records"]}
    failures = [
        {
            "case_id": i,
            "base": index["workflow_base", i]["task_success"],
            "finetuned": index["workflow_qlora_v2", i]["task_success"],
        }
        for i in sorted({r["case_id"] for r in report["records"]})
        if not (index["workflow_base", i]["task_success"] and index["workflow_qlora_v2", i]["task_success"])
    ]
    values = {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "training_sha256": sha(REPORTS / "training.json"),
        "summary": summary,
        "paired": pairs,
        "failures": failures,
        "identical_action_repeats": {
            v: sum(repeated_actions(r) for r in report["records"] if r["variant"] == v) for v in ARMS
        },
        "family_success": {
            f: {
                v: {
                    "passed": sum(
                        r["task_success"] for r in report["records"] if r["variant"] == v and r["family"] == f
                    ),
                    "tasks": sum(r["variant"] == v and r["family"] == f for r in report["records"]),
                }
                for v in ARMS
            }
            for f in FAMILIES
        },
        "analysis_source_sha256": sha(__file__),
        "csv_sha256": sha(folder / "per-case.csv"),
        "figure_sha256": {p.name: sha(p) for p in folder.iterdir() if p.suffix in {".png", ".svg"}},
    }
    (folder / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    failure_rows = []
    for r in report["records"]:
        if r["task_success"]:
            continue
        reasons = []
        if not r["clarification_obtained"]:
            reasons.append("required_clarification_missing")
        if not r["identity_and_order_checked"]:
            reasons.append("identity_or_order_not_inspected")
        if not r["catalog_inspected"]:
            reasons.append("exchange_catalog_not_inspected")
        if not r["raw_task_success"]:
            reasons.append("wrong_or_uncompleted_outcome")
        if r["schema_valid"] != r["model_calls"]:
            reasons.append("invalid_action_json")
        if r["status_final"] == "running":
            reasons.append("tool_budget_exhausted")
        if any(
            o["result"].get("handoff") and o["result"].get("error") == "duplicate_read_blocked"
            for o in r["trace"]
        ):
            reasons.append("repetition_handoff")
        failure_rows.append(
            {
                "variant": r["variant"],
                "case_id": r["case_id"],
                "reasons": reasons,
                "last_decision": r["trace"][-1] if r["trace"] else None,
            }
        )
    (REPORTS / "failure-analysis.json").write_text(
        json.dumps(
            {
                "evaluation_sha256": sha(REPORTS / "evaluation.json"),
                "cases": failure_rows,
                "note": "Overlapping categories; all failures retained. New tuning requires a new test.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"paired": pairs, "figures": list(values["figure_sha256"])}))


if __name__ == "__main__":
    main()
