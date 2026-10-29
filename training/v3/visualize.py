"""Export the new paired test, intent confusion and validation-business selection."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import ROOT, sha
from training.v2.visualize import paired, repeated_actions, save
from training.v3.common import REPORTS, verify_frozen
from training.v3.fixtures import FAMILIES

ARMS = ["base", "qlora_v2", "qlora_v3"]
LABELS = ["Base", "Previous QLoRA", "Action-weighted QLoRA"]
COLORS = ["#a7afad", "#a9844c", "#335c62"]


def main():
    verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    folder = ROOT / "docs/figures/v3"
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
            "svg.hashsalt": "resolvedesk-v3",
        }
    )
    summary = report["summary"]
    pairs = {
        "previous_to_new": paired(report["records"], "qlora_v2", "qlora_v3"),
        "base_to_new": paired(report["records"], "base", "qlora_v3"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    specs = [
        (axes[0, 0], "A. All business tasks", "task_success", "tasks"),
        (
            axes[0, 1],
            "B. Available exchanges fulfilled",
            "fulfillable_exchange_success",
            "fulfillable_exchange_tasks",
        ),
        (
            axes[1, 0],
            "C. Exchange incorrectly changed to return",
            "exchange_as_return",
            "fulfillable_exchange_tasks",
        ),
    ]
    for ax, title, numerator, denominator in specs:
        values = [summary[v][numerator] / summary[v][denominator] * 100 for v in ARMS]
        ax.bar(range(3), values, color=COLORS, width=0.6)
        for i, v in enumerate(ARMS):
            ax.text(
                i,
                values[i] + 2,
                f"{summary[v][numerator]}/{summary[v][denominator]}\n{values[i]:.1f}%",
                ha="center",
            )
        ax.set_xticks(range(3), ["Base", "QLoRA v2", "QLoRA v3"])
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel(
            "Cases (%)" if numerator == "exchange_as_return" else "Correct tasks + prerequisites (%)"
        )
        ax.set_title(title, loc="left", pad=15)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    ax = axes[1, 1]
    counts = pairs["previous_to_new"]
    values = [counts[k] for k in ("improved", "regressed", "both_pass", "both_fail")]
    ax.barh(range(4), values, color=[COLORS[2], COLORS[1], COLORS[0], "#d1d5cc"], height=0.6)
    for i, n in enumerate(values):
        ax.text(n + 0.2, i, str(n), va="center")
    ax.set_yticks(range(4), ["Improved", "Regressed", "Both pass", "Both fail"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 4)
    ax.set_xlabel("Paired tasks (N=48)")
    ax.set_title("D. Previous to new adapter, fixed workflow", loc="left", pad=15)
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("ResolveDesk | New 48-task test with the same controller", x=0.075, ha="left", fontsize=18)
    fig.subplots_adjust(left=0.085, right=0.96, top=0.9, bottom=0.15, hspace=0.48, wspace=0.4)
    fig.text(
        0.075,
        0.065,
        "Synthetic paired worlds; one greedy trial per task; not production data.",
        fontsize=9,
    )
    fig.text(
        0.075,
        0.035,
        "Panel C: return proposal on an available-exchange request. Lower is better.",
        fontsize=9,
    )
    save(fig, folder, "evaluation")
    family_success = {
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
    }
    fig, ax = plt.subplots(figsize=(13, 11))
    y = np.arange(len(FAMILIES))
    for j, v in enumerate(ARMS):
        values = [family_success[f][v]["passed"] for f in FAMILIES]
        ax.barh(
            y + (j - 1) * 0.25,
            values,
            0.24,
            color=COLORS[j],
            label=LABELS[j],
            hatch=["", "//", "xx"][j],
            edgecolor="#40545a",
            linewidth=0.5,
        )
        for yi, n in zip(y + (j - 1) * 0.25, values, strict=True):
            ax.text(n + 0.04, yi, str(n), va="center", fontsize=9)
    ax.set_yticks(y, [f.replace("_", " ") for f in FAMILIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 3.5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("Correct tasks / 3 per family")
    ax.set_title("All 16 scenario families | identical cases in every arm", loc="left", pad=50)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=3)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.29, right=0.95, top=0.87, bottom=0.09)
    fig.text(
        0.08,
        0.025,
        "Counterfactual pairs vary stock or customer intent in the same world.",
        fontsize=9,
    )
    save(fig, folder, "families")
    expected = ["cancel", "return", "exchange", "handoff"]
    predicted = [*expected, "unfinished"]
    confusion = {
        v: [
            [
                sum(
                    r["variant"] == v and r["expected"]["kind"] == e and r["predicted_outcome"] == p
                    for r in report["records"]
                )
                for p in predicted
            ]
            for e in expected
        ]
        for v in ARMS
    }
    maximum = max(n for matrix in confusion.values() for row in matrix for n in row)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for j, (ax, v) in enumerate(zip(axes, ARMS, strict=True)):
        matrix = np.array(confusion[v])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, maximum))
        for (i, k), n in np.ndenumerate(matrix):
            ax.text(
                k, i, str(n), ha="center", va="center", color="white" if n > maximum * 0.55 else "#263b41"
            )
        ax.set_xticks(
            range(5), ["cancel", "return", "exchange", "handoff", "unfinished"], rotation=35, ha="right"
        )
        ax.set_yticks(range(4), expected)
        ax.set_title(LABELS[j], pad=12)
        ax.set_xlabel("Final proposal / handoff / unfinished")
        if j == 0:
            ax.set_ylabel("Expected business outcome")
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.28, top=0.82, wspace=0.36)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.025, label="Tasks (common count scale)")
    fig.suptitle("Operation confusion | all 48 cases in each arm", x=0.07, ha="left", fontsize=17)
    fig.text(
        0.07,
        0.045,
        "Final proposed operation, not argument correctness; unfinished has no accepted proposal.",
        fontsize=10,
    )
    save(fig, folder, "confusion")
    losses = [r for r in training["metrics"] if r["kind"] == "weighted_valid_loss"]
    candidates = training["validation_candidates"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.plot([r["iteration"] for r in losses], [r["val_loss"] for r in losses], color=COLORS[2], marker="o")
    ax.set(
        xlabel="Completed optimizer updates",
        ylabel="Tool-weighted validation loss",
        ylim=(0, None),
        title="A. Completion objective (tool-name weight ×8)",
    )
    ax.set_xticks([r["iteration"] for r in losses])
    ax.grid(alpha=0.15)
    ax = axes[1]
    x = np.arange(len(candidates))
    for j, (key, den, label) in enumerate(
        [
            ("fulfillable_exchange_success", "fulfillable_exchange_tasks", "Available exchanges"),
            ("task_success", "tasks", "All validation tasks"),
        ]
    ):
        nums = [c["summary"][key] for c in candidates]
        dens = [c["summary"][den] for c in candidates]
        rates = [n / d * 100 for n, d in zip(nums, dens, strict=True)]
        ax.bar(x + (j - 0.5) * 0.32, rates, 0.31, label=label, color=COLORS[2 - j])
        for xi, n, d, rate in zip(x + (j - 0.5) * 0.32, nums, dens, rates, strict=True):
            ax.text(xi, rate + 2, f"{n}/{d}", ha="center", fontsize=10)
    selected = training["best_checkpoint"]["iteration"]
    ax.set_xticks(
        x,
        [
            f"{c['completed_updates']}" + ("\nSELECTED" if c["completed_updates"] == selected else "")
            for c in candidates
        ],
    )
    ax.set(
        xlabel="Reloaded checkpoint (completed updates)",
        ylabel="Validation task success (%)",
        ylim=(0, 120),
        title="B. Selection prioritizes exchange task completion",
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.85, bottom=0.25, wspace=0.35)
    fig.text(
        0.08,
        0.055,
        "16 validation tasks select weights; tests follow. Update 240 is not a candidate.",
        fontsize=9,
    )
    save(fig, folder, "training")
    columns = [
        "variant",
        "case_id",
        "family",
        "task_success",
        "predicted_outcome",
        "exchange_as_return",
        "unauthorized_return",
        "model_calls",
        "schema_valid",
        "rejected_write_attempts",
        "duplicate_read_executions",
        "latency_ms",
        "status_final",
    ]
    with (folder / "per-case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in report["records"]:
            writer.writerow({k: row[k] for k in columns})
    index = {(r["variant"], r["case_id"]): r for r in report["records"]}
    failures = [
        {
            "case_id": i,
            "base": index["base", i]["task_success"],
            "previous": index["qlora_v2", i]["task_success"],
            "new": index["qlora_v3", i]["task_success"],
        }
        for i in sorted({r["case_id"] for r in report["records"]})
        if not all(index[v, i]["task_success"] for v in ARMS)
    ]
    values = {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "training_sha256": sha(REPORTS / "training.json"),
        "analysis_source_sha256": sha(__file__),
        "shared_visualization_sha256": sha(ROOT / "training/v2/visualize.py"),
        "summary": summary,
        "paired": pairs,
        "family_success": family_success,
        "confusion": confusion,
        "failures": failures,
        "identical_action_repeats": {
            v: sum(repeated_actions(r) for r in report["records"] if r["variant"] == v) for v in ARMS
        },
        "csv_sha256": sha(folder / "per-case.csv"),
        "figure_sha256": {p.name: sha(p) for p in folder.iterdir() if p.suffix in {".png", ".svg"}},
    }
    (folder / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    failure_rows = []
    for r in report["records"]:
        if r["task_success"]:
            continue
        reasons = []
        if r["exchange_as_return"]:
            reasons.append("exchange_misclassified_as_return")
        if not r["clarification_obtained"]:
            reasons.append("required_clarification_missing")
        if not r["catalog_inspected"]:
            reasons.append("catalog_not_inspected")
        if not r["raw_task_success"]:
            reasons.append("wrong_or_uncompleted_business_outcome")
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
                "records": failure_rows,
                "scope": "All failed runs retained; overlapping diagnostics, no additional model calls.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"paired": pairs, "figures": list(values["figure_sha256"])}))


if __name__ == "__main__":
    main()
