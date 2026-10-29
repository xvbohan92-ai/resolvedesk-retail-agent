"""Source-backed static figures for the four-arm controlled experiment."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import ROOT, sha
from training.v2.visualize import paired, save
from training.v4.common import REPORTS, verify_frozen
from training.v4.evaluation import ARMS
from training.v4.fixtures import FAMILIES

LABELS = ["V3 + old flow", "V3 + new flow", "Base + new flow", "V4 + new flow"]
SHORT = ["V3\nold flow", "V3\nnew flow", "Base\nnew flow", "V4\nnew flow"]
COLORS = ["#a7afad", "#718992", "#a9844c", "#335c62"]
HATCHES = ["", "//", "..", "xx"]


def main():
    verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    folder = ROOT / "docs/figures/v4"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#263b41",
            "svg.fonttype": "none",
            "svg.hashsalt": "resolvedesk-v4",
        }
    )
    summary, records = report["summary"], report["records"]
    pairs = {
        "workflow": paired(records, "previous_old_flow", "previous_new_flow"),
        "training": paired(records, "previous_new_flow", "qlora_v4"),
        "base": paired(records, "base_new_flow", "qlora_v4"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    specs = [
        (axes[0, 0], "A. Complete business tasks", "task_success", "tasks"),
        (
            axes[1, 1],
            "D. Available exchange changed to return",
            "exchange_as_return",
            "fulfillable_exchange_tasks",
        ),
    ]
    for ax, title, key, den in specs:
        rates = [summary[v][key] / summary[v][den] * 100 for v in ARMS]
        ax.bar(range(4), rates, color=COLORS, width=0.6)
        for i, v in enumerate(ARMS):
            ax.text(
                i,
                rates[i] + 2,
                f"{summary[v][key]}/{summary[v][den]}\n{rates[i]:.1f}%",
                ha="center",
                fontsize=10,
            )
        ax.set_xticks(range(4), SHORT)
        ax.set_ylim(0, 120)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, loc="left", pad=15)
        ax.set_ylabel("Tasks (%)" if key == "task_success" else "Eligible tasks (%), lower is better")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    ax = axes[0, 1]
    counts = [summary[v]["repeated_question_attempts"] for v in ARMS]
    ax.bar(range(4), counts, color=COLORS, width=0.6)
    for i, n in enumerate(counts):
        ax.text(i, n + max(counts + [1]) * 0.025, str(n), ha="center")
    ax.set_xticks(range(4), SHORT)
    ax.set_ylim(0, max(counts + [1]) * 1.2 + 1)
    ax.set_title("B. Repeated model question-field attempts", loc="left", pad=15)
    ax.set_ylabel("Attempts across 32 tasks, lower is better")
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax = axes[1, 0]
    x = np.arange(2)
    for j, v in enumerate(ARMS):
        nums = [summary[v]["fallback_success"], summary[v]["handoff_success"]]
        dens = [summary[v]["fallback_tasks"], summary[v]["handoff_tasks"]]
        rates = [n / d * 100 for n, d in zip(nums, dens, strict=True)]
        positions = x + (j - 1.5) * 0.19
        ax.bar(
            positions,
            rates,
            0.18,
            color=COLORS[j],
            label=LABELS[j],
            hatch=HATCHES[j],
            edgecolor="#40545a",
            linewidth=0.5,
        )
        for position, n, d, rate in zip(positions, nums, dens, rates, strict=True):
            ax.text(position, rate + 2, f"{n}/{d}", ha="center", fontsize=9)
    ax.set_xticks(x, ["Authorized stockout fallback", "Required model handoff"])
    ax.set_ylim(0, 120)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Correct tasks + prerequisites (%)")
    ax.set_title("C. Boundary handling", loc="left", pad=15)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.94),
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "ResolveDesk | Fresh 32-task test: workflow and training effects", x=0.08, ha="left", fontsize=18
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.84, bottom=0.15, wspace=0.38, hspace=0.48)
    fig.text(
        0.08,
        0.065,
        "Synthetic correlated worlds; one greedy trial/task; same eight-total-call budget.",
        fontsize=10,
    )
    fig.text(
        0.08,
        0.03,
        "All model question attempts remain counted, including exact duplicates blocked from delivery.",
        fontsize=10,
    )
    save(fig, folder, "evaluation")
    families = {
        f: {
            v: {
                "passed": sum(r["task_success"] for r in records if r["variant"] == v and r["family"] == f),
                "tasks": sum(r["variant"] == v and r["family"] == f for r in records),
            }
            for v in ARMS
        }
        for f in FAMILIES
    }
    fig, ax = plt.subplots(figsize=(14, 11))
    y = np.arange(len(FAMILIES))
    for j, v in enumerate(ARMS):
        nums = [families[f][v]["passed"] for f in FAMILIES]
        positions = y + (j - 1.5) * 0.2
        ax.barh(
            positions,
            nums,
            0.19,
            color=COLORS[j],
            hatch=HATCHES[j],
            edgecolor="#40545a",
            linewidth=0.5,
            label=LABELS[j],
        )
        for position, n in zip(positions, nums, strict=True):
            ax.text(n + 0.025, position, str(n), va="center", fontsize=9)
    ax.set_yticks(y, [f.replace("_", " ") for f in FAMILIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 2.35)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Correct tasks / 2 per family")
    ax.set_title("Every scenario family | same new tasks in all four arms", loc="left", pad=52)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=4, fontsize=9)
    ax.grid(axis="x", alpha=0.15)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.29, right=0.95, top=0.87, bottom=0.09)
    fig.text(
        0.08,
        0.025,
        "Counterfactual pairs share worlds within the split. Old experiments use different tasks.",
        fontsize=10,
    )
    save(fig, folder, "families")
    expected = ["cancel", "return", "exchange", "handoff"]
    predicted = [*expected, "unfinished"]
    confusion = {
        v: [
            [
                sum(
                    r["variant"] == v and r["expected"]["kind"] == e and r["predicted_outcome"] == p
                    for r in records
                )
                for p in predicted
            ]
            for e in expected
        ]
        for v in ARMS
    }
    maximum = max(n for matrix in confusion.values() for row in matrix for n in row)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))
    for j, (ax, v) in enumerate(zip(axes, ARMS, strict=True)):
        matrix = np.array(confusion[v])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(maximum, 1))
        for (i, k), n in np.ndenumerate(matrix):
            ax.text(
                k, i, str(n), ha="center", va="center", color="white" if n > maximum * 0.55 else "#263b41"
            )
        ax.set_xticks(range(5), predicted, rotation=35, ha="right")
        ax.set_yticks(range(4), expected)
        ax.set_title(LABELS[j])
        ax.set_xlabel("Final proposed operation / status")
        if j == 0:
            ax.set_ylabel("Expected outcome")
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.29, top=0.8, wspace=0.4)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.025, label="Tasks (common scale)")
    fig.suptitle("Operation confusion | all 32 tasks in each arm", x=0.06, ha="left", fontsize=18)
    fig.text(
        0.06,
        0.04,
        "Operation labels omit argument/clarification correctness; every zero cell is shown.",
        fontsize=11,
    )
    save(fig, folder, "confusion")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    ax = axes[0]
    ax.plot(
        [m["iteration"] for m in training["metrics"]],
        [m["train_loss"] for m in training["metrics"]],
        color=COLORS[-1],
        marker="o",
    )
    ax.set(
        xlabel="Completed optimizer updates",
        ylabel="Tool-weighted training loss",
        ylim=(0, None),
        title="A. SFT objective (tool-name weight ×2)",
    )
    ax.grid(alpha=0.15)
    ax = axes[1]
    pref = training["calibration"]["metrics"]
    ax.plot(
        [m["iteration"] for m in pref],
        [m["loss"] for m in pref],
        color=COLORS[2],
        alpha=0.7,
        linewidth=1,
        label="Joint objective",
    )
    ax.set(
        xlabel="Preference pair updates",
        ylabel="Joint objective (nats)",
        title="B. Calibration trial: " + ("adopted" if training["calibration_adopted"] else "rejected"),
        ylim=(0, None),
    )
    ax.grid(alpha=0.15)
    ax.plot(
        [m["iteration"] for m in pref],
        [m["chosen_completion_nll"] for m in pref],
        color=COLORS[1],
        linestyle="--",
        linewidth=1,
        label="Chosen-completion NLL",
    )
    ax.legend(frameon=False, fontsize=9)
    ax = axes[2]
    candidates = training["validation_candidates"]
    x = np.arange(len(candidates))
    for j, (key, den, label) in enumerate(
        [
            ("fulfillable_exchange_success", "fulfillable_exchange_tasks", "Exchange"),
            ("fallback_success", "fallback_tasks", "Fallback"),
            ("handoff_success", "handoff_tasks", "Handoff"),
        ]
    ):
        nums = [c["summary"][key] for c in candidates]
        dens = [c["summary"][den] for c in candidates]
        rates = [n / d * 100 for n, d in zip(nums, dens, strict=True)]
        positions = x + (j - 1) * 0.24
        ax.bar(positions, rates, 0.23, color=COLORS[j + 1], label=label)
        for position, n, d, rate in zip(positions, nums, dens, rates, strict=True):
            ax.text(position, rate + 2, f"{n}/{d}", ha="center", fontsize=10)
    ax.set_xticks(
        x,
        [
            {
                240: "240 SFT",
                604: "SFT480\n+seq124",
                616: "SFT480\n+action136",
                696: "Above +80\nrecovery",
                744: "Above +48\ncalibration",
            }[c["completed_updates"]]
            + ("\nSELECTED" if c["completed_updates"] == training["best_checkpoint"]["iteration"] else "")
            for c in candidates
        ],
    )
    ax.set(
        xlabel="Reloaded stage checkpoint",
        ylabel="Validation success (%)",
        ylim=(0, 120),
        title="C. Validation before held-out test",
    )
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=3)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.3, wspace=0.35)
    fig.text(
        0.08,
        0.04,
        "Pre-test revisions add facts and recovery; calibration is selected only with validation guards. "
        "Fresh test separates flow/training.",
        fontsize=10,
    )
    save(fig, folder, "training")
    columns = [
        "variant",
        "case_id",
        "family",
        "expected_outcome",
        "task_success",
        "predicted_outcome",
        "exchange_as_return",
        "repeated_question_attempts",
        "repeated_questions_delivered",
        "model_handoff",
        "model_calls",
        "schema_valid",
        "rejected_write_attempts",
        "latency_ms",
        "status_final",
    ]
    with (folder / "per-case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    **{k: row[k] for k in columns if k != "expected_outcome"},
                    "expected_outcome": row["expected"]["kind"],
                }
            )
    index = {(r["variant"], r["case_id"]): r for r in records}
    failures = [
        {"case_id": i, **{v: index[v, i]["task_success"] for v in ARMS}}
        for i in sorted({r["case_id"] for r in records})
        if not all(index[v, i]["task_success"] for v in ARMS)
    ]
    handoff_diagnostics = {}
    for v in ARMS:
        arm = [r for r in records if r["variant"] == v]
        attempted = [r for r in arm if r["model_handoff"]]
        handoff_diagnostics[v] = {
            "model_handoffs": len(attempted),
            "verified_handoffs": sum(
                r["task_success"] and r["expected"]["kind"] == "handoff" for r in attempted
            ),
            "unnecessary_handoffs": sum(r["expected"]["kind"] != "handoff" for r in attempted),
            "non_handoff_tasks": sum(r["expected"]["kind"] != "handoff" for r in arm),
            "unverified_policy_handoffs": sum(
                r["expected"]["kind"] == "handoff" and not r["task_success"] for r in attempted
            ),
        }
    values = {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "training_sha256": sha(REPORTS / "training.json"),
        "analysis_source_sha256": sha(__file__),
        "shared_visualization_sha256": sha(ROOT / "training/v2/visualize.py"),
        "summary": summary,
        "paired": pairs,
        "family_success": families,
        "confusion": confusion,
        "handoff_diagnostics": handoff_diagnostics,
        "failures": failures,
        "identical_action_repeats": {v: summary[v]["identical_action_repeats"] for v in ARMS},
        "csv_sha256": sha(folder / "per-case.csv"),
        "figure_sha256": {p.name: sha(p) for p in folder.iterdir() if p.suffix in {".png", ".svg"}},
    }
    (folder / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    failed = [
        {
            "variant": r["variant"],
            "case_id": r["case_id"],
            "repeated_question_attempts": r["repeated_question_attempts"],
            "missing_required_reply": not r["clarification_obtained"],
            "exchange_as_return": r["exchange_as_return"],
            "last_decision": r["trace"][-1] if r["trace"] else None,
        }
        for r in records
        if not r["task_success"]
    ]
    (REPORTS / "failure-analysis.json").write_text(
        json.dumps(
            {
                "evaluation_sha256": sha(REPORTS / "evaluation.json"),
                "records": failed,
                "scope": "All failed runs retained; overlapping raw-action diagnostics, no extra inference.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"paired": pairs, "figures": list(values["figure_sha256"])}))


if __name__ == "__main__":
    main()
