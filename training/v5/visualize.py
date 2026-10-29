"""Quantitative figures backed by the frozen four-arm raw trajectories."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import ROOT, sha
from training.v5.common import DATA, REPORTS, verify_frozen
from training.v5.evaluation import ARMS
from training.v5.fixtures import FAMILIES, load_cases

LABELS = ["V4 · old flow", "V4 · typed flow", "Base · typed flow", "V5 · typed flow"]
SHORT = ["V4\nold flow", "V4\ntyped flow", "Base\ntyped flow", "V5\ntyped flow"]
COLORS = ["#a7afad", "#718992", "#a9844c", "#335c62"]
HATCH = ["", "//", "..", "xx"]


def paired(records, first, second):
    index = {(r["variant"], r["case_id"]): r["task_success"] for r in records}
    counts = {"improved": 0, "regressed": 0, "both_pass": 0, "both_fail": 0}
    for cid in {r["case_id"] for r in records}:
        a, b = index[first, cid], index[second, cid]
        counts[
            "both_pass" if a and b else "both_fail" if not a and not b else "improved" if b else "regressed"
        ] += 1
    return counts


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main():
    verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    records, s = report["records"], report["summary"]
    cases = {c["id"]: c for c in load_cases("test")}
    permissions = {}
    for variant in ARMS[1:]:
        subset = [r for r in records if r["variant"] == variant]

        def expected(r):
            return cases[r["case_id"]]["intent_initial"]["fallback_return"] == "yes"

        def predicted(r):
            return bool(r["intent_initial"] and r["intent_initial"]["fallback_return"] == "yes")

        permissions[variant] = {
            "authorized_tasks": sum(expected(r) for r in subset),
            "non_authorized_tasks": sum(not expected(r) for r in subset),
            "true_affirmative": sum(expected(r) and predicted(r) for r in subset),
            "unsupported_affirmative": sum(not expected(r) and predicted(r) for r in subset),
            "missed_affirmative": sum(expected(r) and not predicted(r) for r in subset),
        }
    folder = ROOT / "docs/figures/v5"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#263b41",
            "svg.fonttype": "none",
            "svg.hashsalt": "resolvedesk-v5",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    ax = axes[0, 0]
    for i, v in enumerate(ARMS):
        n = s[v]["task_success"]
        ax.bar(i, n / 32 * 100, color=COLORS[i], width=0.6)
        ax.text(i, n / 32 * 100 + 2, f"{n}/32", ha="center")
    ax.set(
        xticks=range(4),
        xticklabels=SHORT,
        ylim=(0, 115),
        ylabel="Correct tasks (%)",
        title="A. Full business outcomes",
    )
    groups = [
        (
            axes[0, 1],
            "B. Core recovery outcomes",
            [
                ("Available\nexchange", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
                ("Authorized\nstockout return", "fallback_success", "fallback_tasks"),
                ("Required\nhandoff", "handoff_success", "handoff_tasks"),
            ],
            True,
        ),
        (
            axes[1, 0],
            "C. Unnecessary handoff",
            [
                ("Model attempts", "unnecessary_handoff_attempts", None),
                ("Actually executed", "unnecessary_handoff_executed", None),
            ],
            False,
        ),
        (
            axes[1, 1],
            "D. Raw safety-related attempts",
            [
                ("Repeated question\nfield attempts", "repeated_question_attempts", None),
                ("Available exchange\nchanged to return", "exchange_as_return", None),
                ("Unauthorized\nreturn attempts", "unauthorized_return", None),
            ],
            False,
        ),
    ]
    for ax, title, specs, percent in groups:
        maximum = 1
        for j, v in enumerate(ARMS):
            x = np.arange(len(specs)) + (j - 1.5) * 0.19
            nums = [s[v][k] for _, k, _ in specs]
            dens = [s[v][d] if d else 1 for _, _, d in specs]
            vals = [n / d * 100 if percent else n for n, d in zip(nums, dens, strict=True)]
            ax.bar(x, vals, 0.18, color=COLORS[j], hatch=HATCH[j], edgecolor="#40545a", linewidth=0.4)
            maximum = max(maximum, *vals)
            for position, value, n, d in zip(x, vals, nums, dens, strict=True):
                ax.text(
                    position,
                    value + (2 if percent else 0.3),
                    f"{n}/{d}" if percent else str(n),
                    ha="center",
                    fontsize=9,
                )
        ax.set(
            xticks=range(len(specs)),
            xticklabels=[g[0] for g in specs],
            ylabel="Correct tasks (%)" if percent else "Count, lower is better",
            ylim=(0, 118 if percent else maximum * 1.22 + 1),
            title=title,
        )
        if not percent:
            from matplotlib.ticker import MaxNLocator

            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.14)
        ax.set_axisbelow(True)
    fig.suptitle(
        "ResolveDesk | Fresh 32-task test: typed intent and training", x=0.07, ha="left", fontsize=18
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i], label=LABELS[i]) for i in range(4)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 0.94), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.14, hspace=0.4, wspace=0.3)
    fig.text(
        0.07,
        0.035,
        "Self-authored synthetic cases; 12 total calls including intent; one greedy trial/task.",
        fontsize=10,
    )
    save(fig, folder, "evaluation")

    families = {
        f: {
            v: sum(r["task_success"] for r in records if r["family"] == f and r["variant"] == v) for v in ARMS
        }
        for f in FAMILIES
    }
    fig, ax = plt.subplots(figsize=(14, 11))
    y = np.arange(len(FAMILIES))
    for j, v in enumerate(ARMS):
        vals = [families[f][v] for f in FAMILIES]
        positions = y + (j - 1.5) * 0.17
        ax.barh(
            positions,
            vals,
            0.16,
            color=COLORS[j],
            hatch=HATCH[j],
            label=LABELS[j],
            edgecolor="#40545a",
            linewidth=0.3,
        )
        for pos, n in zip(positions, vals, strict=True):
            ax.text(n + 0.025, pos, str(n), va="center", fontsize=9)
    ax.set(
        yticks=y,
        yticklabels=[f.replace("_", " ") for f in FAMILIES],
        xlim=(0, 2.35),
        xticks=[0, 1, 2],
        xlabel="Correct tasks / 2 per family",
    )
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False, fontsize=10)
    fig.suptitle("Every family | same fresh tasks in every arm", x=0.31, ha="left", fontsize=17)
    fig.subplots_adjust(left=0.31, top=0.88, bottom=0.08)
    save(fig, folder, "families")

    fields = list(s["qlora_v5"]["intent_field_correct"])
    fig, ax = plt.subplots(figsize=(13, 6))
    for j, v in enumerate(ARMS[1:]):
        nums = [s[v]["intent_field_correct"][k] for k in fields] + [s[v]["intent_joint_correct"]]
        positions = np.arange(len(nums)) + (j - 1) * 0.24
        ax.bar(positions, np.array(nums) / 32 * 100, 0.23, color=COLORS[j + 1], label=LABELS[j + 1])
        for x, n in zip(positions, nums, strict=True):
            ax.text(x, n / 32 * 100 + 1.7, str(n), ha="center", fontsize=9)
    ax.set(
        xticks=range(len(fields) + 1),
        xticklabels=[k.replace("_", "\n") for k in fields] + ["All seven\njointly"],
        ylim=(0, 116),
        ylabel="Correct initial intent fields (%), counts /32 above",
        title="Initial intent extraction | includes missing values and refused alternatives",
    )
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21))
    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.08,
        0.015,
        "Exact quote provenance is not semantic consent. Missing/invalid initial records fail.",
        fontsize=10,
    )
    save(fig, folder, "intent")

    expected, predicted = (
        ["cancel", "return", "exchange", "handoff"],
        ["cancel", "return", "exchange", "handoff", "unfinished"],
    )
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, v, label in zip(axes, ARMS, LABELS, strict=True):
        matrix = np.array(confusion[v])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, maximum))
        for (y, x), n in np.ndenumerate(matrix):
            ax.text(
                x, y, str(n), ha="center", va="center", color="white" if n > maximum * 0.55 else "#263b41"
            )
        ax.set(xticks=range(5), yticks=range(4), yticklabels=expected, title=label)
        ax.set_xticklabels(predicted, rotation=35, ha="right")
        ax.set_xlabel("Proposed operation / final status")
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.025, label="Tasks (common scale)")
    fig.suptitle("Operation confusion | all 32 tasks per arm", x=0.04, ha="left", fontsize=17)
    fig.subplots_adjust(left=0.04, right=0.88, bottom=0.25, wspace=0.42)
    fig.text(
        0.04, 0.015, "Operation type alone omits argument, intent and clarification correctness.", fontsize=10
    )
    save(fig, folder, "confusion")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    metrics = training["metrics"]
    losses = np.array([m["loss"] for m in metrics])
    axes[0].plot(range(1, len(losses) + 1), losses, color=COLORS[1], alpha=0.25, linewidth=0.7)
    if len(losses) >= 16:
        axes[0].plot(
            range(16, len(losses) + 1),
            np.convolve(losses, np.ones(16) / 16, mode="valid"),
            color=COLORS[-1],
            label="16-update mean",
        )
    axes[0].set(
        xlabel="Completed optimizer updates",
        ylabel="Completion NLL (symlog)",
        title="A. Uniform full-completion supervision",
    )
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False)
    candidates = training["validation_candidates"]
    for j, (key, den, label) in enumerate(
        [
            ("task_success", "tasks", "Overall"),
            ("fulfillable_exchange_success", "fulfillable_exchange_tasks", "Exchange"),
            ("fallback_success", "fallback_tasks", "Fallback"),
            ("handoff_success", "handoff_tasks", "Handoff"),
        ]
    ):
        nums = [c["summary"][key] for c in candidates]
        dens = [c["summary"][den] for c in candidates]
        rates = [n / d * 100 for n, d in zip(nums, dens, strict=True)]
        positions = np.arange(len(candidates)) + (j - 1.5) * 0.19
        axes[1].bar(positions, rates, 0.18, color=COLORS[j], label=label)
        for x, n, d, rate in zip(positions, nums, dens, rates, strict=True):
            axes[1].text(x, rate + 2, f"{n}/{d}", ha="center", fontsize=9)
    axes[1].set(
        xticks=range(len(candidates)),
        xticklabels=[
            f"{c['iteration']} updates"
            + ("\nSELECTED" if c["iteration"] == training["best_checkpoint"]["iteration"] else "")
            for c in candidates
        ],
        ylim=(0, 118),
        ylabel="Validation task success (%)",
        title="B. Validation-only selection before new test",
    )
    axes[1].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=9)
    fig.subplots_adjust(bottom=0.25, wspace=0.25)
    save(fig, folder, "training")

    columns = [
        "variant",
        "case_id",
        "family",
        "task_success",
        "predicted_outcome",
        "model_calls",
        "repeated_question_attempts",
        "repeated_questions_delivered",
        "exchange_as_return",
        "unnecessary_handoff_attempts",
        "unnecessary_handoff_executed",
        "unrequested_substitute_attempts",
        "policy_gate_rejections",
        "quote_rejections",
        "latency_ms",
    ]
    with (folder / "per-case.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in records)
    index = {(r["variant"], r["case_id"]): r["task_success"] for r in records}
    failures = [
        {"case_id": cid, **{v: index[v, cid] for v in ARMS}}
        for cid in sorted({r["case_id"] for r in records})
        if not all(index[v, cid] for v in ARMS)
    ]
    values = {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "training_sha256": sha(REPORTS / "training.json"),
        "manifest_sha256": sha(DATA / "manifest.json"),
        "analysis_source_sha256": sha(__file__),
        "csv_sha256": sha(folder / "per-case.csv"),
        "summary": s,
        "family_success": families,
        "confusion": confusion,
        "permission_diagnostics": permissions,
        "failures": failures,
        "paired": {
            "end_to_end": paired(records, ARMS[0], ARMS[3]),
            "workflow": paired(records, ARMS[0], ARMS[1]),
            "training": paired(records, ARMS[1], ARMS[3]),
            "base": paired(records, ARMS[2], ARMS[3]),
        },
        "figure_sha256": {p.name: sha(p) for p in folder.iterdir() if p.suffix in {".png", ".svg"}},
    }
    (folder / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    (REPORTS / "failure-analysis.json").write_text(
        json.dumps(
            {
                "evaluation_sha256": values["evaluation_sha256"],
                "records": [
                    {
                        "variant": r["variant"],
                        "case_id": r["case_id"],
                        "initial_intent": r["intent_initial"],
                        "intent_field_correct": r["intent_field_correct"],
                        "last_decision": r["trace"][-1],
                        "status": r["status_final"],
                    }
                    for r in records
                    if not r["task_success"]
                ],
                "scope": "All failures retained; post-hoc diagnostics never select the checkpoint.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"paired": values["paired"], "figures": list(values["figure_sha256"])}))


if __name__ == "__main__":
    main()
