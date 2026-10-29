import csv
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import ROOT, sha
from training.fixtures import FAMILIES


def main():
    report = json.loads((ROOT / "reports/evaluation-graded.json").read_text())
    train = json.loads((ROOT / "reports/training.json").read_text())
    assert report["training_sha256"] == sha(ROOT / "reports/training.json")
    folder = ROOT / "docs/figures"
    folder.mkdir(parents=True, exist_ok=True)
    indexed = {(r["variant"], r["case_id"]): r for r in report["records"]}
    ids = sorted({r["case_id"] for r in report["records"]})
    pairs = [
        {
            "case_id": i,
            "base": indexed[("base", i)]["task_success"],
            "finetuned": indexed[("finetuned", i)]["task_success"],
        }
        for i in ids
    ]
    paired = Counter((r["base"], r["finetuned"]) for r in pairs)
    pair_counts = {
        "improved": paired[(False, True)],
        "regressed": paired[(True, False)],
        "both_pass": paired[(True, True)],
        "both_fail": paired[(False, False)],
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#344950",
            "text.color": "#263b41",
            "axes.edgecolor": "#9ca5a3",
            "svg.fonttype": "none",
            "svg.hashsalt": "resolvedesk-v0.1",
        }
    )
    base, fine, accent = "#a7afad", "#335c62", "#b27942"
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [0.8, 1.25, 1]})
    a, b, c = axes
    sums = report["summary"]
    rates = [sums[v]["task_success"] / sums[v]["tasks"] * 100 for v in ("base", "finetuned")]
    a.bar([0, 1], rates, color=[base, fine], width=0.58)
    for x, v in enumerate(("base", "finetuned")):
        a.text(
            x,
            rates[x] + 2,
            f"{sums[v]['task_success']}/{sums[v]['tasks']}\n{rates[x]:.1f}%",
            ha="center",
            fontsize=12,
        )
    a.set_xticks([0, 1], ["Base", "QLoRA"])
    a.set_ylim(0, 115)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("Tasks meeting outcome + prerequisites (%)")
    a.set_title("A. Frozen task outcomes", loc="left", fontsize=12, pad=20)
    y = np.arange(len(FAMILIES))
    height = 0.33
    for j, v in enumerate(("base", "finetuned")):
        counts = [
            sum(r["task_success"] for r in report["records"] if r["variant"] == v and r["family"] == f)
            for f in FAMILIES
        ]
        b.barh(y + (j - 0.5) * height, counts, height, color=[base, fine][j], label=["Base", "QLoRA"][j])
        for yi, count in zip(y + (j - 0.5) * height, counts, strict=True):
            b.text(count + 0.06, yi, str(count), va="center", fontsize=9)
    b.set_yticks(y, [f.replace("_", " ") for f in FAMILIES])
    b.invert_yaxis()
    b.set_xlim(0, 3.6)
    b.set_xticks([0, 1, 2, 3])
    b.set_xlabel("Correct tasks / 3 per family")
    b.set_title("B. Same cases and policy guards", loc="left", fontsize=12, pad=20)
    b.legend(frameon=False, loc="lower right", fontsize=9)
    labels = ["Improved", "Regressed", "Both pass", "Both fail"]
    vals = [pair_counts[k] for k in ("improved", "regressed", "both_pass", "both_fail")]
    c.barh(range(4), vals, color=[fine, accent, base, "#d1d5cc"], height=0.6)
    for i, n in enumerate(vals):
        c.text(n + 0.2, i, str(n), va="center")
    c.set_yticks(range(4), labels)
    c.invert_yaxis()
    c.set_xlim(0, max(5, max(vals) + 3))
    c.set_xticks(range(0, max(6, max(vals) + 4), 3))
    c.set_xlabel(f"Paired tasks (N={len(ids)})")
    c.set_title("C. Before/after pairing", loc="left", fontsize=12, pad=20)
    for ax in axes:
        ax.set_axisbelow(True)
    a.grid(axis="y", alpha=0.16)
    b.grid(axis="x", alpha=0.16)
    c.grid(axis="x", alpha=0.16)
    fig.suptitle(
        "ResolveDesk | Qwen3-4B local retail action evaluation", fontsize=19, x=0.05, ha="left", y=0.98
    )
    fig.subplots_adjust(left=0.065, right=0.97, top=0.82, bottom=0.24, wspace=0.7)
    fig.text(
        0.05,
        0.085,
        "27 self-authored English tasks; shared task families, schema and author. "
        "One greedy trial per task; no real retailer data.",
        fontsize=10,
    )
    fig.text(
        0.05,
        0.045,
        "Clarification-aware grading; same MLX base and policy guards. Scripted replies/confirmations. "
        "Ollama integration uses a separate decoding path.",
        fontsize=10,
    )
    for extension in ("png", "svg"):
        fig.savefig(
            folder / f"evaluation.{extension}",
            dpi=160,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
    plt.close(fig)
    records = [r for r in train["metrics"] if r["kind"] == "valid"]
    x = [r["iteration"] for r in records]
    loss = [r["val_loss"] for r in records]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, loss, color=fine, marker="o", linewidth=1.8)
    selected = train["best_checkpoint"]
    ax.scatter(
        [selected["iteration"]],
        [selected["val_loss"]],
        color=accent,
        marker="*",
        s=180,
        zorder=5,
        label=f"Selected callback iteration {selected['iteration']}",
    )
    ax.set(
        xlabel="MLX-LM reported iteration (initial 0 and explicit final evaluation included)",
        ylabel="Full validation completion loss",
        title="Validation-only checkpoint selection | 64 held-out decision snapshots",
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.17)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.24, top=0.86)
    fig.text(
        0.1,
        0.07,
        "4-bit QLoRA · 160 optimizer updates · rank 8 · last 8 layers · "
        "test model outputs generated only after selection.",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        fig.savefig(
            folder / f"training.{extension}",
            dpi=160,
            facecolor="white",
            metadata={"Date": None} if extension == "svg" else None,
        )
    plt.close(fig)
    columns = [
        "variant",
        "case_id",
        "family",
        "task_success",
        "proposal_correct",
        "model_calls",
        "schema_valid",
        "reference_next_action_matches",
        "rejected_write_attempts",
        "latency_ms",
        "output_tokens",
        "status_final",
    ]
    with (folder / "per-case.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in report["records"]:
            writer.writerow({k: row[k] for k in columns})
    # Matplotlib emits trailing spaces inside SVG path data; canonicalize before hashing.
    for path in folder.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    values = {
        "evaluation_sha256": sha(ROOT / "reports/evaluation-graded.json"),
        "training_sha256": sha(ROOT / "reports/training.json"),
        "csv_sha256": sha(folder / "per-case.csv"),
        "summary": report["summary"],
        "paired": pair_counts,
        "failures": [r for r in pairs if not (r["base"] and r["finetuned"])],
        "figure_sha256": {
            n: sha(folder / n) for n in ("evaluation.png", "evaluation.svg", "training.png", "training.svg")
        },
    }
    (folder / "values.json").write_text(json.dumps(values, indent=2) + "\n")
    print(json.dumps({"paired": pair_counts, "figures": list(values["figure_sha256"])}))


if __name__ == "__main__":
    main()
