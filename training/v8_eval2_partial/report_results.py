"""Generate authorized retrospective common40 partial figures/CSVs only after independent full replay."""

import argparse
import csv
import hashlib
import io
import json
import os
import statistics
import tempfile
from pathlib import Path

from training.common import ROOT
from training.cuda_eval.prepare import denominators
from training.v8_eval2.verify_results import AFTER_SHA, BEFORE_SHA, PHYSICAL_PROOF_SHA
from training.v8_eval2_partial.metrics import (
    ANALYSIS_ID,
    ARMS,
    FIELDS,
    RUN_ID,
    derived_values,
    raw_diagnostics,
)
from training.v8_eval2_partial.metrics import LABELS as ARM_LABELS

REPORTS = "reports/v8-eval2-partial40"
OUTPUT = "docs/figures/v8-eval2-partial40"
SOURCE_FOLDER = "reports/v8-eval2/cloud-evidence/snapshots/20261030T150701917677Z"
SOURCE_REPORT = SOURCE_FOLDER + "/reports/v8-eval2/evaluation-partial.json"
SOURCE_JOURNAL = SOURCE_FOLDER + "/reports/v8-eval2/call-journal.json"
FIGURES = ("evaluation", "intent", "recovery", "families", "consent", "training")
CSV_NAMES = ("per-case.csv", "case-mapping.csv", "metrics.csv", "training.csv")
LABELS = tuple(ARM_LABELS[a] for a in ARMS)
COLORS = ("#8b99a3", "#376786", "#af7a30")
METRIC_SPECS = (
    ("Correctly handled selected tasks", "task_success", "tasks"),
    ("Available exchange", "fulfillable_exchange_success", "fulfillable_exchange_tasks"),
    ("Stockout fallback", "fallback_success", "fallback_tasks"),
    ("Necessary handoff", "handoff_success", "handoff_tasks"),
    ("Missing email", "missing_email_success", "missing_email_tasks"),
    ("Joint first raw fields", "intent_joint_correct", "intent_tasks"),
    ("Explicit initial YES", "true_permission_yes", "authorized_tasks"),
    ("Unsupported initial YES (lower)", "unsupported_permission_yes", "non_authorized_tasks"),
    ("First raw parse failures (lower)", "initial_raw_parse_failures", "intent_tasks"),
    ("First reason enum errors (lower)", "initial_reason_enum_error_tasks", "intent_tasks"),
)


def load_verified(root=ROOT):
    """Require independently audited136 rows before analyzing only the fixed paired120."""
    root = Path(root).resolve()
    proof = read(root, f"{REPORTS}/partial-evidence-verification.json")
    require(proof.get("status") == "verified_partial_evidence", "No independent partial-evidence receipt")
    require(proof.get("analysis_id") == ANALYSIS_ID and proof.get("run_id") == RUN_ID, "Different analysis")
    require(
        proof.get("counts")
        == {
            "audited_completed_records": 136,
            "paired_records": 120,
            "cases": 40,
            "completed_calls": 765,
            "ambiguous_calls": 1,
            "excluded_completed_records": 16,
            "unstarted_tasks": 7,
            "ambiguous_pending_tasks": 1,
        },
        "Independent audit did not cover every known row and the unknown call",
    )
    require(
        proof.get("original_full_result_verified") is False
        and proof.get("original144_gate_complete") is False
        and proof.get("original_run_modified") is False,
        "The original144 must remain incomplete",
    )
    require(
        proof.get("new_model_or_gpu_calls") == 0 and proof.get("training_updates") == 0,
        "This is CPU-only saved-evidence analysis",
    )
    require(
        proof.get("replay")
        == {
            "known_business_actions_events_confirmation": True,
            "known_actual_input_and_output_tokens": True,
            "known_scoped_consent": True,
            "unknown_request_only": True,
            "unknown_response_tokens_or_business_reconstructed": False,
        },
        "Known-record replay or unknown-request handling is incomplete",
    )
    plan = read(root, f"{REPORTS}/analysis-plan.json")
    require(
        plan.get("status") == "frozen_partial_analysis" and plan.get("sealed") is True,
        "The post-hoc analysis plan must be sealed",
    )
    require(
        plan.get("human_scope_approval", {}).get("explicit") is True, "No explicit partial40 authorization"
    )
    require(plan.get("analysis_id") == ANALYSIS_ID and plan.get("run_id") == RUN_ID, "Different sealed scope")
    require(
        proof["analysis_plan_sha256"] == sha(root, f"{REPORTS}/analysis-plan.json"), "Analysis plan changed"
    )
    require(
        plan["selection_sha256"] == proof["selection_sha256"] == sha(root, f"{REPORTS}/selection.json"),
        "Selection changed",
    )
    require(
        plan["selection_audit_sha256"] == sha(root, plan["selection_audit_path"]), "Selection audit changed"
    )
    selection = read(root, plan["selection_path"])
    require(
        plan["original_experiment"]["raw_report_path"] == SOURCE_REPORT
        and plan["original_experiment"]["call_journal_path"] == SOURCE_JOURNAL,
        "The plan points to a different retained snapshot",
    )
    required = {f"{REPORTS}/{n}.json" for n in ("analysis-plan", "selection", "selection-audit")}
    required |= {
        SOURCE_REPORT,
        SOURCE_JOURNAL,
        SOURCE_FOLDER + "/snapshot-receipt.json",
        SOURCE_FOLDER + "/import-verification.json",
        "training/v8_eval2/datasets/cases.json",
        "reports/v8-eval2/analysis-plan.json",
        "reports/v8-eval2/artifact-identity.json",
        "training/v8_eval2_partial/verify_results.py",
        "training/v8_eval2/verify_results.py",
    }
    require(required <= set(proof["files"]), "The partial audit omitted a source binding")
    require(
        set(plan["source_inputs_sha256"]) <= set(proof["files"]), "The new source closure was not verified"
    )
    for name, expected in {**proof["files"], **plan["source_inputs_sha256"]}.items():
        require(sha(root, name) == expected, "Verified source changed: " + name)
    require(
        proof["verifier_source_sha256"] == sha(root, "training/v8_eval2_partial/verify_results.py"),
        "Partial verifier source changed",
    )
    all_cases = read(root, "training/v8_eval2/datasets/cases.json")
    require(
        len(all_cases) == 48 and selection["case_ids"] == [c["id"] for c in all_cases[:40]],
        "Only the fixed common first40 authored cases are allowed",
    )
    cases = all_cases[:40]
    require(
        digest_bytes(canonical(cases)) == selection["canonical_selected_cases_sha256"],
        "Selected case content changed",
    )
    require(plan["denominators"] == denominators(cases), "Subset cohort denominators changed")
    original = read(root, SOURCE_REPORT)
    require(original["status"] == "partial" and len(original["records"]) == 136, "Wrong raw evidence")
    require(
        len(proof["record_audits"]) == 136 and len(proof["selected_records"]) == 120,
        "Every known record, including excluded tails, must be audited",
    )
    for i, audit in enumerate(proof["record_audits"]):
        row = original["records"][i]
        require(
            audit["source_record_index"] == i
            and audit["source_record_sha256"] == digest_bytes(canonical(row)),
            "Raw record identity differs",
        )
        require(
            (audit["variant"], audit["case_id"]) == (row["variant"], row["case_id"]),
            "Record audit points to another case",
        )
        require(
            all(row[key] == val for key, val in audit["derived_scoring"].items()), "Replayed score differs"
        )
    records, pointers = [], []
    require(len(selection["records"]) == 120, "Missing paired record selection")
    for ref, audit in zip(selection["records"], proof["selected_records"], strict=True):
        i = ref["original_record_index"]
        row = original["records"][i]
        require(
            ref["canonical_sha256"] == audit["source_record_sha256"] == digest_bytes(canonical(row)),
            "Selected raw record differs",
        )
        require(
            i == audit["source_record_index"] and audit["selected_for_paired40"] is True,
            "Audit/selection source indices differ",
        )
        require((ref["variant"], ref["case_id"]) == (row["variant"], row["case_id"]), "Selected key differs")
        records.append(row)
        pointers.append(f"{SOURCE_REPORT}#/records/{i}/trace")
    require(
        [(r["variant"], r["case_id"]) for r in records] == [(a, c["id"]) for a in ARMS for c in cases],
        "Paired cases/order differ",
    )
    require(
        digest_bytes(canonical(records)) == selection["canonical_selected_records_sha256"],
        "The canonical paired120 selection changed",
    )
    require(
        proof["selected_protocol_records"] == [r["consent_audit"] for r in proof["selected_records"]],
        "Selected consent records differ",
    )
    protocol = {"records": proof["selected_protocol_records"]}
    selected_report = {"records": records, "record_sources": pointers}
    values = derived_values(selected_report, {c["id"]: c for c in cases}, protocol, all_cases=all_cases)
    for arm, cohort in proof["selected_protocol_summary"].items():
        require(
            all(values["protocol_summary"][arm][k] == v for k, v in cohort.items()),
            "The verified selected consent summary differs",
        )
    values["coverage"]["selection_audit_path"] = plan["selection_audit_path"]
    values["coverage"]["selection_audit_sha256"] = plan["selection_audit_sha256"]
    selection_audit = read(root, plan["selection_audit_path"])
    values["coverage"]["focus_coverage"] = {
        name: {k: item[k] for k in ("original48_tasks", "selected40_tasks")}
        for name, item in selection_audit["focus_coverage"].items()
    }
    original_plan = read(root, "reports/v8-eval2/analysis-plan.json")
    artifact_paths = original_plan["artifact_bindings"]
    require(
        {v["runtime_path"] for v in artifact_paths.values()} <= set(proof["files"]),
        "Saved historical training proof was not bound",
    )
    training = read(root, artifact_paths["training-report.json"]["runtime_path"])
    physical = read(root, artifact_paths["final-training-physical-verification.json"]["runtime_path"])
    require(
        sha(root, artifact_paths["final-training-physical-verification.json"]["runtime_path"])
        == PHYSICAL_PROOF_SHA,
        "The retained physical training proof changed",
    )
    require(
        proof["saved_weight_audit"]["before_sha256"] == BEFORE_SHA
        and proof["saved_weight_audit"]["after_sha256"] == AFTER_SHA,
        "Saved weight identities changed",
    )
    sources = {
        n: sha(root, n)
        for n in ("training/v8_eval2_partial/report_results.py", "training/v8_eval2_partial/metrics.py")
    }
    values.update(
        families=values["family_success"],
        selection_denominators=plan["denominators"],
        exclusions={"complete_old_arm_tail_records": 16, "unstarted_tasks": 7, "ambiguous_pending_tasks": 1},
        training={
            "historical_saved_run_only": True,
            "new_updates_this_experiment": 0,
            "updates": training["completed_true_optimizer_updates"],
            "sampling": training["sampling"],
            "parameters": 3_670_016,
            "tensors": 112,
            "adapter_l2_change": physical["training"]["adapter_l2_change"],
            "elapsed_loop_gpu_sync_seconds": training["elapsed_loop_gpu_sync_seconds"],
            "timing_scope": training["timing_scope"],
        },
        saved_weights={"before": BEFORE_SHA, "after": AFTER_SHA},
        files={
            **proof["files"],
            **sources,
            f"{REPORTS}/partial-evidence-verification.json": sha(
                root, f"{REPORTS}/partial-evidence-verification.json"
            ),
        },
        verification_is_quality_pass=False,
        visual_qa_performed=False,
        report_only=True,
        original144_gate_complete=False,
        deployment_or_default_provider_change_allowed=False,
        latency_scope="Fixed arm-block order and nonrandom truncation; trajectories/logging confound latency",
        consent_scope="Attempts/proposals/commits/final orders are distinct; zeros cannot certify safety",
    )
    return {"values": values, "test": selected_report, "cases": cases, "training": training}


def canonical(value):
    # Match the frozen runner's value_hash, including JSON's default ASCII escaping.
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(root, relative):
    """Reject symlinks and paths escaping the explicitly selected checkout."""
    root = Path(root).resolve()
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Evidence paths must be relative to the checkout")
    path = root / name
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError("Symlink evidence is not accepted")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Evidence path escaped the checkout")
    return path


def sha(root, relative):
    h = hashlib.sha256()
    with checked_path(root, relative).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(root, relative):
    return json.loads(checked_path(root, relative).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def per_case_rows(report, cases):
    by_id = {c["id"]: c for c in cases}
    rows = []
    for i, row in enumerate(report["records"]):
        case = by_id[row["case_id"]]
        raw = raw_diagnostics(row, case)
        rows.append(
            {
                "variant": row["variant"],
                "case_id": row["case_id"],
                "family": row["family"],
                "expected_kind": case["expected"]["kind"],
                "confirmation_choice": case["confirmation"],
                "task_success": row["task_success"],
                "status_final": row["status_final"],
                "model_calls": row["model_calls"],
                "latency_ms": row["latency_ms"],
                "joint_initial_raw_fields_correct": all(raw["intent_field_correct"].values()),
                "initial_raw_parse_failure": raw["initial_raw_parse_failure"],
                "initial_enum_errors": json.dumps(raw["initial_enum_errors"], sort_keys=True),
                "initial_permission_gold": case["intent_initial"]["fallback_return"],
                "initial_permission_prediction": (raw["intent_initial"] or {}).get(
                    "fallback_return", "PARSE_FAILURE"
                ),
                **{f"initial_{key}_correct": value for key, value in raw["intent_field_correct"].items()},
                **{
                    key: row[key]
                    for key in (
                        "quote_rejections",
                        "enum_error_attempts",
                        "record_rejections",
                        "repeated_question_attempts",
                        "repeated_questions_delivered",
                        "identical_action_repeats",
                        "unnecessary_handoff_attempts",
                        "unnecessary_handoff_executed",
                    )
                },
                "raw_trace": report["record_sources"][i],
            }
        )
    return rows


def render(bundle, directory):
    """Called only after load_verified succeeds. All marks derive from saved evidence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "svg.hashsalt": ANALYSIS_ID,
            "text.color": "#253744",
        }
    )
    values, summary = bundle["values"], bundle["values"]["summary"]
    directory = Path(directory)

    def save(fig, name, note):
        if bundle.get("layout_fixture_only"):
            fig.text(
                0.02,
                0.985,
                "CPU FIXTURE LAYOUT ONLY — NOT MODEL RESULTS",
                fontsize=11,
                va="top",
                color="#8d371f",
                weight="bold",
            )
        fig.text(0.02, 0.015, note, fontsize=10, va="bottom", wrap=True)
        fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    def bars(ax, groups, title):
        max_value = max(1, *(summary[arm][key] for _, key in groups for arm in ARMS))
        for i, arm in enumerate(ARMS):
            x = np.arange(len(groups)) + (i - 1) * 0.25
            y = [summary[arm][key] for _, key in groups]
            ax.bar(x, y, width=0.23, color=COLORS[i], label=LABELS[i], hatch=("//", "", "..")[i])
            for position, count in zip(x, y, strict=True):
                ax.text(position, count + max_value * 0.025, str(count), ha="center", fontsize=10)
        ax.set(
            xticks=range(len(groups)),
            xticklabels=[label for label, _ in groups],
            title=title,
            ylabel="Count (units named on x-axis)",
            ylim=(0, max_value * 1.23 + 1),
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)

    def rates(ax, specs, title):
        positions = np.arange(len(specs))
        for i, arm in enumerate(ARMS):
            for j, (_, numerator, denominator) in enumerate(specs):
                n, d = summary[arm][numerator], summary[arm][denominator]
                require(d > 0, "A displayed rate has no cohort")
                y = j + (i - 1) * 0.24
                ax.barh(
                    y,
                    100 * n / d,
                    height=0.22,
                    color=COLORS[i],
                    hatch=("//", "", "..")[i],
                    label=LABELS[i] if j == 0 else None,
                )
                ax.text(min(100 * n / d + 1.5, 102), y, f"{n}/{d}", va="center", fontsize=10)
        ax.set(
            yticks=positions,
            yticklabels=[s[0] for s in specs],
            xlim=(0, 119),
            xticks=[0, 25, 50, 75, 100],
            xlabel="Percent of the named cohort",
            title=title,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    rates(ax, METRIC_SPECS[:5], "Stage-only paired40 comparison: saved weights and controller flows")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.57, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(left=0.27, bottom=0.16, top=0.81)
    save(
        fig,
        "evaluation",
        "40 nonrandom common-prefix worlds per arm; outcomes include correctly declined proposals. "
        "Not a full48 or original144 comparison.",
    )

    fig, axes = plt.subplots(1, 3, figsize=(19, 7.3))
    field_specs = [(key.replace("_", " "), key, "intent_tasks") for key in FIELDS]
    field_summary = {arm: {**summary[arm], **summary[arm]["intent_field_correct"]} for arm in ARMS}
    old_summary, summary = summary, field_summary
    rates(axes[0], field_specs, "First raw field accuracy")
    summary = old_summary
    rates(axes[1], METRIC_SPECS[5:8], "Joint fields and explicit permission")
    bars(
        axes[2],
        [
            ("Parse\nfailures", "initial_raw_parse_failures"),
            ("Any enum\nerror tasks", "initial_enum_error_tasks"),
            ("Reason enum\nerror tasks", "initial_reason_enum_error_tasks"),
        ],
        f"First-output failures /{summary[ARMS[0]]['intent_tasks']}",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(left=0.1, right=0.98, wspace=0.70, bottom=0.2, top=0.81)
    save(
        fig,
        "intent",
        "Only the first model output is scored here; "
        "later corrections and backend recovery receive no first-output credit.",
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    bars(
        axes[0, 0],
        [
            ("Quote\nrejections", "quote_rejections"),
            ("Record\nrejections", "record_rejections"),
            ("Enum error\nattempts", "enum_error_attempts"),
            ("Enum repaired\ntasks", "enum_recovered_tasks"),
        ],
        "All-turn validation and repairs",
    )
    bars(
        axes[0, 1],
        [
            ("Question\nattempt repeats", "repeated_question_attempts"),
            ("Question\ndelivered repeats", "repeated_questions_delivered"),
            ("All identical\naction repeats", "identical_action_repeats"),
        ],
        "Repetition: different units",
    )
    bars(
        axes[1, 0],
        [
            ("Unnecessary\nhandoff attempts", "unnecessary_handoff_attempts"),
            ("Unnecessary\nhandoffs executed", "unnecessary_handoff_executed"),
        ],
        "Unnecessary handoffs",
    )
    rates(
        axes[1, 1],
        [
            (
                "Required questions obtained",
                "required_clarification_fields_obtained",
                "required_clarification_fields",
            ),
            (
                "Tasks obtaining all required answers",
                "required_clarification_tasks_obtained",
                "required_clarification_tasks",
            ),
        ],
        "Required clarifications",
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.65, bottom=0.15, top=0.85)
    save(
        fig,
        "recovery",
        "Attempts, executed actions, fields and tasks are separate measures. "
        "Corrective calls consume the same total12-call budget.",
    )

    family = values["families"]
    names = list(values["coverage"]["per_family"])
    max_cases = max(f["tasks_per_arm"] for f in family.values())
    matrix = np.array([[family[name][arm] if name in family else np.nan for arm in ARMS] for name in names])
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#e1e5e7")
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.imshow(matrix, vmin=0, vmax=max_cases, cmap=cmap, aspect="auto")
    ax.set(
        xticks=range(3),
        xticklabels=["Prior adapter\nV7 flow", "Prior adapter\nV8 flow", "Saved V8 adapter\nV8 flow"],
        yticks=range(len(names)),
        yticklabels=[
            name.replace("_", " ") + (" (excluded)" if name not in family else "") for name in names
        ],
        title="Stage-only family coverage and correctly handled selected tasks",
    )
    for y in range(len(names)):
        for x in range(3):
            ax.text(
                x,
                y,
                (
                    f"{int(matrix[y, x])}/{family[names[y]]['tasks_per_arm']}"
                    if names[y] in family
                    else "Excluded"
                ),
                ha="center",
                va="center",
                color="white" if names[y] in family and matrix[y, x] == max_cases else "#253744",
            )
    fig.colorbar(image, ax=ax, ticks=list(range(max_cases + 1)), label="Correctly handled selected tasks")
    fig.subplots_adjust(left=0.36, bottom=0.17)
    save(
        fig,
        "families",
        "All original families are shown. Grey excluded rows have no estimate, not zero successes. "
        "All failures in the selected prefix are retained.",
    )

    p = values["protocol_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    protocol_specs = [
        ("Raw return\nattempts", "raw_return_attempts"),
        ("Return\nproposals", "return_proposals"),
        ("Return\ncommits", "committed_returns"),
        ("Final return\norders", "final_return_orders"),
    ]
    violation_specs = [
        ("Attempts\nwithout\nscoped YES", "return_attempts_without_scoped_yes"),
        ("Proposals\nwithout\nscoped YES", "return_proposals_without_scoped_yes"),
        ("Commits\nwithout\nscoped YES", "committed_returns_without_scoped_yes"),
        ("Final orders\nno verified\ncommit", "final_returns_without_verified_commit"),
        ("Rejected\nattempts; no\nscoped YES", "rejected_return_attempts_without_scoped_yes"),
    ]
    old_summary, summary = summary, p
    bars(axes[0], protocol_specs, "Exchange-origin return paths: distinct units")
    bars(axes[1], violation_specs, "Supplemental scoped-consent diagnostics")
    summary = old_summary
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.55, 0.95),
        ncol=3,
        fontsize=10,
        frameon=False,
    )
    fig.subplots_adjust(wspace=0.30, bottom=0.25, top=0.81)
    save(
        fig,
        "consent",
        f"Cohort:{p[ARMS[0]]['gold_exchange_tasks']} selected exchange-origin worlds per arm. "
        "Counts are never summed across units; "
        "zeros do not prove production safety or fix known bypasses.",
    )

    train = bundle["training"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    history = train["metrics"]
    axes[0, 0].plot(
        [r["iteration"] for r in history], [r["loss"] for r in history], color=COLORS[2], linewidth=0.9
    )
    axes[0, 0].set(
        title="Historical saved-V8 training loss (not rerun)",
        xlabel="True optimizer update",
        ylabel="Weighted token loss",
    )
    sampling = train["sampling"]
    axes[0, 1].bar(
        ["Physical SFT pairs", "Unique pairs sampled"],
        [sampling["physical_rows"], sampling["unique_rows_sampled"]],
        color=[COLORS[0], COLORS[2]],
    )
    for i, n in enumerate([sampling["physical_rows"], sampling["unique_rows_sampled"]]):
        axes[0, 1].text(i, n + 15, str(n), ha="center")
    axes[0, 1].set(
        title=f"Measured sampling coverage: {sampling['unique_cases_sampled']} source cases",
        ylabel="Pair count",
        ylim=(0, sampling["physical_rows"] * 1.15),
    )
    for ax, key, title, unit in [
        (axes[1, 0], "model_calls", "Calls across actual trajectories", "Total calls per task"),
        (axes[1, 1], "latency_ms", "Descriptive serial task latency", "Seconds per task"),
    ]:
        arrays = [
            [
                r[key] / (1000 if key == "latency_ms" else 1)
                for r in bundle["test"]["records"]
                if r["variant"] == arm
            ]
            for arm in ARMS
        ]
        box = ax.boxplot(
            arrays, tick_labels=["Prior / V7", "Prior / V8", "V8 / V8"], patch_artist=True, showmeans=False
        )
        for body, color in zip(box["boxes"], COLORS, strict=True):
            body.set_facecolor(color)
            body.set_alpha(0.6)
        for i, array in enumerate(arrays, 1):
            ax.text(
                i, max(array) * 1.05 + 0.2, f"median {statistics.median(array):.2f}", ha="center", fontsize=10
            )
        ax.set(title=title, ylabel=unit, ylim=(0, max(max(a) for a in arrays) * 1.2 + 1))
        if key == "model_calls":
            ax.axhline(12, color="#4e5961", linestyle="--", linewidth=1)
            ax.set_ylim(0, max(13, max(max(a) for a in arrays) * 1.2 + 1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(hspace=0.42, wspace=0.3, bottom=0.15)
    save(
        fig,
        "training",
        "Saved historical training only; this run performs no training or validation. Loss is not accuracy. "
        "Latency includes serial order, trajectory and logging effects; no speedup claim.",
    )


def csv_exports(bundle):
    rows = per_case_rows(bundle["test"], bundle["cases"])
    mapping = bundle["values"]["cases"]
    metrics = [
        {
            "variant": arm,
            "metric": label,
            "numerator": bundle["values"]["summary"][arm][n],
            "denominator": bundle["values"]["summary"][arm][d],
            "percent": 100 * bundle["values"]["summary"][arm][n] / bundle["values"]["summary"][arm][d],
        }
        for arm in ARMS
        for label, n, d in METRIC_SPECS
    ]
    history = bundle["training"]["metrics"]
    return {
        "per-case.csv": csv_bytes(rows, list(rows[0])),
        "case-mapping.csv": csv_bytes(mapping, list(mapping[0])),
        "metrics.csv": csv_bytes(metrics, list(metrics[0])),
        "training.csv": csv_bytes(history, list(history[0])),
    }


def deliver(root=ROOT):
    """Create new generated assets; refuse partial evidence and refuse replacing an export."""
    root = Path(root).resolve()
    bundle = load_verified(root)
    destination = checked_path(root, OUTPUT)
    require(not destination.exists(), "Never overwrite a delivered partial40 asset directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v8-eval2-partial40-report-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "delivery"
        stage.mkdir()
        for name, raw in csv_exports(bundle).items():
            (stage / name).write_bytes(raw)
        render(bundle, stage)
        expected = {*CSV_NAMES, *(f"{name}.{suffix}" for name in FIGURES for suffix in ("png", "svg"))}
        require({p.name for p in stage.iterdir()} == expected, "Incomplete six-group export")
        values = bundle["values"]
        values["output_files_sha256"] = {
            p.name: digest_bytes(p.read_bytes()) for p in sorted(stage.iterdir())
        }
        values["reporting_source_sha256"] = sha(root, "training/v8_eval2_partial/report_results.py")
        values["metrics_source_sha256"] = sha(root, "training/v8_eval2_partial/metrics.py")
        (stage / "values.json").write_text(json.dumps(values, indent=2) + "\n")
        receipt = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "analysis_id": ANALYSIS_ID,
            "status": "generated_partial40_awaiting_visual_qa",
            "original144_gate_complete": False,
            "report_only": True,
            "verification_is_quality_pass": False,
            "visual_qa_performed": False,
            "files": {p.name: digest_bytes(p.read_bytes()) for p in sorted(stage.iterdir())},
            "inputs": values["files"],
            "reporting_source_sha256": values["reporting_source_sha256"],
        }
        (stage / "report-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        for name, expected in values["files"].items():
            require(sha(root, name) == expected, "Input changed during rendering: " + name)
        os.rename(stage, destination)
    return receipt


def verify_delivery(root=ROOT, *, write_receipt=True):
    """Reconcile every CSV/asset hash and source value; visual inspection remains separate."""
    root = Path(root).resolve()
    bundle = load_verified(root)
    values = read(root, OUTPUT + "/values.json")
    receipt = read(root, OUTPUT + "/report-receipt.json")
    require(values["run_id"] == receipt["run_id"] == RUN_ID, "Different generated run")
    for key, expected in bundle["values"].items():
        require(values.get(key) == expected, "Exported source values differ: " + key)
    require(receipt["inputs"] == values["files"], "Report input receipt differs")
    require(
        values["reporting_source_sha256"] == sha(root, "training/v8_eval2_partial/report_results.py"),
        "Reporting source changed",
    )
    require(
        values["metrics_source_sha256"] == sha(root, "training/v8_eval2_partial/metrics.py"),
        "Metrics changed",
    )
    for name, expected in receipt["files"].items():
        require(sha(root, OUTPUT + "/" + name) == expected, "Asset differs: " + name)
    expected_assets = {*CSV_NAMES, *(f"{n}.{s}" for n in FIGURES for s in ("png", "svg"))}
    require(set(values["output_files_sha256"]) == expected_assets, "Incomplete generated asset ledger")
    for name, expected in values["output_files_sha256"].items():
        require(sha(root, OUTPUT + "/" + name) == expected, "Asset hash differs: " + name)
    for name, expected in csv_exports(bundle).items():
        require(
            checked_path(root, OUTPUT + "/" + name).read_bytes() == expected,
            "CSV does not match every raw source row: " + name,
        )
    for name in FIGURES:
        require(
            checked_path(root, OUTPUT + f"/{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"),
            "PNG signature differs",
        )
        require("<svg" in checked_path(root, OUTPUT + f"/{name}.svg").read_text(), "Missing SVG root")
    verification = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "verified_partial40_hashes_csv_and_values",
        "analysis_id": ANALYSIS_ID,
        "original144_gate_complete": False,
        "counts": {
            "paired_records": len(bundle["test"]["records"]),
            "cases": len(bundle["cases"]),
            "png": 6,
            "svg": 6,
            "csv": 4,
        },
        "visual_qa_performed": False,
        "verification_is_quality_pass": False,
        "values_sha256": sha(root, OUTPUT + "/values.json"),
        "report_receipt_sha256": sha(root, OUTPUT + "/report-receipt.json"),
        "result_verification_sha256": sha(root, REPORTS + "/partial-evidence-verification.json"),
        "partial_evidence_verification_sha256": sha(root, REPORTS + "/partial-evidence-verification.json"),
        "files": receipt["files"],
        "source_sha256": sha(root, "training/v8_eval2_partial/report_results.py"),
    }
    if write_receipt:
        from training.v8_eval2.verify_results import write_once

        write_once(root / REPORTS / "figure-verification.json", verification)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-figures", action="store_true")
    args = parser.parse_args()
    if args.verify_figures:
        output = verify_delivery(args.root)
    elif args.check_only:
        output = {
            "ready": True,
            "run_id": RUN_ID,
            "records": len(load_verified(args.root)["test"]["records"]),
        }
    else:
        output = deliver(args.root)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
