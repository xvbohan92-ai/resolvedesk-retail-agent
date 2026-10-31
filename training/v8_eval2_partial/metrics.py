"""Dynamic metrics for an explicitly authorized, retrospective common-case prefix.

Reuse the original raw-first/graded-outcome definitions without mutating their
module globals. Selection and independent replay are prerequisites, not outputs
of these functions. Nothing here infers performance on the omitted cases.
"""

from collections import Counter

from training.v8_eval2.metrics import (
    ARMS,
    COMPARISONS,
    FIELDS,
    LABELS,
    paired,
    protocol_summary,
    raw_diagnostics,
    summarize,
)

RUN_ID = "resolvedesk-v8-eval-002"
ANALYSIS_ID = "resolvedesk-v8-eval-002-partial40"

__all__ = [
    "ANALYSIS_ID",
    "ARMS",
    "COMPARISONS",
    "FIELDS",
    "LABELS",
    "RUN_ID",
    "coverage",
    "derived_values",
    "paired",
    "protocol_summary",
    "raw_diagnostics",
    "summarize",
]


def coverage(all_cases, selected_ids):
    selected = set(selected_ids)
    original = Counter(c["family"] for c in all_cases)
    included = Counter(c["family"] for c in all_cases if c["id"] in selected)
    return {
        "included_families": [name for name in original if included[name]],
        "excluded_families": [name for name in original if not included[name]],
        "per_family": {
            name: {
                "original_cases": count,
                "selected_cases": included[name],
                "excluded_cases": count - included[name],
            }
            for name, count in original.items()
        },
        "scope": "Coverage counts only; omitted cases are absent, never zero-scored failures.",
    }


def derived_values(report, cases, protocol, *, all_cases):
    rows = report["records"]
    ids = list(cases)
    assert len(ids) == len(set(ids))
    assert [(r["variant"], r["case_id"]) for r in rows] == [(arm, cid) for arm in ARMS for cid in ids]
    index = {(r["variant"], r["case_id"]): r for r in rows}
    cov = coverage(all_cases, ids)
    return {
        "run_id": RUN_ID,
        "analysis_id": ANALYSIS_ID,
        "paired_cases": len(ids),
        "paired_records": len(rows),
        "summary": {arm: summarize([r for r in rows if r["variant"] == arm], cases) for arm in ARMS},
        "protocol_summary": protocol_summary(protocol["records"]),
        "paired": {key: paired(rows, *pair) for key, pair in COMPARISONS.items()},
        "coverage": cov,
        "family_success": {
            family: {
                "tasks_per_arm": sum(c["family"] == family for c in cases.values()),
                **{
                    arm: sum(r["task_success"] for r in rows if r["variant"] == arm and r["family"] == family)
                    for arm in ARMS
                },
            }
            for family in cov["included_families"]
        },
        "cases": [
            {
                "case_id": cid,
                "family": cases[cid]["family"],
                **{arm: index[arm, cid]["task_success"] for arm in ARMS},
            }
            for cid in ids
        ],
        "failures": [
            {"case_id": cid, **{arm: index[arm, cid]["task_success"] for arm in ARMS}}
            for cid in ids
            if not all(index[arm, cid]["task_success"] for arm in ARMS)
        ],
        "failed_records": [
            {
                "variant": r["variant"],
                "case_id": r["case_id"],
                "status": r["status_final"],
                "model_calls": r["model_calls"],
            }
            for r in rows
            if not r["task_success"]
        ],
        "scope": {
            "kind": "authorized_retrospective_common40_prefix",
            "same_cases_per_arm": True,
            "fixed_arm_block_order": True,
            "random_truncation": False,
            "extrapolation_to_original48_allowed": False,
            "original144_gate_complete": False,
            "missing_records_are_model_failures": False,
            "old_interrupted_run_results_pooled": False,
            "external_benchmark_or_production_data": False,
        },
        "baseline_is_already_finetuned": True,
        "new_training_performed_in_this_run": False,
    }
