"""Independent run002 metrics; first raw predictions and old evidence remain untouched."""

import json
import statistics
from collections import Counter

from training.common import ROOT
from training.v7.protocol_audit import COUNTS as PROTOCOL_COUNTS

ARMS = ("predecessor_fp16", "cloud1344_fp16")
RUN_ID = "resolvedesk-cuda-eval-002"
FIGURE_NAMES = ("evaluation", "intent", "recovery", "families", "consent", "training")
LABELS = {ARMS[0]: "Predecessor · FP16", ARMS[1]: "After 1,344 updates · FP16"}
REPORTS = ROOT / "reports/cuda-eval2"
FIGURES = ROOT / "docs/figures/cuda-eval2"
FIELDS = ("operation", "email", "order_id", "requested_option", "payment_method", "reason", "fallback_return")
ENUMS = {
    "operation": {"cancel", "return", "exchange", "handoff", "unknown"},
    "payment_method": {"original", "gift_card", "unknown"},
    "reason": {"no longer needed", "ordered by mistake", "unknown"},
    "fallback_return": {"yes", "no", "unknown"},
}


def first_output(raw):
    try:
        action = json.loads(raw)
    except (TypeError, ValueError):
        action = None
    args = action.get("arguments") if isinstance(action, dict) else None
    is_record = isinstance(action, dict) and action.get("tool") == "record_request" and isinstance(args, dict)
    errors = {
        key: args[key]
        for key, allowed in ENUMS.items()
        if is_record and key in args and (not isinstance(args[key], str) or args[key] not in allowed)
    }
    complete = (
        is_record
        and set(args) == set(FIELDS) | {"evidence"}
        and all(isinstance(v, str) for v in args.values())
    )
    return ({key: args[key] for key in FIELDS} if complete else None), errors


def raw_diagnostics(record, case):
    trace = record["trace"]
    initial, errors = first_output(trace[0]["raw"] if trace else None)
    expected = case["intent_initial"]
    return {
        "intent_initial": initial,
        "initial_enum_errors": errors,
        "initial_raw_parse_failure": initial is None,
        "initial_reason_enum_error": "reason" in errors,
        "intent_enum_valid": initial is not None and not errors,
        "intent_field_correct": {key: bool(initial and initial[key] == expected[key]) for key in FIELDS},
        "permission_authorized": expected["fallback_return"] == "yes",
        "permission_yes": bool(initial and initial["fallback_return"] == "yes"),
    }


def summarize(records, cases):
    """Independent numerator/denominator calculations at one authored task per arm."""
    assert len({r["case_id"] for r in records}) == len(records)
    assert all(r["case_id"] in cases for r in records)
    diagnostics = {r["case_id"]: raw_diagnostics(r, cases[r["case_id"]]) for r in records}

    def cohort(predicate):
        subset = [r for r in records if predicate(cases[r["case_id"]])]
        return len(subset), sum(bool(r["task_success"]) for r in subset)

    available = cohort(lambda c: c["expected"]["kind"] == "exchange")
    fallback = cohort(
        lambda c: c["expected"]["kind"] == "return" and c["intent_initial"]["operation"] == "exchange"
    )
    handoff = cohort(lambda c: c["expected"]["kind"] == "handoff")
    email = cohort(lambda c: "email" in c["required_answers"])
    result = {
        "tasks": len(records),
        "intent_tasks": len(records),
        "task_success": sum(bool(r["task_success"]) for r in records),
        "fulfillable_exchange_tasks": available[0],
        "fulfillable_exchange_success": available[1],
        "fallback_tasks": fallback[0],
        "fallback_success": fallback[1],
        "handoff_tasks": handoff[0],
        "handoff_success": handoff[1],
        "missing_email_tasks": email[0],
        "missing_email_success": email[1],
        "authorized_tasks": sum(d["permission_authorized"] for d in diagnostics.values()),
        "non_authorized_tasks": sum(not d["permission_authorized"] for d in diagnostics.values()),
        "true_permission_yes": sum(
            d["permission_authorized"] and d["permission_yes"] for d in diagnostics.values()
        ),
        "unsupported_permission_yes": sum(
            not d["permission_authorized"] and d["permission_yes"] for d in diagnostics.values()
        ),
        "initial_raw_parse_failures": sum(d["initial_raw_parse_failure"] for d in diagnostics.values()),
        "initial_enum_error_tasks": sum(bool(d["initial_enum_errors"]) for d in diagnostics.values()),
        "initial_reason_enum_error_tasks": sum(d["initial_reason_enum_error"] for d in diagnostics.values()),
        "intent_field_correct": {
            key: sum(d["intent_field_correct"][key] for d in diagnostics.values()) for key in FIELDS
        },
        "intent_joint_correct": sum(all(d["intent_field_correct"].values()) for d in diagnostics.values()),
        "declared_accept_choice_tasks": sum(cases[r["case_id"]]["confirmation"] for r in records),
        "declared_decline_choice_tasks": sum(not cases[r["case_id"]]["confirmation"] for r in records),
        "accepted_confirmation_tasks": sum(
            any(event["kind"] == "operation_committed" for event in r["events"]) for r in records
        ),
        "declined_confirmation_tasks": sum(
            any(event["kind"] == "confirmation_declined" for event in r["events"]) for r in records
        ),
        "required_clarification_tasks": sum(bool(cases[r["case_id"]]["required_answers"]) for r in records),
        "required_clarification_fields": sum(len(cases[r["case_id"]]["required_answers"]) for r in records),
        "required_clarification_tasks_obtained": sum(
            bool(cases[r["case_id"]]["required_answers"]) and r["clarification_obtained"] for r in records
        ),
        "required_clarification_fields_obtained": sum(
            field in r["questions"] for r in records for field in cases[r["case_id"]]["required_answers"]
        ),
        "model_calls": sum(r["model_calls"] for r in records),
        "schema_valid": sum(r["schema_valid"] for r in records),
        "initial_records_valid": sum(r["initial_record_valid"] for r in records),
        "input_tokens": sum(o["timing"].get("input_tokens", 0) for r in records for o in r["trace"]),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "median_model_calls": statistics.median(r["model_calls"] for r in records) if records else None,
        "latency_p50_ms": statistics.median(r["latency_ms"] for r in records) if records else None,
    }
    for key in [
        "repeated_question_attempts",
        "repeated_questions_delivered",
        "identical_action_repeats",
        "unnecessary_handoff_attempts",
        "unnecessary_handoff_executed",
        "enum_error_attempts",
        "reason_enum_errors",
        "record_rejections",
        "quote_rejections",
        "slot_normalizations",
        "policy_gate_rejections",
        "confirmation_preserved",
        "exchange_as_return",
        "unauthorized_return",
    ]:
        result[key] = sum(r[key] for r in records)
    result.update(
        enum_error_tasks=sum(r["enum_error_attempts"] > 0 for r in records),
        enum_recovered_tasks=sum(r["enum_recovered"] for r in records),
        reason_enum_error_tasks=sum(r["reason_enum_errors"] > 0 for r in records),
        reason_enum_recovered_tasks=sum(r["reason_enum_recovered"] for r in records),
    )
    return result


def protocol_summary(rows):
    output = {}
    for arm in ARMS:
        subset = [row for row in rows if row["variant"] == arm]
        output[arm] = {
            "tasks": len(subset),
            "gold_exchange_tasks": sum(row["audited"] for row in subset),
            **{key: sum(row[key] for row in subset) for key in PROTOCOL_COUNTS},
            "rejected_return_attempts_without_scoped_yes": sum(
                attempt["raw_return_attempt"]
                and not attempt["scoped_yes_verified"]
                and not attempt["return_proposal_created"]
                and bool(attempt["backend_error"])
                for row in subset
                for attempt in row["attempts"]
            ),
        }
    return output


def paired(records):
    index = {(r["variant"], r["case_id"]): bool(r["task_success"]) for r in records}
    result = Counter({"both_pass": 0, "both_fail": 0, "improved": 0, "regressed": 0})
    for cid in {r["case_id"] for r in records}:
        before, after = (index[arm, cid] for arm in ARMS)
        result[
            "both_pass"
            if before and after
            else "both_fail"
            if not before and not after
            else "improved"
            if after
            else "regressed"
        ] += 1
    return dict(result)


def derived_values(report, cases, protocol):
    index = {(r["variant"], r["case_id"]): bool(r["task_success"]) for r in report["records"]}
    return {
        "summary": {
            arm: summarize([r for r in report["records"] if r["variant"] == arm], cases) for arm in ARMS
        },
        "protocol_summary": protocol_summary(protocol["records"]),
        "paired": paired(report["records"]),
        "family_success": {
            family: {
                arm: sum(
                    bool(r["task_success"])
                    for r in report["records"]
                    if r["variant"] == arm and r["family"] == family
                )
                for arm in ARMS
            }
            for family in sorted({case["family"] for case in cases.values()})
        },
        "failures": [
            {"case_id": cid, **{arm: index[arm, cid] for arm in ARMS}}
            for cid in sorted(cases)
            if not all(index[arm, cid] for arm in ARMS)
        ],
        "failed_records": [
            {
                "variant": r["variant"],
                "case_id": r["case_id"],
                "status": r["status_final"],
                "model_calls": r["model_calls"],
            }
            for r in report["records"]
            if not r["task_success"]
        ],
    }
