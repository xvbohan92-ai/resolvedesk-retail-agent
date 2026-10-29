"""Frozen before inference: same independent grader and total-call budget for four arms."""

import json
import time
from collections import Counter

from training.v4.evaluation import summarize as old_summary
from training.v7.scoring import score

ARMS = ["previous_old_flow", "previous_new_flow", "base_new_flow", "qlora_v7"]


REQUEST_FIELDS = (
    "operation",
    "email",
    "order_id",
    "requested_option",
    "payment_method",
    "reason",
    "fallback_return",
    "evidence",
)
ENUMS = {
    "operation": {"cancel", "return", "exchange", "handoff", "unknown"},
    "payment_method": {"original", "gift_card", "unknown"},
    "reason": {"no longer needed", "ordered by mistake", "unknown"},
    "fallback_return": {"yes", "no", "unknown"},
}


def raw_action(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def enum_errors(action):
    if not action or action.get("tool") != "record_request" or not isinstance(action.get("arguments"), dict):
        return {}
    args = action["arguments"]
    return {
        k: args[k]
        for k, choices in ENUMS.items()
        if k in args and (not isinstance(args[k], str) or args[k] not in choices)
    }


def run_case(store, case, provider, prompt_fn, max_steps=12):
    state, questions = store.create(case), []
    started = time.perf_counter()
    for _ in range(max_steps):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
        if state["status"] != "running":
            break
        raw, timing = provider.generate(prompt_fn(state))
        state = store.apply(state["id"], raw, timing, state["revision"])
    before, proposal = store.orders(state["id"]), state.get("proposal")
    status_before = state["status"]
    if status_before == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = store.get(state["id"])
    after = store.orders(state["id"])
    judged = score(case, {**state, "proposal": proposal}, after, before, questions)
    obs = state["observations"]
    proposals = [o["result"]["proposal"] for o in obs if "proposal" in o["result"]]
    fields = [
        o["action"]["arguments"]["field"]
        for o in obs
        if o["action"] and o["action"]["tool"] == "ask_customer"
    ]
    delivered = [
        o["result"]["field"] for o in obs if "question" in o["result"] and "error" not in o["result"]
    ]
    attempted_return = any(o["action"] and o["action"]["tool"] == "propose_return" for o in obs)
    seen, duplicate_attempts, duplicate_executions = set(), 0, 0
    for o in obs:
        a, r = o["action"], o["result"]
        if a and a["tool"] in {"identify_customer", "get_order", "get_product"}:
            k = json.dumps(a, sort_keys=True)
            if k in seen:
                duplicate_attempts += 1
                duplicate_executions += "error" not in r
            if "error" not in r:
                seen.add(k)
    base = {
        "case_id": case["id"],
        "family": case["family"],
        "expected": case["expected"],
        "confirmation": case["confirmation"],
        **judged,
        "status_before_confirmation": status_before,
        "status_final": state["status"],
        "model_calls": len(obs),
        "schema_valid": sum(o["action"] is not None and not enum_errors(raw_action(o["raw"])) for o in obs),
        "rejected_write_attempts": sum(
            bool(o["action"] and o["action"]["tool"].startswith("propose_") and "error" in o["result"])
            for o in obs
        ),
        "duplicate_read_attempts": duplicate_attempts,
        "duplicate_read_executions": duplicate_executions,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "output_tokens": sum(o["timing"].get("output_tokens", 0) or 0 for o in obs),
        "questions": questions,
        "trace": obs,
        "order_before_confirmation": before,
        "order_after_confirmation": after,
        "events": store.timeline(state["id"]),
        "exchange_as_return": bool(case["expected"]["kind"] == "exchange" and attempted_return),
        "unauthorized_return": bool(
            case["family"].startswith("exchange")
            and case["expected"]["kind"] != "return"
            and attempted_return
        ),
        "direct_return": case["expected"]["kind"] == "return" and not case["family"].startswith("exchange"),
        "predicted_outcome": proposals[-1]["kind"]
        if proposals
        else ("handoff" if state["status"] == "handoff" else "unfinished"),
        "repeated_question_attempts": sum(n - 1 for n in Counter(fields).values()),
        "repeated_questions_delivered": sum(n - 1 for n in Counter(delivered).values()),
        "identical_action_repeats": sum(
            n - 1
            for n in Counter(json.dumps(o["action"], sort_keys=True) for o in obs if o["action"]).values()
        ),
        "model_handoff": any(o["action"] and o["action"]["tool"] == "handoff" for o in obs),
    }

    first = raw_action(obs[0]["raw"]) if obs else None
    args = first.get("arguments") if first else None
    initial = (
        {k: v for k, v in args.items() if k != "evidence"}
        if first
        and first.get("tool") == "record_request"
        and isinstance(args, dict)
        and set(args) == set(REQUEST_FIELDS)
        and all(isinstance(v, str) for v in args.values())
        else None
    )
    initial_errors = enum_errors(first)
    enum_attempts = sum(bool(enum_errors(raw_action(o["raw"]))) for o in obs)
    reason_attempts = sum("reason" in enum_errors(raw_action(o["raw"])) for o in obs)
    recovered, reason_recovered, pending, pending_reason = False, False, False, False
    for event in store.timeline(state["id"]):
        if event["kind"] == "customer_message":
            pending, pending_reason = False, False
        elif event["kind"] == "model_tool":
            o = event["detail"]
            errors = enum_errors(raw_action(o["raw"]))
            pending = pending or bool(errors)
            pending_reason = pending_reason or "reason" in errors
            if "resolution_request" in o["result"] and not errors:
                recovered = recovered or pending
                reason_recovered = reason_recovered or pending_reason
                pending, pending_reason = False, False
    return {
        **base,
        "intent_initial": initial,
        "initial_raw_parse_failure": initial is None,
        "initial_parse_missing": initial is None,
        "initial_enum_errors": initial_errors,
        "initial_record_valid": bool(obs and "resolution_request" in obs[0]["result"]),
        "intent_enum_valid": initial is not None and not initial_errors,
        "enum_error_attempts": enum_attempts,
        "enum_recovered": recovered,
        "reason_enum_errors": reason_attempts,
        "reason_enum_recovered": reason_recovered,
        "record_rejections": sum(
            bool(
                raw_action(o["raw"])
                and raw_action(o["raw"]).get("tool") == "record_request"
                and "error" in o["result"]
            )
            for o in obs
        ),
        "intent_field_correct": {
            k: bool(initial and initial[k] == v) for k, v in case["intent_initial"].items()
        },
        "unnecessary_handoff_attempts": sum(
            bool(o["action"] and o["action"]["tool"] == "handoff" and case["expected"]["kind"] != "handoff")
            for o in obs
        ),
        "unnecessary_handoff_executed": int(
            case["expected"]["kind"] != "handoff" and any(o["result"].get("handoff") for o in obs)
        ),
        "policy_gate_rejections": sum(o["result"].get("error") == "resolution_gate_rejection" for o in obs),
        "unrequested_substitute_attempts": sum(
            bool(
                o["action"]
                and o["action"]["tool"] == "propose_exchange"
                and case["family"].startswith("exchange")
                and o["action"]["arguments"]["new_item_id"] != case["expected"]["new_item_id"]
            )
            for o in obs
        ),
        "intent_record_attempts": sum(
            bool(raw_action(o["raw"]) and raw_action(o["raw"]).get("tool") == "record_request") for o in obs
        ),
        "quote_rejections": sum("Evidence must exactly" in o["result"].get("detail", "") for o in obs),
        "missing_email_case": "email" in case["required_answers"],
        "permission_authorized": case["intent_initial"]["fallback_return"] == "yes",
        "permission_yes": bool(initial and initial["fallback_return"] == "yes"),
        "consent_questions": sum(
            o["result"].get("field") == "fallback_return" and "question" in o["result"] for o in obs
        ),
        "slot_normalizations": sum(len(o["result"].get("slot_normalizations", [])) for o in obs),
    }


def summarize(rows):
    s = old_summary(rows)
    for key in [
        "unnecessary_handoff_attempts",
        "unnecessary_handoff_executed",
        "policy_gate_rejections",
        "unrequested_substitute_attempts",
        "intent_record_attempts",
        "quote_rejections",
        "slot_normalizations",
        "enum_error_attempts",
        "record_rejections",
        "reason_enum_errors",
    ]:
        s[key] = sum(r[key] for r in rows)
    intents = rows if rows and "variant" in rows[0] else rows
    s["intent_tasks"] = len(intents)
    s["intent_joint_correct"] = sum(all(r["intent_field_correct"].values()) for r in intents)
    s["intent_field_correct"] = (
        {k: sum(r["intent_field_correct"][k] for r in intents) for k in rows[0]["intent_field_correct"]}
        if rows
        else {}
    )
    fallback = [r for r in rows if r["expected"]["kind"] == "return" and r["family"].startswith("exchange")]
    email = [r for r in rows if r["missing_email_case"]]
    s.update(
        fallback_tasks=len(fallback),
        fallback_success=sum(r["task_success"] for r in fallback),
        missing_email_tasks=len(email),
        missing_email_success=sum(r["task_success"] for r in email),
        unsupported_permission_yes=sum(r["permission_yes"] and not r["permission_authorized"] for r in rows),
        non_authorized_tasks=sum(not r["permission_authorized"] for r in rows),
        true_permission_yes=sum(r["permission_yes"] and r["permission_authorized"] for r in rows),
        authorized_tasks=sum(r["permission_authorized"] for r in rows),
        consent_questions=sum(r["consent_questions"] for r in rows),
    )
    s.update(
        initial_raw_parse_failures=sum(r["initial_raw_parse_failure"] for r in rows),
        initial_enum_error_tasks=sum(bool(r["initial_enum_errors"]) for r in rows),
        enum_error_tasks=sum(r["enum_error_attempts"] > 0 for r in rows),
        enum_recovered_tasks=sum(r["enum_recovered"] for r in rows),
        reason_enum_error_tasks=sum(r["reason_enum_errors"] > 0 for r in rows),
        reason_enum_recovered_tasks=sum(r["reason_enum_recovered"] for r in rows),
        initial_records_valid=sum(r["initial_record_valid"] for r in rows),
    )
    return s


def selection_key(summary):
    return (
        int(summary["exchange_as_return"] == summary["unauthorized_return"] == 0),
        -summary["unsupported_permission_yes"],
        summary["true_permission_yes"] / summary["authorized_tasks"],
        min(
            summary[k] / summary[d]
            for k, d in [
                ("fulfillable_exchange_success", "fulfillable_exchange_tasks"),
                ("fallback_success", "fallback_tasks"),
                ("handoff_success", "handoff_tasks"),
                ("missing_email_success", "missing_email_tasks"),
            ]
        ),
        summary["task_success"],
        -summary["reason_enum_errors"],
        -summary["initial_enum_error_tasks"],
        -summary["initial_raw_parse_failures"],
        -summary["unnecessary_handoff_executed"],
        -summary["unnecessary_handoff_attempts"],
        -summary["repeated_question_attempts"],
        -summary["model_calls"],
    )


def promotion(records):
    s = {v: summarize([r for r in records if r["variant"] == v]) for v in ARMS}
    candidate = s["qlora_v7"]
    checks = {
        "all_stockout_fallback_correct": candidate["fallback_success"] == candidate["fallback_tasks"],
        "all_14_available_exchanges": candidate["fulfillable_exchange_success"] == 14,
        "all_16_required_handoffs": candidate["handoff_success"] == 16,
        "all_8_missing_email_cases": candidate["missing_email_success"] == 8,
        "zero_unsupported_permission_yes": candidate["unsupported_permission_yes"] == 0,
        "all_12_explicit_grants_recognized": candidate["true_permission_yes"] == 12,
        "zero_initial_parse_failures": candidate["initial_raw_parse_failures"] == 0,
        "zero_initial_enum_errors": candidate["initial_enum_error_tasks"] == 0,
        "zero_reason_enum_errors": candidate["reason_enum_errors"] == 0,
        "zero_unnecessary_handoff": candidate["unnecessary_handoff_executed"] == 0,
        "zero_repeated_field_attempts": candidate["repeated_question_attempts"] == 0,
        "zero_exchange_as_return": candidate["exchange_as_return"] == 0,
        "zero_unauthorized_return_attempts": candidate["unauthorized_return"] == 0,
        "confirmation_preserved": candidate["confirmation_preserved"] == 48,
        "overall_not_below_new_flow_comparators": candidate["task_success"]
        >= max(s[v]["task_success"] for v in ["previous_new_flow", "base_new_flow"]),
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "scope": (
            "Predeclared absolute prototype targets plus relative overall guard; not production evidence."
        ),
    }
