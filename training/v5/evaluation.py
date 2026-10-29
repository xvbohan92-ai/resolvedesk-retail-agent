"""Frozen before inference: same independent grader and total-call budget for four arms."""

import json
import time
from collections import Counter

from training.v2.scoring import score
from training.v4.evaluation import summarize as old_summary

ARMS = ["previous_old_flow", "previous_new_flow", "base_new_flow", "qlora_v5"]


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
        "schema_valid": sum(o["action"] is not None for o in obs),
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

    initial = next(
        (
            o["result"]["resolution_request"]
            for o in obs
            if "resolution_request" in o["result"] and o["result"]["source_message_count"] == 1
        ),
        None,
    )
    return {
        **base,
        "intent_initial": initial,
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
            bool(o["action"] and o["action"]["tool"] == "record_request") for o in obs
        ),
        "quote_rejections": sum("Evidence must exactly" in o["result"].get("detail", "") for o in obs),
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
    ]:
        s[key] = sum(r[key] for r in rows)
    intents = (
        [r for r in rows if r["variant"] != "previous_old_flow"] if rows and "variant" in rows[0] else rows
    )
    s["intent_tasks"] = len(intents)
    s["intent_joint_correct"] = sum(all(r["intent_field_correct"].values()) for r in intents)
    s["intent_field_correct"] = (
        {k: sum(r["intent_field_correct"][k] for r in intents) for k in rows[0]["intent_field_correct"]}
        if rows
        else {}
    )
    return s


def selection_key(summary):
    return (
        int(summary["exchange_as_return"] == summary["unauthorized_return"] == 0),
        min(
            summary[k] / summary[d]
            for k, d in [
                ("fulfillable_exchange_success", "fulfillable_exchange_tasks"),
                ("fallback_success", "fallback_tasks"),
                ("handoff_success", "handoff_tasks"),
            ]
        ),
        summary["task_success"],
        -summary["unnecessary_handoff_executed"],
        -summary["unnecessary_handoff_attempts"],
        -summary["repeated_question_attempts"],
        -summary["model_calls"],
    )


def promotion(records):
    s = {v: summarize([r for r in records if r["variant"] == v]) for v in ARMS}
    candidate = s["qlora_v5"]
    checks = {
        "all_stockout_fallback_correct": candidate["fallback_success"] == candidate["fallback_tasks"],
        "at_least_8_of_10_available_exchanges": candidate["fulfillable_exchange_success"] >= 8,
        "at_least_7_of_8_required_handoffs": candidate["handoff_success"] >= 7,
        "at_most_one_unnecessary_handoff": candidate["unnecessary_handoff_executed"] <= 1,
        "zero_repeated_field_attempts": candidate["repeated_question_attempts"] == 0,
        "zero_exchange_as_return": candidate["exchange_as_return"] == 0,
        "zero_unauthorized_return_attempts": candidate["unauthorized_return"] == 0,
        "confirmation_preserved": candidate["confirmation_preserved"] == 32,
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
