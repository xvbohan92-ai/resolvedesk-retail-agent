import json
import statistics
import time
from collections import Counter

from resolve_desk.progress import messages
from resolve_desk.workflow import messages as old_messages
from training.v2.scoring import score
from training.v3.evaluation import summarize as previous_summary

ARMS = ["previous_old_flow", "previous_new_flow", "base_new_flow", "qlora_v4"]


def run_case(store, case, provider, new_flow=True, max_steps=8):
    state, questions = store.create(case), []
    started = time.perf_counter()
    for _ in range(max_steps):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
        if state["status"] != "running":
            break
        raw, timing = provider.generate(messages(state) if new_flow else old_messages(state))
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
    return {
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


def summarize(rows):
    fallback = [r for r in rows if r["family"] == "exchange_fallback"]
    handoffs = [r for r in rows if r["expected"]["kind"] == "handoff"]
    return {
        **previous_summary(rows),
        "fallback_tasks": len(fallback),
        "fallback_success": sum(r["task_success"] for r in fallback),
        "handoff_tasks": len(handoffs),
        "handoff_success": sum(r["task_success"] and r["model_handoff"] for r in handoffs),
        "repeat_free_success": sum(r["task_success"] and not r["repeated_question_attempts"] for r in rows),
        "repeated_question_attempts": sum(r["repeated_question_attempts"] for r in rows),
        "repeated_questions_delivered": sum(r["repeated_questions_delivered"] for r in rows),
        "identical_action_repeats": sum(r["identical_action_repeats"] for r in rows),
        "median_model_calls": statistics.median(r["model_calls"] for r in rows),
    }


def selection_key(summary, loss):
    minimum = min(
        summary[k] / summary[d]
        for k, d in [
            ("fulfillable_exchange_success", "fulfillable_exchange_tasks"),
            ("fallback_success", "fallback_tasks"),
            ("handoff_success", "handoff_tasks"),
        ]
    )
    return (
        int(summary["exchange_as_return"] == 0),
        minimum,
        summary["task_success"],
        -summary["repeated_question_attempts"],
        -loss,
    )


def promotion(records):
    summaries = {v: summarize([r for r in records if r["variant"] == v]) for v in ARMS}
    old, base, new = [summaries[v] for v in ["previous_new_flow", "base_new_flow", "qlora_v4"]]
    checks = {
        k: new[k] >= max(old[k], base[k])
        for k in ["task_success", "fulfillable_exchange_success", "fallback_success", "handoff_success"]
    }
    checks.update(
        zero_exchange_as_return=new["exchange_as_return"] == 0,
        repeated_questions_not_increased=new["repeated_question_attempts"]
        <= old["repeated_question_attempts"],
        repeated_questions_reduced_if_present=not old["repeated_question_attempts"]
        or new["repeated_question_attempts"] < old["repeated_question_attempts"],
        confirmation_preserved=new["confirmation_preserved"] == new["tasks"],
    )
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "scope": "Fixed local prototype rule on correlated synthetic tasks, not production certification.",
    }
