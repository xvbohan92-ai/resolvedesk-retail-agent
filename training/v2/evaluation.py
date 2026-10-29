import json
import statistics
import time

from resolve_desk.protocol import messages as legacy_messages
from resolve_desk.workflow import messages
from training.v2.scoring import score


def run_case(store, case, provider, workflow, max_steps=8):
    state, questions = store.create(case), []
    started = time.perf_counter()
    for _ in range(max_steps):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
        if state["status"] != "running":
            break
        raw, timing = provider.generate(messages(state) if workflow else legacy_messages(state))
        state = store.apply(state["id"], raw, timing, state["revision"])
    before = store.orders(state["id"])
    proposal, status_before = state.get("proposal"), state["status"]
    if status_before == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = store.get(state["id"])
    after = store.orders(state["id"])
    judged = score(case, {**state, "proposal": proposal}, after, before, questions)
    observations = state["observations"]
    seen, duplicate_attempts, duplicate_executions = set(), 0, 0
    for o in observations:
        a, r = o["action"], o["result"]
        if a and a["tool"] in {"identify_customer", "get_order", "get_product"}:
            key = json.dumps(a, sort_keys=True)
            if key in seen:
                duplicate_attempts += 1
                duplicate_executions += "error" not in r
            if "error" not in r:
                seen.add(key)
    return {
        "case_id": case["id"],
        "family": case["family"],
        "expected": case["expected"],
        "confirmation": case["confirmation"],
        **judged,
        "status_before_confirmation": status_before,
        "status_final": state["status"],
        "model_calls": len(observations),
        "schema_valid": sum(o["action"] is not None for o in observations),
        "rejected_write_attempts": sum(
            bool(o["action"] and o["action"]["tool"].startswith("propose_") and "error" in o["result"])
            for o in observations
        ),
        "duplicate_read_attempts": duplicate_attempts,
        "duplicate_read_executions": duplicate_executions,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "output_tokens": sum(o["timing"].get("output_tokens", 0) or 0 for o in observations),
        "questions": questions,
        "trace": observations,
        "order_before_confirmation": before,
        "order_after_confirmation": after,
        "events": store.timeline(state["id"]),
    }


def summarize(rows):
    required = [r for r in rows if r["required_clarification"]]
    exchange = [r for r in rows if r["family"].startswith("exchange")]
    fulfill = [r for r in rows if r["expected"]["kind"] == "exchange"]
    return {
        "tasks": len(rows),
        "task_success": sum(r["task_success"] for r in rows),
        "exchange_related_tasks": len(exchange),
        "exchange_related_success": sum(r["task_success"] for r in exchange),
        "fulfillable_exchange_tasks": len(fulfill),
        "fulfillable_exchange_success": sum(r["task_success"] for r in fulfill),
        "required_clarifications": len(required),
        "clarifications_obtained": sum(r["clarification_obtained"] for r in required),
        **{
            k: sum(r[k] for r in rows)
            for k in (
                "model_calls",
                "schema_valid",
                "duplicate_read_attempts",
                "duplicate_read_executions",
                "rejected_write_attempts",
                "output_tokens",
                "confirmation_preserved",
            )
        },
        "latency_p50_ms": statistics.median(r["latency_ms"] for r in rows),
        "median_model_calls": statistics.median(r["model_calls"] for r in rows),
    }


def promotion(rows):
    base = {r["case_id"]: r for r in rows if r["variant"] == "workflow_base"}
    fine = {r["case_id"]: r for r in rows if r["variant"] == "workflow_qlora_v2"}
    a, b = summarize(list(base.values())), summarize(list(fine.values()))
    regressions = [
        i
        for i in base
        if not base[i]["family"].startswith("exchange")
        and base[i]["task_success"]
        and not fine[i]["task_success"]
    ]
    checks = {
        "overall_improved": b["task_success"] > a["task_success"],
        "exchange_improved": b["exchange_related_success"] > a["exchange_related_success"],
        "confirmation_preserved": b["confirmation_preserved"] == b["tasks"],
        "no_nonexchange_paired_regressions": not regressions,
        "rejected_writes_not_increased": b["rejected_write_attempts"] <= a["rejected_write_attempts"],
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "nonexchange_regressions": regressions,
        "scope": "Predeclared conservative prototype rule; not a statistical or production readiness claim.",
    }
