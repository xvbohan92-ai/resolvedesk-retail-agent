import json
import time

from resolve_desk.protocol import messages, parse_action
from training.fixtures import next_reference
from training.scoring import score


def reference_match(actual, expected):
    if not actual or actual["tool"] != expected["tool"]:
        return False
    if actual["tool"] == "handoff":
        return True
    if actual["tool"] == "ask_customer":
        return actual["arguments"]["field"] == expected["arguments"]["field"]
    return actual["arguments"] == expected["arguments"]


def run_case(store, case, provider, max_steps=8):
    state = store.create(case)
    decisions, questions = [], []
    started = time.perf_counter()
    for _ in range(max_steps):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
            state["messages"][-1]["field"] = field
            with store.connect() as c:
                c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
        if state["status"] != "running":
            break
        expected = next_reference(case, state)
        raw, timing = provider.generate(messages(state))
        try:
            actual = parse_action(raw)
        except ValueError:
            actual = None
        decisions.append(
            {
                "reference_next_action": expected,
                "actual": actual,
                "reference_match": reference_match(actual, expected),
            }
        )
        state = store.apply(state["id"], raw, timing, state["revision"])
    before = store.orders(state["id"])
    proposal = state.get("proposal")
    before_status = state["status"]
    if before_status == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], case["confirmation"])
        state = store.get(state["id"])
    grading_state = {**state, "proposal": proposal}
    after = store.orders(state["id"])
    judged = score(case, grading_state, after, before)
    observations = state["observations"]
    return {
        "case_id": case["id"],
        "family": case["family"],
        "expected": case["expected"],
        "confirmation": case["confirmation"],
        **judged,
        "status_before_confirmation": before_status,
        "status_final": state["status"],
        "model_calls": len(observations),
        "schema_valid": sum(o["action"] is not None for o in observations),
        "reference_next_action_matches": sum(d["reference_match"] for d in decisions),
        "rejected_write_attempts": sum(
            bool(o["action"] and o["action"]["tool"].startswith("propose_") and "error" in o["result"])
            for o in observations
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "output_tokens": sum(o["timing"].get("output_tokens", 0) or 0 for o in observations),
        "questions": questions,
        "decisions": decisions,
        "trace": observations,
        "order_before_confirmation": before,
        "order_after_confirmation": after,
        "events": store.timeline(state["id"]),
    }
