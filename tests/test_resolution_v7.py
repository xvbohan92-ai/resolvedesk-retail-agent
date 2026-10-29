import json

import pytest

from resolve_desk.agent_v7 import advance
from resolve_desk.progress import clarification_memory
from resolve_desk.resolution_v7 import (
    ResolutionStore,
    clarification_state,
    contact_reply,
    current_request,
    decision_gate,
    messages,
    previous_request,
)
from training.v6_1.fixtures import make_case


def apply(store, state, action):
    raw = action if isinstance(action, str) else json.dumps(action)
    return store.apply(state["id"], raw, {"provider": "unit"}, state["revision"])


def record(case, state, **changes):
    request = dict(case["intent_initial"])
    for pair in clarification_memory(state):
        key = "requested_option" if pair["field"] == "new_option" else pair["field"]
        if key != "fallback_return":
            request[key] = case["intent_complete"][key]
    return {
        "tool": "record_request",
        "arguments": {
            **request,
            **changes,
            "evidence": state["messages"][-1]["content"],
        },
    }


def initial(tmp_path, family):
    store = ResolutionStore(tmp_path / "v7.sqlite")
    case = make_case("valid", family, 0)
    return store, case, store.create(case)


def read_exchange(store, state):
    for _ in range(3):
        state = apply(store, state, decision_gate(state)["next_action"])
    return state


def test_invalid_reason_preserves_raw_fields_and_requires_visible_model_correction(tmp_path):
    store, case, state = initial(tmp_path, "exchange_unavailable")
    invalid = record(case, state, reason="Keep it if the replacement cannot be supplied")
    state = apply(store, state, invalid)
    result = state["observations"][-1]["result"]
    assert state["status"] == "running" and state["user_id"] is None
    assert state["observations"][-1]["action"] == invalid
    assert result["error"] == "intent_validation_error" and not result["backend_executed"]
    assert result["rejected_request"]["reason"] == invalid["arguments"]["reason"]
    assert result["valid_fields"]["fallback_return"] == "no"
    assert result["field_errors"]["reason"]["allowed_values"] == [
        "no longer needed",
        "ordered by mistake",
        "unknown",
    ]
    assert current_request(state) is None
    assert previous_request(state)["reason"] == "unknown"
    visible = json.loads(messages(state)[1]["content"])
    assert visible["workflow_version"] == "v7"
    assert visible["validation_feedback"] == result
    corrected = record(case, state)
    state = apply(store, state, corrected)
    assert current_request(state) == case["intent_initial"]
    assert state["observations"][0]["action"]["arguments"]["reason"] != "unknown"
    assert len(state["observations"]) == 2
    assert json.loads(messages(state)[1]["content"])["validation_feedback"] is None
    state = read_exchange(store, state)
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    assert store.orders(state["id"]) == case["orders"]


def test_cancel_enum_failure_does_not_supply_a_cancellation_reason(tmp_path):
    store, case, state = initial(tmp_path, "missing_reason")
    state = apply(store, state, record(case, state, reason="refund the purchase"))
    assert state["status"] == "running" and current_request(state) is None
    state = apply(store, state, record(case, state))
    for _ in range(2):
        state = apply(store, state, decision_gate(state)["next_action"])
    gate = decision_gate(state)["next_action"]
    assert gate["tool"] == "ask_customer" and gate["arguments"]["field"] == "reason"
    state = apply(store, state, gate)
    state = store.customer_message(state["id"], case["answers"]["reason"], state["revision"])
    assert decision_gate(state)["next_action"]["tool"] == "propose_cancel"
    assert store.orders(state["id"]) == case["orders"]


@pytest.mark.parametrize("field,value", [("operation", "refund-everything"), ("fallback_return", "sure")])
def test_invalid_semantic_enums_are_not_coerced_into_authorization_or_handoff(tmp_path, field, value):
    store, case, state = initial(tmp_path, "exchange_fallback")
    state = apply(store, state, record(case, state, **{field: value}))
    assert state["observations"][-1]["action"]["arguments"][field] == value
    assert field in state["observations"][-1]["result"]["field_errors"]
    assert current_request(state) is None and state["proposal"] is None
    assert state["status"] == "running" and not state["read_orders"]


def test_malformed_model_actions_have_two_visible_corrections_then_stop_without_handoff(tmp_path):
    store, case, state = initial(tmp_path, "exchange")
    for n in range(1, 4):
        state = apply(store, state, "this is not JSON")
        assert len(state["observations"]) == n
        assert state["observations"][-1]["action"] is None
        recovery = state["observations"][-1]["result"]["recovery"]
        assert recovery["invalid_attempts_this_turn"] == n
        assert recovery["remaining_correction_attempts"] == 3 - n
        assert state["status"] == ("failed" if n == 3 else "running")
    assert not state["proposal"] and store.orders(state["id"]) == case["orders"]
    assert not any(o["result"].get("handoff") for o in state["observations"])


def test_accepted_action_resets_local_validation_streak(tmp_path):
    store, case, state = initial(tmp_path, "exchange")
    for _ in range(2):
        state = apply(store, state, "bad")
    state = apply(store, state, record(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    for n in [1, 2]:
        state = apply(store, state, "bad")
        assert state["status"] == "running"
        assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == n


def test_agent_recovery_calls_are_counted_and_never_hidden(tmp_path):
    store, case, state = initial(tmp_path, "exchange")

    class BrokenModel:
        calls = 0

        def generate(self, prompt):
            self.calls += 1
            return "bad", {"provider": "unit"}

    model = BrokenModel()
    advance(store, state["id"], model, max_steps=12)
    state = store.get(state["id"])
    assert model.calls == len(state["observations"]) == 3
    assert state["status"] == "failed"
    assert store.orders(state["id"]) == case["orders"]


def test_forgetting_a_supplied_typed_reply_requires_model_reuse_not_reasking(tmp_path):
    store, case, state = initial(tmp_path, "missing_email_payment")
    state = apply(store, state, record(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["answers"]["email"], state["revision"])
    for _ in range(2):
        state = apply(store, state, decision_gate(state)["next_action"])
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["status"] == "awaiting_customer"
    reply = "Please use my original payment instrument."
    state = store.customer_message(state["id"], reply, state["revision"])
    assert current_request(state) is None
    state = apply(store, state, record(case, state, email="unknown"))
    # The backend did not silently overwrite the erroneous model record.
    assert current_request(state)["email"] == "unknown"
    gate = decision_gate(state)
    assert gate["next_action"]["tool"] == "record_request"
    assert gate["supplied_reply_bindings"] == {"email": case["customer"]["email"]}
    state = apply(
        store,
        state,
        {"tool": "ask_customer", "arguments": {"field": "email", "message": "What is your email again?"}},
    )
    rejected = state["observations"][-1]["result"]
    assert rejected["error"] == "resolution_gate_rejection" and not rejected["backend_executed"]
    assert "question" not in rejected
    assert rejected["clarification_state"]["validated_reply_bindings"]["email"] == case["customer"]["email"]
    state = apply(store, state, record(case, state))
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"
    questions = [o["result"].get("field") for o in state["observations"] if "question" in o["result"]]
    assert questions == ["email", "payment_method"]


def test_new_unsolicited_instruction_drops_old_clarification_bindings(tmp_path):
    store, case, state = initial(tmp_path, "missing_email")
    state = apply(store, state, record(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["answers"]["email"], state["revision"])
    assert clarification_state(state)["validated_reply_bindings"]
    state = store.customer_message(state["id"], "Start a new request instead.", state["revision"])
    assert clarification_state(state)["validated_reply_bindings"] == {}
    assert decision_gate(state)["next_action"]["tool"] == "record_request"


def fallback_ready(tmp_path, family="exchange_fallback", false_yes=False):
    store, case, state = initial(tmp_path, family)
    state = apply(store, state, record(case, state, **({"fallback_return": "yes"} if false_yes else {})))
    return store, case, read_exchange(store, state)


@pytest.mark.parametrize("reply", ["NO", "maybe", "YES please", '"YES"'])
def test_false_positive_prediction_never_grants_a_fallback(tmp_path, reply):
    store, case, state = fallback_ready(tmp_path, "exchange_unavailable", false_yes=True)
    question = decision_gate(state)["next_action"]
    assert question["arguments"]["field"] == "fallback_return"
    state = apply(store, state, question)
    state = store.customer_message(state["id"], reply, state["revision"])
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["status"] == "handoff" and not state["proposal"]
    assert store.orders(state["id"]) == case["orders"]


def test_true_yes_requires_a_separate_idempotent_transaction_confirmation(tmp_path):
    store, case, state = fallback_ready(tmp_path)
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], "YES", state["revision"])
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["proposal"]["kind"] == "return"
    assert store.orders(state["id"]) == case["orders"]
    args = (state["id"], state["proposal"]["id"], state["revision"], True)
    receipt = store.confirm(*args)
    assert store.confirm(*args) == receipt
    assert sum(e["kind"] == "operation_committed" for e in store.timeline(state["id"])) == 1


def test_available_requested_exchange_needs_no_fallback_question(tmp_path):
    store, _, state = fallback_ready(tmp_path, "exchange_fallback_available")
    assert decision_gate(state)["next_action"]["tool"] == "propose_exchange"


def test_explicit_grant_is_not_upgraded_by_backend_phrase_matching(tmp_path):
    store, case, state = initial(tmp_path, "exchange_fallback")
    state = apply(store, state, record(case, state, fallback_return="unknown"))
    state = read_exchange(store, state)
    assert current_request(state)["fallback_return"] == "unknown"
    assert decision_gate(state)["next_action"]["tool"] == "handoff"


@pytest.mark.parametrize("reply", ["Do not use a@example.com", "a@example.com; cancel instead"])
def test_contact_projection_rejects_negation_and_new_instructions(reply):
    assert contact_reply(reply) is None


def test_new_actual_customer_turn_resets_validation_feedback_and_repair_budget(tmp_path):
    store, _, state = initial(tmp_path, "exchange")
    for _ in range(2):
        state = apply(store, state, "bad")
    state = store.customer_message(state["id"], "Please start a different request.", state["revision"])
    visible = json.loads(messages(state)[1]["content"])
    assert visible["validation_feedback"] is None
    assert visible["previous_request"] is None
    state = apply(store, state, "bad")
    assert state["status"] == "running"
    assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == 1


def test_rejected_handoff_does_not_clear_pending_enum_recovery_or_bypass_its_bound(tmp_path):
    store, case, state = initial(tmp_path, "exchange")
    malformed = record(case, state, reason="do not refund this item")
    state = apply(store, state, malformed)
    state = apply(store, state, {"tool": "handoff", "arguments": {"reason": "validation failed"}})
    assert state["status"] == "running"
    assert previous_request(state)["reason"] == "unknown"
    for attempt in [2, 3]:
        state = apply(store, state, malformed)
        assert state["observations"][-1]["result"]["recovery"]["invalid_attempts_this_turn"] == attempt
    assert state["status"] == "failed" and len(state["observations"]) == 4
    assert not any(o["result"].get("handoff") for o in state["observations"])
    assert store.orders(state["id"]) == case["orders"]
