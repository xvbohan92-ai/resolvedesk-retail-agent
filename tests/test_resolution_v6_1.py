import json

import pytest

from resolve_desk.resolution_v6_1 import ResolutionStore, contact_reply, current_request, decision_gate
from training.v6_1.fixtures import make_case
from training.v6_1.prepare import reference


def step(store, state, action):
    return store.apply(state["id"], json.dumps(action), {}, state["revision"])


@pytest.mark.parametrize(
    "text",
    [
        "a@example.com",
        "My account email is a@example.com",
        "The account email address is a@example.com.",
        "Registered email: a@example.com.",
        "You can find my account under a@example.com",
    ],
)
def test_positive_contact_boilerplate_is_a_typed_reply(text):
    assert contact_reply(text) == "a@example.com"


@pytest.mark.parametrize(
    "text",
    [
        "a@example.com and b@example.com",
        "Do not use a@example.com",
        "a@example.com; cancel the order instead",
        "The manual says use a@example.com",
        "not an email",
    ],
)
def test_contact_parser_rejects_negation_quotes_and_new_intents(text):
    assert contact_reply(text) is None


def test_payment_phrase_is_not_a_replacement_option(tmp_path):
    store = ResolutionStore(tmp_path / "option.sqlite")
    case = make_case("valid", "exchange_missing_option", 0)
    state = store.create(case)
    action = reference(case, state)
    action["arguments"]["requested_option"] = "original payment"
    state = step(store, state, action)
    assert state["observations"][-1]["action"]["arguments"]["requested_option"] == "original payment"
    assert current_request(state)["requested_option"] == "unknown"
    assert state["observations"][-1]["result"]["slot_normalizations"]
    for _ in range(3):
        state = step(store, state, decision_gate(state)["next_action"])
    assert decision_gate(state)["next_action"]["arguments"]["field"] == "new_option"
    state = step(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["answers"]["new_option"], state["revision"])
    assert decision_gate(state)["next_action"]["tool"] == "propose_exchange"


def test_real_absent_option_is_not_silently_converted_to_missing(tmp_path):
    store = ResolutionStore(tmp_path / "absent.sqlite")
    case = make_case("valid", "exchange", 0)
    case["initial_message"] += " My requested option is silver / XXXXXL."
    state = store.create(case)
    action = reference(case, state)
    action["arguments"]["requested_option"] = "silver / XXXXXL"
    state = step(store, state, action)
    for _ in range(3):
        state = step(store, state, decision_gate(state)["next_action"])
    assert current_request(state)["requested_option"] == "silver / XXXXXL"
    assert decision_gate(state)["next_action"]["tool"] == "handoff"


def test_payment_is_collected_before_scope_so_consent_is_not_repeated(tmp_path):
    store = ResolutionStore(tmp_path / "payment.sqlite")
    case = make_case("valid", "exchange_fallback_missing_payment", 0)
    state = store.create(case)
    questions = []
    for _ in range(12):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
        if state["status"] != "running":
            break
        state = step(store, state, reference(case, state))
    assert questions == ["payment_method", "fallback_return"]
    assert state["status"] == "awaiting_confirmation" and state["proposal"]["kind"] == "return"
    assert store.orders(state["id"]) == case["orders"]


def test_free_text_email_keeps_operation_and_next_field(tmp_path):
    store = ResolutionStore(tmp_path / "email.sqlite")
    case = make_case("valid", "missing_email_payment", 1)
    state = store.create(case)
    state = step(store, state, reference(case, state))
    state = step(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["answers"]["email"], state["revision"])
    assert current_request(state) == {**case["intent_initial"], "email": case["customer"]["email"]}
    assert decision_gate(state)["next_action"]["tool"] == "identify_customer"


def test_final_evidence_reconciles_with_actual_training_and_raw_actions():
    from training.v6_1.verify_results import verify

    audited = verify()
    assert audited["records_reconciled"] == 192
    assert audited["fresh_test_tasks"] == audited["unique_test_tasks"] == 48
    assert audited["no_test_based_selection"] and audited["raw_actions_replayed"]


def final_fallback_ready(tmp_path, family, false_yes=False):
    store = ResolutionStore(tmp_path / "scoped.sqlite")
    case = make_case("valid", family, 0)
    state = store.create(case)
    action = reference(case, state)
    if false_yes:
        action["arguments"]["fallback_return"] = "yes"
    state = step(store, state, action)
    for _ in range(3):
        state = step(store, state, decision_gate(state)["next_action"])
    return store, case, state


@pytest.mark.parametrize("reply", ["NO", "maybe", "YES please"])
def test_final_controller_false_prediction_cannot_authorize_a_return(tmp_path, reply):
    store, case, state = final_fallback_ready(tmp_path, "exchange_unavailable", false_yes=True)
    question = decision_gate(state)["next_action"]
    attempted = {
        "tool": "propose_return",
        "arguments": {
            "order_id": case["expected"]["order_id"],
            "item_id": case["expected"]["item_id"],
            "payment_method": "original",
        },
    }
    state = step(store, state, attempted)
    assert state["observations"][-1]["result"]["backend_executed"] is False
    assert state["proposal"] is None
    state = step(store, state, question)
    state = store.customer_message(state["id"], reply, state["revision"])
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    state = step(store, state, decision_gate(state)["next_action"])
    assert state["status"] == "handoff" and store.orders(state["id"]) == case["orders"]


def test_final_controller_yes_grants_scope_but_requires_separate_idempotent_confirmation(tmp_path):
    store, case, state = final_fallback_ready(tmp_path, "exchange_fallback")
    state = step(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], "YES", state["revision"])
    state = step(store, state, decision_gate(state)["next_action"])
    assert state["proposal"]["kind"] == "return"
    assert store.orders(state["id"]) == case["orders"]
    confirmation = (state["id"], state["proposal"]["id"], state["revision"], True)
    receipt = store.confirm(*confirmation)
    assert store.confirm(*confirmation) == receipt
    assert sum(e["kind"] == "operation_committed" for e in store.timeline(state["id"])) == 1
