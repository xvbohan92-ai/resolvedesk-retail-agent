import copy
import json

import pytest

from resolve_desk.resolution import (
    ResolutionStore,
    current_request,
    decision_gate,
    messages,
    previous_request,
)
from resolve_desk.store import Conflict
from training.v5.evaluation import promotion, selection_key
from training.v5.fixtures import make_case
from training.v5.prepare import reference


def act(store, state, action):
    return store.apply(state["id"], json.dumps(action), {}, state["revision"])


def prepared(tmp_path, family):
    store = ResolutionStore(tmp_path / "db")
    case = make_case("valid", family, 0)
    state = store.create(case)
    while state["status"] == "running":
        action = reference(case, state)
        if action["tool"].startswith("propose_") or action["tool"] in {"handoff", "ask_customer"}:
            break
        state = act(store, state, action)
    return store, case, state


def test_available_exchange_with_fallback_does_not_turn_into_return_or_handoff(tmp_path):
    store, case, state = prepared(tmp_path, "exchange_fallback_available")
    gate = decision_gate(state)["next_action"]
    assert gate["tool"] == "propose_exchange"
    assert gate["arguments"]["new_item_id"] == case["expected"]["new_item_id"]
    for wrong in [
        {"tool": "handoff", "arguments": {"reason": "Cannot help."}},
        {
            "tool": "propose_return",
            "arguments": {
                "order_id": case["answers"]["order_id"],
                "item_id": case["answers"]["item_id"],
                "payment_method": "original",
            },
        },
    ]:
        state = act(store, state, wrong)
        assert state["status"] == "running"
        assert state["observations"][-1]["result"]["error"] == "resolution_gate_rejection"
    assert store.orders(state["id"]) == case["orders"]


def test_authorized_stockout_returns_original_item_without_substitution(tmp_path):
    store, case, state = prepared(tmp_path, "exchange_fallback")
    action = decision_gate(state)["next_action"]
    assert action["tool"] == "propose_return"
    state = act(store, state, action)
    assert state["proposal"]["amount_cents"] == case["orders"][0]["items"][0]["price_cents"]
    assert store.orders(state["id"]) == case["orders"]
    body = (state["id"], state["proposal"]["id"], state["revision"], True)
    first = store.confirm(*body)
    assert store.confirm(*body) == first
    assert len([e for e in store.timeline(state["id"]) if e["kind"] == "operation_committed"]) == 1


@pytest.mark.parametrize(
    "family", ["exchange_unavailable", "exchange_shortfall", "foreign_order", "forbidden_status"]
)
def test_true_policy_boundaries_allow_handoff_without_irrelevant_questions(tmp_path, family):
    store, case, state = prepared(tmp_path, family)
    action = decision_gate(state)["next_action"]
    assert action["tool"] == "handoff"
    state = act(store, state, action)
    assert state["status"] == "handoff" and store.orders(state["id"]) == case["orders"]
    assert not any(o["action"]["tool"] == "ask_customer" for o in state["observations"])


def test_missing_preference_is_not_inferred_from_original_payment_instrument(tmp_path):
    store, case, state = prepared(tmp_path, "missing_payment")
    ask = decision_gate(state)["next_action"]
    assert ask["tool"] == "ask_customer" and ask["arguments"]["field"] == "payment_method"
    state = act(store, state, ask)
    state = store.customer_message(state["id"], "original", state["revision"])
    assert current_request(state) is None and previous_request(state)["payment_method"] == "unknown"
    assert decision_gate(state)["next_action"]["tool"] == "record_request"
    # A duplicate attempt consumes a real step but is not delivered.
    state = act(store, state, ask)
    assert state["status"] == "running" and state["observations"][-1]["result"]["backend_executed"] is False
    state = act(store, state, reference(case, state))
    assert decision_gate(state)["next_action"]["tool"] == "propose_return"


def test_unrequested_in_stock_substitute_and_wrong_payment_are_blocked(tmp_path):
    store, case, state = prepared(tmp_path, "exchange_gift")
    action = decision_gate(state)["next_action"]
    assert action["arguments"]["payment_method"] == case["customer"]["gift_card"]["id"]
    wrong = copy.deepcopy(action)
    wrong["arguments"]["payment_method"] = "original"
    state = act(store, state, wrong)
    assert state["proposal"] is None
    wrong = copy.deepcopy(action)
    wrong["arguments"]["new_item_id"] = next(
        v["item_id"]
        for v in next(iter(case["products"].values()))["variants"]
        if v["item_id"] not in {case["answers"]["item_id"], case["expected"]["new_item_id"]}
    )
    state = act(store, state, wrong)
    assert state["proposal"] is None and store.orders(state["id"]) == case["orders"]


@pytest.mark.parametrize("corruption", ["quote", "email", "order"])
def test_recorded_intent_cannot_invent_customer_text_or_identifiers(tmp_path, corruption):
    store = ResolutionStore(tmp_path / "db")
    case = make_case("valid", "exchange", 0)
    state = store.create(case)
    action = reference(case, state)
    key, value = {
        "quote": ("evidence", "This exact sentence was never said."),
        "email": ("email", "invented@retail.example"),
        "order": ("order_id", "ORD-INVENTED"),
    }[corruption]
    action["arguments"][key] = value
    state = act(store, state, action)
    assert "error" in state["observations"][-1]["result"] and current_request(state) is None


def test_quote_provenance_is_not_misrepresented_as_semantic_consent(tmp_path):
    store = ResolutionStore(tmp_path / "db")
    case = make_case("valid", "exchange_unavailable", 0)
    state = store.create(case)
    action = reference(case, state)
    action["arguments"]["fallback_return"] = "yes"  # Deliberately wrong model interpretation.
    state = act(store, state, action)
    assert current_request(state)["fallback_return"] == "yes"
    assert "unverified" in state["observations"][-1]["result"]["provenance"]
    # Gold-independent controller cannot semantically certify a model's field.
    # End-to-end labels and explicit proposal review must expose this failure.
    assert case["intent_initial"]["fallback_return"] == "no"


def test_new_unsolicited_intent_revokes_proposal_and_does_not_reuse_old_slots(tmp_path):
    store, case, state = prepared(tmp_path, "exchange")
    state = act(store, state, reference(case, state))
    pid, rev = state["proposal"]["id"], state["revision"]
    state = store.customer_message(state["id"], "Actually return this item to original payment.", rev)
    assert current_request(state) is None and previous_request(state) is None
    assert state["proposal"] is None
    with pytest.raises(Conflict):
        store.confirm(state["id"], pid, rev, True)


def test_gold_metadata_never_changes_model_prompt_or_policy_action(tmp_path):
    store = ResolutionStore(tmp_path / "db")
    case = make_case("valid", "exchange_fallback", 0)
    altered = copy.deepcopy(case)
    altered.update(
        family="fake",
        expected={"kind": "handoff"},
        answers={"email": "SECRET"},
        intent_complete={"operation": "cancel"},
    )
    a, b = store.create(case), store.create(altered)
    assert messages(a) == messages(b)
    visible = json.loads(messages(a)[1]["content"])
    assert not {"expected", "answers", "family", "case_id", "intent_initial"} & visible.keys()


def test_validation_rule_prefers_balanced_business_outcomes_over_total_only():
    base = {
        "exchange_as_return": 0,
        "unauthorized_return": 0,
        "fulfillable_exchange_success": 10,
        "fulfillable_exchange_tasks": 10,
        "fallback_success": 0,
        "fallback_tasks": 2,
        "handoff_success": 8,
        "handoff_tasks": 8,
        "task_success": 28,
        "unnecessary_handoff_executed": 0,
        "unnecessary_handoff_attempts": 0,
        "repeated_question_attempts": 0,
        "model_calls": 160,
    }
    balanced = {**base, "fulfillable_exchange_success": 8, "fallback_success": 2, "task_success": 27}
    assert selection_key(balanced) > selection_key(base)


def test_promotion_cannot_pass_with_zero_fallback_even_when_all_baselines_fail(monkeypatch):
    from training.v5 import evaluation

    stats = {
        "fallback_success": 0,
        "fallback_tasks": 2,
        "fulfillable_exchange_success": 10,
        "handoff_success": 8,
        "unnecessary_handoff_executed": 0,
        "repeated_question_attempts": 0,
        "exchange_as_return": 0,
        "unauthorized_return": 0,
        "confirmation_preserved": 32,
        "task_success": 30,
    }
    monkeypatch.setattr(evaluation, "summarize", lambda _: stats)
    result = promotion([])
    assert not result["promoted"] and not result["checks"]["all_stockout_fallback_correct"]


def test_delivered_v5_results_reconcile():
    from training.v5.verify_results import verify

    assert verify()["records_reconciled"] == 128
