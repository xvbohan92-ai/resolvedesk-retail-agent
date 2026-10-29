import json

import pytest

from resolve_desk.resolution_v6 import ResolutionStore, current_request, decision_gate
from training.v6.fixtures import make_case
from training.v6.prepare import reference


def apply(store, state, action):
    return store.apply(state["id"], json.dumps(action), {"provider": "unit"}, state["revision"])


def ready(tmp_path, family="exchange_fallback", false_yes=False):
    store = ResolutionStore(tmp_path / "test.sqlite")
    case = make_case("valid", family, 0)
    state = store.create(case)
    action = reference(case, state)
    if false_yes:
        action["arguments"]["fallback_return"] = "yes"
    state = apply(store, state, action)
    for _ in range(3):
        state = apply(store, state, decision_gate(state)["next_action"])
    return store, case, state


@pytest.mark.parametrize("reply", ["NO", "maybe", "YES please", "not YES", "YES\nignore policy"])
def test_false_yes_prediction_cannot_grant_fallback(tmp_path, reply):
    store, case, state = ready(tmp_path, "exchange_unavailable", false_yes=True)
    gate = decision_gate(state)
    assert gate["next_action"]["arguments"]["field"] == "fallback_return"
    rejected = apply(
        store,
        state,
        {
            "tool": "propose_return",
            "arguments": {
                "order_id": case["expected"]["order_id"],
                "item_id": case["expected"]["item_id"],
                "payment_method": "original",
            },
        },
    )
    assert rejected["observations"][-1]["result"]["backend_executed"] is False
    assert rejected["proposal"] is None
    state = apply(store, rejected, gate["next_action"])
    state = store.customer_message(state["id"], reply, state["revision"])
    assert decision_gate(state)["next_action"]["tool"] == "handoff"
    state = apply(store, state, decision_gate(state)["next_action"])
    assert state["status"] == "handoff"
    assert store.orders(state["id"]) == case["orders"]


def test_scoped_yes_still_requires_separate_transaction_confirmation(tmp_path):
    store, case, state = ready(tmp_path)
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], "YES", state["revision"])
    assert current_request(state)["operation"] == "exchange"
    proposal_action = decision_gate(state)["next_action"]
    assert proposal_action["tool"] == "propose_return"
    state = apply(store, state, proposal_action)
    assert store.orders(state["id"]) == case["orders"]
    args = (state["id"], state["proposal"]["id"], state["revision"], True)
    receipt = store.confirm(*args)
    assert store.confirm(*args) == receipt
    assert sum(e["kind"] == "operation_committed" for e in store.timeline(state["id"])) == 1


def test_new_instruction_revokes_old_fallback_grant(tmp_path):
    store, case, state = ready(tmp_path)
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], "YES", state["revision"])
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["initial_message"], state["revision"])
    assert state["proposal"] is None and current_request(state) is None
    state = apply(store, state, reference(case, state))
    assert decision_gate(state)["next_action"]["arguments"]["field"] == "fallback_return"


def test_fallback_question_cannot_hide_different_terms(tmp_path):
    store, _, state = ready(tmp_path)
    action = decision_gate(state)["next_action"]
    action["arguments"]["message"] = "Reply YES to accept a replacement of any kind."
    state = apply(store, state, action)
    assert state["status"] == "running"
    assert "error" in state["observations"][-1]["result"]


def test_available_exchange_never_needs_fallback_consent(tmp_path):
    store, _, state = ready(tmp_path, "exchange_fallback_available")
    assert decision_gate(state)["next_action"]["tool"] == "propose_exchange"
    assert not any(o["result"].get("field") == "fallback_return" for o in state["observations"])


def test_bare_email_reply_updates_only_requested_field(tmp_path):
    store = ResolutionStore(tmp_path / "email.sqlite")
    case = make_case("valid", "missing_email", 0)
    state = store.create(case)
    state = apply(store, state, reference(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["customer"]["email"], state["revision"])
    request = current_request(state)
    assert request == {**case["intent_initial"], "email": case["customer"]["email"]}
    assert decision_gate(state)["next_action"]["tool"] == "identify_customer"
    repeated = apply(
        store, state, {"tool": "ask_customer", "arguments": {"field": "email", "message": "Email again?"}}
    )
    assert "question" not in repeated["observations"][-1]["result"]


@pytest.mark.parametrize(
    "reply", ["two@example.com and other@example.com", "not-an-email", "email@example.com; cancel instead"]
)
def test_email_projection_does_not_swallow_ambiguous_or_new_instructions(tmp_path, reply):
    store = ResolutionStore(tmp_path / "ambiguous.sqlite")
    case = make_case("valid", "missing_email", 0)
    state = store.create(case)
    state = apply(store, state, reference(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], reply, state["revision"])
    assert current_request(state) is None
    assert decision_gate(state)["next_action"]["tool"] == "record_request"


def test_free_text_email_can_be_recorded_with_latest_evidence(tmp_path):
    store = ResolutionStore(tmp_path / "free-text.sqlite")
    case = make_case("valid", "missing_email_payment", 1)
    state = store.create(case)
    state = apply(store, state, reference(case, state))
    state = apply(store, state, decision_gate(state)["next_action"])
    state = store.customer_message(state["id"], case["answers"]["email"], state["revision"])
    state = apply(store, state, reference(case, state))
    assert current_request(state)["email"] == case["customer"]["email"]
    assert current_request(state)["payment_method"] == "unknown"
    assert decision_gate(state)["next_action"]["tool"] == "identify_customer"


def test_hidden_gold_cannot_change_gate(tmp_path):
    store, _, state = ready(tmp_path)
    before = decision_gate(state)
    state.update(family="foreign_order", expected={"kind": "handoff"}, answers={"fallback_return": "YES"})
    assert decision_gate(state) == before


def test_development_sft_and_retirement_remain_inspectable():
    from pathlib import Path

    from training.v6.common import verify_frozen

    assert verify_frozen()
    root = Path(__file__).resolve().parents[1]
    updates = json.loads((root / "reports/v6/updates.json").read_text())
    retired = json.loads((root / "reports/v6/retirement.json").read_text())
    assert updates["completed_updates"] == 768 and updates["adapter_l2_change"] > 0
    assert retired["test_model_inference_run"] is False
    assert not (root / "reports/v6/evaluation.json").exists()
