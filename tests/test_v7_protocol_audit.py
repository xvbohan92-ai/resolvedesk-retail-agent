"""CPU reference traces expose protocol failures without changing frozen scores."""

import copy
import json

import pytest

from resolve_desk.resolution_v6_1 import ResolutionStore as PreviousStore
from resolve_desk.resolution_v7 import ResolutionStore, decision_gate
from training.v7.fixtures import make_case
from training.v7.prepare import reference
from training.v7.protocol_audit import audit_record
from training.v7.scoring import score


def reference_trace(
    tmp_path,
    *,
    previous=False,
    switch_operation=False,
    early_attempt=False,
    reply_no=False,
    reset_after_grant=False,
):
    case = make_case("valid", "exchange_fallback_missing_payment", 1)
    store = (PreviousStore if previous else ResolutionStore)(tmp_path / "reference.sqlite")
    state, questions = store.create(case), []
    injected = False

    def apply(state, action):
        return store.apply(state["id"], json.dumps(action), {"provider": "cpu_reference"}, state["revision"])

    def attempted_return(state):
        return apply(
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

    for _ in range(20):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            questions.append(field)
            reply = "NO" if reply_no and field == "fallback_return" else case["answers"][field]
            state = store.customer_message(state["id"], reply, state["revision"])
            if field == "payment_method" and switch_operation:
                changed = reference(case, state)
                assert changed["tool"] == "record_request"  # Noncanonical actual payment reply.
                changed["arguments"]["operation"] = "return"
                state = apply(state, changed)
            if field == "fallback_return" and reset_after_grant:
                state = store.customer_message(
                    state["id"],
                    case["initial_message"] + " Please review this request again.",
                    state["revision"],
                )
                changed = reference(case, state)
                changed["arguments"]["operation"] = "return"
                changed["arguments"]["payment_method"] = "original"
                state = apply(state, changed)
            if field == "fallback_return" and reply_no:
                state = attempted_return(state)
        if state["status"] != "running":
            break
        gate = decision_gate(state)["next_action"]
        if (
            early_attempt
            and not injected
            and gate["tool"] == "ask_customer"
            and gate["arguments"]["field"] == "fallback_return"
        ):
            state = attempted_return(state)
            injected = True
        state = apply(state, reference(case, state))
    assert state["status"] in {"awaiting_confirmation", "handoff"}
    before, proposal = store.orders(state["id"]), state["proposal"]
    if state["status"] == "awaiting_confirmation":
        store.confirm(state["id"], proposal["id"], state["revision"], True)
        state = store.get(state["id"])
    after = store.orders(state["id"])
    business = score(case, {**state, "proposal": proposal}, after, before, questions)
    record = {
        "variant": "previous_old_flow" if previous else "qlora_v7",
        "case_id": case["id"],
        "events": store.timeline(state["id"]),
        "status_final": state["status"],
        "order_after_confirmation": after,
    }
    return case, record, business


@pytest.mark.parametrize("previous", [False, True])
def test_operation_switch_without_scoped_yes_is_visible_despite_frozen_business_success(tmp_path, previous):
    case, record, business = reference_trace(tmp_path, previous=previous, switch_operation=True)
    original = copy.deepcopy(record)
    result = audit_record(record, case)
    assert business["task_success"], "Demonstrate the frozen outcome grader's protocol blind spot"
    assert result["raw_return_attempts"] == result["return_attempts_without_scoped_yes"] == 1
    assert result["return_proposals_without_scoped_yes"] == 1
    assert result["committed_returns_without_scoped_yes"] == 1
    assert result["final_returns_without_verified_commit"] == 1
    assert "current_operation_bypasses_exchange_scope" in result["attempts"][0]["reasons"]
    assert result["attempts"][0]["matching_response"] is None
    assert record == original


@pytest.mark.parametrize("previous", [False, True])
def test_matching_actual_scoped_yes_is_verified_on_old_and_new_workflows(tmp_path, previous):
    case, record, business = reference_trace(tmp_path, previous=previous)
    result = audit_record(record, case)
    assert business["task_success"]
    assert result["raw_return_attempts"] == result["return_proposals"] == result["committed_returns"] == 1
    assert result["return_attempts_without_scoped_yes"] == result["return_proposals_without_scoped_yes"] == 0
    assert (
        result["committed_returns_without_scoped_yes"] == result["final_returns_without_verified_commit"] == 0
    )
    assert result["attempts"][0]["matching_response"]["reply"] == "YES"
    assert result["attempts"][0]["scoped_yes_verified"]


def test_rejected_model_attempt_is_separate_from_later_correctly_authorized_execution(tmp_path):
    case, record, business = reference_trace(tmp_path, early_attempt=True)
    result = audit_record(record, case)
    assert business["task_success"]
    assert result["raw_return_attempts"] == 2 and result["return_attempts_without_scoped_yes"] == 1
    assert result["return_proposals"] == result["committed_returns"] == 1
    assert (
        result["return_proposals_without_scoped_yes"] == result["committed_returns_without_scoped_yes"] == 0
    )
    assert result["attempts"][0]["backend_error"] == "resolution_gate_rejection"
    assert result["attempts"][1]["scoped_yes_verified"]


def test_actual_no_reply_never_counts_as_scoped_yes(tmp_path):
    case, record, _ = reference_trace(tmp_path, reply_no=True)
    result = audit_record(record, case)
    assert result["return_attempts_without_scoped_yes"] == 1
    assert result["return_proposals"] == result["committed_returns"] == 0
    assert "actual_scoped_reply_is_not_exact_yes" in result["attempts"][0]["reasons"]


def test_unsolicited_customer_turn_revokes_an_earlier_actual_yes_in_audit(tmp_path):
    case, record, business = reference_trace(tmp_path, reset_after_grant=True)
    result = audit_record(record, case)
    assert business["task_success"]
    assert any(e["kind"] == "customer_message" and e["detail"]["content"] == "YES" for e in record["events"])
    assert result["attempts"][0]["matching_response"] is None
    assert (
        result["return_proposals_without_scoped_yes"] == result["committed_returns_without_scoped_yes"] == 1
    )
