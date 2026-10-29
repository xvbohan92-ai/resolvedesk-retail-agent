"""CPU reference traces verify receipt integrity; no model or experiment reports."""

import copy
import itertools
import json

import pytest

from resolve_desk.resolution_v6_1 import ResolutionStore as PreviousStore
from resolve_desk.resolution_v7 import ResolutionStore, decision_gate
from training.v7 import verify_results
from training.v7.fixtures import make_case
from training.v7.prepare import reference


@pytest.fixture(autouse=True)
def distinct_deterministic_ids(monkeypatch):
    # Guarantee different source/replay opaque IDs without depending on randomness.
    counter = itertools.count(1)
    monkeypatch.setattr("resolve_desk.store.secrets.token_hex", lambda n: format(next(counter), f"0{2 * n}x"))


def reference_record(tmp_path, *, accepted=True, previous=False, replace_proposal=False):
    case = make_case("valid", "exchange", 0)
    store = (PreviousStore if previous else ResolutionStore)(tmp_path / "reference.sqlite")
    state = store.create(case)

    def reach_proposal(state):
        for _ in range(12):
            if state["status"] == "awaiting_customer":
                field = state["observations"][-1]["result"]["field"]
                state = store.customer_message(state["id"], case["answers"][field], state["revision"])
            if state["status"] != "running":
                break
            action = reference(case, state)
            state = store.apply(
                state["id"], json.dumps(action), {"provider": "cpu_reference"}, state["revision"]
            )
        assert state["status"] == "awaiting_confirmation", decision_gate(state)
        return state

    state = reach_proposal(state)
    if replace_proposal:
        first = copy.deepcopy(state["proposal"])
        state = store.customer_message(
            state["id"], case["initial_message"] + " Please review this request again.", state["revision"]
        )
        state = reach_proposal(state)
        assert first["id"] != state["proposal"]["id"]
        assert first["fingerprint"] == state["proposal"]["fingerprint"]
    store.confirm(state["id"], state["proposal"]["id"], state["revision"], accepted)
    state = store.get(state["id"])
    return case, {
        "variant": "previous_old_flow" if previous else "qlora_v7",
        "events": store.timeline(state["id"]),
        "status_final": state["status"],
        "order_after_confirmation": store.orders(state["id"]),
    }


def confirmation(record):
    return next(
        e for e in reversed(record["events"]) if e["kind"] in {"operation_committed", "confirmation_declined"}
    )


def proposals(record):
    return [
        e["detail"]["result"]["proposal"]
        for e in record["events"]
        if e["kind"] == "model_tool" and "proposal" in e["detail"]["result"]
    ]


@pytest.mark.parametrize("previous", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_replay_maps_only_opaque_ids_and_keeps_record_and_business_fingerprints(tmp_path, previous, accepted):
    case, record = reference_record(tmp_path, accepted=accepted, previous=previous)
    original_case, original_record = copy.deepcopy(case), copy.deepcopy(record)
    verify_results.replay_actions(case, record)
    assert record == original_record and case == original_case
    assert proposals(record)[0]["fingerprint"]


@pytest.mark.parametrize("corruption", ["order", "amount", "operation"])
def test_confirmation_event_business_corruption_is_detected_even_with_correct_order_snapshot(
    tmp_path, corruption
):
    case, record = reference_record(tmp_path)
    payload = confirmation(record)["detail"]
    if corruption == "order":
        payload["order_id"] = "ORD-WRONG"
    elif corruption == "amount":
        payload["operation"]["amount_cents"] += 1
    else:
        payload["operation"]["kind"] = "return"
    unchanged = copy.deepcopy(record)
    with pytest.raises(AssertionError, match="Confirmation event payload differs"):
        verify_results.replay_actions(case, record)
    assert record == unchanged


def test_unseen_confirmation_proposal_id_is_rejected(tmp_path):
    case, record = reference_record(tmp_path)
    confirmation(record)["detail"]["proposal_id"] = "not-a-recorded-proposal"
    with pytest.raises(AssertionError):
        verify_results.replay_actions(case, record)


def test_previously_seen_but_revoked_proposal_cannot_confirm_current_proposal(tmp_path):
    case, record = reference_record(tmp_path, replace_proposal=True)
    old, current = proposals(record)
    assert confirmation(record)["detail"]["proposal_id"] == current["id"]
    confirmation(record)["detail"]["proposal_id"] = old["id"]
    with pytest.raises(AssertionError, match="different proposal"):
        verify_results.replay_actions(case, record)


def test_decline_event_must_reference_the_recorded_pending_proposal(tmp_path):
    case, record = reference_record(tmp_path, accepted=False)
    confirmation(record)["detail"]["proposal_id"] = "other-proposal"
    with pytest.raises(AssertionError):
        verify_results.replay_actions(case, record)


def test_business_fingerprint_is_checked_instead_of_discarded(tmp_path):
    case, record = reference_record(tmp_path)
    proposals(record)[0]["fingerprint"] = "0" * 64
    with pytest.raises(AssertionError):
        verify_results.replay_actions(case, record)


@pytest.mark.parametrize("field", ["amount_cents", "fingerprint"])
def test_committed_order_operation_fields_remain_strictly_compared(tmp_path, field):
    case, record = reference_record(tmp_path)
    operation = record["order_after_confirmation"][0]["operation"]
    operation[field] = operation[field] + 1 if field == "amount_cents" else "0" * 64
    with pytest.raises(AssertionError):
        verify_results.replay_actions(case, record)


@pytest.mark.parametrize("accepted", [False, True])
def test_confirm_return_value_is_verified_separately_from_its_recorded_event(tmp_path, monkeypatch, accepted):
    case, record = reference_record(tmp_path, accepted=accepted)

    class WrongReturnStore(ResolutionStore):
        def confirm(self, *args, **kwargs):
            receipt = copy.deepcopy(super().confirm(*args, **kwargs))
            if accepted:
                receipt["order_id"] = "ORD-WRONG"
            else:
                receipt["order_changed"] = True
            return receipt

    monkeypatch.setattr(verify_results, "ResolutionStore", WrongReturnStore)
    with pytest.raises(AssertionError):
        verify_results.replay_actions(case, record)
