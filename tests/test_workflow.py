import json

import pytest

from resolve_desk.protocol import messages, parse_action
from resolve_desk.store import Conflict, Store
from training.fixtures import FAMILIES, make_case, next_reference
from training.scoring import score


def walk(store, case):
    state = store.create(case)
    for _ in range(10):
        if state["status"] == "awaiting_customer":
            field = state["observations"][-1]["result"]["field"]
            state = store.customer_message(state["id"], case["answers"][field], state["revision"])
            state["messages"][-1]["field"] = field
            with store.connect() as c:
                c.execute("UPDATE sessions SET body=? WHERE id=?", (json.dumps(state), state["id"]))
        if state["status"] != "running":
            return state
        state = store.apply(state["id"], json.dumps(next_reference(case, state)), {}, state["revision"])
    raise AssertionError("Demonstrator exceeded budget")


@pytest.mark.parametrize("family", FAMILIES)
def test_reference_can_complete_tasks_without_mutating_before_confirmation(tmp_path, family):
    store = Store(tmp_path / "db.sqlite")
    case = make_case("test", family, 0)
    state = walk(store, case)
    before = store.orders(state["id"])
    assert before == case["orders"]
    if state["proposal"]:
        store.confirm(state["id"], state["proposal"]["id"], state["revision"], True)
        state = store.get(state["id"])
    assert score(case, state, store.orders(state["id"]), before)["task_success"]


def test_confirmation_is_idempotent_and_atomic_even_if_response_is_lost(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    state = walk(store, make_case("valid", "return", 0))
    pid, sid, rev = state["proposal"]["id"], state["id"], state["revision"]
    first = store.confirm(sid, pid, rev, True)
    assert store.confirm(sid, pid, rev, True) == first
    assert sum(e["kind"] == "operation_committed" for e in store.timeline(sid)) == 1


def test_commit_failure_rolls_back_order_proposal_receipt_and_audit(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    case = make_case("valid", "cancel", 0)
    state = walk(store, case)

    def fault():
        raise RuntimeError("Injected failure after order write")

    with pytest.raises(RuntimeError):
        store.confirm(state["id"], state["proposal"]["id"], state["revision"], True, fault)
    assert store.orders(state["id"]) == case["orders"]
    assert store.get(state["id"])["status"] == "awaiting_confirmation"
    assert not any(e["kind"] == "operation_committed" for e in store.timeline(state["id"]))
    assert (
        store.confirm(state["id"], state["proposal"]["id"], state["revision"], True)["status"] == "committed"
    )


def test_decline_and_changed_intent_revoke_old_proposal(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    case = make_case("valid", "cancel", 0)
    state = walk(store, case)
    pid = state["proposal"]["id"]
    state = store.customer_message(state["id"], "Wait, keep my order.", state["revision"])
    with pytest.raises(Conflict):
        store.confirm(state["id"], pid, state["revision"], True)
    assert store.orders(state["id"]) == case["orders"]


def test_stale_order_and_cross_session_confirmation_are_rejected(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    case = make_case("valid", "exchange", 0)
    state = walk(store, case)
    another = store.create(case)
    with pytest.raises(Conflict):
        store.confirm(another["id"], state["proposal"]["id"], another["revision"], True)
    with store.connect() as c:
        c.execute(
            "UPDATE orders SET revision=revision+1 WHERE id=?",
            (state["order_keys"][case["orders"][0]["id"]],),
        )
    with pytest.raises(Conflict):
        store.confirm(state["id"], state["proposal"]["id"], state["revision"], True)
    assert store.orders(state["id"]) == case["orders"]


def test_stale_generated_action_and_forged_confirmation_tool_do_not_execute(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    case = make_case("valid", "return", 0)
    state = store.create(case)
    store.customer_message(state["id"], "One more detail.", state["revision"])
    with pytest.raises(Conflict):
        store.apply(state["id"], json.dumps(next_reference(case, state)), {}, state["revision"])
    current = store.get(state["id"])
    current = store.apply(state["id"], '{"tool":"commit_operation","arguments":{}}', {}, current["revision"])
    assert current["status"] == "failed"
    assert store.orders(state["id"]) == case["orders"]


@pytest.mark.parametrize(
    "text",
    [
        "```json\n{}\n```",
        '{"tool":"get_order","arguments":{"order_id":12}}',
        '{"tool":"get_order","arguments":{"order_id":"1","extra":"x"}}',
    ],
)
def test_strict_action_contract_has_no_repair_fallback(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_model_context_excludes_gold_fields_private_database_and_demo_tags(tmp_path):
    case = make_case("valid", "return", 0)
    state = Store(tmp_path / "db").create(case)
    state["messages"][0]["field"] = "hidden_tag"
    payload = json.loads(messages(state)[1]["content"])
    assert set(payload) == {"messages", "observations"}
    assert "field" not in payload["messages"][0]
    assert "expected" not in payload and "orders" not in payload and "products" not in payload
