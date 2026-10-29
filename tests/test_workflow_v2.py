import json

from resolve_desk.store import Store
from resolve_desk.workflow import WorkflowStore, allowed_tools, messages
from training.fixtures import load_cases


def call(store, state, tool, **arguments):
    return store.apply(state["id"], json.dumps({"tool": tool, "arguments": arguments}), {}, state["revision"])


def test_duplicate_successful_lookup_is_not_executed_again_and_stops_after_feedback(tmp_path, monkeypatch):
    store = WorkflowStore(tmp_path / "retail.sqlite")
    case = load_cases("valid")[0]
    state = store.create(case)
    calls, original = [], Store.tool

    def counted(self, c, s, name, args):
        calls.append(name)
        return original(self, c, s, name, args)

    monkeypatch.setattr(Store, "tool", counted)
    for _ in range(3):
        state = call(store, state, "identify_customer", email=case["customer"]["email"])
    assert calls == ["identify_customer"]
    assert state["status"] == "handoff" and not state["proposal"]
    assert all(o["result"]["backend_executed"] is False for o in state["observations"][1:])
    assert store.orders(state["id"])[0]["status"] == case["orders"][0]["status"]


def test_refresh_of_changed_order_is_not_a_duplicate(tmp_path):
    store = WorkflowStore(tmp_path / "retail.sqlite")
    case = load_cases("valid")[0]
    state = call(store, store.create(case), "identify_customer", email=case["customer"]["email"])
    oid = case["orders"][0]["id"]
    state = call(store, state, "get_order", order_id=oid)
    with store.connect() as c:
        c.execute("UPDATE orders SET revision=revision+1 WHERE id=?", (state["order_keys"][oid],))
    state = call(store, state, "get_order", order_id=oid)
    assert state["observations"][-1]["result"]["revision"] == 1


def test_exchange_requires_catalog_and_exposes_all_candidates_without_selecting(tmp_path):
    store = WorkflowStore(tmp_path / "retail.sqlite")
    case = next(c for c in load_cases("valid") if c["family"] == "exchange_fallback")
    state = call(store, store.create(case), "identify_customer", email=case["customer"]["email"])
    state = call(store, state, "get_order", order_id=case["orders"][0]["id"])
    assert "propose_exchange" not in allowed_tools(state)
    state = call(store, state, "get_product", product_id=case["orders"][0]["items"][0]["product_id"])
    candidates = state["observations"][-1]["result"]["exchange_candidates"]
    assert len(candidates) == 2 and any(not c["available"] for c in candidates)
    assert {c["price_difference_cents"] for c in candidates} == {500, -300}
    assert "propose_exchange" in allowed_tools(state) and state["proposal"] is None
    payload = json.loads(messages({**state, "expected": {"secret": "gold"}})[1]["content"])
    assert set(payload) == {"messages", "completed_reads", "recent_events", "allowed_tools"}
    assert "gold" not in json.dumps(payload)


def test_error_is_recoverable_instead_of_cached_as_success(tmp_path):
    store = WorkflowStore(tmp_path / "retail.sqlite")
    case = load_cases("valid")[0]
    state = call(store, store.create(case), "identify_customer", email="wrong@retail.example")
    state = call(store, state, "identify_customer", email=case["customer"]["email"])
    assert state["user_id"] == case["customer"]["id"] and state["status"] == "running"
