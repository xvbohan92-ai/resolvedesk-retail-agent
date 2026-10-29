import json

import pytest

from resolve_desk.agent import advance
from resolve_desk.protocol import messages
from resolve_desk.providers import local_url
from resolve_desk.store import Store
from training.fixtures import make_case


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1:11434/?key=x",
        "http://user:secret@localhost:11434",
        "file:///tmp/model",
    ],
)
def test_model_transport_cannot_send_customer_messages_to_remote_urls(url):
    with pytest.raises(ValueError):
        local_url(url)


def apply(store, state, tool, **arguments):
    return store.apply(state["id"], json.dumps({"tool": tool, "arguments": arguments}), {}, state["revision"])


@pytest.mark.parametrize("error", ["unavailable_variant", "same_variant", "wrong_refund_method"])
def test_invalid_write_proposals_do_not_change_sandbox_orders(tmp_path, error):
    case = make_case("valid", "exchange_fallback", 0)
    store = Store(tmp_path / "db")
    state = store.create(case)
    state = apply(store, state, "identify_customer", email=case["customer"]["email"])
    state = apply(store, state, "get_order", order_id=case["orders"][0]["id"])
    state = apply(store, state, "get_product", product_id=case["orders"][0]["items"][0]["product_id"])
    args = {
        "order_id": case["orders"][0]["id"],
        "item_id": case["expected"]["item_id"],
        "new_item_id": case["expected"]["new_item_id"],
        "payment_method": "original",
    }
    if error == "same_variant":
        args["new_item_id"] = args["item_id"]
    if error == "wrong_refund_method":
        args["payment_method"] = "another_customer_card"
    state = apply(store, state, "propose_exchange", **args)
    assert "error" in state["observations"][-1]["result"]
    assert state["proposal"] is None and store.orders(state["id"]) == case["orders"]


def test_production_agent_never_calls_the_demonstrator_or_uses_expected_answers(monkeypatch, tmp_path):
    import training.fixtures as fixtures

    case = make_case("valid", "return", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Demonstrator leaked into production")

    monkeypatch.setattr(fixtures, "next_reference", forbidden)
    actions = [
        {"tool": "identify_customer", "arguments": {"email": case["customer"]["email"]}},
        {"tool": "get_order", "arguments": {"order_id": case["orders"][0]["id"]}},
        {
            "tool": "propose_return",
            "arguments": {
                "order_id": case["orders"][0]["id"],
                "item_id": case["orders"][0]["items"][0]["item_id"],
                "payment_method": "original",
            },
        },
    ]

    class Provider:
        def generate(self, prompt):
            visible = json.loads(prompt[1]["content"])
            assert set(visible) == {
                "messages",
                "completed_reads",
                "recent_events",
                "allowed_tools",
                "clarification_memory",
                "observed_constraints",
                "decision_facts",
            }
            return json.dumps(actions.pop(0)), {"latency_ms": 0, "provider": "injected_test_double"}

    store = Store(tmp_path / "db")
    state = store.create(case)
    result = advance(store, state["id"], Provider())
    assert result["status"] == "awaiting_confirmation" and not actions
    assert store.orders(state["id"]) == case["orders"]
    assert messages(state)[0]["role"] == "system"
