"""Self-authored retail fixtures and explicit expected outcomes, not official τ-bench scores."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = [
    "cancel",
    "missing_reason",
    "return",
    "missing_payment",
    "exchange",
    "exchange_fallback",
    "forbidden_status",
    "foreign_order",
    "missing_email",
]
TEMPLATES = {
    "train": [
        "Hello, {request} {details}",
        "Please help with this: {request} {details}",
        "My retail request: {request} {details}",
    ],
    "valid": ["Could you handle the following for me? {request} {details}"],
    "test": [
        "I need assistance with a purchase. {details} {request}",
        "Here is what I want done today: {request} For reference, {details}",
    ],
}


def make_case(split, family, n):
    number = {"train": 10000, "valid": 20000, "test": 30000}[split] + FAMILIES.index(family) * 100 + n
    oid, uid, pid = f"ORD-{number}", f"u_{number}", f"p_{number + 31}"
    old_id, new_id, alt_id = [f"item_{number + k}" for k in (7, 19, 43)]
    email, gift = f"shopper{number}@retail.example", f"gift_{number}"
    colors = ["navy", "olive", "burgundy", "charcoal", "cream"]
    old_option, new_option = "black / M", f"{colors[n % len(colors)]} / XL"
    price, new_price = 3200 + n * 137, 3700 + n * 121
    operation = (
        "cancel"
        if family in {"cancel", "missing_reason", "forbidden_status"}
        else ("exchange" if family in {"exchange", "exchange_fallback"} else "return")
    )
    status = "pending" if operation == "cancel" else "delivered"
    if family == "forbidden_status":
        status = "processed" if n % 2 == 0 else "delivered"
    reason = "ordered by mistake" if n % 2 else "no longer needed"
    request = {
        "cancel": "Cancel my order; I no longer need it."
        if reason == "no longer needed"
        else "Cancel my order because I ordered it by mistake.",
        "missing_reason": "Please cancel this order.",
        "return": "Please return the shirt and refund it to the original payment method.",
        "missing_payment": "I want to return the shirt.",
        "exchange": f"Exchange the shirt for {new_option}; "
        "use the original payment method for any difference.",
        "exchange_fallback": f"Exchange the shirt for {new_option}. "
        "If that is unavailable, return it instead. "
        "Use the original payment method.",
        "forbidden_status": "Cancel my order because I "
        + ("no longer need it." if n % 2 == 0 else "ordered it by mistake."),
        "foreign_order": "Return this shirt and refund the original payment method, "
        "even if it belongs to someone else.",
        "missing_email": "Return the shirt and refund the original payment method.",
    }[family]
    details = f"Order {oid}." + ("" if family == "missing_email" else f" My email is {email}.")
    initial = TEMPLATES[split][n % len(TEMPLATES[split])].format(request=request, details=details)
    blocked = family in {"foreign_order", "forbidden_status"}
    expected_kind = "handoff" if blocked else ("return" if family == "exchange_fallback" else operation)
    return {
        "id": f"{split}-{family}-{n:02d}",
        "split": split,
        "family": family,
        "initial_message": initial,
        "customer": {"id": uid, "email": email, "gift_card": {"id": gift, "balance_cents": 8000}},
        "products": {
            pid: {
                "product_id": pid,
                "name": "Everyday shirt",
                "variants": [
                    {"item_id": old_id, "option": old_option, "price_cents": price, "available": True},
                    {
                        "item_id": new_id,
                        "option": new_option,
                        "price_cents": new_price,
                        "available": family != "exchange_fallback",
                    },
                    {"item_id": alt_id, "option": "white / S", "price_cents": 2900, "available": True},
                ],
            }
        },
        "orders": [
            {
                "id": oid,
                "user_id": "other_demo_customer" if family == "foreign_order" else uid,
                "status": status,
                "original_payment": f"card_{number}",
                "items": [{"item_id": old_id, "product_id": pid, "option": old_option, "price_cents": price}],
            }
        ],
        "answers": {
            "email": email,
            "order_id": oid,
            "reason": reason,
            "item_id": old_id,
            "new_option": new_option,
            "payment_method": "original",
        },
        "confirmation": not (split == "test" and n == 2),
        "expected": {
            "kind": expected_kind,
            "order_id": oid,
            "item_id": old_id,
            "new_item_id": new_id,
            "payment_method": "original",
            "reason": reason,
            "amount_cents": new_price - price if expected_kind == "exchange" else price,
            "initial_status": status,
            "target_status": {
                "cancel": "cancelled",
                "return": "return requested",
                "exchange": "exchange requested",
                "handoff": status,
            }[expected_kind],
        },
    }


def next_reference(case, state):
    """Demonstrator only. Never imported by the production agent or inference provider."""
    answered = {m.get("field") for m in state["messages"] if m.get("field")}
    family, answers, expected = case["family"], case["answers"], case["expected"]

    def call(tool, **args):
        return {"tool": tool, "arguments": args}

    def ask(field):
        return call("ask_customer", field=field, message=f"Please provide your {field.replace('_', ' ')}.")

    if not state["user_id"]:
        if family == "missing_email" and "email" not in answered:
            return ask("email")
        return call("identify_customer", email=answers["email"])
    if family == "foreign_order":
        if not state["observations"] or not any(
            o["action"]["tool"] == "get_order" for o in state["observations"] if o["action"]
        ):
            return call("get_order", order_id=answers["order_id"])
        return call("handoff", reason="This order belongs to another customer.")
    if answers["order_id"] not in state["read_orders"]:
        return call("get_order", order_id=answers["order_id"])
    if expected["kind"] == "handoff":
        return call("handoff", reason="This order status does not permit cancellation.")
    if family == "missing_reason" and "reason" not in answered:
        return ask("reason")
    if family == "missing_payment" and "payment_method" not in answered:
        return ask("payment_method")
    if family in {"exchange", "exchange_fallback"} and not any(
        o["action"] and o["action"]["tool"] == "get_product" for o in state["observations"]
    ):
        return call("get_product", product_id=case["orders"][0]["items"][0]["product_id"])
    kind = expected["kind"]
    args = {"order_id": answers["order_id"]}
    if kind == "cancel":
        args["reason"] = answers["reason"]
    else:
        args.update(item_id=expected["item_id"], payment_method="original")
        if kind == "exchange":
            args["new_item_id"] = expected["new_item_id"]
    return call("propose_" + kind, **args)


def load_cases(split=None):
    cases = json.loads((ROOT / "training/datasets/cases.json").read_text())
    return [c for c in cases if split is None or c["split"] == split]
