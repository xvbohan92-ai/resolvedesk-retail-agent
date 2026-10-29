import random

from training.common import ROOT

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
    "exchange_missing_option",
    "exchange_missing_payment",
    "exchange_gift",
    "exchange_unavailable",
    "exchange_shortfall",
]
TEMPLATES = {
    "train": [
        "Hi support. {request} {details}",
        "For my purchase: {details} {request}",
        "Can you help? {request} Details: {details}",
        "About this order, {request} {details}",
    ],
    "valid": [
        "I'd appreciate help with this purchase: {request} {details}",
        "Customer service request. {details} {request}",
    ],
    "test": [
        "I am contacting the shop today. {details} {request}",
        "Here are my purchase details: {details} My request is as follows: {request}",
        "After checking my purchase, I decided: {request} You can locate it with {details}",
    ],
}


def make_case(split, family, n):
    seed = {"train": 501, "valid": 1401, "test": 2901}[split] * 10000 + FAMILIES.index(family) * 100 + n
    rng = random.Random(seed)

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=5))

    oid, uid, pid = ident("ORD-"), ident("u_"), ident("p_")
    old, new, alternative = [ident("item_") for _ in range(3)]
    email, gift = ident("buyer").lower() + "@retail.example", ident("gift_")
    colors = {
        "train": ["navy", "olive", "red", "gray", "cream"],
        "valid": ["teal", "cobalt", "tan"],
        "test": ["plum", "moss", "orange"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['S', 'L', 'XL'][n % 3]}"
    old_option = "black / M"
    price = rng.randrange(2000, 7500)
    delta = [-450, 625, 1100][n % 3]
    new_price = price + delta
    exchange = family.startswith("exchange")
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    operation = "exchange" if exchange else ("cancel" if cancel else "return")
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ["processed", "delivered"][n % 2]
    reason = ["no longer needed", "ordered by mistake"][n % 2]
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    available = family not in {"exchange_fallback", "exchange_unavailable"}
    if family == "exchange_shortfall":
        new_price, delta = price + 1600, 1600
    balance = 200 if family == "exchange_shortfall" else 8500
    request = {
        "cancel": "Cancel this order because I "
        + ("don't need it anymore." if n % 2 == 0 else "placed a duplicate order."),
        "missing_reason": "I would like to cancel this order.",
        "return": "Return the item and refund my original payment method.",
        "missing_payment": "I would like to return the item.",
        "exchange": (
            f"Swap the item for {option}. Charge or refund the difference to the original payment method."
        ),
        "exchange_fallback": (
            f"Please exchange this item for {option}. If it is out of stock, a "
            f"return is OK. Use my original payment method."
        ),
        "forbidden_status": "Cancel this order; I no longer need it.",
        "foreign_order": (
            "Return the item to the original payment method, even if the order is someone else's."
        ),
        "missing_email": "Please return my item to the original payment method.",
        "exchange_missing_option": (
            "I want to exchange the item for a different variant. Use the original payment method."
        ),
        "exchange_missing_payment": f"I would like to exchange my item for {option}.",
        "exchange_gift": f"Exchange this for {option}; use my existing gift card {gift} for the difference.",
        "exchange_unavailable": (
            f"Exchange the item for {option} only. Do not return it or choose "
            f"another variant. Use original payment."
        ),
        "exchange_shortfall": (
            f"Exchange the item for {option}. Only use gift card {gift}; do not use another payment method."
        ),
    }[family]
    if split == "test" and family == "cancel":
        request = (
            "Cancel the purchase; it is no longer needed."
            if n % 2 == 0
            else "I bought this twice by mistake, so cancel this purchase."
        )
    details = f"Order {oid}." + ("" if family == "missing_email" else f" Email {email}.")
    expected_kind = (
        "handoff"
        if family in {"foreign_order", "forbidden_status", "exchange_unavailable", "exchange_shortfall"}
        else ("return" if family == "exchange_fallback" else operation)
    )
    variants = [
        {"item_id": old, "option": old_option, "price_cents": price, "available": True},
        {"item_id": new, "option": option, "price_cents": new_price, "available": available},
        {"item_id": alternative, "option": "white / XS", "price_cents": price + 275, "available": True},
    ]
    rng.shuffle(variants)
    missing = {
        "missing_email": "email",
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
    }.get(family)
    return {
        "id": f"v2-{split}-{family}-{n:02d}",
        "split": split,
        "family": family,
        "initial_message": TEMPLATES[split][n % len(TEMPLATES[split])].format(
            request=request, details=details
        ),
        "customer": {"id": uid, "email": email, "gift_card": {"id": gift, "balance_cents": balance}},
        "products": {pid: {"product_id": pid, "name": "Everyday shirt", "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": "other_demo_customer" if family == "foreign_order" else uid,
                "status": status,
                "original_payment": ident("card_"),
                "items": [{"item_id": old, "product_id": pid, "option": old_option, "price_cents": price}],
            }
        ],
        "answers": {
            "email": email,
            "order_id": oid,
            "reason": reason,
            "item_id": old,
            "new_option": option,
            "payment_method": payment,
        },
        "required_clarification": missing,
        "confirmation": not (split == "test" and n == 2),
        "expected": {
            "kind": expected_kind,
            "order_id": oid,
            "item_id": old,
            "new_item_id": new,
            "payment_method": payment,
            "reason": reason,
            "amount_cents": delta if expected_kind == "exchange" else price,
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
    """Programmatic demonstrator, not available to the serving agent."""
    answered = {m.get("field") for m in state["messages"]}
    answers, expected = case["answers"], case["expected"]

    def call(tool, **args):
        return {"tool": tool, "arguments": args}

    def ask(field):
        return call("ask_customer", field=field, message=f"Please provide your {field.replace('_', ' ')}.")

    if not state["user_id"]:
        if case["required_clarification"] == "email" and "email" not in answered:
            return ask("email")
        return call("identify_customer", email=answers["email"])
    if case["family"] == "foreign_order":
        attempted = any(o["action"] and o["action"]["tool"] == "get_order" for o in state["observations"])
        return (
            call("handoff", reason="This order belongs to another customer.")
            if attempted
            else call("get_order", order_id=answers["order_id"])
        )
    if answers["order_id"] not in state["read_orders"]:
        return call("get_order", order_id=answers["order_id"])
    if case["family"] == "forbidden_status":
        return call("handoff", reason="This status does not permit cancellation.")
    missing = case["required_clarification"]
    if missing and missing not in answered:
        return ask(missing)
    if case["family"].startswith("exchange") and not any(
        o["action"] and o["action"]["tool"] == "get_product" and "error" not in o["result"]
        for o in state["observations"]
    ):
        return call("get_product", product_id=case["orders"][0]["items"][0]["product_id"])
    kind = expected["kind"]
    if kind == "handoff":
        return call(
            "handoff", reason="Requested variant is unavailable or the authorized payment is insufficient."
        )
    args = {"order_id": answers["order_id"]}
    if kind == "cancel":
        args["reason"] = answers["reason"]
    else:
        args.update(item_id=expected["item_id"], payment_method=answers["payment_method"])
        if kind == "exchange":
            args["new_item_id"] = expected["new_item_id"]
    return call("propose_" + kind, **args)


def load_cases(split=None):
    import json

    rows = json.loads((ROOT / "training/v2/datasets/cases.json").read_text())
    return [c for c in rows if split is None or c["split"] == split]
