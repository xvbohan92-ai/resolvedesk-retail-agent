# ruff: noqa: E501 -- Authored synthetic utterances retain full source lines.
import json
import random
from copy import deepcopy

from training.common import ROOT
from training.v2.fixtures import FAMILIES as OLD_FAMILIES

FAMILIES = [*OLD_FAMILIES, "exchange_fallback_available", "return_reject_exchange"]
OPENINGS = {
    "train": [
        "Retail help request: {details} {request}",
        "Please assist with my purchase. {request} {details}",
    ],
    "valid": [
        "Hello store team. {request} You can find the purchase using {details}",
        "I need the following resolution. {details} {request}",
    ],
    "test": [
        "For this order, here is the resolution I am asking for. {request} Reference information: {details}",
        "A purchase needs your help. {details} My chosen resolution is: {request}",
    ],
}


def request(split, family, option, gift, reason, n):
    exchange = {
        "train": f"Change the item to {option} and apply the difference to original payment.",
        "valid": f"Please arrange a variant swap to {option}; debit or credit original payment for the difference.",
        "test": f"The item should be exchanged for {option}. Settle its price adjustment with original payment.",
    }[split]
    returned = {
        "train": "Take the item back and reimburse original payment.",
        "valid": "Please process an item return to the original payment method.",
        "test": "I am sending the whole purchased item back for a refund to original payment.",
    }[split]
    fallback = {
        "train": "If the requested replacement is unavailable, I accept an item return instead.",
        "valid": "Only lack of stock for that precise option permits a return fallback.",
        "test": "A full-item return is my fallback solely when that exact replacement is not available.",
    }[split]
    if family in {"exchange_fallback", "exchange_fallback_available"}:
        return exchange + " " + fallback
    if family == "exchange_unavailable":
        return exchange + " No item return and no substitute option are authorized."
    if family == "exchange_shortfall":
        return (
            f"Swap my item to {option}, charging gift card {gift} only. I refuse another payment or a return."
        )
    if family == "exchange_gift":
        return f"Please exchange to {option}; the price difference must use existing gift card {gift}."
    if family == "exchange_missing_option":
        return {
            "train": "Exchange my item to a different option, with original payment for the difference.",
            "valid": "I would like a variant swap, settling with original payment.",
            "test": "An exchange to another option is needed; use original payment for the adjustment.",
        }[split]
    if family == "exchange_missing_payment":
        return f"An exchange is needed: replace the item with {option}."
    if family == "exchange":
        return exchange
    if family == "return_reject_exchange":
        return returned + f" I reject a replacement, including {option}; this is not a swap request."
    if family in {"return", "missing_email"}:
        return returned
    if family == "missing_payment":
        return {
            "train": "I need to return my item.",
            "valid": "Please accept this item back.",
            "test": "I want this purchase processed as an item return.",
        }[split]
    if family == "foreign_order":
        return returned + " Do it even if this order is not owned by my account."
    if family == "missing_reason":
        return {
            "train": "Cancel my purchase.",
            "valid": "Please arrange an order cancellation.",
            "test": "I want this order canceled.",
        }[split]
    return "Please cancel my purchase because " + (
        "it is no longer needed." if reason == "no longer needed" else "it was ordered by mistake."
    )


def make_case(split, family, n):
    if family in {"exchange_fallback_available", "return_reject_exchange"}:
        parent = "exchange_fallback" if family == "exchange_fallback_available" else "exchange"
        case = deepcopy(make_case(split, parent, n))
        case["family"], case["id"] = family, f"v4-{split}-{family}-{n:02d}"
        original = case["orders"][0]["items"][0]["price_cents"]
        if family == "exchange_fallback_available":
            variant = next(
                v
                for v in next(iter(case["products"].values()))["variants"]
                if v["item_id"] == case["expected"]["new_item_id"]
            )
            variant["available"] = True
            case["expected"].update(
                kind="exchange",
                amount_cents=variant["price_cents"] - original,
                target_status="exchange requested",
            )
        else:
            case["expected"].update(kind="return", amount_cents=original, target_status="return requested")
            details = f"Order {case['answers']['order_id']}. Email {case['answers']['email']}."
            case["initial_message"] = OPENINGS[split][n % len(OPENINGS[split])].format(
                request=request(
                    split,
                    family,
                    case["answers"]["new_option"],
                    case["customer"]["gift_card"]["id"],
                    case["answers"]["reason"],
                    n,
                ),
                details=details,
            )
        return case
    rng = random.Random(
        {"train": 71001, "valid": 81001, "test": 91001}[split] * 10000 + FAMILIES.index(family) * 100 + n
    )

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))

    oid, uid, pid, old, new, alt = [ident(p) for p in ("ORD-", "u_", "p_", "item_", "item_", "item_")]
    email, gift = ident("buyer").lower() + "@retail.example", ident("gift_")
    colors = {
        "train": ["burgundy", "lime", "ivory", "denim"],
        "valid": ["maroon", "azure"],
        "test": ["saffron", "charcoal"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['S', 'L', 'XL'][n % 3]}"
    price = rng.randrange(2400, 8200)
    delta = [-525, 875, 1450][(FAMILIES.index(family) + n) % 3]
    if family == "exchange_shortfall":
        delta = 1800
    available = family not in {"exchange_fallback", "exchange_unavailable"}
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    reason = "no longer needed" if n % 2 == 0 else "ordered by mistake"
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ["processed", "delivered"][n % 2]
    kind = (
        "handoff"
        if family in {"foreign_order", "forbidden_status", "exchange_unavailable", "exchange_shortfall"}
        else "return"
        if family == "exchange_fallback" or not (cancel or family.startswith("exchange"))
        else "cancel"
        if cancel
        else "exchange"
    )
    variants = [
        {"item_id": old, "option": "black / M", "price_cents": price, "available": True},
        {"item_id": new, "option": option, "price_cents": price + delta, "available": available},
        {"item_id": alt, "option": "white / XS", "price_cents": price + 225, "available": True},
    ]
    rng.shuffle(variants)
    details = f"Order {oid}." + ("" if family == "missing_email" else f" Email {email}.")
    missing = {
        "missing_email": "email",
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
    }.get(family)
    return {
        "id": f"v4-{split}-{family}-{n:02d}",
        "family": family,
        "split": split,
        "initial_message": OPENINGS[split][n % len(OPENINGS[split])].format(
            request=request(split, family, option, gift, reason, n), details=details
        ),
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {"id": gift, "balance_cents": 200 if family == "exchange_shortfall" else 9000},
        },
        "products": {pid: {"product_id": pid, "name": "Everyday shirt", "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": "other_demo_customer" if family == "foreign_order" else uid,
                "status": status,
                "original_payment": ident("card_"),
                "items": [{"item_id": old, "product_id": pid, "option": "black / M", "price_cents": price}],
            }
        ],
        "answers": {
            "email": email,
            "order_id": oid,
            "item_id": old,
            "new_option": option,
            "reason": reason,
            "payment_method": payment,
        },
        "required_clarification": missing,
        "confirmation": not (split == "test" and n == 1),
        "expected": {
            "kind": kind,
            "order_id": oid,
            "item_id": old,
            "new_item_id": new,
            "payment_method": payment,
            "reason": reason,
            "amount_cents": delta if kind == "exchange" else price,
            "initial_status": status,
            "target_status": {
                "exchange": "exchange requested",
                "return": "return requested",
                "cancel": "cancelled",
                "handoff": status,
            }[kind],
        },
    }


def next_reference(case, state):
    from training.v2.fixtures import next_reference as old_reference

    # The v2 reference already handles generic expected outcomes and missing fields.
    # The additional return contrast inspects a catalog without changing its return intent.
    action = old_reference(case, state)
    if (
        case["family"] == "return_reject_exchange"
        and state["user_id"]
        and case["answers"]["order_id"] in state["read_orders"]
        and not any(
            o["action"] and o["action"]["tool"] == "get_product" and "error" not in o["result"]
            for o in state["observations"]
        )
    ):
        return {
            "tool": "get_product",
            "arguments": {"product_id": case["orders"][0]["items"][0]["product_id"]},
        }
    return action


def load_cases(split=None):
    rows = json.loads((ROOT / "training/v4/datasets/cases.json").read_text())
    return [r for r in rows if split is None or r["split"] == split]
