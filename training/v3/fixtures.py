import json
import random
from copy import deepcopy

from training.common import ROOT
from training.v2.fixtures import FAMILIES as OLD_FAMILIES

FAMILIES = [*OLD_FAMILIES, "exchange_fallback_available", "return_reject_exchange"]
OPENINGS = {
    "train": ["Store support: {request} {details}", "Regarding my item, {details} {request}"],
    "valid": ["Please resolve this purchase issue. {details} {request}"],
    "test": [
        "This is the outcome I want for my purchase: {request} My details: {details}",
        "I checked the item and decided as follows. {details} {request}",
        "Could the shop handle this request? {request} Here is the reference: {details}",
    ],
}


def request(split, family, option, gift, reason, n):
    # Split-specific request phrasings, not just a different greeting.
    exchange = {
        "train": f"Exchange my current item for {option}; use the original payment for any price difference.",
        "valid": f"Replace this variant with {option}. Settle only the difference on the original payment.",
        "test": [
            f"Swap the item to {option}, using original payment for any additional charge or partial credit.",
            f"I need {option} instead of this variant, not a whole-item refund. Use original payment.",
            f"Please exchange my item for {option}; any price adjustment should use original payment.",
        ][n % 3],
    }[split]
    direct_return = {
        "train": "Return this item and refund the original payment method.",
        "valid": "I want a return of the purchased item, refunded to original payment.",
        "test": "Refund the entire purchased item to its original payment method; I am returning it.",
    }[split]
    fallback = {
        "train": "Only if that exact variant is unavailable, return the item instead.",
        "valid": "A return is acceptable only when the requested replacement is out of stock.",
        "test": "If that option cannot be supplied, I authorize an item return as fallback.",
    }[split]
    if family in {"exchange_fallback", "exchange_fallback_available"}:
        return exchange + " " + fallback
    if family == "exchange_unavailable":
        return exchange + " I do not authorize a return or another variant."
    if family == "exchange_shortfall":
        return (
            f"Exchange the item for {option}. Use only my gift card {gift}; no alternative payment or return."
        )
    if family == "exchange_gift":
        return f"Exchange my item for {option}. Use my existing gift card {gift} for the price difference."
    if family == "exchange_missing_option":
        return {
            "train": "I want an exchange to another variant; use original payment.",
            "valid": "Replace my item with a different option. Settle with original payment.",
            "test": "Arrange a variant exchange for this purchase, charged or credited to original payment.",
        }[split]
    if family == "exchange_missing_payment":
        return f"I want a replacement variant: {option}. Please exchange this item."
    if family == "exchange":
        return exchange
    if family == "return_reject_exchange":
        return direct_return + f" Do not exchange it for {option}; I do not want a replacement."
    if family == "return":
        return direct_return
    if family == "missing_email":
        return direct_return
    if family == "missing_payment":
        return {
            "train": "Please return this item.",
            "valid": "I would like to send this purchase back.",
            "test": "Arrange a return of my purchased item.",
        }[split]
    if family == "foreign_order":
        return direct_return + " Process it even if another customer owns this order."
    if family == "missing_reason":
        return {
            "train": "Cancel this order.",
            "valid": "I am requesting cancellation of the order.",
            "test": "Please stop and cancel my purchase.",
        }[split]
    return "Cancel this order because I " + (
        "no longer need it." if reason == "no longer needed" else "ordered it by mistake."
    )


def make_case(split, family, n):
    if family in {"exchange_fallback_available", "return_reject_exchange"}:
        parent = "exchange_fallback" if family == "exchange_fallback_available" else "exchange"
        case = deepcopy(make_case(split, parent, n))
        case["family"], case["id"] = family, f"v3-{split}-{family}-{n:02d}"
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
        {"train": 31001, "valid": 41001, "test": 51001}[split] * 10000 + FAMILIES.index(family) * 100 + n
    )

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=6))

    oid, uid, pid, old, new, alt = [ident(p) for p in ("ORD-", "u_", "p_", "item_", "item_", "item_")]
    email, gift = ident("buyer").lower() + "@retail.example", ident("gift_")
    colors = {
        "train": ["navy", "olive", "coral", "cream"],
        "valid": ["teal", "cobalt"],
        "test": ["plum", "moss", "amber"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['S', 'L', 'XL'][n % 3]}"
    price = rng.randrange(2300, 7800)
    delta = [-375, 550, 1250][(FAMILIES.index(family) + n) % 3]
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
        "id": f"v3-{split}-{family}-{n:02d}",
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
        "confirmation": not (split == "test" and n == 2),
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
    rows = json.loads((ROOT / "training/v3/datasets/cases.json").read_text())
    return [r for r in rows if split is None or r["split"] == split]
