"""New seeded entities, prices, wording and independently authored intent labels."""

import json
import random

from training.common import ROOT
from training.v4.fixtures import FAMILIES


def make_case(split, family, n):
    parent = {"exchange_fallback_available": "exchange_fallback", "return_reject_exchange": "exchange"}.get(
        family, family
    )
    rng = random.Random(f"resolvedesk-round-five:{split}:{parent}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=8))

    oid, uid, pid, old, new, alt = [ident(p) for p in ("ORD-", "u_", "p_", "item_", "item_", "item_")]
    email, gift = ident("buyer").lower() + "@retail.example", ident("gift_")
    colors = {
        "train": ["cobalt", "linen", "olive", "plum"],
        "valid": ["coral", "moss"],
        "test": ["copper", "indigo"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['XS', 'M', 'XXL'][n % 3]}"
    price = rng.randrange(2700, 9600)
    delta = [-650, 1200, 1850][(FAMILIES.index(parent) + n) % 3]
    if family == "exchange_shortfall":
        delta = 2100
    balance = 275 if family == "exchange_shortfall" else 10000
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    available = family not in {"exchange_fallback", "exchange_unavailable"}
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ["processed", "delivered"][n % 2]
    reason = "no longer needed" if n % 2 == 0 else "ordered by mistake"
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    exchange = {
        "train": f"Swap the purchased item for {option}; use original payment for the price adjustment.",
        "valid": f"I would like an exchange to {option}, settling the difference using original payment.",
        "test": f"Keep this as a swap to {option}. Apply its price difference to original payment.",
    }[split]
    returned = {
        "train": "Please return the item and refund original payment.",
        "valid": "I want to send the item back, reimbursing original payment.",
        "test": "Take the purchased item back for a full refund to original payment.",
    }[split]
    fallback = {
        "train": "If that exact option is out of stock, a return with refund is my authorized backup.",
        "valid": "A return is acceptable only if that specific replacement cannot be supplied.",
        "test": (
            "My second choice is a full-item return, but only when that precise replacement is unavailable."
        ),
    }[split]
    if split == "train":
        fallback = [
            fallback,
            "I authorize returning the item instead only if the exact requested variant is unavailable.",
            "If you cannot supply that exact option, refund the whole item rather than choose a substitute.",
            "Use an item return as backup solely when the specified replacement is out of stock.",
        ][n % 4]
    if family in {"exchange_fallback", "exchange_fallback_available"}:
        text = exchange + " " + fallback
    elif family == "exchange_unavailable":
        text = exchange + " Do not return it or pick a substitute. If unavailable, get human help."
    elif family == "exchange_shortfall":
        text = f"Exchange to {option}. Use only existing gift card {gift}; no other payment or return."
    elif family == "exchange_gift":
        text = f"Exchange to {option} and settle the difference only with my existing gift card {gift}."
    elif family == "exchange_missing_option":
        text = "I need a different color or size through an exchange; use original payment."
    elif family == "exchange_missing_payment":
        text = f"Please exchange the item for {option}."
    elif family == "exchange":
        text = exchange
    elif family == "return_reject_exchange":
        text = returned + f" I do not want a swap to {option} or any replacement."
    elif family in {"return", "missing_email"}:
        text = returned
    elif family == "missing_payment":
        text = {
            "train": "Please take this item back.",
            "valid": "I would like an item return.",
            "test": "The purchase needs to be returned.",
        }[split]
    elif family == "foreign_order":
        text = returned + " Process it even if the order is attached to someone else's account."
    elif family == "missing_reason":
        text = {
            "train": "Cancel this order.",
            "valid": "I need an order cancellation.",
            "test": "Please have this purchase canceled.",
        }[split]
    else:
        text = "Please cancel the order because it was " + (
            "ordered by mistake." if n % 2 else "no longer needed."
        )
    details = f"Order {oid}." + ("" if family == "missing_email" else f" Email {email}.")
    openings = {
        "train": [
            f"{details} {text}",
            f"Customer request: {text} Purchase details: {details}",
            f"Can you handle this? {details} {text}",
            f"{text} Find the purchase here: {details}",
        ],
        "valid": [
            f"Hello support. {text} My purchase reference: {details}",
            f"Here is what I need. {details} {text}",
        ],
        "test": [
            f"Resolution for my purchase: {text} Account information: {details}",
            f"Please help with this transaction. {details} Requested outcome: {text}",
        ],
    }
    message = openings[split][n % len(openings[split])]
    missing = {
        "missing_email": "email",
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
    }.get(family)
    kind = (
        "handoff"
        if family in {"forbidden_status", "foreign_order", "exchange_unavailable", "exchange_shortfall"}
        else "return"
        if family == "exchange_fallback"
        else primary
    )
    initial = {
        "operation": primary,
        "email": "unknown" if missing == "email" else email,
        "order_id": oid,
        "requested_option": option if primary == "exchange" and missing != "new_option" else "unknown",
        "payment_method": (
            "unknown"
            if cancel or missing == "payment_method"
            else "gift_card"
            if payment == gift
            else "original"
        ),
        "reason": reason if cancel and missing != "reason" else "unknown",
        "fallback_return": (
            "yes"
            if family in {"exchange_fallback", "exchange_fallback_available"}
            else "no"
            if family in {"exchange_unavailable", "exchange_shortfall"}
            else "unknown"
        ),
    }
    complete = dict(initial)
    if missing:
        key = "requested_option" if missing == "new_option" else missing
        complete[key] = {
            "email": email,
            "reason": reason,
            "payment_method": "original",
            "new_option": option,
        }[missing]
    variants = [
        {"item_id": old, "option": "black / L", "price_cents": price, "available": True},
        {"item_id": new, "option": option, "price_cents": price + delta, "available": available},
        {"item_id": alt, "option": "white / S", "price_cents": price + 350, "available": True},
    ]
    rng.shuffle(variants)
    return {
        "id": f"v5-{split}-{family}-{n:02d}",
        "family": family,
        "split": split,
        "initial_message": message,
        "customer": {"id": uid, "email": email, "gift_card": {"id": gift, "balance_cents": balance}},
        "products": {pid: {"product_id": pid, "name": "Cotton overshirt", "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": "other_demo_customer" if family == "foreign_order" else uid,
                "status": status,
                "original_payment": ident("card_"),
                "items": [{"item_id": old, "product_id": pid, "option": "black / L", "price_cents": price}],
            }
        ],
        "answers": {
            "email": email,
            "order_id": oid,
            "reason": reason,
            "new_option": option,
            "item_id": old,
            "payment_method": payment,
            "operation": primary,
        },
        "evidence_initial": text,
        "intent_initial": initial,
        "intent_complete": complete,
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
                "cancel": "cancelled",
                "return": "return requested",
                "exchange": "exchange requested",
                "handoff": status,
            }[kind],
        },
    }


def load_cases(split=None):
    cases = json.loads((ROOT / "training/v5/datasets/cases.json").read_text())
    return [c for c in cases if split is None or c["split"] == split]
