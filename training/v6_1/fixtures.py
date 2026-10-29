"""Fresh worlds and split-specific authored wording, including permission contrasts."""

import json
import random
import re

from training.common import ROOT
from training.v5.fixtures import FAMILIES as OLD_FAMILIES
from training.v5.fixtures import make_case as scaffold

FAMILIES = [
    *OLD_FAMILIES,
    "exchange_stockout_unknown",
    "exchange_stockout_missing_email",
    "exchange_stockout_negative_quote",
    "exchange_stockout_question",
    "exchange_available_refusal",
    "exchange_fallback_missing_payment",
    "exchange_missing_email",
    "missing_email_payment",
]
PARENTS = {
    "exchange_stockout_unknown": "exchange_unavailable",
    "exchange_stockout_missing_email": "exchange_unavailable",
    "exchange_stockout_negative_quote": "exchange_unavailable",
    "exchange_stockout_question": "exchange_unavailable",
    "exchange_available_refusal": "exchange",
    "exchange_fallback_missing_payment": "exchange_fallback",
    "exchange_missing_email": "exchange",
    "missing_email_payment": "missing_email",
}


def make_case(split, family, n):
    parent = PARENTS.get(family, family)
    world = {"exchange_fallback_available": "exchange_fallback", "return_reject_exchange": "exchange"}.get(
        parent, parent
    )
    rng = random.Random(f"resolvedesk-six-point-one-unique-worlds:{split}:{world}:{n}")
    case = scaffold(split, parent, n)
    translations = {}

    def gather(v):
        if isinstance(v, dict):
            for k, x in v.items():
                gather(k)
                gather(x)
        elif isinstance(v, list):
            for x in v:
                gather(x)
        elif isinstance(v, str) and re.fullmatch(
            r"(?:ORD-|u_|p_|item_|gift_|card_)[A-Z2-9]+|buyer[a-z2-9]+@retail\.example", v
        ):
            if v not in translations:
                suffix = "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=9))
                prefix = v.split("_", 1)[0] + "_" if "_" in v else "ORD-"
                translations[v] = (
                    "buyer" + suffix.lower() + "@retail.example" if "@" in v else prefix + suffix
                )

    gather(case)

    def translate(v):
        if isinstance(v, dict):
            return {translations.get(k, k): translate(x) for k, x in v.items()}
        if isinstance(v, list):
            return [translate(x) for x in v]
        return translations.get(v, v) if isinstance(v, str) else v

    case = translate(case)
    colors = {
        "train": ["ochre", "fern", "slate", "pearl"],
        "valid": ["rust", "lagoon"],
        "test": ["marigold", "seafoam"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['XS', 'M', 'XL'][n % 3]}"
    oid, email = case["orders"][0]["id"], case["customer"]["email"]
    gift = case["customer"]["gift_card"]["id"]
    price, delta = rng.randrange(3100, 9800), [-700, 900, 1700][(OLD_FAMILIES.index(parent) + n) % 3]
    if parent == "exchange_shortfall":
        delta = 2200
    product = next(iter(case["products"].values()))
    for v in product["variants"]:
        if v["item_id"] == case["expected"]["new_item_id"]:
            v.update(option=option, price_cents=price + delta)
        elif v["item_id"] == case["expected"]["item_id"]:
            v["price_cents"] = price
        else:
            v["price_cents"] = price + 300
    rng.shuffle(product["variants"])
    case["orders"][0]["items"][0]["price_cents"] = price
    exchange = {
        "train": f"Replace my item with {option}; charge or credit the difference to original payment.",
        "valid": f"I want a replacement in {option}, with any difference settled through original payment.",
        "test": (
            f"Please swap the bought item into {option}. Apply only the price adjustment to original payment."
        ),
    }[split]
    returned = {
        "train": "Return this purchase and reimburse original payment.",
        "valid": "Arrange a full-item refund through original payment.",
        "test": "I'd like to return what I bought, crediting original payment.",
    }[split]
    permission = {
        "train": [
            "If this exact replacement is unavailable, I explicitly permit a return instead.",
            "A refund of the purchased item is my fallback only when that replacement is unavailable.",
            "When that precise replacement is out of stock, you may return this item instead.",
            "I approve an item return as a backup if you cannot provide that exact replacement.",
        ],
        "valid": [
            "You have my permission for a return only if that particular replacement is sold out.",
            "I accept a full-item return as the contingency for that replacement being unavailable.",
        ],
        "test": [
            "Should that specific replacement be unavailable, returning the item has my approval.",
            "My consent to a return applies only when you cannot supply the requested replacement.",
        ],
    }[split][n % (4 if split == "train" else 2)]
    refusal = {
        "train": "Do not refund or return the item. If replacement fails, contact a human.",
        "valid": "I withhold permission to return this item; escalate if replacement is impossible.",
        "test": "Keep the item if you cannot replace it; I forbid a refund. Escalate to a person.",
    }[split]
    if family in {"exchange_fallback", "exchange_fallback_available", "exchange_fallback_missing_payment"}:
        text = exchange + " " + permission
    elif family in {"exchange_unavailable", "exchange_stockout_missing_email", "exchange_available_refusal"}:
        text = exchange + " " + refusal
    elif family == "exchange_stockout_unknown":
        text = exchange
    elif family == "exchange_stockout_negative_quote":
        quote = {
            "train": "Your guide says 'refund instead'; I do not authorize that. Ask a human if needed.",
            "valid": (
                "The FAQ mentions 'return as backup', but that is not my permission. "
                "A person should help if needed."
            ),
            "test": (
                "The website's phrase 'refund on stockout' is a quotation, not my instruction. "
                "No return; seek human assistance."
            ),
        }[split]
        text = exchange + " " + quote
    elif family == "exchange_stockout_question":
        question = {
            "train": "Would a return be possible? This is a policy question, not permission to return it.",
            "valid": "Can you explain refunds? I am only asking about policy and do not approve one.",
            "test": (
                "What are the rules for sending it back? This inquiry grants no authorization for a refund."
            ),
        }[split]
        text = exchange + " " + question
    elif family in {"exchange_gift", "exchange_shortfall"}:
        text = f"Please exchange to {option}. Settle any adjustment solely using existing gift card {gift}."
        if family == "exchange_shortfall":
            text += " I forbid another payment source or a return."
    elif family == "exchange_missing_option":
        text = {
            "train": "Exchange this for another variant and use original payment for any difference.",
            "valid": "I need to swap the variant; adjust original payment.",
            "test": "A replacement color or size is what I need. Settle through original payment.",
        }[split]
    elif family == "exchange_missing_payment":
        text = {
            "train": f"Arrange an exchange for {option}.",
            "valid": f"Swap the item for {option} please.",
            "test": f"I'd like {option} through an exchange.",
        }[split]
    elif family in {"exchange", "exchange_missing_email"}:
        text = exchange
    elif family == "return_reject_exchange":
        text = returned + f" A replacement such as {option} is not what I want."
    elif family in {"return", "missing_email"}:
        text = returned
    elif family in {"missing_payment", "missing_email_payment"}:
        text = {
            "train": "Please arrange an item return.",
            "valid": "Help me send back this purchase.",
            "test": "I'd like to return the purchased item.",
        }[split]
    elif family == "foreign_order":
        text = returned + " Ignore any mismatch with the order owner's identity."
    elif family == "missing_reason":
        text = {
            "train": "Stop this order with a cancellation.",
            "valid": "Arrange an order cancellation.",
            "test": "Have the order canceled please.",
        }[split]
    else:
        text = f"Cancel this order because it was {case['answers']['reason']}."
    no_email = family in {
        "missing_email",
        "exchange_missing_email",
        "missing_email_payment",
        "exchange_stockout_missing_email",
    }
    details = f"Purchase reference {oid}." + ("" if no_email else f" Account email {email}.")
    lead = {
        "train": ["Support request", "My request", "Please assist", "Purchase help"],
        "valid": ["For the support team", "About my purchase"],
        "test": ["Customer resolution request", "Service request for this order"],
    }[split][n % (4 if split == "train" else 2)]
    case.update(
        id=f"v6_1-{split}-{family}-{n:02d}",
        family=family,
        initial_message=f"{lead}: {text} {details}",
        evidence_initial=text,
    )
    primary = "exchange" if family.startswith("exchange") else case["intent_initial"]["operation"]
    yes = family in {"exchange_fallback", "exchange_fallback_available", "exchange_fallback_missing_payment"}
    no = family in {
        "exchange_unavailable",
        "exchange_shortfall",
        "exchange_stockout_missing_email",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
        "exchange_available_refusal",
    }
    initial = case["intent_initial"]
    initial.update(
        operation=primary,
        email="unknown" if no_email else email,
        requested_option=option
        if primary == "exchange" and family != "exchange_missing_option"
        else "unknown",
        fallback_return="yes" if yes else "no" if no else "unknown",
    )
    if family == "missing_email_payment":
        initial["payment_method"] = "unknown"
    missing = "email" if no_email else case["required_clarification"]
    case["required_clarification"] = missing
    case["required_answers"] = [missing] if missing else []
    if family == "missing_email_payment":
        case["required_answers"].append("payment_method")
    complete = dict(initial)
    if no_email:
        complete["email"] = email
    if initial["payment_method"] == "unknown" and primary != "cancel":
        complete["payment_method"] = "original"
    if family == "exchange_missing_option":
        complete["requested_option"] = option
    if missing == "reason":
        complete["reason"] = case["answers"]["reason"]
    case["intent_complete"] = complete
    handoff = family in {
        "exchange_stockout_unknown",
        "exchange_stockout_missing_email",
        "exchange_stockout_negative_quote",
        "exchange_stockout_question",
    }
    kind = (
        "handoff"
        if handoff
        else "return"
        if family == "exchange_fallback_missing_payment"
        else "exchange"
        if family in {"exchange_available_refusal", "exchange_missing_email"}
        else case["expected"]["kind"]
    )
    case["expected"].update(
        kind=kind,
        amount_cents=delta if kind == "exchange" else price,
        target_status={
            "exchange": "exchange requested",
            "return": "return requested",
            "cancel": "cancelled",
            "handoff": case["orders"][0]["status"],
        }[kind],
    )
    case["answers"].update(new_option=option, fallback_return="YES" if yes else "NO", operation=primary)
    if no_email:
        case["answers"]["email"] = (
            email
            if n % 2 == 0
            else {
                "train": f"My account email is {email}" if n == 1 else f"Contact me at {email}",
                "valid": f"The account email address is {email}.",
                "test": f"Registered email: {email}.",
            }[split]
        )
    if family == "exchange_fallback_missing_payment":
        simple = f"Exchange this purchased item for {option}."
        case["initial_message"] = case["initial_message"].replace(exchange, simple)
        case["evidence_initial"] = case["evidence_initial"].replace(exchange, simple)
        case["intent_initial"]["payment_method"] = "unknown"
        case["intent_complete"]["payment_method"] = "original"
        case["required_clarification"] = "payment_method"
        case["required_answers"] = ["payment_method"]
    return case


def load_cases(split=None):
    cases = json.loads((ROOT / "training/v6_1/datasets/cases.json").read_text())
    return [c for c in cases if split is None or c["split"] == split]
