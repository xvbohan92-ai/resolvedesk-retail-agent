"""New independent worlds and split-specific language; no historical test aliases."""

import json
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
    "exchange_fallback_available",
    "return_reject_exchange",
    "exchange_stockout_unknown",
    "exchange_stockout_missing_email",
    "exchange_stockout_negative_quote",
    "exchange_stockout_question",
    "exchange_available_refusal",
    "exchange_fallback_missing_payment",
    "exchange_missing_email",
    "missing_email_payment",
]
YES_FAMILIES = {
    "exchange_fallback",
    "exchange_fallback_available",
    "exchange_fallback_missing_payment",
    "exchange_missing_option",
    "exchange_missing_payment",
    "exchange_missing_email",
}
NO_FAMILIES = {
    "exchange_unavailable",
    "exchange_shortfall",
    "exchange_stockout_missing_email",
    "exchange_stockout_negative_quote",
    "exchange_stockout_question",
    "exchange_available_refusal",
}
MISSING_EMAIL = {
    "missing_email",
    "exchange_missing_email",
    "missing_email_payment",
    "exchange_stockout_missing_email",
}
HANDOFF = {
    "forbidden_status",
    "foreign_order",
    "exchange_unavailable",
    "exchange_shortfall",
    "exchange_stockout_unknown",
    "exchange_stockout_missing_email",
    "exchange_stockout_negative_quote",
    "exchange_stockout_question",
}
STOCKOUT = {
    "exchange_fallback",
    "exchange_fallback_missing_payment",
    "exchange_unavailable",
    "exchange_stockout_unknown",
    "exchange_stockout_missing_email",
    "exchange_stockout_negative_quote",
    "exchange_stockout_question",
}
GRANTS = {
    "train": [
        (
            "I give conditional approval for returning the original article if your stock "
            "cannot fill that exact exchange."
        ),
        (
            "My authorization includes a full return, contingent on the specified "
            "replacement being impossible to supply."
        ),
        (
            "A return of the whole purchase is approved by me solely in the event that the "
            "requested variant is sold out."
        ),
        (
            "Where you cannot deliver precisely that replacement, my instruction is to "
            "return the original item instead."
        ),
        "The approval I am giving for a return is limited to the requested replacement being unavailable.",
        "If inventory rules out that exact replacement, you are authorized by me to arrange an item return.",
        (
            "Only when this particular exchange cannot be fulfilled do I approve taking the "
            "original article back."
        ),
        "Treat a return as authorized by me conditional upon the exact replacement being out of stock.",
    ],
    "valid": [
        (
            "My approval for sending the item back is conditional: the specified exchange "
            "must be unfulfillable first."
        ),
        (
            "A full return carries my authorization if, and only if, that precise "
            "replacement cannot be obtained."
        ),
        (
            "Should inventory prevent this particular exchange, you have my approval to "
            "arrange the item's return."
        ),
        (
            "I permit the original purchase to be returned on the condition that the named "
            "replacement is unavailable."
        ),
        (
            "The sole circumstance in which I authorize a return is failure to supply the "
            "exact replacement I named."
        ),
        (
            "You may arrange a full-item return as my approved contingency when that "
            "replacement cannot be fulfilled."
        ),
    ],
    "test": [
        (
            "Returning the original article is within my authorization, provided the exact "
            "exchange option cannot be supplied."
        ),
        "An unavailable specified replacement activates my permission to send the complete item back.",
        "I authorize a full-item return subject to a stockout of precisely the replacement requested here.",
        "When inventory cannot support the named replacement, arranging a return has my explicit approval.",
        (
            "My authorization extends to the item's return only after the exact requested "
            "swap proves unavailable."
        ),
        (
            "A return is an approved alternative in the specific circumstance that you "
            "cannot fulfill my named replacement."
        ),
    ],
}


def make_case(split, family, n):
    assert family in FAMILIES and split in {"train", "valid", "test"}
    rng = random.Random(f"resolvedesk-seven-independent-policy-world:{split}:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=10))

    oid, uid, pid, old, new, alt = [ident(p) for p in ("ORD-", "u_", "p_", "item_", "item_", "item_")]
    email, gift, card = ident("buyer").lower() + "@retail.example", ident("gift_"), ident("card_")
    colors = {
        "train": ["saffron", "denim", "spruce", "porcelain"],
        "valid": ["terracotta", "celadon"],
        "test": ["amethyst", "sandstone"],
    }[split]
    option = f"{colors[n % len(colors)]} / {['S', 'L', 'XXL', 'M'][n % 4]}"
    price, delta = rng.randrange(3300, 11000), [-800, 1100, 1950][(FAMILIES.index(family) + n) % 3]
    if family == "exchange_shortfall":
        delta = 2400
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ["processed", "delivered"][n % 2]
    reason = "ordered by mistake" if n % 2 else "no longer needed"
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    missing = {
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
        "exchange_fallback_missing_payment": "payment_method",
        "missing_email_payment": "email",
    }.get(family)
    if family in MISSING_EMAIL:
        missing = "email"
    exchange = {
        "train": (
            f"Exchange the article I bought for {option}; apply its price difference to original payment."
        ),
        "valid": (
            f"Replace the purchased article with {option} and settle the difference through original payment."
        ),
        "test": f"The exchange I am requesting is {option}, with the adjustment on original payment.",
    }[split]
    returned = {
        "train": "Please initiate a return of the complete article, reimbursing original payment.",
        "valid": "I am requesting the item's return with reimbursement through original payment.",
        "test": "Arrange to take the entire purchased item back and credit original payment.",
    }[split]
    refusal = {
        "train": (
            "If the exchange fails, retain my current item. I do not consent to a return; a "
            "human should assist."
        ),
        "valid": (
            "Leave my current article alone if replacement cannot be filled. No item return "
            "is authorized; involve a person."
        ),
        "test": (
            "Retain my original article when replacement fails. Sending it back is "
            "forbidden; let a person handle the issue."
        ),
    }[split]
    if family == "exchange_missing_option":
        text = {
            "train": "I need an exchange to another option; use original payment for its price difference.",
            "valid": (
                "An exchange to another color or size is needed, with the adjustment through "
                "original payment."
            ),
            "test": (
                "I am requesting a different option by exchange. Settle its adjustment on original payment."
            ),
        }[split]
    elif family in {"exchange_missing_payment", "exchange_fallback_missing_payment"}:
        text = {
            "train": f"Please replace the purchased article with {option}.",
            "valid": f"The replacement I want by exchange is {option}.",
            "test": f"Arrange an item exchange into {option} for me.",
        }[split]
    elif family in {"exchange_gift", "exchange_shortfall"}:
        text = {
            "train": (
                f"Exchange my item for {option}. My permission concerns only the adjustment using "
                f"existing gift card {gift}."
            ),
            "valid": (
                f"Use existing gift card {gift} solely to settle the price difference on an "
                f"exchange to {option}."
            ),
            "test": (
                f"For an exchange into {option}, debit or credit the difference exclusively on "
                f"existing gift card {gift}."
            ),
        }[split]
        if family == "exchange_shortfall":
            text += " I authorize neither a different payment source nor an item return."
    elif primary == "exchange":
        text = exchange
    elif family == "missing_reason":
        text = {
            "train": "Please arrange cancellation of this order.",
            "valid": "Have this purchase order canceled for me.",
            "test": "I am asking you to cancel this order.",
        }[split]
    elif cancel:
        text = f"Cancel the purchase order: it was {reason}."
    elif family in {"missing_payment", "missing_email_payment"}:
        text = {
            "train": "I want to send the purchased article back.",
            "valid": "Please begin a return for this article.",
            "test": "The requested resolution is to return my purchased article.",
        }[split]
    else:
        text = returned
    if family in YES_FAMILIES:
        index = sorted(YES_FAMILIES).index(family)
        text += " " + GRANTS[split][(index + n) % len(GRANTS[split])]
    elif family in {"exchange_unavailable", "exchange_stockout_missing_email", "exchange_available_refusal"}:
        text += " " + refusal
    elif family == "exchange_stockout_negative_quote":
        text += (
            " "
            + {
                "train": (
                    "The policy lists 'take it back on stockout'. That quoted wording is not my "
                    "approval. No return; ask a human."
                ),
                "valid": (
                    "A help page says 'send the item back when stock is missing'. I am quoting it, "
                    "not giving permission. Keep mine and ask a person."
                ),
                "test": (
                    "Someone else's message said 'refund it when unavailable'. That is their "
                    "suggestion, not my authorization. Do not return mine; escalate."
                ),
            }[split]
        )
    elif family == "exchange_stockout_question":
        text += (
            " "
            + {
                "train": (
                    "I am asking how a return would work, without approving one. Please seek a human "
                    "if the swap is impossible."
                ),
                "valid": (
                    "What would the return process involve? This question does not authorize any "
                    "return. A person can help if needed."
                ),
                "test": (
                    "Could you describe the policy for a full return? I am requesting information "
                    "and withholding authorization; contact a human if necessary."
                ),
            }[split]
        )
    elif family == "return_reject_exchange":
        text += f" A substitution into {option} is specifically rejected; I want no replacement."
    elif family == "foreign_order":
        text += " Please disregard any ownership conflict with the account."
    details = f"Order reference: {oid}." + ("" if family in MISSING_EMAIL else f" Customer email: {email}.")
    lead = {
        "train": [
            "Retail service request",
            "Resolution instructions",
            "Please handle my purchase",
            "For the service agent",
        ],
        "valid": ["A request about the following purchase", "Customer support instructions"],
        "test": ["Requested handling of my purchase", "Please resolve the following order issue"],
    }[split][n % (4 if split == "train" else 2)]
    message = f"{lead}: {text} {details}" if n % 2 == 0 else f"{lead}: {details} {text}"
    required = [missing] if missing else []
    if family == "missing_email_payment":
        required.append("payment_method")
    initial = {
        "operation": primary,
        "email": "unknown" if family in MISSING_EMAIL else email,
        "order_id": oid,
        "requested_option": option if primary == "exchange" and missing != "new_option" else "unknown",
        "payment_method": "unknown"
        if cancel or "payment_method" in required
        else "gift_card"
        if payment == gift
        else "original",
        "reason": reason if cancel and missing != "reason" else "unknown",
        "fallback_return": "yes" if family in YES_FAMILIES else "no" if family in NO_FAMILIES else "unknown",
    }
    complete = dict(initial)
    for field in required:
        complete["requested_option" if field == "new_option" else field] = {
            "email": email,
            "new_option": option,
            "payment_method": "original",
            "reason": reason,
        }[field]
    kind = (
        "handoff"
        if family in HANDOFF
        else "return"
        if family in {"exchange_fallback", "exchange_fallback_missing_payment"}
        else primary
    )
    customer = {
        "id": uid,
        "email": email,
        "gift_card": {"id": gift, "balance_cents": 200 if family == "exchange_shortfall" else 12500},
    }
    variants = [
        {"item_id": old, "option": "graphite / M", "price_cents": price, "available": True},
        {"item_id": new, "option": option, "price_cents": price + delta, "available": family not in STOCKOUT},
        {"item_id": alt, "option": "ivory / XS", "price_cents": price + 450, "available": True},
    ]
    rng.shuffle(variants)
    answers = {
        "email": email,
        "order_id": oid,
        "reason": reason,
        "new_option": option,
        "item_id": old,
        "payment_method": payment,
        "operation": primary,
        "fallback_return": "YES" if family in YES_FAMILIES else "NO",
    }
    # Reply style varies by split and index; gold field values remain separate.
    if family in MISSING_EMAIL and n % 2:
        answers["email"] = {
            "train": f"Use this account email address: {email}.",
            "valid": f"You can find the account under {email}.",
            "test": f"The email on file for me is {email}.",
        }[split]
    if "payment_method" in required and n % 2:
        answers["payment_method"] = {
            "train": "Please use the original payment method.",
            "valid": "Put it on the original payment, please.",
            "test": "Use original payment for that adjustment.",
        }[split]
    if missing == "new_option" and n % 2:
        answers["new_option"] = {
            "train": f"The replacement option I choose is {option}.",
            "valid": f"I would like the {option} variant.",
            "test": f"My choice of replacement is {option}.",
        }[split]
    if missing == "reason" and n % 2:
        answers["reason"] = {
            "train": "It was ordered by mistake.",
            "valid": "The order was placed by mistake.",
            "test": "I placed this order in error.",
        }[split]
    return {
        "id": f"v7-{split}-{family}-{n:02d}",
        "family": family,
        "split": split,
        "initial_message": message,
        "evidence_initial": text,
        "customer": customer,
        "products": {pid: {"product_id": pid, "name": "Linen utility shirt", "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": ident("other_") if family == "foreign_order" else uid,
                "status": status,
                "original_payment": card,
                "items": [
                    {"item_id": old, "product_id": pid, "option": "graphite / M", "price_cents": price}
                ],
            }
        ],
        "answers": answers,
        "intent_initial": initial,
        "intent_complete": complete,
        "required_clarification": missing,
        "required_answers": required,
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


def load_cases(split=None):
    cases = json.loads((ROOT / "training/v7/datasets/cases.json").read_text())
    return [c for c in cases if split is None or c["split"] == split]
