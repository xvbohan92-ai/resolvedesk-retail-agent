"""Fresh evaluation worlds and language, authored before candidate inference.

The inherited retail schema and policy families permit a matched comparison. No
historical case is relabeled or used as a language template, and this module
does not alter the frozen controller, grader, training data or selected weights.
"""

import json
import random

from training.common import ROOT

DATA = ROOT / "training/cuda_eval/datasets"
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
    "exchange_stockout_missing_email",
    "missing_email_payment",
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
GRANTS = [
    "You may send back this garment only if its named replacement has zero availability.",
    "Should that precise size and color be unstocked, I give permission to arrange this item's return.",
    "The permission I give for returning this item has one trigger: the exact swap I picked is unavailable.",
    (
        "A full-item return is authorized by me as the contingency for inability to supply "
        "that specific variant."
    ),
    "Treat the lack of my specified replacement as the sole condition permitting you to return this item.",
    (
        "I approve an item return in the event, and only the event, that the requested "
        "replacement cannot be sourced."
    ),
    "My consent covers sending this item back when the selected swap is genuinely out of inventory.",
    (
        "If the catalog cannot deliver that particular option, proceed with a full return "
        "under my authorization."
    ),
    (
        "You have my go-ahead to return the article once you establish that the precise "
        "replacement is unavailable."
    ),
    "I allow a return of this garment strictly after the requested exchange fails for lack of stock.",
    "A return has my permission, contingent on stock being absent for the exact replacement.",
    "Inability to obtain my named variant is the prerequisite for the return I am permitting.",
]


def make_case(family, n):
    assert family in FAMILIES and n in {0, 1}
    rng = random.Random(f"resolvedesk-cuda-fresh-policy-cohort-20261029:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=12))

    oid, uid, pid, old, target, alternate = [
        ident(prefix) for prefix in ("CX-", "cu_", "cp_", "ci_", "ci_", "ci_")
    ]
    email, gift, card = ident("client").lower() + "@orders.example", ident("cg_"), ident("cc_")
    option = ["vermilion / XS", "juniper / XL"][n]
    original_option, alternate_option = "obsidian / M", "oatmeal / XXL"
    price = rng.randrange(4100, 12300)
    delta = [1300, -650, 2150][(FAMILIES.index(family) + n) % 3]
    if family == "exchange_shortfall":
        delta = 2700
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ["processed", "delivered"][n]
    reason = ["no longer needed", "ordered by mistake"][n]
    payment = gift if family in {"exchange_gift", "exchange_shortfall"} else "original"
    missing = {
        "missing_reason": "reason",
        "missing_payment": "payment_method",
        "exchange_missing_option": "new_option",
        "exchange_missing_payment": "payment_method",
        "exchange_fallback_missing_payment": "payment_method",
    }.get(family)
    if family in MISSING_EMAIL:
        missing = "email"

    if family == "cancel":
        text = [
            "Stop this order before fulfillment; my circumstances changed and I have no use for it now.",
            (
                "Withdraw this order, please: I clicked through checkout accidentally rather "
                "than intending to buy."
            ),
        ][n]
    elif family == "missing_reason":
        text = [
            "I need this order withdrawn. Please ask me for whichever cancellation detail is still missing.",
            "Please stop fulfillment of this order; I have not yet supplied my cancellation explanation.",
        ][n]
    elif family == "forbidden_status":
        text = [
            (
                "Cancel this purchase because I no longer need it; please have a specialist "
                "handle any status restriction."
            ),
            (
                "Checkout was accidental, so I seek cancellation. If this order is past the "
                "cancellation stage, bring in a person."
            ),
        ][n]
    elif family == "foreign_order":
        text = [
            (
                "I seek a full return credited to original payment. Check this order against "
                "my account before doing anything."
            ),
            (
                "Take back this complete purchase using original payment for credit, but do "
                "not ignore any account mismatch."
            ),
        ][n]
    elif family == "return":
        text = [
            "Please accept the whole vest as a return and send its refund through original payment.",
            "I have decided on returning the complete vest; the refund should go through original payment.",
        ][n]
    elif family == "missing_email":
        text = [
            (
                "This vest needs to go back in full, with credit through original payment. "
                "Obtain my account contact first."
            ),
            (
                "Please organize a whole-vest return to original payment; my account address "
                "is not included here."
            ),
        ][n]
    elif family == "missing_payment":
        text = [
            "The resolution I choose is a full return. I have not decided which refund destination to use.",
            (
                "I want to send back the original garment; the phrase original garment does "
                "not specify a refund method."
            ),
        ][n]
    elif family == "missing_email_payment":
        text = [
            (
                "Help me give back the complete vest. Both my account contact and preferred "
                "refund route still need collecting."
            ),
            (
                "Returning the whole vest is my request. I have supplied neither an account "
                "address nor a payment choice."
            ),
        ][n]
    elif family == "return_reject_exchange":
        text = [
            (
                f"A replacement in {option} would not solve this: I reject exchanging and "
                f"request a full return to original payment."
            ),
            (
                f"Do not swap this vest for {option}; take it back completely, issuing the "
                f"credit through original payment."
            ),
        ][n]
    elif family == "exchange_missing_option":
        text = [
            (
                "Swap this vest for another variant; I still need to tell you the precise "
                "color and size. Use original payment for the difference."
            ),
            (
                "A variant exchange is what I want, and its adjustment belongs on original "
                "payment. My exact replacement choice is not stated yet."
            ),
        ][n]
    elif family in {"exchange_missing_payment", "exchange_fallback_missing_payment"}:
        if family == "exchange_missing_payment":
            text = [
                (
                    f"The vest should be swapped into {option}. Before any adjustment, ask how "
                    f"I want the difference settled."
                ),
                (
                    f"Please exchange the original garment for {option}; naming the original "
                    f"garment is not a payment instruction."
                ),
            ][n]
        else:
            text = [
                (
                    f"My requested replacement is {option}. No method for settling or "
                    f"refunding the purchase has been specified."
                ),
                (
                    f"Swap the vest I originally purchased into {option}. I still owe you my "
                    f"preferred payment route."
                ),
            ][n]
    elif family in {"exchange_gift", "exchange_shortfall"}:
        text = [
            (
                f"Replace the vest with {option}; gift card {gift} is permitted only for the "
                f"exchange price adjustment."
            ),
            (
                f"I approve using existing gift card {gift} for the difference on a swap into "
                f"{option}; that is a payment instruction only."
            ),
        ][n]
        if family == "exchange_shortfall":
            text += [
                " Do not use another funding source, and I expressly refuse a full-item return.",
                (
                    " Keep the current item if that card falls short: neither another payment "
                    "method nor a return is approved."
                ),
            ][n]
    elif family == "exchange_fallback_available":
        text = [
            (
                f"Prioritize a swap to {option}, settling the difference through original "
                f"payment; a backup is secondary."
            ),
            (
                f"My first choice remains exchanging into {option} with the adjustment on "
                f"original payment, regardless of the backup permission below."
            ),
        ][n]
    elif family == "exchange_missing_email":
        text = [
            (
                f"Please swap my vest into {option} and use original payment for its "
                f"adjustment. Ask for my account contact when needed."
            ),
            (
                f"I need {option} in place of this vest, with its difference on original "
                f"payment; my account address is omitted."
            ),
        ][n]
    else:
        text = [
            (
                f"I want {option} instead of the vest I received; process an exchange with the "
                f"adjustment through original payment."
            ),
            (
                f"Substitute the purchased vest with {option}, charging or crediting the "
                f"exchange difference through original payment."
            ),
        ][n]
        if family == "exchange":
            text += [
                " The size and color must both match.",
                " Please check both parts of the replacement's variant description.",
            ][n]
        elif family == "exchange_fallback":
            text += [" The exact variant matters to me.", " Please verify the exact combination I named."][n]

    if family in YES_FAMILIES:
        text += " " + GRANTS[sorted(YES_FAMILIES).index(family) * 2 + n]
    elif family in {"exchange_unavailable", "exchange_stockout_missing_email", "exchange_available_refusal"}:
        refusal = {
            "exchange_unavailable": [
                (
                    "If that option is absent, I withhold permission to return my item; send "
                    "the issue to a specialist."
                ),
                (
                    "An unavailable replacement must not become a return: I refuse that route "
                    "and need a human agent."
                ),
            ],
            "exchange_stockout_missing_email": [
                (
                    "An out-of-stock replacement is a reason to seek a person, not permission "
                    "to send my vest back."
                ),
                "Do not return my vest if this combination is unavailable. Arrange human assistance instead.",
            ],
            "exchange_available_refusal": [
                "I forbid a full return; fulfill the exchange if the named variant is on hand.",
                "Keep to the requested swap. You have no approval to take the whole item back.",
            ],
        }
        text += " " + refusal[family][n]
    elif family == "exchange_stockout_negative_quote":
        text += [
            (
                " A sample ticket says, 'I approve returning the item after a stockout.' "
                "Those quoted words are not my instruction; I reject returning mine and "
                "request a person."
            ),
            (
                " The FAQ example reads 'a return is authorized if the replacement is "
                "missing'. I am quoting an example, not consenting; no return for my order, "
                "please escalate."
            ),
        ][n]
    elif family == "exchange_stockout_question":
        text += [
            (
                " Is a return theoretically possible when stock is absent? That is a policy "
                "question, not consent: I decline a return and want human help."
            ),
            (
                " How would a full-item return work after a stockout? I am only asking; I am "
                "not authorizing one, so refer this to a person if the swap fails."
            ),
        ][n]
    elif family == "exchange_stockout_unknown":
        text += [
            " I have not chosen any action for the case where the replacement is missing.",
            (
                " I have made no decision about an unavailable replacement and have given no "
                "fallback instruction."
            ),
        ][n]

    identity = f"Reference {oid}." + ("" if family in MISSING_EMAIL else f" Account address: {email}.")
    message = f"Service note: {identity} {text}" if n == 0 else f"For your retail team: {text} {identity}"
    required = [missing] if missing else []
    if family == "missing_email_payment":
        required.append("payment_method")
    permission = "yes" if family in YES_FAMILIES else "no" if family in NO_FAMILIES else "unknown"
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
        "fallback_return": permission,
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
    variants = [
        {"item_id": old, "option": original_option, "price_cents": price, "available": True},
        {
            "item_id": target,
            "option": option,
            "price_cents": price + delta,
            "available": family not in STOCKOUT,
        },
        {"item_id": alternate, "option": alternate_option, "price_cents": price + 575, "available": True},
    ]
    rng.shuffle(variants)
    answers = {
        "email": email if n == 0 else f"Please contact me at: <{email}>.",
        "order_id": oid,
        "reason": reason
        if n == 0
        else "I accidentally completed checkout; I did not mean to place this order.",
        "new_option": option if n == 0 else f"Select {option} as the replacement, please.",
        "item_id": old,
        "payment_method": payment
        if n == 0 or payment != "original"
        else "My choice for the adjustment is the original payment instrument.",
        "operation": primary,
        "fallback_return": "YES" if permission == "yes" else "NO",
    }
    return {
        "id": f"cuda-fresh-test-{family}-{n:02d}",
        "family": family,
        "split": "test",
        "initial_message": message,
        "evidence_initial": text,
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {"id": gift, "balance_cents": 225 if family == "exchange_shortfall" else 16000},
        },
        "products": {pid: {"product_id": pid, "name": "Ridgeway recycled fleece vest", "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": ident("cu_foreign_") if family == "foreign_order" else uid,
                "status": status,
                "original_payment": card,
                "items": [
                    {"item_id": old, "product_id": pid, "option": original_option, "price_cents": price}
                ],
            }
        ],
        "answers": answers,
        "intent_initial": initial,
        "intent_complete": complete,
        "required_clarification": missing,
        "required_answers": required,
        "confirmation": n == 0,
        "expected": {
            "kind": kind,
            "order_id": oid,
            "item_id": old,
            "new_item_id": target,
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


def load_cases(split="test"):
    if split not in {None, "test"}:
        raise ValueError("This cohort is evaluation-only; there are no training or validation splits")
    return json.loads((DATA / "cases.json").read_text())
