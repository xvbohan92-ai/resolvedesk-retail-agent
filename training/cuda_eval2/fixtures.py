"""New English requests and catalog worlds for a separately authorized evaluation.

Scenario policies and denominators are inherited for comparison. The language,
identities and worlds below are newly authored, not old cases with replacement
IDs. No prior model outputs, partial scores or failure rates are inputs here.
"""

import json
import random

from training.common import ROOT
from training.cuda_eval.fixtures import (
    FAMILIES as POLICY_FAMILIES,
)
from training.cuda_eval.fixtures import (
    HANDOFF,
    MISSING_EMAIL,
    NO_FAMILIES,
    STOCKOUT,
    YES_FAMILIES,
)

DATA = ROOT / "training/cuda_eval2/datasets"
FAMILIES = tuple(POLICY_FAMILIES)
GRANTS = (
    "If there is no stock in the exact color and size above, I am happy for you to set up a return instead.",
    (
        "My fallback instruction is to return this jacket should you be unable to provide "
        "that exact replacement."
    ),
    (
        "You can return the jacket with my approval, but only after confirming that the "
        "replacement I selected cannot be supplied."
    ),
    (
        "Please treat a sold-out match for my chosen variant as permission from me to switch "
        "to a complete return."
    ),
    "I consent to giving the whole jacket back if stock prevents the particular swap I am asking for.",
    "Provided the specified replacement is unavailable, I agree to resolve this by returning the jacket.",
    "I authorize taking this jacket back as my backup for a stockout of the exact exchange selection.",
    (
        "If you cannot fill the requested size and color combination, returning this jacket "
        "is the alternative I approve."
    ),
    "Only a failure to obtain that precise replacement would activate my consent to a full return.",
    "In the event that my chosen exchange option is sold out, I want you to proceed with an item return.",
    "The jacket may be returned under my permission when the specific replacement cannot be fulfilled.",
    (
        "If the listed variant cannot actually be supplied, I give the go-ahead for a full "
        "return of this jacket."
    ),
)


def make_case(family, n):
    assert family in FAMILIES and n in {0, 1}
    rng = random.Random(f"resolvedesk-cuda-eval-002-new-authored-world:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=13))

    oid, uid, pid, old, target, alternate = [
        ident(prefix) for prefix in ("NE-", "ne_u_", "ne_p_", "ne_i_", "ne_i_", "ne_i_")
    ]
    email = ident("shopper").lower() + "@purchases.example"
    gift, card = ident("ne_gift_"), ident("ne_card_")
    option = ("topaz / M", "eucalyptus / S")[n]
    original_option, alternate_option = "hematite / XL", "buttermilk / XS"
    price = rng.randrange(5800, 14500)
    delta = (1550, -975, 2300)[(FAMILIES.index(family) + n) % 3]
    if family == "exchange_shortfall":
        delta = 3200
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ("delivered", "processed")[n]
    reason = ("ordered by mistake", "no longer needed")[n]
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
        text = (
            (
                "This order went through after an accidental checkout tap. Please cancel it; "
                "I did not intend to buy the jacket."
            ),
            (
                "Please cancel this purchase. The event I bought it for was called off, so I "
                "do not need the jacket now."
            ),
        )[n]
    elif family == "missing_reason":
        text = (
            "Could you cancel my jacket order? I can explain why when you ask.",
            "I am contacting you to stop this purchase order. You have not received my reason yet.",
        )[n]
    elif family == "forbidden_status":
        text = (
            (
                "I made this purchase unintentionally and would like it canceled. If its "
                "current status prevents cancellation, please connect me with support staff."
            ),
            (
                "There is no longer any need for this jacket, so I request cancellation. A "
                "human can help if fulfillment has already progressed too far."
            ),
        )[n]
    elif family == "foreign_order":
        text = (
            (
                "Please look into returning this jacket and refunding original payment, "
                "starting with a check that the order belongs to my account."
            ),
            (
                "I would like a full return, with credit to original payment. Verify the "
                "account holder before offering any transaction."
            ),
        )[n]
    elif family == "return":
        text = (
            (
                "I have the jacket and want to return all of it. Please issue the refund back "
                "to original payment."
            ),
            "A complete jacket return is my decision; put the reimbursement through original payment.",
        )[n]
    elif family == "missing_email":
        text = (
            (
                "I am sending back the entire jacket for a refund to original payment. I have "
                "not included the email needed to find my account."
            ),
            (
                "Can you arrange a full jacket return to original payment? First collect the "
                "account address you need from me."
            ),
        )[n]
    elif family == "missing_payment":
        text = (
            "I would like to return the jacket. I can choose the refund route once you have found the order.",
            (
                "Please take back the jacket from my original order. We have not discussed "
                "how the refund should be issued."
            ),
        )[n]
    elif family == "missing_email_payment":
        text = (
            (
                "Please help with a return of the whole jacket. You will need to ask for my "
                "account email and where I want the refund."
            ),
            (
                "I want this jacket returned in full; the account contact and refund "
                "preference are both still to be supplied."
            ),
        )[n]
    elif family == "return_reject_exchange":
        text = (
            (
                f"Even {option} would not be suitable. I want to return the jacket to original "
                f"payment, without an exchange."
            ),
            (
                f"Please refund original payment for the whole jacket rather than replacing it "
                f"with {option}. I am not interested in a swap."
            ),
        )[n]
    elif family == "exchange_missing_option":
        text = (
            (
                "I would like a jacket exchange, with the difference handled through original "
                "payment. I have yet to give you my new size and color."
            ),
            (
                "Please arrange a swap to another jacket variant and use original payment for "
                "its adjustment; ask me which variant I want."
            ),
        )[n]
    elif family == "exchange_missing_payment":
        text = (
            (
                f"Swap the jacket for {option}; let me choose how to settle the difference "
                f"after you check the order."
            ),
            (
                f"I would like {option} as the replacement. I will confirm the payment method "
                f"for any balance once you look it up."
            ),
        )[n]
    elif family == "exchange_fallback_missing_payment":
        text = (
            f"Could you switch the jacket to {option}? I am not choosing a payment source in this message.",
            (
                f"The {option} jacket is the replacement I need. Let us settle the payment "
                f"details after you locate my order."
            ),
        )[n]
    elif family in {"exchange_gift", "exchange_shortfall"}:
        text = (
            (
                f"For a swap to {option}, use gift card {gift} for any price difference. That "
                f"permission is about the exchange payment."
            ),
            (
                f"Please change the jacket to {option}, applying the adjustment to gift card "
                f"{gift}; the card instruction concerns that adjustment alone."
            ),
        )[n]
        if family == "exchange_shortfall":
            text += (
                (
                    " If the card cannot cover it, do not use another method or begin a "
                    "return; ask a representative to help."
                ),
                (
                    " I decline both a different payment source and a full return if that "
                    "card does not have enough funds."
                ),
            )[n]
    elif family == "exchange_fallback_available":
        text = (
            (
                f"My preferred resolution is a swap into {option}, with the adjustment on "
                f"original payment. Use the backup only if needed."
            ),
            (
                f"I still want to exchange for {option} and settle the difference through "
                f"original payment; the contingency below does not replace that first choice."
            ),
        )[n]
    elif family == "exchange_missing_email":
        text = (
            (
                f"I need a swap into {option}, using original payment for the difference. "
                f"Please obtain my account email before proceeding."
            ),
            (
                f"Exchange this jacket for {option} and adjust original payment. You will need "
                f"my account contact, which I have not given here."
            ),
        )[n]
    else:
        text = (
            (
                f"Please exchange the jacket I received for {option}; apply any price "
                f"adjustment through original payment."
            ),
            (
                f"I would like to change my jacket to {option}, with the exchange difference "
                f"settled using original payment."
            ),
        )[n]
        if family == "exchange":
            text += (
                " Please check the stated size as well as the color.",
                " Both variant attributes should match my selection.",
            )[n]
        elif family == "exchange_fallback":
            text += (" I need that particular combination.", " Check that precise option, please.")[n]

    if family in YES_FAMILIES:
        text += " " + GRANTS[sorted(YES_FAMILIES).index(family) * 2 + n]
    elif family in {"exchange_unavailable", "exchange_stockout_missing_email", "exchange_available_refusal"}:
        text += {
            "exchange_unavailable": (
                (
                    " If you cannot supply it, do not start a return. I want a support "
                    "representative to take over."
                ),
                (
                    " A stockout does not give you my agreement to return the jacket; I "
                    "refuse that and request human assistance."
                ),
            ),
            "exchange_stockout_missing_email": (
                (
                    " Leave my jacket as it is if stock is missing; I do not approve a "
                    "return, so ask a person to assist."
                ),
                (
                    " Please involve a human if this option is sold out. I am explicitly "
                    "declining any full-item return."
                ),
            ),
            "exchange_available_refusal": (
                " A full jacket return is not approved by me; fulfill the specified swap if it is available.",
                (
                    " I am saying no to an item return. Please carry out the named exchange "
                    "when inventory allows it."
                ),
            ),
        }[family][n]
    elif family == "exchange_stockout_negative_quote":
        text += (
            (
                " An earlier agent typed 'I agree to return it if the size is missing' as an "
                "example. That was not my statement: I do not consent to a return and want a "
                "person to help."
            ),
            (
                " The website example says 'you may return the item after an unavailable "
                "exchange'. I am quoting its example, not agreeing; do not return mine, "
                "please escalate to staff."
            ),
        )[n]
    elif family == "exchange_stockout_question":
        text += (
            (
                " What is the return time frame if the replacement is unavailable? I am "
                "asking for information only. Do not submit a return; please get a human if "
                "the exchange cannot happen."
            ),
            (
                " Would a full return be an option after a stockout? This is only a question, "
                "and I decline permission to perform it. Refer me to a person if necessary."
            ),
        )[n]
    elif family == "exchange_stockout_unknown":
        text += (
            " A stock problem would need a separate conversation; I have not decided on a backup.",
            (
                " I have not thought through what to do if this choice is not in inventory, "
                "so there is no fallback decision yet."
            ),
        )[n]

    contact = "" if family in MISSING_EMAIL else f" You can find my account using {email}."
    message = (
        f"Hello support, {text} The order reference is {oid}.{contact}"
        if n == 0
        else f"Regarding order {oid}.{contact}\n{text}"
    )
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
        {"item_id": alternate, "option": alternate_option, "price_cents": price + 625, "available": True},
    ]
    rng.shuffle(variants)
    answers = {
        "email": f"Contact me here: <{email}>." if n == 0 else email,
        "order_id": oid,
        "reason": "That checkout was unintentional; it was an accidental order." if n == 0 else reason,
        "new_option": f"I would like the {option} one, please." if n == 0 else option,
        "item_id": old,
        "payment_method": "Please put the credit or adjustment back through the payment I originally used."
        if n == 0 and payment == "original"
        else payment,
        "operation": primary,
        "fallback_return": "YES" if permission == "yes" else "NO",
    }
    return {
        "id": f"cuda-fresh2-test-{family}-{n:02d}",
        "family": family,
        "split": "test",
        "initial_message": message,
        "evidence_initial": text,
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {"id": gift, "balance_cents": 350 if family == "exchange_shortfall" else 18500},
        },
        "products": {
            pid: {"product_id": pid, "name": "Northbridge packable rain jacket", "variants": variants}
        },
        "orders": [
            {
                "id": oid,
                "user_id": ident("ne_foreign_") if family == "foreign_order" else uid,
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
        raise ValueError("This separate cohort has only a test split; no training or validation data")
    return json.loads((DATA / "cases.json").read_text())
