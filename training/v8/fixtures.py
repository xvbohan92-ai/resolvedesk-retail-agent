"""Authored train/validation worlds for V8, with no access to the new test cohort.

Six materially distinct language profiles per business family are used: four
training profiles and two validation profiles. This remains same-author synthetic
retail data under the inherited policy, not an external or human benchmark.
"""

import json
import random

from training.v7.fixtures import FAMILIES, HANDOFF, MISSING_EMAIL, NO_FAMILIES, STOCKOUT, YES_FAMILIES
from training.v8.common import DATA

SPLITS = {"v8-train": 4, "v8-valid": 2}
PRODUCTS = {
    "v8-train": ("Summitline brushed-cotton hoodie", "hoodie", "quartz / XXXL", "mulberry / XXS"),
    "v8-valid": ("Brookhaven zip-neck pullover", "pullover", "basalt / XXS", "tangerine / XXXL"),
}
OPTIONS = {
    "v8-train": ("carmine / XS", "cerulean / XL", "pewter / S", "wisteria / XXL"),
    "v8-valid": ("malachite / L", "sunflower / M"),
}
EXCHANGE_TEXT = (
    (
        "Would you swap my {noun} for the {surface}? Charge or credit the difference to the "
        "original payment method."
    ),
    (
        "The {surface} is the replacement I am choosing; reconcile any price change on the "
        "payment method used at checkout."
    ),
    (
        "Please process an exchange into {surface}, with the price adjustment going back "
        "through my original payment method."
    ),
    (
        "I want to trade the delivered {noun} for {surface}. Use the original payment method "
        "for either a debit or a credit."
    ),
    "Replace the {noun} I received with {surface}; my selected route for the adjustment is original payment.",
    (
        "My requested replacement is {surface}, and I choose the original payment method to "
        "settle its price difference."
    ),
)
GIFT_EXCHANGE_TEXT = (
    "Would you swap my {noun} for the {surface}? Use existing gift card {gift} for its adjustment.",
    "The replacement I choose is {surface}, with any price change settled on my existing gift card {gift}.",
    "Process an exchange into {surface}; debit or credit only the existing gift card {gift}.",
    "Trade my delivered {noun} for {surface}, using gift card {gift} to settle the price difference.",
    "My selected replacement is {surface} and my chosen adjustment route is existing gift card {gift}.",
    "Exchange the {noun} into {surface}. I select gift card {gift} for any difference in price.",
)
RETURN_TEXT = (
    "Take the delivered {noun} back, with reimbursement to my original payment method.",
    (
        "I am ending this purchase by returning the {noun}; send the credit to the payment "
        "method used at checkout."
    ),
    "Please open an item return for my {noun} and reimburse the original payment method.",
    "A refund to my original payment method is what I want for sending the whole {noun} back.",
    "Accept the purchased {noun} as a return and route its refund to original payment.",
    "The resolution I choose is a full return of the {noun}, credited through my original payment method.",
)
CANCEL_TEXT = (
    "Stop this purchase before fulfillment: I {reason_clause}.",
    "Please cancel the pending purchase because I {reason_clause}.",
    "Withdraw this order from processing; I {reason_clause}.",
    "I need the order canceled, as I {reason_clause}.",
    "End this order before it ships. The explanation is that I {reason_clause}.",
    "Cancel this purchase rather than send it out; I {reason_clause}.",
)
NO_REASON_TEXT = (
    "Please stop the pending order; I have not supplied the reason for canceling it yet.",
    "I am requesting cancellation and will explain why when you ask.",
    "Cancel the pending purchase, but first ask me which cancellation reason applies.",
    "Withdraw my order. My cancellation reason is missing from this message.",
    "I need this purchase canceled; please obtain my reason before proposing the cancellation.",
    "The action I want is cancellation. You still need to find out why I am canceling.",
)
PAYMENT_UNKNOWN_RETURN = (
    (
        "I want the {noun} from my original order taken back. Calling it the original order "
        "says nothing about where to send a refund."
    ),
    "Please return the {noun} on my original purchase. I have not chosen a refund payment method.",
    "The {noun} from the original order should be returned; ask me separately how to issue the credit.",
    "Arrange a return of my original {noun} purchase. The destination of the refund is still undecided.",
    (
        "Take back the {noun} associated with the original order. I am identifying the "
        "purchase, not selecting a payment channel."
    ),
    (
        "My original purchase needs an item return. Before the proposal, obtain my "
        "preference for receiving the refund."
    ),
)
PAYMENT_UNKNOWN_EXCHANGE = (
    (
        "Exchange the {noun} from my original order into {surface}. I have not selected how "
        "to settle the price adjustment."
    ),
    "My original purchase needs replacing with {surface}; the payment route for any difference is undecided.",
    (
        "I choose {surface} as the exchange for the original {noun}. Please ask where its "
        "debit or credit should go."
    ),
    "Trade the original-order {noun} for {surface}, after collecting the missing payment preference.",
    (
        "The {surface} is my replacement choice for this original purchase; I have yet to "
        "authorize an adjustment method."
    ),
    (
        "For the {noun} on my original order, arrange an exchange into {surface}. Which "
        "payment channel to use is a separate unanswered question."
    ),
)
OPTION_UNKNOWN = (
    (
        "I need a replacement {noun}, but have not picked its exact color and size. Use the "
        "original payment method for the adjustment."
    ),
    (
        "I am considering {option} or {alternative} for the exchange and have not chosen "
        "between them. Settle any price difference on original payment."
    ),
    (
        "Please exchange my {noun}. The replacement variant is undecided; the adjustment "
        "should use original payment."
    ),
    (
        "Either {option} or {alternative} might work as a replacement, but neither is a "
        "final selection. My payment preference is original payment."
    ),
    (
        "I want a {noun} exchange, with no definite replacement option yet. Apply the "
        "eventual price change to original payment."
    ),
    (
        "My shortlist is {option} and {alternative}; that list is not a choice. An exchange "
        "is wanted and its adjustment should go to original payment."
    ),
)
GRANTS = (
    (
        "My advance instruction is to return the item only if the particular replacement I "
        "selected cannot be furnished."
    ),
    "I am agreeing to the item's return solely as a contingency for a stockout of my chosen replacement.",
    (
        "A return has my permission in the limited circumstance that inventory cannot supply "
        "the variant I requested."
    ),
    (
        "I authorize taking the item back if the selected exchange cannot be met; that is "
        "the condition attached to my consent."
    ),
    (
        "You have my conditional go-ahead for an item return should the precise replacement "
        "choice prove unavailable."
    ),
    (
        "My consent covers returning this item only when the exact exchange I specified is "
        "impossible to fulfill."
    ),
)
REFUSALS = (
    "Do not take my item back if that swap cannot happen. Retain it and refer the issue to a person.",
    "A failed exchange does not carry my permission to return anything; I want human assistance instead.",
    (
        "There is no return authorization from me. If the named variant is unavailable, a "
        "specialist must handle it."
    ),
    (
        "Keep the existing item in place if you cannot fill the swap. I decline any return "
        "fallback and request a person."
    ),
    "I forbid a return as the alternative to a stockout; let a human support agent resolve that situation.",
    "My decision is no return when the replacement is missing. Preserve the item and involve a staff member.",
)
FAMILY_NOTES = {
    "cancel": (
        "The supplied reason is my cancellation reason.",
        "This concerns cancellation before shipment.",
    ),
    "missing_reason": (
        "Please do not invent an explanation for me.",
        "There is no cancellation explanation to infer yet.",
    ),
    "return": ("I am not asking for another size or color.", "No exchange is part of this request."),
    "missing_payment": (
        "A stored card on the order is not my current refund choice.",
        "The order history must not stand in for my missing payment preference.",
    ),
    "exchange": (
        "The replacement is my preferred outcome.",
        "Please fulfill the chosen exchange if it can be supplied.",
    ),
    "exchange_fallback": (
        "Check the exact selected replacement before using that contingency.",
        "The stock decision must concern the replacement I named.",
    ),
    "forbidden_status": (
        "Please check the actual fulfillment status before doing anything.",
        "Whether the order still permits cancellation must be verified.",
    ),
    "foreign_order": (
        "I know the order may be registered to another customer, but I am asking anyway.",
        "The account on that order might be someone else's; do not skip the ownership check.",
    ),
    "missing_email": (
        "I have omitted the account email and expect you to ask for it.",
        "You still need my account email to locate me.",
    ),
    "exchange_missing_option": (
        "The named possibilities must not be treated as a completed selection.",
        "Ask for a single final variant instead of guessing one.",
    ),
    "exchange_missing_payment": (
        "Do not substitute order-history payment information for my answer.",
        "The payment selection has to come from me, not the order record.",
    ),
    "exchange_gift": (
        "My approval concerns the price adjustment only, not returning the item.",
        "This gift-card permission is not permission for an item-return fallback.",
    ),
    "exchange_unavailable": (
        "Do not choose an available substitute on my behalf.",
        "Another in-stock variant would not satisfy my selection.",
    ),
    "exchange_shortfall": (
        "I authorize neither an extra payment source nor an item return if the card falls short.",
        "If the gift-card funds are inadequate, do not invent another charge source or return permission.",
    ),
    "exchange_fallback_available": (
        "An available exact replacement takes priority over the return contingency.",
        "The contingency does not replace an exchange that can actually be fulfilled.",
    ),
    "return_reject_exchange": (
        "An exchange into {option} is explicitly rejected; I want the item returned.",
        "Do not replace this with {option}; my selection is a return, not an exchange.",
    ),
    "exchange_stockout_unknown": (
        "Please check availability first. I am not making a decision about any other resolution here.",
        "I have only chosen the exchange and have made no statement about an alternative.",
    ),
    "exchange_stockout_missing_email": (
        "My email is not included; request it before looking up the order.",
        "You need to collect the missing account email, even if stock cannot satisfy the replacement.",
    ),
    "exchange_stockout_negative_quote": (
        (
            "A chatbot draft said 'authorize an item return if unavailable'; I reject that "
            "draft as an instruction for my order. I do not authorize any item return."
        ),
        (
            "The phrase 'a stockout permits a return' came from someone else's note, not my "
            "consent. I do not authorize it."
        ),
    ),
    "exchange_stockout_question": (
        (
            "Does a return have a deadline if the exchange fails? I am asking for "
            "information and expressly not approving a return."
        ),
        (
            "What would taking the item back involve? That question grants no permission; I "
            "decline a return for now."
        ),
    ),
    "exchange_available_refusal": (
        "My refusal of a return must not prevent an available exact exchange.",
        "I still want the specified exchange when available, despite refusing a return alternative.",
    ),
    "exchange_fallback_missing_payment": (
        "Even with that return contingency, a payment choice is still missing.",
        "My contingency authorization does not select a payment method.",
    ),
    "exchange_missing_email": (
        "The account email is absent from this request and must be collected.",
        "I have not provided the email for identifying my account.",
    ),
    "missing_email_payment": (
        "Both the account email and the refund payment choice need separate answers.",
        "Collect my omitted email as well as the unresolved payment preference.",
    ),
}


def make_case(split, family, n):
    if split not in SPLITS or family not in FAMILIES or not 0 <= n < SPLITS[split]:
        raise ValueError("V8 development construction accepts only its declared train/valid cases")
    profile = n if split == "v8-train" else n + 4
    rng = random.Random(f"resolvedesk-v8dev-authored-world:{split}:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=11))

    oid, uid, pid, old, new, alt = [
        ident(prefix) for prefix in ("ORD-V8D", "v8dev_u_", "v8dev_p_", "v8dev_i_", "v8dev_i_", "v8dev_i_")
    ]
    email = ident("v8dev_buyer_").lower() + "@purchase.example"
    gift, card = ident("v8dev_gift_"), ident("v8dev_card_")
    name, noun, old_option, alternative = PRODUCTS[split]
    option = OPTIONS[split][n]
    surface = option + (f" {noun}" if profile % 3 != 2 else "")
    price = rng.randrange(3900, 12500)
    delta = 2700 if family == "exchange_shortfall" else (-650, 850, 1750)[(FAMILIES.index(family) + n) % 3]
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    operation = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    reason = "no longer needed" if profile % 2 == 0 else "ordered by mistake"
    reason_clause = "no longer need it" if reason == "no longer needed" else "placed it by mistake"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = "processed" if n % 2 == 0 else "delivered"
    required = []
    if family in MISSING_EMAIL:
        required.append("email")
    if family == "missing_reason":
        required.append("reason")
    if family == "exchange_missing_option":
        required.append("new_option")
    if family in {
        "missing_payment",
        "exchange_missing_payment",
        "exchange_fallback_missing_payment",
        "missing_email_payment",
    }:
        required.append("payment_method")
    values = {
        "noun": noun,
        "surface": surface,
        "option": option,
        "alternative": alternative,
        "reason_clause": reason_clause,
        "gift": gift,
    }
    if family == "missing_reason":
        text = NO_REASON_TEXT[profile]
    elif cancel:
        text = CANCEL_TEXT[profile].format(**values)
    elif family == "exchange_missing_option":
        text = OPTION_UNKNOWN[profile].format(**values)
        if profile in {0, 2, 4}:
            text += f" The sample label '{option}' is quoted from a listing, not selected as my replacement."
    elif operation == "exchange" and "payment_method" in required:
        text = PAYMENT_UNKNOWN_EXCHANGE[profile].format(**values)
    elif family in {"exchange_gift", "exchange_shortfall"}:
        text = GIFT_EXCHANGE_TEXT[profile].format(**values)
    elif operation == "exchange":
        text = EXCHANGE_TEXT[profile].format(**values)
    elif "payment_method" in required:
        text = PAYMENT_UNKNOWN_RETURN[profile].format(**values)
    else:
        text = RETURN_TEXT[profile].format(**values)
    if family in YES_FAMILIES:
        text += " " + GRANTS[profile]
    elif family in {"exchange_unavailable", "exchange_stockout_missing_email", "exchange_available_refusal"}:
        text += " " + REFUSALS[profile]
    text += " " + FAMILY_NOTES[family][profile % 2].format(**values)
    details = f"Purchase identifier {oid}."
    if family not in MISSING_EMAIL:
        details += f" Account contact {email}."
    message = f"{text} {details}" if profile % 2 == 0 else f"{details} {text}"
    gift_payment = family in {"exchange_gift", "exchange_shortfall"}
    payment = gift if gift_payment else "original"
    initial = {
        "operation": operation,
        "email": "unknown" if "email" in required else email,
        "order_id": oid,
        "requested_option": option if operation == "exchange" and "new_option" not in required else "unknown",
        "payment_method": "unknown"
        if cancel or "payment_method" in required
        else "gift_card"
        if gift_payment
        else "original",
        "reason": reason if cancel and "reason" not in required else "unknown",
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
    answers = {
        "email": f"My account email is {email}.",
        "order_id": oid,
        "reason": "I no longer need this purchase."
        if reason == "no longer needed"
        else "I placed the purchase accidentally.",
        "new_option": f"I choose the {option} {noun}, just that variant.",
        "item_id": old,
        "payment_method": "Please settle it on the original payment method.",
        "operation": operation,
        "fallback_return": "YES" if family in YES_FAMILIES else "NO",
    }
    # Each entry is an authored answer to an ACTUAL requested field, not a
    # proactive customer turn. Unknown replies must not supply the final slot.
    reply_sequences = {
        field: [
            {
                "text": answers[field],
                "value": complete["requested_option" if field == "new_option" else field],
            }
        ]
        for field in required
    }
    # Available only after an ACTUAL operation-confirmation question. This is
    # the initial requested operation, not the business expected outcome.
    reply_sequences["operation"] = [{"text": operation, "value": operation}]
    if family == "exchange_missing_option" and profile in {1, 3, 5}:
        reply_sequences["new_option"].insert(
            0,
            {
                "text": f"I am still weighing {option} against {alternative}; I cannot select one yet.",
                "value": "unknown",
            },
        )
    if family in {"missing_payment", "exchange_missing_payment", "missing_email_payment"} and profile in {
        0,
        2,
        4,
    }:
        reply_sequences["payment_method"].insert(
            0,
            {
                "text": (
                    "I meant the original order number, not a payment preference; "
                    "I have not decided that yet."
                ),
                "value": "unknown",
            },
        )
    if family in {"missing_email", "exchange_missing_email"} and profile in {1, 3, 5}:
        reply_sequences["email"].insert(
            0, {"text": "I have not found the account address yet.", "value": "unknown"}
        )
    kind = (
        "handoff"
        if family in HANDOFF
        else "return"
        if family in {"exchange_fallback", "exchange_fallback_missing_payment"}
        else operation
    )
    variants = [
        {"item_id": old, "option": old_option, "price_cents": price, "available": True},
        {"item_id": new, "option": option, "price_cents": price + delta, "available": family not in STOCKOUT},
        {"item_id": alt, "option": alternative, "price_cents": price + 500, "available": True},
    ]
    rng.shuffle(variants)
    return {
        "id": f"{split}-{family}-{n:02d}",
        "family": family,
        "split": split,
        "development_profile": profile,
        "initial_message": message,
        "instruction_text": text,
        "evidence_initial": " ".join(text.split()[:14]),
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {"id": gift, "balance_cents": 150 if family == "exchange_shortfall" else 18000},
        },
        "products": {pid: {"product_id": pid, "name": name, "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": ident("v8dev_foreign_") if family == "foreign_order" else uid,
                "status": status,
                "original_payment": card,
                "items": [{"item_id": old, "product_id": pid, "option": old_option, "price_cents": price}],
            }
        ],
        "answers": answers,
        "reply_sequences": reply_sequences,
        "intent_initial": initial,
        "intent_complete": complete,
        "required_clarification": required[0] if required else None,
        "required_answers": required,
        "confirmation": profile % 2 == 0,
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
    normalized = {"train": "v8-train", "valid": "v8-valid"}.get(split, split)
    if normalized is not None and normalized not in SPLITS:
        raise ValueError("V8 development fixtures never load held-out test")
    cases = json.loads((DATA / "cases.json").read_text())
    return [case for case in cases if normalized is None or case["split"] == normalized]
