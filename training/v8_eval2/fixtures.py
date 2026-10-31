"""Fresh authored requests and independent retail worlds; no training split.

The business families deliberately match the prior evaluation. Wording, product
facts, catalog values and entity namespaces are newly authored, not an old case
with a different ID. No earlier model predictions or success rates are inputs.
"""

# ruff: noqa: E501 -- Preserve literal authored customer utterances.
import json
import random

from training.common import ROOT
from training.cuda_eval.fixtures import (
    FAMILIES as POLICY_FAMILIES,
)
from training.cuda_eval.fixtures import HANDOFF, MISSING_EMAIL, NO_FAMILIES, STOCKOUT, YES_FAMILIES

DATA = ROOT / "training/v8_eval2/datasets"
FAMILIES = tuple(POLICY_FAMILIES)
SEED_NAMESPACE = "resolvedesk-v8-eval2-authored-worlds-20261029"

PRODUCTS = (
    ("Alderreach quilted jacket", "jacket"),
    ("Brackenmere cotton blouse", "blouse"),
    ("Cairnford cable sweater", "sweater"),
    ("Dovecross twill overshirt", "overshirt"),
    ("Elmstead fleece gilet", "gilet"),
    ("Fennbrook jersey dress", "dress"),
    ("Gorsehaven rugby shirt", "rugby shirt"),
    ("Hawthornquay linen tunic", "tunic"),
    ("Iverstone knitted polo", "polo"),
    ("Junebridge denim skirt", "skirt"),
    ("Kilmere canvas anorak", "anorak"),
    ("Larchgate wool waistcoat", "waistcoat"),
    ("Mallowbank corduroy shirt", "shirt"),
    ("Northwren merino cardigan", "cardigan"),
    ("Oakenshore jersey hoodie", "hoodie"),
    ("Pineford chambray smock", "smock"),
    ("Quillhaven cotton pullover", "pullover"),
    ("Reedcross raincoat", "raincoat"),
    ("Stonecove brushed sweatshirt", "sweatshirt"),
    ("Ternbrook seersucker top", "top"),
    ("Umberfield knitted vest", "vest"),
    ("Valewick poplin blouse", "blouse"),
    ("Willowreach flannel overshirt", "overshirt"),
    ("Yarrowford padded jacket", "jacket"),
    ("Ashenquay pique polo", "polo"),
    ("Briarcross wool sweater", "sweater"),
    ("Cedarhaven cotton cardigan", "cardigan"),
    ("Dunlinmere linen dress", "dress"),
    ("Eidergate terry hoodie", "hoodie"),
    ("Foxglovecove cotton tunic", "tunic"),
    ("Granitebrook windbreaker", "windbreaker"),
    ("Hazelshore woven shirt", "shirt"),
    ("Ivycross textured pullover", "pullover"),
    ("Jackdawfield jersey skirt", "skirt"),
    ("Knotgrassquay quilted gilet", "gilet"),
    ("Lindenreach cotton smock", "smock"),
    ("Meadowgate wool waistcoat", "waistcoat"),
    ("Nettleford knit top", "top"),
    ("Orchardcove brushed anorak", "anorak"),
    ("Plovermere denim shirt", "shirt"),
    ("Rowanfield fleece sweatshirt", "sweatshirt"),
    ("Sorrelcross rain jacket", "jacket"),
    ("Thistlegate jersey blouse", "blouse"),
    ("Vervainshore cotton polo", "polo"),
    ("Wheatbrook knitted dress", "dress"),
    ("Yewhaven twill tunic", "tunic"),
    ("Zephyrford linen cardigan", "cardigan"),
    ("Bitternquay wool pullover", "pullover"),
)
TARGET_COLORS = (
    "porcelain blue",
    "terracotta rose",
    "glacial mint",
    "saffron cream",
    "dusk violet",
    "saltwater green",
    "burnished copper",
    "cherrywood red",
    "pale pewter",
    "fern smoke",
    "dried apricot",
    "riverglass blue",
    "heather charcoal",
    "moss silver",
    "mulberry clay",
    "buttercup gold",
    "storm lilac",
    "seagrass ivory",
    "rosewood tan",
    "cobalt haze",
    "faded marigold",
    "lapis grey",
    "chestnut blush",
    "reed yellow",
    "shell pink",
    "basalt olive",
    "winter persimmon",
    "harvest teal",
    "flax plum",
    "chalk jade",
    "aubergine mist",
    "pearl russet",
    "moraine beige",
    "thunder sage",
    "cerulean sand",
    "cinnamon slate",
    "marsh lavender",
    "antique coral",
    "sleet turquoise",
    "pomegranate ash",
    "dandelion smoke",
    "peacock stone",
    "silt raspberry",
    "copperleaf blue",
    "misted ochre",
    "woodland blush",
    "bluebell rust",
    "almond marine",
)
ORIGINAL_COLORS = (
    "deep auburn",
    "quarry green",
    "tobacco grey",
    "raven mauve",
    "cinder cream",
    "juniper sand",
    "dark marzipan",
    "currant bronze",
    "loden pearl",
    "charred jade",
    "mulch blue",
    "umber fog",
    "plum granite",
    "walnut ice",
    "willow burgundy",
    "bronzed oat",
    "inked copper",
    "hazelnut sky",
    "smoked quartz",
    "peat blossom",
    "tar sage",
    "pumice grape",
    "russet dusk",
    "lichen navy",
    "coffee frost",
    "ebony clay",
    "truffle jade",
    "burnt agate",
    "shadow maize",
    "iron blush",
    "bark azure",
    "carbon wheat",
    "mushroom wine",
    "charcoal peach",
    "lichen chestnut",
    "taupe emerald",
    "dried fern",
    "umber pearl",
    "malt indigo",
    "raven beige",
    "slate nutmeg",
    "walnut seafoam",
    "deep cassis",
    "charred linen",
    "lichen espresso",
    "plum graphite",
    "mossed copper",
    "night saffron",
)
ALTERNATE_COLORS = (
    "citrine mist",
    "opal bark",
    "frost plum",
    "pistachio dusk",
    "seashell brass",
    "morning garnet",
    "silver fig",
    "turmeric frost",
    "petal steel",
    "verdigris oat",
    "amber cloud",
    "bleached mulberry",
    "moonlit maize",
    "coastal lilac",
    "salmon quartz",
    "icicle rust",
    "sageberry pink",
    "chalk persimmon",
    "granite coral",
    "honeydew slate",
    "frosted carmine",
    "mauve cedar",
    "biscuit lapis",
    "tea rose grey",
    "ecru scarlet",
    "glazed juniper",
    "hemp sapphire",
    "pebble fuchsia",
    "wheat aquamarine",
    "zinc apricot",
    "stone tangerine",
    "misty currant",
    "bamboo crimson",
    "pale damson",
    "sanded cobalt",
    "creamy pine",
    "powder cinnamon",
    "olive pearl",
    "ginger glass",
    "grape flax",
    "pistachio ash",
    "celadon blush",
    "chalky wine",
    "soft ambergreen",
    "granite poppy",
    "peony shale",
    "dusty canary",
    "ivory petrol",
)

GRANTS = (
    "For the single contingency where my chosen replacement has no stock, you may return this item instead.",
    "I am authorizing a return of this item only after the replacement selection is confirmed unavailable.",
    "A stock failure for the replacement I finally select is the sole event in which I give you leave to return mine.",
    "My instruction includes a return contingency, but it takes effect only when my selected exchange cannot be supplied.",
    "You have my permission to arrange a return if, and only if, inventory lacks the exact replacement I pick.",
    "If the variant I choose turns out to be unstocked, consider the item return authorized by me.",
    "I consent to sending the item back as a return when the precise exchange option proves unavailable, and under no other condition.",
    "Only a confirmed stockout of my requested variant activates my authorization for an item return.",
    "Provided the selected replacement is unavailable, I authorize a return; a replacement in stock must still be exchanged.",
    "For an unavailable replacement of the specified size and color, my approved backup is a full-item return.",
    "My approval to return this purchase is restricted to a verified lack of the replacement I choose.",
    "You may turn the exchange into an item return solely when the chosen catalog variant cannot be supplied from stock.",
)


def make_case(family, n):
    if family not in FAMILIES or n not in (0, 1):
        raise ValueError("Expected a policy family and within-family index0/1")
    index = FAMILIES.index(family) * 2 + n
    rng = random.Random(f"{SEED_NAMESPACE}:{family}:{n}")

    def ident(prefix):
        return prefix + "".join(rng.choices("ABCDEFGHJKLMNPQRSTUVWXYZ23456789", k=13))

    oid, uid, pid, old, target, alternate = [
        ident(prefix)
        for prefix in (
            "R8E2-",
            "v8e2_user_",
            "v8e2_product_",
            "v8e2_sku_",
            "v8e2_sku_",
            "v8e2_sku_",
        )
    ]
    email = ident("v8e2shopper").lower() + "@cohort-retail.example"
    gift, card = ident("v8e2_gift_"), ident("v8e2_card_")
    product_name, noun = PRODUCTS[index]
    sizes = ("XS", "S", "M", "L", "XL", "XXL")
    option = f"{TARGET_COLORS[index]} / {sizes[(index + 2) % 6]}"
    original_option = f"{ORIGINAL_COLORS[index]} / {sizes[index % 6]}"
    alternate_option = f"{ALTERNATE_COLORS[index]} / {sizes[(index + 4) % 6]}"
    price = 4850 + index * 173 + rng.randrange(100, 450)
    delta = (-850, 1400, 2650, 0)[index % 4]
    if family == "exchange_shortfall":
        delta = 3950 + n * 275
    cancel = family in {"cancel", "missing_reason", "forbidden_status"}
    primary = "cancel" if cancel else "exchange" if family.startswith("exchange") else "return"
    status = "pending" if cancel else "delivered"
    if family == "forbidden_status":
        status = ("processed", "delivered")[n]
    reason = ("no longer needed", "ordered by mistake")[n]
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
    language = {
        "cancel": (
            f"The event that required this {noun} is no longer happening. Please cancel its order because I no longer need the purchase.",
            f"I meant to save the {noun} in my cart, but bought it by mistake. Cancel that accidental order for me.",
        ),
        "missing_reason": (
            f"Please put through a cancellation for the {noun}. I have not told you why I am canceling yet.",
            f"The {noun} order is the one I want canceled. Before closing it, ask me for the cancellation explanation.",
        ),
        "return": (
            f"After trying the {noun}, I have decided to send it back as a full return. Credit the refund to original payment.",
            f"I am keeping neither the item nor a replacement: return the whole {noun}, reimbursing original payment.",
        ),
        "missing_payment": (
            f"I want the {noun} from my original order returned in full. Original refers to that order; I still need to choose the refund destination.",
            f"Please arrange the return of this {noun}. I have not chosen a payment method for receiving the money back.",
        ),
        "exchange": (
            f"The {noun} I received is not the combination I want to keep. Exchange it for {option}, with any difference settled on original payment.",
            f"Please replace my {noun} with the {option} {noun}. Use original payment to charge or refund the exchange adjustment.",
        ),
        "exchange_fallback": (
            f"For the {noun}, first seek the {option} replacement and settle through original payment. I am asking for an exchange of that exact version.",
            f"I have picked the {option} {noun} for a swap. Adjust original payment and check its stock before proceeding.",
        ),
        "forbidden_status": (
            f"The reason for canceling this {noun} is that it is no longer needed. If its fulfillment status bars cancellation, a human should handle the request.",
            f"This {noun} was ordered by mistake, so I request cancellation. If the existing order status prevents it, connect me with staff.",
        ),
        "foreign_order": (
            f"Can you return this entire {noun} with the credit on original payment? Verify that the receipt actually belongs to my account before taking any action.",
            f"I would like a full {noun} return to original payment, subject to an ownership check. A receipt for a different customer needs staff attention.",
        ),
        "missing_email": (
            f"My decision is a full return of the {noun}, credited to original payment. I will give you the account email when you ask for it.",
            f"Take the {noun} back and refund original payment. The account email needed to locate me is still missing from my message.",
        ),
        "exchange_missing_option": (
            f"For this {noun}, I am weighing {option} against {alternate_option}; neither choice is final. I want an exchange through original payment, but you need to ask which version I choose.",
            f"I want to exchange the {noun}, settling original payment, but cannot yet choose between {alternate_option} and {option}. Obtain that choice rather than deciding for me.",
        ),
        "exchange_missing_payment": (
            f"Swap the {noun} on my original order for {option}. I have selected the replacement but no payment route for the price adjustment.",
            f"The replacement should be the {option} {noun}. This concerns the original purchase; I have not said how to settle an exchange difference.",
        ),
        "exchange_gift": (
            f"The swap I request is the {option} {noun}. Settle its price difference with gift card {gift}; that is a payment instruction for the exchange only.",
            f"Use gift card {gift} for an exchange into the {option} {noun}. Permission to use that card says nothing about returning the item.",
        ),
        "exchange_unavailable": (
            f"Please attempt the {option} {noun} exchange via original payment. I will not accept an item return as the fallback; a stock failure should go to a person.",
            f"An original-payment swap into {option} is my request for the {noun}. Do not arrange a return when it is unstocked; ask support to take over.",
        ),
        "exchange_shortfall": (
            f"Try to exchange my {noun} for {option}, using only gift card {gift} for the difference. If it cannot cover the cost, I do not approve a return or another funding source; transfer me to staff.",
            f"I choose the {option} {noun} with the adjustment on gift card {gift}. A low balance should mean human assistance, not a return or a different payment method.",
        ),
        "exchange_fallback_available": (
            f"My requested outcome is the {option} {noun} exchange using original payment. Supply that variant if it is in stock; the contingency must not replace an available swap.",
            f"Check whether the {option} {noun} can be exchanged through original payment. A possible backup is not a request to abandon a replacement that can be supplied.",
        ),
        "return_reject_exchange": (
            f"I have considered the {option} {noun} and decided against exchanging. Please return the whole item and refund original payment instead.",
            f"Do not place an exchange for {option}. The {noun} should be a complete return, with its refund on original payment.",
        ),
        "exchange_stockout_unknown": (
            f"Seek a {option} {noun} exchange and use original payment for its difference. I am not ready to make any decision about a stockout fallback.",
            f"I would like the {noun} replaced with {option}, adjusted through original payment. If that cannot be sourced, my next preference remains undecided.",
        ),
        "exchange_stockout_missing_email": (
            f"Collect my account email, then try exchanging the {noun} for {option} through original payment. I refuse an item return if stock is missing; have a representative handle that situation.",
            f"I have not provided my mailbox yet. I want the {option} {noun} as an original-payment exchange, and any stockout must go to support because I do not authorize a return.",
        ),
        "exchange_stockout_negative_quote": (
            f"A help article prints 'YES, return the item if the replacement is sold out'. That is a quoted example, not my consent. I want the {option} {noun} exchange through original payment; refuse a fallback return and contact a person if needed.",
            f"For the {noun}, request {option} and settle original payment. The phrase 'I approve a return after a stockout' appears in a sample chat I am quoting. I do not approve that return for mine; let staff handle an unavailable exchange.",
        ),
        "exchange_stockout_question": (
            f"Is an item return an option if the {option} {noun} is sold out? I am asking about policy, not authorizing a return. My request is an original-payment exchange, and I want staff help if it cannot be filled.",
            f"Please check the {option} {noun} exchange using original payment. What steps would a refund require after a stockout? Do not execute a return on the strength of that question; I decline it and prefer a person.",
        ),
        "exchange_available_refusal": (
            f"I approve the {option} {noun} exchange, with the adjustment on original payment. Keep that exchange if available: I am explicitly ruling out a full return.",
            f"Exchange this {noun} into {option} and settle through original payment. A full-item return is not permitted by me, even though the swap remains requested.",
        ),
        "exchange_fallback_missing_payment": (
            f"The {option} {noun} is my exchange selection for the original order. Before proposing anything, collect the payment method I want to use for its adjustment.",
            f"I want my original {noun} replaced by {option}. No funding or refund destination has been selected for this exchange yet.",
        ),
        "exchange_missing_email": (
            f"Please exchange the {noun} for the {option} {noun}, settling any difference through original payment. You still need to obtain my account email.",
            f"My replacement choice is {option} for the {noun}, and the adjustment belongs on original payment. Ask for the missing account contact first.",
        ),
        "missing_email_payment": (
            f"Return the {noun} from that original purchase. I have provided neither an account email nor a choice of where the refund should go.",
            f"I wish to return the original {noun} in full. You need two details from me: the account mailbox and my refund method, which I have not chosen here.",
        ),
    }
    body = language[family][n]
    if family in YES_FAMILIES:
        body += " " + GRANTS[sorted(YES_FAMILIES).index(family) * 2 + n]
    contact = "" if family in MISSING_EMAIL else f"\nAccount contact: {email}."
    message = (
        f"Order reference {oid}.\n{body}{contact}" if n == 0 else f"{body}\nReceipt number: {oid}.{contact}"
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
        {"item_id": old, "option": original_option, "price_cents": price, "available": index % 3 != 0},
        {
            "item_id": target,
            "option": option,
            "price_cents": price + delta,
            "available": family not in STOCKOUT,
        },
        {
            "item_id": alternate,
            "option": alternate_option,
            "price_cents": price + 620 + n * 330,
            "available": True,
        },
    ]
    rng.shuffle(variants)
    return {
        "id": f"v8-eval2-test-{family}-{n:02d}",
        "family": family,
        "split": "test",
        "initial_message": message,
        "evidence_initial": body,
        "coverage_tags": ["fresh_eval_only", "literal_quote_provenance", "product_noun_context"]
        + (["two_unchosen_variants"] if missing == "new_option" else [])
        + (["unspecified_payment_despite_original_order"] if "payment_method" in required else []),
        "customer": {
            "id": uid,
            "email": email,
            "gift_card": {
                "id": gift,
                "balance_cents": 390 if family == "exchange_shortfall" else 23100 + index * 25,
            },
        },
        "products": {pid: {"product_id": pid, "name": product_name, "variants": variants}},
        "orders": [
            {
                "id": oid,
                "user_id": ident("v8e2_other_") if family == "foreign_order" else uid,
                "status": status,
                "original_payment": card,
                "items": [
                    {"item_id": old, "product_id": pid, "option": original_option, "price_cents": price}
                ],
            }
        ],
        "answers": {
            "email": f"Please use {email} as my account email."
            if n == 0
            else f"My account email is <{email}>.",
            "order_id": oid,
            "reason": reason,
            "new_option": f"My choice is {option} for this {noun}.",
            "item_id": old,
            "payment_method": "original" if payment == "original" else gift,
            "operation": primary,
            "fallback_return": "YES" if permission == "yes" else "NO",
        },
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
    if split not in (None, "test"):
        raise ValueError("This evaluation-only cohort has only a test split")
    return json.loads((DATA / "cases.json").read_text())
