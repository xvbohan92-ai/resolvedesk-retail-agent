"""Arithmetic facts from completed tools; no customer-intent or answer routing."""

import json

CONTRACT = """
Decision recovery contract:
decision_facts lists every observed exchange candidate, not a recommended option.
Match the customer's exact requested option; another available variant is not a substitute.
For the matched option, inspect available and the authorized gift card's covers_difference.
If unavailable, a return requires explicit customer return-fallback authorization; otherwise handoff.
If the authorized gift card cannot cover the difference and no alternative was authorized, handoff.
Never change an exchange into a return just to avoid a payment shortfall.
An order's original_payment identifies a payment instrument, not a customer refund preference.
Do not invent email, cancellation reason, requested option or payment preference. If missing,
ask that field even when you already inspected the order/catalog. Once the customer replies,
reuse clarification_memory; do not ask the answered field again. An observed ownership or
status boundary takes priority over irrelevant reason/payment questions.
"""


def augment_messages(prompt):
    visible = json.loads(prompt[1]["content"])
    reads = visible.get("completed_reads", [])
    cards = [
        o["result"]["gift_card"]
        for o in reads
        if o["action"]["tool"] == "identify_customer" and "gift_card" in o["result"]
    ]
    candidates = []
    for o in reads:
        if o["action"]["tool"] != "get_product":
            continue
        for c in o["result"].get("exchange_candidates", []):
            delta = c["price_difference_cents"]
            candidates.append(
                {
                    "order_id": c["order_id"],
                    "current_item_id": c["current_item_id"],
                    "candidate_item_id": c["item_id"],
                    "option": c["option"],
                    "available": c["available"],
                    "price_difference_cents": delta,
                    "gift_card_coverage": [
                        {
                            "id": card["id"],
                            "balance_cents": card["balance_cents"],
                            "covers_difference": delta <= card["balance_cents"],
                        }
                        for card in cards
                    ],
                }
            )
    visible["decision_facts"] = candidates
    system = prompt[0]["content"]
    if CONTRACT not in system:
        system += CONTRACT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(visible, separators=(",", ":"))},
    ]
