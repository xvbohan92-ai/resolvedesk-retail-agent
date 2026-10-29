"""V2 progress-aware tools; v1 policy/store remain intact for historical reproduction.

Only customer utterances and observed tool results enter the model prompt. Progress
gates enforce tool prerequisites, never choose the customer's requested operation.
"""

import json

from .protocol import SYSTEM
from .store import Store, canonical

READS = {"identify_customer", "get_order", "get_product"}
GUIDANCE = """
V2 execution guidance:
Use completed_reads as memory. Do not repeat an identical completed lookup.
Use allowed_tools to decide what is possible at this point. An error is not a completed step.
Copy order_id, product_id and item_id exactly from customer text and inspected facts.
After get_order, a pending cancellation needs propose_cancel; a delivered exchange needs
get_product with the order item's product_id, then propose_exchange with the matching
candidate's item_id. Do not replace an exchange by a return unless the customer explicitly
accepted that fallback AND the requested variant is unavailable. A different price is NOT
unavailability. A negative price_difference_cents is a refund, not an error.
The exchange_candidates are catalog facts, not a selected answer. Match the customer's
requested color/size to option, check available and copy candidate item_id. Never use a
product_id as an item_id. Do not ask for a cancellation reason when processing an exchange.
"Duplicate order" means cancellation reason "ordered by mistake"; "don't need it" means
"no longer needed". Missing reasons still require a question, never a guess.
An existing gift card may pay an exchange difference only if it covers a positive difference.
If recovery cannot progress, handoff with the observed problem. Never claim success.
"""


def completed_reads(state):
    reads = {}
    for o in state["observations"]:
        a, r = o["action"], o["result"]
        if a and a["tool"] in READS and "error" not in r:
            reads[canonical(a)] = {"action": a, "result": r}
    return list(reads.values())


def allowed_tools(state):
    allowed = ["ask_customer", "handoff"]
    reads = completed_reads(state)
    if not state["user_id"]:
        return ["identify_customer", *allowed]
    allowed.append("get_order")  # A genuinely changed order can be refreshed.
    orders = [o["result"] for o in reads if o["action"]["tool"] == "get_order"]
    products = {o["action"]["arguments"]["product_id"] for o in reads if o["action"]["tool"] == "get_product"}
    if any(o["status"] == "pending" for o in orders):
        allowed.append("propose_cancel")
    delivered = [o for o in orders if o["status"] == "delivered"]
    if delivered:
        allowed.extend(["get_product", "propose_return"])
        if any(i["product_id"] in products for o in delivered for i in o["items"]):
            allowed.append("propose_exchange")
    return allowed


def messages(state):
    reads = completed_reads(state)
    recent = [
        {"action": o["action"], "result": o["result"]}
        for o in state["observations"]
        if not o["action"] or o["action"]["tool"] not in READS or "error" in o["result"]
    ][-3:]
    visible = {
        "messages": [{"role": m["role"], "content": m["content"]} for m in state["messages"]],
        "completed_reads": reads,
        "recent_events": recent,
        "allowed_tools": allowed_tools(state),
    }
    return [
        {"role": "system", "content": SYSTEM + GUIDANCE},
        {"role": "user", "content": json.dumps(visible, separators=(",", ":"))},
    ]


class WorkflowStore(Store):
    def apply(self, sid, raw, timing, revision):
        return super().apply(sid, raw, {**timing, "workflow": "v2"}, revision)

    def tool(self, c, state, name, args):
        action = {"tool": name, "arguments": args}
        hit = next((o for o in reversed(completed_reads(state)) if o["action"] == action), None)
        if hit and name == "get_order":
            key = state["order_keys"].get(args["order_id"])
            row = c.execute("SELECT revision FROM orders WHERE id=?", (key,)).fetchone()
            if not row or row[0] != hit["result"]["revision"]:
                hit = None
        if hit:
            blocks = sum(
                o["action"] == action and o["result"].get("error") == "duplicate_read_blocked"
                for o in state["observations"]
            )
            result = {
                "error": "duplicate_read_blocked",
                "backend_executed": False,
                "cached_result": hit["result"],
                "detail": "Lookup already succeeded. Use the cached facts and choose another action.",
            }
            if blocks >= 1:
                state["status"] = "handoff"
                result.update(handoff=True, reason="Repeated completed lookup after recovery feedback.")
            return result
        if name not in allowed_tools(state):
            return {
                "error": "step_prerequisite_rejection",
                "backend_executed": False,
                "allowed_tools": allowed_tools(state),
                "detail": "Inspect identity, order and exchange catalog before proposing.",
            }
        result = super().tool(c, state, name, args)
        if name == "get_product":
            orders = [o["result"] for o in completed_reads(state) if o["action"]["tool"] == "get_order"]
            candidates = []
            for order in orders:
                for item in order["items"]:
                    if item["product_id"] == args["product_id"]:
                        candidates.extend(
                            {
                                "order_id": order["id"],
                                "current_item_id": item["item_id"],
                                **v,
                                "price_difference_cents": v["price_cents"] - item["price_cents"],
                            }
                            for v in result["variants"]
                            if v["item_id"] != item["item_id"]
                        )
            result = {**result, "exchange_candidates": candidates}
        return result
