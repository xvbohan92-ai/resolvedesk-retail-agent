"""V4 observed clarification memory; never select the customer's operation in code."""

import json

from .store import Store
from .workflow import WorkflowStore, completed_reads
from .workflow import messages as previous_messages

GUIDANCE = """
V4 recovery contract:
clarification_memory contains actual customer replies, not verified facts. Read their
meaning and reuse usable answers. A new customer instruction takes priority. Do not
ask an already answered question again. If an answer is unusable, ask a different,
specific clarification; if you cannot progress, handoff with the observed problem.
An order-access denial is a policy boundary: handoff, never request another payment.
Order status facts are authoritative. Cancellation is permitted only for pending;
return/exchange only for delivered. If the requested operation is ineligible, handoff
instead of repeatedly requesting a reason or payment.
For exchanges inspect the catalog and match the exact requested option. Availability
is separate from price. Available requested option: exchange; unavailable requested
option: return ONLY if the customer authorized a return fallback, otherwise handoff.
Do not substitute another in-stock option. An insufficient authorized gift card needs
handoff when no alternative was authorized. A real return request remains a return.
After receiving a missing field, continue to the next read or proposal, not another
copy of the same question. A blocked duplicate question did not reach the customer.
All operations still require an explicit customer-confirmed proposal.
"""


def clarification_memory(state):
    pairs = []
    for observation in state["observations"]:
        result = observation["result"]
        index = result.get("asked_at_message_count")
        if "error" in result or not isinstance(index, int) or index >= len(state["messages"]):
            continue
        reply = state["messages"][index]
        if reply["role"] == "customer":
            pairs.append(
                {
                    "field": result["field"],
                    "question": result["question"],
                    "reply": reply["content"],
                    "reply_index": index,
                }
            )
    # An unsolicited new customer turn starts a new context; do not reuse stale slots.
    reply_indices = {p["reply_index"] for p in pairs}
    reset = max((i for i in range(len(state["messages"])) if i not in reply_indices), default=0)
    return [p for p in pairs if p["reply_index"] > reset]


def observed_constraints(state):
    constraints = []
    for o in completed_reads(state):
        if o["action"]["tool"] == "get_order":
            order = o["result"]
            constraints.append(
                {
                    "order_id": order["id"],
                    "status": order["status"],
                    "cancellation_eligible": order["status"] == "pending",
                    "return_exchange_eligible": order["status"] == "delivered",
                }
            )
    for o in state["observations"]:
        if (
            o["action"]
            and o["action"]["tool"] == "get_order"
            and "Other customers" in o["result"].get("detail", "")
        ):
            constraints.append({"order_id": o["action"]["arguments"]["order_id"], "access_denied": True})
    return constraints


def messages(state):
    original = previous_messages(state)
    visible = json.loads(original[1]["content"])
    visible["clarification_memory"] = clarification_memory(state)
    visible["observed_constraints"] = observed_constraints(state)
    return [
        {"role": "system", "content": original[0]["content"] + GUIDANCE},
        {"role": "user", "content": json.dumps(visible, separators=(",", ":"))},
    ]


class ProgressStore(WorkflowStore):
    def apply(self, sid, raw, timing, revision):
        return Store.apply(self, sid, raw, {**timing, "workflow": "v4"}, revision)

    def tool(self, c, state, name, args):
        if name == "ask_customer":
            hit = next(
                (
                    p
                    for p in reversed(clarification_memory(state))
                    if p["field"] == args["field"] and p["question"] == args["message"]
                ),
                None,
            )
            if hit:
                return {
                    "error": "duplicate_clarification_blocked",
                    "backend_executed": False,
                    "cached_customer_reply": hit,
                    "detail": "Reuse the reply or ask a different precise question.",
                }
            result = super().tool(c, state, name, args)
            result["asked_at_message_count"] = len(state["messages"])
            return result
        return super().tool(c, state, name, args)
