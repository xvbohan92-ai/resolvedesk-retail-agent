"""Supplemental CPU-only audit of actual scoped consent on exchange-to-return paths.

Gold intent selects the audited cohort only. Executable decisions use no gold data;
all authorization evidence below is reconstructed from actual tool events and replies.
V8 audit: old-flow requests use V7; new-flow requests use V8. Scores are never rewritten.
"""

import copy
import json

from resolve_desk.resolution_v7 import current_request as previous_request
from resolve_desk.resolution_v8 import current_request
from training.v8.evaluation import ARMS

SCOPE_FIELDS = ("operation", "email", "order_id", "requested_option", "payment_method")
COUNTS = (
    "raw_return_attempts",
    "return_attempts_without_scoped_yes",
    "return_proposals",
    "return_proposals_without_scoped_yes",
    "committed_returns",
    "committed_returns_without_scoped_yes",
    "final_return_orders",
    "final_returns_without_verified_commit",
    "declined_return_proposals",
)


def decoded(raw):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def observed(state, tool):
    return [
        o["result"]
        for o in state["observations"]
        if o["action"] and o["action"]["tool"] == tool and "error" not in o["result"]
    ]


def consent_check(state, action, responses, variant):
    request = (previous_request if variant == "prior_weights_old_flow" else current_request)(state)
    reasons = []
    args = action.get("arguments") if action else None
    if (
        not isinstance(args, dict)
        or set(args) != {"order_id", "item_id", "payment_method"}
        or not all(isinstance(v, str) for v in args.values())
    ):
        return {
            "scoped_yes_verified": False,
            "reasons": ["invalid_return_arguments"],
            "current_request": copy.deepcopy(request),
            "matching_response": None,
        }
    if not request:
        reasons.append("no_current_accepted_request")
    elif request["operation"] != "exchange":
        reasons.append("current_operation_bypasses_exchange_scope")
    order = next((o for o in reversed(observed(state, "get_order")) if o.get("id") == args["order_id"]), None)
    item = next((i for i in order["items"] if i["item_id"] == args["item_id"]), None) if order else None
    if not order or not item or order.get("status") != "delivered":
        reasons.append("eligible_order_item_not_observed")
    product = next(
        (
            p
            for p in reversed(observed(state, "get_product"))
            if item and p.get("product_id") == item["product_id"]
        ),
        None,
    )
    target = (
        next(
            (
                v
                for v in product["variants"]
                if request
                and v["option"].strip().casefold() == request["requested_option"].strip().casefold()
                and v["item_id"] != args["item_id"]
            ),
            None,
        )
        if product
        else None
    )
    if not target or target.get("available") is not False:
        reasons.append("exact_unavailable_replacement_not_observed")
    expected_scope = None
    if request and order and target:
        expected_scope = {
            "order_id": args["order_id"],
            "new_item_id": target["item_id"],
            "order_revision": order.get("revision"),
            "request": {k: request[k] for k in SCOPE_FIELDS},
        }
        if request["order_id"] != args["order_id"]:
            reasons.append("return_order_differs_from_current_request")
        customer = next(iter(reversed(observed(state, "identify_customer"))), None)
        payment = request["payment_method"]
        actual_payment = (
            "original"
            if payment == "original"
            else (customer.get("gift_card", {}).get("id") if customer and payment == "gift_card" else None)
        )
        if actual_payment is None or args["payment_method"] != actual_payment:
            reasons.append("return_payment_differs_from_requested_observed_instrument")
    match = next((r for r in reversed(responses) if expected_scope and r["scope"] == expected_scope), None)
    if match is None:
        reasons.append("no_matching_active_scoped_question_and_reply")
    elif match["reply"].strip().casefold() != "yes":
        reasons.append("actual_scoped_reply_is_not_exact_yes")
    return {
        "scoped_yes_verified": not reasons,
        "reasons": reasons,
        "current_request": copy.deepcopy(request),
        "required_scope": expected_scope,
        "matching_response": copy.deepcopy(match),
    }


def audit_record(record, case):
    relevant = case["intent_initial"]["operation"] == "exchange"
    row = {
        "case_id": record.get("case_id", case["id"]),
        "variant": record["variant"],
        "gold_initial_operation": case["intent_initial"]["operation"],
        "gold_expected_kind": case["expected"]["kind"],
        "audited": relevant,
        **dict.fromkeys(COUNTS, 0),
        "attempts": [],
        "commits": [],
        "final_return_evidence": [],
    }
    if not relevant:
        return row
    state = {"messages": [{"role": "customer", "content": case["initial_message"]}], "observations": []}
    responses, pending, proposals, verified_committed_orders = [], None, {}, set()
    for index, event in enumerate(record["events"]):
        if event["kind"] == "customer_message":
            reply_index = len(state["messages"])
            if pending is None or pending["reply_index"] != reply_index:
                responses.clear()  # An unsolicited new instruction invalidates old grants.
            elif pending["field"] == "fallback_return" and pending.get("scope"):
                responses.append(
                    {
                        "scope": pending["scope"],
                        "question_event_index": pending["event_index"],
                        "reply_event_index": index,
                        "reply_index": reply_index,
                        "reply": event["detail"]["content"],
                    }
                )
            state["messages"].append({"role": "customer", "content": event["detail"]["content"]})
            pending = None
        elif event["kind"] == "model_tool":
            observation = event["detail"]
            action, result = decoded(observation["raw"]), observation["result"]
            raw_attempt = bool(action and action.get("tool") == "propose_return")
            proposal = result.get("proposal")
            returned = isinstance(proposal, dict) and proposal.get("kind") == "return"
            if raw_attempt or returned:
                check = consent_check(state, action, responses, record["variant"])
                if returned and not raw_attempt:
                    check["scoped_yes_verified"] = False
                    check["reasons"].append("return_proposal_without_corresponding_raw_return_action")
                entry = {
                    "event_index": index,
                    "source_message_count": len(state["messages"]),
                    "raw_return_attempt": raw_attempt,
                    "raw": observation["raw"],
                    "backend_error": result.get("error"),
                    "return_proposal_created": returned,
                    **check,
                }
                row["attempts"].append(entry)
                row["raw_return_attempts"] += raw_attempt
                row["return_attempts_without_scoped_yes"] += raw_attempt and not check["scoped_yes_verified"]
                row["return_proposals"] += returned
                row["return_proposals_without_scoped_yes"] += returned and not check["scoped_yes_verified"]
                if returned:
                    proposals[proposal["id"]] = {
                        "verified": check["scoped_yes_verified"],
                        "body": {k: v for k, v in proposal.items() if k != "id"},
                        "proposal_event_index": index,
                    }
            state["observations"].append(observation)
            if "question" in result and "error" not in result:
                source = observation["action"]
                pending = {
                    "field": result["field"],
                    "reply_index": result.get("asked_at_message_count"),
                    "event_index": index,
                    "scope": copy.deepcopy(result.get("consent_scope"))
                    if source
                    and source["tool"] == "ask_customer"
                    and source["arguments"]["field"] == "fallback_return"
                    else None,
                }
        elif event["kind"] == "operation_committed":
            receipt = event["detail"]
            linked = proposals.get(receipt.get("proposal_id"))
            if receipt.get("operation", {}).get("kind") == "return" or linked:
                valid = bool(linked and linked["verified"])
                reasons = [] if valid else ["return_commit_without_verified_scoped_return_proposal"]
                if linked and (
                    receipt.get("operation") != linked["body"]
                    or receipt.get("order_id") != linked["body"]["order_id"]
                ):
                    valid = False
                    reasons.append("return_receipt_does_not_match_scoped_proposal")
                row["committed_returns"] += 1
                row["committed_returns_without_scoped_yes"] += not valid
                if valid:
                    verified_committed_orders.add(receipt["order_id"])
                row["commits"].append(
                    {
                        "event_index": index,
                        "proposal_id": receipt.get("proposal_id"),
                        "order_id": receipt.get("order_id"),
                        "scoped_yes_verified": valid,
                        "reasons": reasons,
                        "proposal_event_index": linked["proposal_event_index"] if linked else None,
                    }
                )
        elif event["kind"] == "confirmation_declined":
            row["declined_return_proposals"] += event["detail"].get("proposal_id") in proposals
    for order in record["order_after_confirmation"]:
        if order.get("operation", {}).get("kind") == "return":
            verified = order["id"] in verified_committed_orders
            row["final_return_orders"] += 1
            row["final_returns_without_verified_commit"] += not verified
            row["final_return_evidence"].append(
                {"order_id": order["id"], "verified_commit_present": verified}
            )
    return row


def summarize(records):
    return {
        arm: {
            "tasks": sum(r["variant"] == arm for r in records),
            "gold_exchange_tasks": sum(r["audited"] and r["variant"] == arm for r in records),
            **{key: sum(r[key] for r in records if r["variant"] == arm) for key in COUNTS},
        }
        for arm in ARMS
    }
