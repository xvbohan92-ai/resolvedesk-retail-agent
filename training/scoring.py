"""Expected task facts are authored independently of the executable tool validator."""


def score(case, state, orders, before_confirmation=None):
    expected = case["expected"]
    order = next(o for o in orders if o["id"] == expected["order_id"])
    result = {"proposal_correct": False, "task_success": False, "confirmation_preserved": True}
    if expected["kind"] == "handoff":
        ok = state["status"] == "handoff" and order == case["orders"][0]
        result.update(proposal_correct=ok, task_success=ok)
        return result
    proposal = state.get("proposal")
    if not proposal:
        return result
    ok = proposal["kind"] == expected["kind"] and proposal["order_id"] == expected["order_id"]
    ok &= proposal["amount_cents"] == expected["amount_cents"]
    args = proposal["arguments"]
    if expected["kind"] == "cancel":
        ok &= args.get("reason") == expected["reason"]
    else:
        ok &= args.get("item_id") == expected["item_id"]
        ok &= args.get("payment_method") == expected["payment_method"]
        if expected["kind"] == "exchange":
            ok &= args.get("new_item_id") == expected["new_item_id"]
    preserved = before_confirmation == case["orders"] if before_confirmation is not None else True
    expected_status = expected["target_status"] if case["confirmation"] else expected["initial_status"]
    final_ok = state["status"] == ("completed" if case["confirmation"] else "awaiting_customer")
    result.update(
        proposal_correct=bool(ok),
        confirmation_preserved=preserved,
        task_success=bool(ok and preserved and final_ok and order["status"] == expected_status),
    )
    return result
