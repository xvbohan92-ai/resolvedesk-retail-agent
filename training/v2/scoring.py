from training.scoring import score as outcome_score


def score(case, state, after, before, questions):
    result = outcome_score(case, state, after, before)
    required = case["required_clarification"]
    observations = state["observations"]
    identified = any(
        o["action"]
        and o["action"]["tool"] == "identify_customer"
        and o["result"].get("user_id") == case["customer"]["id"]
        for o in observations
    )
    inspected = any(
        o["action"]
        and o["action"]["tool"] == "get_order"
        and o["action"]["arguments"].get("order_id") == case["expected"]["order_id"]
        and (
            o["result"].get("id") == case["expected"]["order_id"]
            or (case["family"] == "foreign_order" and "Other customers" in o["result"].get("detail", ""))
        )
        for o in observations
    )
    catalog = not case["family"].startswith("exchange") or any(
        o["action"]
        and o["action"]["tool"] == "get_product"
        and o["result"].get("product_id") == case["orders"][0]["items"][0]["product_id"]
        for o in observations
    )
    clarified = required is None or required in questions
    return {
        **result,
        "raw_task_success": result["task_success"],
        "task_success": bool(result["task_success"] and clarified and identified and inspected and catalog),
        "required_clarification": required,
        "clarification_obtained": clarified,
        "identity_and_order_checked": bool(identified and inspected),
        "catalog_inspected": catalog,
    }
