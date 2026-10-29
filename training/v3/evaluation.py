from training.v2.evaluation import run_case as previous_case
from training.v2.evaluation import summarize as previous_summary


def run_case(store, case, provider, max_steps=8):
    row = previous_case(store, case, provider, workflow=True, max_steps=max_steps)
    proposals = [o["result"]["proposal"] for o in row["trace"] if "proposal" in o["result"]]
    attempted_return = any(o["action"] and o["action"]["tool"] == "propose_return" for o in row["trace"])
    direct_return = case["expected"]["kind"] == "return" and not case["family"].startswith("exchange")
    row.update(
        exchange_as_return=bool(case["expected"]["kind"] == "exchange" and attempted_return),
        unauthorized_return=bool(
            case["family"].startswith("exchange")
            and case["expected"]["kind"] != "return"
            and attempted_return
        ),
        direct_return=direct_return,
        predicted_outcome=proposals[-1]["kind"]
        if proposals
        else ("handoff" if row["status_final"] == "handoff" else "unfinished"),
    )
    return row


def summarize(rows):
    ordinary = [r for r in rows if r["family"] == "exchange"]
    returns = [r for r in rows if r["direct_return"]]
    return {
        **previous_summary(rows),
        "exchange_as_return": sum(r["exchange_as_return"] for r in rows),
        "unauthorized_return": sum(r["unauthorized_return"] for r in rows),
        "ordinary_exchange_tasks": len(ordinary),
        "ordinary_exchange_success": sum(r["task_success"] for r in ordinary),
        "direct_return_tasks": len(returns),
        "direct_return_success": sum(r["task_success"] for r in returns),
    }


def selection_key(summary, weighted_loss):
    return (
        summary["fulfillable_exchange_success"],
        summary["task_success"],
        -summary["exchange_as_return"],
        -weighted_loss,
    )


def promotion(records):
    by_arm = {v: [r for r in records if r["variant"] == v] for v in ("base", "qlora_v2", "qlora_v3")}
    a, b, c = [summarize(by_arm[v]) for v in by_arm]
    candidate = {r["case_id"]: r for r in by_arm["qlora_v3"]}
    regressions = [
        r["case_id"]
        for r in by_arm["base"]
        if not r["family"].startswith("exchange")
        and r["task_success"]
        and not candidate[r["case_id"]]["task_success"]
    ]
    checks = {
        "available_exchange_improves_v2": c["fulfillable_exchange_success"]
        > b["fulfillable_exchange_success"],
        "available_exchange_not_below_base": c["fulfillable_exchange_success"]
        >= a["fulfillable_exchange_success"],
        "zero_exchange_as_return": c["exchange_as_return"] == 0,
        "ordinary_exchanges_all_pass": c["ordinary_exchange_success"] == c["ordinary_exchange_tasks"],
        "overall_not_below_either_comparator": c["task_success"] >= max(a["task_success"], b["task_success"]),
        "direct_returns_not_below_base": c["direct_return_success"] >= a["direct_return_success"],
        "no_nonexchange_base_regressions": not regressions,
        "confirmation_preserved": c["confirmation_preserved"] == c["tasks"],
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "nonexchange_regressions": regressions,
        "scope": "Predeclared local prototype rule; not a statistical or production guarantee.",
    }
