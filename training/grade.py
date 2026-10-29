"""Clarification-aware correction of outcome grading, applied equally to saved trajectories.

The frozen system policy already requires asking for missing reason/payment/email and
inspecting an order. Initial outcome grading omitted those prerequisites. No model output
is regenerated, and no prompt, checkpoint, dataset or tool behavior changes in this step.
"""

import copy
import json
import statistics

from training.common import ROOT, sha, verify_frozen
from training.fixtures import load_cases


def prerequisites(case, row):
    required = {"missing_reason": "reason", "missing_payment": "payment_method", "missing_email": "email"}
    field = required.get(case["family"])
    clarified = field is None or field in row["questions"]
    observations = row["trace"]
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
            or (case["family"] == "foreign_order" and "error" in o["result"])
        )
        for o in observations
    )
    return {
        "required_clarification": field,
        "clarification_obtained": clarified,
        "identity_and_order_checked": bool(identified and inspected),
    }


def summarize(rows):
    calls = sum(r["model_calls"] for r in rows)
    required = [r for r in rows if r["required_clarification"]]
    return {
        "tasks": len(rows),
        "task_success": sum(r["task_success"] for r in rows),
        "raw_outcome_matches": sum(r["raw_task_success"] for r in rows),
        "proposal_correct": sum(r["proposal_correct"] for r in rows),
        "model_calls": calls,
        "schema_valid": sum(r["schema_valid"] for r in rows),
        "reference_next_action_matches": sum(r["reference_next_action_matches"] for r in rows),
        "required_clarifications": len(required),
        "clarifications_obtained": sum(r["clarification_obtained"] for r in required),
        "identity_and_order_checked": sum(r["identity_and_order_checked"] for r in rows),
        "rejected_write_attempts": sum(r["rejected_write_attempts"] for r in rows),
        "confirmation_preserved": sum(r["confirmation_preserved"] for r in rows),
        "latency_p50_ms": statistics.median(r["latency_ms"] for r in rows),
        "median_model_calls": statistics.median(r["model_calls"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
    }


def main():
    verify_frozen()
    raw = json.loads((ROOT / "reports/evaluation.json").read_text())
    path = ROOT / "reports/evaluation-graded.json"
    if path.exists():
        raise ValueError("Corrected evaluation already exists; preserve previous results")
    cases = {c["id"]: c for c in load_cases("test")}
    records = []
    for row in raw["records"]:
        corrected = copy.deepcopy(row)
        checks = prerequisites(cases[row["case_id"]], row)
        corrected.update(checks, raw_task_success=row["task_success"])
        corrected["task_success"] = bool(
            row["task_success"] and checks["clarification_obtained"] and checks["identity_and_order_checked"]
        )
        records.append(corrected)
    report = {
        **raw,
        "records": records,
        "raw_evaluation_sha256": sha(ROOT / "reports/evaluation.json"),
        "grading_source_sha256": sha(__file__),
        "grading_correction": "Initial end-state grading omitted required information acquisition and "
        "could reward guessed missing cancellation reasons/payment methods. Corrected uniformly from "
        "saved traces using the already-frozen system policy. Original scores retained, no new model "
        "calls, retraining, checkpoint selection, prompt changes, tool changes or case changes.",
        "summary": {v: summarize([r for r in records if r["variant"] == v]) for v in ("base", "finetuned")},
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
