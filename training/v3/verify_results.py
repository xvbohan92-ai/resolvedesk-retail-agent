"""Reconcile raw trajectories, semantic confusion, validation selection and export assets."""

import csv
import json
from collections import Counter

from resolve_desk.evidence import validated_v3_report
from training.common import ROOT, sha
from training.common import verify_frozen as verify_v1
from training.v2.common import verify_frozen as verify_v2
from training.v3.common import REPORTS, verify_frozen
from training.v3.fixtures import load_cases


def verify():
    verify_v1()
    verify_v2()
    manifest = verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    values = json.loads((ROOT / "docs/figures/v3/values.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    arms = ["base", "qlora_v2", "qlora_v3"]
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == Counter(
        {(v, i): 1 for v in arms for i in cases}
    )
    assert len(report["records"]) == 144
    assert training["dataset"] == report["dataset"] == manifest
    assert not training["test_used_for_selection"] and training["adapter_l2_change"] > 0
    assert training["initial_adapter_sha256"] != training["selected_adapter_sha256"]
    candidates = training["validation_candidates"]
    assert {r["completed_updates"] for r in candidates} == {119, 239}

    def key(r):
        s = r["summary"]
        return (
            s["fulfillable_exchange_success"],
            s["task_success"],
            -s["exchange_as_return"],
            -r["weighted_valid_loss"],
        )

    selected = max(candidates, key=key)
    assert training["best_checkpoint"]["iteration"] == selected["completed_updates"]
    selected_full = json.loads((REPORTS / f"validation-{selected['completed_updates']}.json").read_text())
    assert selected_full["adapter_sha256"] == training["selected_adapter_sha256"]
    assert not selected_full["preflight_only"] and len(selected_full["records"]) == 16
    with (ROOT / "docs/figures/v3/per-case.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 144
    for arm in arms:
        rows = [r for r in report["records"] if r["variant"] == arm]
        s = report["summary"][arm]
        assert len(rows) == s["tasks"] == 48
        assert sum(r["task_success"] for r in rows) == s["task_success"]
        assert sum(r["task_success"] == "True" for r in csv_rows if r["variant"] == arm) == s["task_success"]
        expected_count = 0
        for row in rows:
            c = cases[row["case_id"]]
            is_error = c["expected"]["kind"] == "exchange" and any(
                o["action"] and o["action"]["tool"] == "propose_return" for o in row["trace"]
            )
            expected_count += is_error
            assert row["exchange_as_return"] == bool(is_error)
            assert row["order_before_confirmation"] == c["orders"]
            events = [e for e in row["events"] if e["kind"] == "operation_committed"]
            if not c["confirmation"]:
                assert not events and row["order_after_confirmation"] == c["orders"]
            for o in row["trace"]:
                if o["action"]:
                    assert json.loads(o["raw"]) == o["action"]
            if row["task_success"]:
                field = c["required_clarification"]
                assert field is None or field in row["questions"]
                if c["expected"]["kind"] == "handoff":
                    assert row["status_final"] == "handoff" and row["order_after_confirmation"] == c["orders"]
                else:
                    proposal = next(
                        o["result"]["proposal"] for o in reversed(row["trace"]) if "proposal" in o["result"]
                    )
                    assert (
                        proposal["kind"] == c["expected"]["kind"]
                        and proposal["amount_cents"] == c["expected"]["amount_cents"]
                    )
                    if c["expected"]["kind"] == "exchange":
                        assert proposal["new_item_id"] == c["expected"]["new_item_id"]
                    if c["confirmation"]:
                        assert len(events) == 1 and row["status_final"] == "completed"
        assert s["exchange_as_return"] == expected_count
        assert sum(sum(r) for r in values["confusion"][arm]) == 48
        confusion = Counter((r["expected"]["kind"], r["predicted_outcome"]) for r in rows)
        for i, e in enumerate(["cancel", "return", "exchange", "handoff"]):
            for j, p in enumerate(["cancel", "return", "exchange", "handoff", "unfinished"]):
                assert values["confusion"][arm][i][j] == confusion[e, p]
    assert validated_v3_report(ROOT)["available"]
    return {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "source_sha256": sha(__file__),
        "paired_records": 144,
        "new_tasks_per_arm": 48,
        "candidate_selection_uses_validation_only": True,
        "semantic_metrics_and_business_state_checks_pass": True,
        "scope": "Source/trajectory/CSV/figure reconciliation, not independent human adjudication.",
    }


if __name__ == "__main__":
    result = verify()
    (REPORTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
