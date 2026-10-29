"""Reconcile delivered numbers and successful business outcomes without model inference."""

import csv
import json
from collections import Counter

from resolve_desk.evidence import validated_v2_report
from training.common import ROOT, sha
from training.common import verify_frozen as verify_v1
from training.v2.common import REPORTS, verify_frozen
from training.v2.fixtures import load_cases


def verify():
    verify_v1()
    manifest = verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    arms = ["legacy_base", "workflow_base", "workflow_qlora_v2"]
    counts = Counter((r["variant"], r["case_id"]) for r in report["records"])
    assert counts == Counter({(v, i): 1 for v in arms for i in cases})
    assert len(report["records"]) == 126
    assert training["dataset"] == report["dataset"] == manifest
    assert not training["test_used_for_selection"] and not training["cloud_compute_used"]
    assert training["adapter_l2_change"] > 0
    assert training["initial_adapter_sha256"] != training["selected_adapter_sha256"]
    valid_losses = [r["val_loss"] for r in training["metrics"] if r["kind"] == "valid" and r["iteration"] > 0]
    assert training["best_checkpoint"]["val_loss"] == min(valid_losses)
    protocol = json.loads((REPORTS / "training-protocol.json").read_text())
    assert protocol["runner_sha256"] == sha(ROOT / "training/v2/train.py")
    assert protocol["common_sha256"] == sha(ROOT / "training/common.py")
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    with (ROOT / "docs/figures/v2/per-case.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 126
    for arm in arms:
        rows = [r for r in report["records"] if r["variant"] == arm]
        summary = report["summary"][arm]
        assert len(rows) == summary["tasks"] == 42
        assert summary["task_success"] == sum(r["task_success"] for r in rows)
        assert summary["exchange_related_success"] == sum(
            r["task_success"] for r in rows if r["family"].startswith("exchange")
        )
        assert summary["model_calls"] == sum(len(r["trace"]) for r in rows)
        assert summary["task_success"] == sum(
            r["task_success"] == "True" for r in csv_rows if r["variant"] == arm
        )
        for row in rows:
            case = cases[row["case_id"]]
            for observation in row["trace"]:
                if observation["action"]:
                    assert json.loads(observation["raw"]) == observation["action"]
            assert row["order_before_confirmation"] == case["orders"]
            events = [e for e in row["events"] if e["kind"] == "operation_committed"]
            if not case["confirmation"]:
                assert row["order_after_confirmation"] == case["orders"] and not events
            if not row["task_success"]:
                continue
            expected = case["expected"]
            assert (
                case["required_clarification"] is None or case["required_clarification"] in row["questions"]
            )
            if expected["kind"] == "handoff":
                assert row["status_final"] == "handoff" and row["order_after_confirmation"] == case["orders"]
            else:
                proposal = next(
                    o["result"]["proposal"] for o in reversed(row["trace"]) if "proposal" in o["result"]
                )
                assert proposal["kind"] == expected["kind"]
                assert proposal["order_id"] == expected["order_id"]
                assert proposal["amount_cents"] == expected["amount_cents"]
                if expected["kind"] == "cancel":
                    assert proposal["arguments"]["reason"] == expected["reason"]
                else:
                    assert proposal["arguments"]["item_id"] == expected["item_id"]
                    assert proposal["arguments"]["payment_method"] == expected["payment_method"]
                if expected["kind"] == "exchange":
                    assert proposal["new_item_id"] == expected["new_item_id"]
                if case["confirmation"]:
                    assert len(events) == 1 and row["status_final"] == "completed"
                    assert row["order_after_confirmation"][0]["status"] == expected["target_status"]
                else:
                    assert row["status_final"] == "awaiting_customer"
    assert validated_v2_report(ROOT)["available"]
    result = {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "source_sha256": sha(__file__),
        "complete_paired_records": 126,
        "frozen_cases_per_arm": 42,
        "training_validation_and_business_state_checks_pass": True,
        "scope": "Checks sources and business states; not a human or production evaluation.",
    }
    return result


if __name__ == "__main__":
    result = verify()
    (REPORTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
