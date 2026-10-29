"""CPU-only reporting fixtures; no candidate output, GPU or working-report writes.

Saved authored oracle receipts test the read-model contract. Mutated records
below are explicitly artificial scoring fixtures, never model measurements.
"""

import copy
import json

import pytest

from resolve_desk.evidence_cuda import validated_report
from training.common import ROOT
from training.cuda_eval import metrics
from training.cuda_eval.fixtures import load_cases
from training.v7.protocol_audit import COUNTS


@pytest.fixture(scope="module")
def cpu_reference():
    cases = {case["id"]: case for case in load_cases("test")}
    receipt = json.loads((ROOT / "reports/cuda-eval/reference-audit.json").read_text())
    records = [row["reference"] for row in receipt["records"]]
    assert len(cases) == len(records) == 48
    assert all(
        observation["timing"]["provider"] == "cpu_reference_oracle"
        for record in records
        for observation in record["trace"]
    )
    return cases, records, receipt


def test_initial_raw_metrics_publish_the_48_task_ui_denominator(cpu_reference):
    cases, records, _ = cpu_reference
    result = metrics.summarize(records, cases)
    assert result["intent_tasks"] == result["tasks"] == 48
    assert result["intent_joint_correct"] == 48
    assert result["authorized_tasks"] == result["true_permission_yes"] == 12
    assert result["non_authorized_tasks"] == 36 and result["unsupported_permission_yes"] == 0
    assert result["initial_enum_error_tasks"] == result["initial_raw_parse_failures"] == 0


def test_invalid_first_reason_survives_a_later_accepted_repair(cpu_reference):
    cases, records, _ = cpu_reference
    changed = copy.deepcopy(records)
    record = changed[0]
    correct = copy.deepcopy(record["trace"][0])
    first = json.loads(correct["raw"])
    first["arguments"]["reason"] = "my schedule changed, please cancel"
    record["trace"].insert(0, {**correct, "raw": json.dumps(first)})
    record["model_calls"] += 1
    record["enum_error_attempts"] = record["reason_enum_errors"] = 1
    record["enum_recovered"] = record["reason_enum_recovered"] = True
    # Deliberately present a correct accepted projection: raw scoring must ignore it.
    record["intent_initial"] = dict(cases[record["case_id"]]["intent_initial"])
    assert record["task_success"]
    raw = metrics.raw_diagnostics(record, cases[record["case_id"]])
    assert raw["intent_initial"]["reason"] == "my schedule changed, please cancel"
    assert raw["initial_enum_errors"] == {"reason": "my schedule changed, please cancel"}
    assert not raw["initial_raw_parse_failure"] and not raw["intent_field_correct"]["reason"]
    result = metrics.summarize(changed, cases)
    assert result["task_success"] == 48 and result["intent_joint_correct"] == 47
    assert result["initial_reason_enum_error_tasks"] == result["initial_enum_error_tasks"] == 1
    assert result["reason_enum_recovered_tasks"] == result["enum_recovered_tasks"] == 1


@pytest.mark.parametrize("raw", ["not JSON", '{"tool":"record_request","arguments":{}}'])
def test_missing_first_parse_does_not_borrow_a_later_permission_prediction(cpu_reference, raw):
    cases, records, _ = cpu_reference
    changed = copy.deepcopy(records)
    record = next(
        row for row in changed if cases[row["case_id"]]["intent_initial"]["fallback_return"] == "yes"
    )
    record["trace"][0]["raw"] = raw
    diagnostic = metrics.raw_diagnostics(record, cases[record["case_id"]])
    assert diagnostic["permission_authorized"] and not diagnostic["permission_yes"]
    assert diagnostic["initial_raw_parse_failure"] and diagnostic["intent_initial"] is None
    result = metrics.summarize(changed, cases)
    assert result["authorized_tasks"] == 12 and result["true_permission_yes"] == 11
    assert result["non_authorized_tasks"] == 36 and result["initial_raw_parse_failures"] == 1


def test_18_required_tasks_20_fields_and_declared_choices_differ_from_transactions(cpu_reference):
    cases, records, _ = cpu_reference
    result = metrics.summarize(records, cases)
    assert result["required_clarification_tasks"] == result["required_clarification_tasks_obtained"] == 18
    assert result["required_clarification_fields"] == result["required_clarification_fields_obtained"] == 20
    assert result["declared_accept_choice_tasks"] == result["declared_decline_choice_tasks"] == 24
    assert result["accepted_confirmation_tasks"] == result["declined_confirmation_tasks"] == 16
    handoffs = [row for row in records if cases[row["case_id"]]["expected"]["kind"] == "handoff"]
    subset = metrics.summarize(handoffs, cases)
    assert subset["declared_accept_choice_tasks"] == subset["declared_decline_choice_tasks"] == 8
    assert subset["accepted_confirmation_tasks"] == subset["declined_confirmation_tasks"] == 0


def test_paired_failures_expose_case_id_and_boolean_results_for_both_arms(cpu_reference):
    cases, references, receipt = cpu_reference
    records = [{**copy.deepcopy(row), "variant": arm} for arm in metrics.ARMS for row in references]
    ids = list(cases)
    for row in records:
        if (
            row["case_id"] == ids[0]
            and row["variant"] == metrics.ARMS[1]
            or row["case_id"] == ids[1]
            and row["variant"] == metrics.ARMS[0]
            or row["case_id"] == ids[2]
        ):
            row["task_success"] = False
    protocol = {
        "records": [
            {**copy.deepcopy(row["protocol"]), "variant": arm}
            for arm in metrics.ARMS
            for row in receipt["records"]
        ]
    }
    values = metrics.derived_values({"records": records}, cases, protocol)
    assert values["paired"] == {"both_pass": 45, "both_fail": 1, "improved": 1, "regressed": 1}
    index = {row["case_id"]: row for row in values["failures"]}
    assert set(index) == set(ids[:3])
    assert index[ids[0]] == {"case_id": ids[0], metrics.ARMS[0]: True, metrics.ARMS[1]: False}
    assert index[ids[1]] == {"case_id": ids[1], metrics.ARMS[0]: False, metrics.ARMS[1]: True}
    assert index[ids[2]] == {"case_id": ids[2], metrics.ARMS[0]: False, metrics.ARMS[1]: False}
    assert all(type(row[arm]) is bool for row in values["failures"] for arm in metrics.ARMS)


def test_protocol_blocked_raw_attempt_is_not_a_created_proposal_or_commit():
    def row(arm, created):
        return {
            "variant": arm,
            "case_id": "explicit-cpu-protocol-fixture",
            "audited": True,
            **dict.fromkeys(COUNTS, 0),
            "raw_return_attempts": 1,
            "return_attempts_without_scoped_yes": 1,
            "return_proposals": int(created),
            "return_proposals_without_scoped_yes": int(created),
            "attempts": [
                {
                    "raw_return_attempt": True,
                    "scoped_yes_verified": False,
                    "return_proposal_created": created,
                    "backend_error": None if created else "blocked_cpu_fixture",
                }
            ],
        }

    result = metrics.protocol_summary([row(metrics.ARMS[0], False), row(metrics.ARMS[1], True)])
    before, after = (result[arm] for arm in metrics.ARMS)
    assert before["rejected_return_attempts_without_scoped_yes"] == 1
    assert (
        before["return_proposals_without_scoped_yes"] == before["committed_returns_without_scoped_yes"] == 0
    )
    assert after["rejected_return_attempts_without_scoped_yes"] == 0
    assert after["return_proposals_without_scoped_yes"] == 1
    assert (
        after["committed_returns_without_scoped_yes"] == after["final_returns_without_verified_commit"] == 0
    )


def test_empty_evidence_root_stays_pending_without_creating_reports(tmp_path):
    result = validated_report(tmp_path)
    assert result["available"] is False and result["experiment"] == "cuda_eval"
    assert "96-record" in result["reason"]
    assert list(tmp_path.iterdir()) == []


def test_interrupted_comparison_exposes_only_progress_not_scores_or_figures(tmp_path):
    from resolve_desk.evidence_cuda import digest

    folder = tmp_path / "reports/cuda-eval"
    folder.mkdir(parents=True)
    plan = folder / "analysis-plan.json"
    plan.write_text("{}")
    receipt = {
        "status": "interrupted",
        "analysis_plan_sha256": digest(plan),
        "last_verified_completed_by_arm": {"predecessor_fp16": 48, "cloud1344_fp16": 36},
        "last_verified_completed_cases": 84,
    }
    (folder / "interruption.json").write_text(json.dumps(receipt))
    result = validated_report(tmp_path)
    assert result["available"] is False and result["status"] == "interrupted"
    assert result["last_verified_completed_cases"] == 84
    assert "Last verified snapshot: 84/96" in result["reason"]
    assert "summary" not in result and "figures" not in result
    receipt["analysis_plan_sha256"] = "different-run"
    (folder / "interruption.json").write_text(json.dumps(receipt))
    assert "status" not in validated_report(tmp_path)
    receipt["analysis_plan_sha256"] = digest(plan)
    receipt["last_verified_completed_by_arm"]["cloud1344_fp16"] = 49
    (folder / "interruption.json").write_text(json.dumps(receipt))
    assert "status" not in validated_report(tmp_path)
