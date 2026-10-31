"""CPU-only contracts using label-aware oracle fixtures, never actual model evidence."""

import copy
import json

import pytest

from training.v7.evaluation import run_case
from training.v8.protocol_audit import audit_record
from training.v8_eval2 import fixtures, prepare
from training.v8_eval2_partial import metrics, report_results, write_summary


@pytest.fixture(scope="module")
def cpu_bundle(tmp_path_factory):
    all_cases = fixtures.load_cases("test")
    cases = all_cases[:40]
    folder = tmp_path_factory.mktemp("partial40-cpu-oracle-only")
    records, protocols = [], []
    for arm in metrics.ARMS:
        Store, messages = prepare.runtime("v7" if arm == metrics.ARMS[0] else "v8")
        store = Store(folder / (arm + ".sqlite"))
        for case in cases:
            row = {"variant": arm, **run_case(store, case, prepare.ReferenceProvider(case), messages)}
            assert row["task_success"]
            assert all(o["timing"]["provider"] == "cpu_reference_oracle" for o in row["trace"])
            records.append(row)
            protocols.append(audit_record(row, case))
    report = {"records": records, "record_sources": [f"CPU_FIXTURE#/records/{i}/trace" for i in range(120)]}
    values = metrics.derived_values(
        report, {c["id"]: c for c in cases}, {"records": protocols}, all_cases=all_cases
    )
    values["training"] = {
        "sampling": {"unique_rows_sampled": 888, "physical_rows": 1100, "unique_cases_sampled": 96}
    }
    return {
        "test": report,
        "cases": cases,
        "values": values,
        "training": {"metrics": [{"iteration": 1, "loss": 0.1}]},
    }


def test_prefix40_is_dynamic_not_original48(cpu_bundle):
    values = cpu_bundle["values"]
    for arm in metrics.ARMS:
        summary = values["summary"][arm]
        assert summary["tasks"] == summary["intent_tasks"] == summary["task_success"] == 40
        assert summary["authorized_tasks"] + summary["non_authorized_tasks"] == 40
        assert summary["authorized_tasks"] == 8 and summary["non_authorized_tasks"] == 32
        assert summary["fulfillable_exchange_tasks"] == 10 and summary["fallback_tasks"] == 2
        assert summary["missing_email_tasks"] == 4 and summary["handoff_tasks"] == 16
        assert summary["required_clarification_tasks"] == summary["required_clarification_fields"] == 12
        assert summary["true_permission_yes"] == summary["authorized_tasks"]
        assert summary["unsupported_permission_yes"] == 0
        assert summary["declared_accept_choice_tasks"] == summary["declared_decline_choice_tasks"] == 20
        assert summary["accepted_confirmation_tasks"] < 20  # Handoff fixture choices are not transactions.
    assert values["paired_records"] == 120 and values["paired_cases"] == 40
    assert values["scope"]["original144_gate_complete"] is False
    assert values["scope"]["extrapolation_to_original48_allowed"] is False


def test_omitted_family_has_no_zero_success_value(cpu_bundle):
    values = cpu_bundle["values"]
    assert len(values["coverage"]["included_families"]) == 20
    assert values["coverage"]["excluded_families"] == [
        "exchange_available_refusal",
        "exchange_fallback_missing_payment",
        "exchange_missing_email",
        "missing_email_payment",
    ]
    assert all(name not in values["family_success"] for name in values["coverage"]["excluded_families"])
    assert all(
        values["coverage"]["per_family"][f]["selected_cases"] == 0
        for f in values["coverage"]["excluded_families"]
    )


def test_selected_raw_error_never_borrows_later_repair(cpu_bundle):
    rows = copy.deepcopy(cpu_bundle["test"]["records"][:40])
    row = rows[0]
    first = copy.deepcopy(row["trace"][0])
    action = json.loads(first["raw"])
    action["arguments"]["reason"] = "event canceled"
    row["trace"].insert(0, {**first, "raw": json.dumps(action)})
    row["model_calls"] += 1
    summary = metrics.summarize(rows, {c["id"]: c for c in cpu_bundle["cases"]})
    assert summary["tasks"] == 40 and summary["initial_reason_enum_error_tasks"] == 1
    assert summary["intent_joint_correct"] == 39
    assert summary["task_success"] == 40  # Artificial raw mutation is not newly graded model evidence.


def test_failures_and_all_three_paired_directions_retained(cpu_bundle):
    bundle = copy.deepcopy(cpu_bundle)
    rows = bundle["test"]["records"]
    rows[0]["task_success"] = False
    rows[40 + 1]["task_success"] = False
    rows[80 + 1]["task_success"] = False
    cases = {c["id"]: c for c in bundle["cases"]}
    protocols = [audit_record(r, cases[r["case_id"]]) for r in rows]
    values = metrics.derived_values(
        bundle["test"], cases, {"records": protocols}, all_cases=fixtures.load_cases("test")
    )
    assert len(values["failed_records"]) == 3 and len(values["failures"]) == 2
    assert values["paired"]["workflow"]["improved"] == values["paired"]["workflow"]["regressed"] == 1
    assert values["paired"]["training_under_v8"]["both_fail"] == 1
    assert all(p["tasks"] == 40 for p in values["paired"].values())


def test_csv_keeps120_source_pointers_and40_mapping(cpu_bundle):
    import csv
    import io

    exports = report_results.csv_exports(cpu_bundle)
    per_case = list(csv.DictReader(io.StringIO(exports["per-case.csv"].decode())))
    assert len(per_case) == 120
    assert per_case[-1]["raw_trace"] == "CPU_FIXTURE#/records/119/trace"
    assert len(list(csv.DictReader(io.StringIO(exports["case-mapping.csv"].decode())))) == 40
    denominators = {
        row["denominator"] for row in csv.DictReader(io.StringIO(exports["metrics.csv"].decode()))
    }
    assert "40" in denominators and "48" not in denominators


def test_no_receipt_fails_before_output_or_renderer(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No renderer before independent partial verification")

    monkeypatch.setattr(report_results, "render", forbidden)
    with pytest.raises(FileNotFoundError):
        report_results.deliver(tmp_path)
    with pytest.raises(FileNotFoundError):
        write_summary.write(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change",
    [
        {"status": "verified"},
        {"original144_gate_complete": True},
        {"original_full_result_verified": True},
        {"new_model_or_gpu_calls": 1},
        {"counts": {"cases": 48}},
        {"analysis_id": "the-original-full144"},
    ],
)
def test_partial_receipt_cannot_promote_original_gate_or_change_scope(tmp_path, monkeypatch, change):
    proof = {
        "status": "verified_partial_evidence",
        "analysis_id": metrics.ANALYSIS_ID,
        "run_id": metrics.RUN_ID,
        "counts": {
            "audited_completed_records": 136,
            "paired_records": 120,
            "cases": 40,
            "completed_calls": 765,
            "ambiguous_calls": 1,
            "excluded_completed_records": 16,
            "unstarted_tasks": 7,
            "ambiguous_pending_tasks": 1,
        },
        "original_full_result_verified": False,
        "original144_gate_complete": False,
        "original_run_modified": False,
        "new_model_or_gpu_calls": 0,
        "training_updates": 0,
    }
    proof.update(change)

    def only_proof(root, name):
        assert name.endswith("partial-evidence-verification.json"), "Invalid proof advanced to source reads"
        return proof

    monkeypatch.setattr(report_results, "read", only_proof)
    with pytest.raises(ValueError):
        report_results.load_verified(tmp_path)


def test_bilingual_scope_declines_exclusions_and_headline(cpu_bundle):
    for zh in (False, True):
        text = write_summary.document(cpu_bundle, zh=zh)
        for name in cpu_bundle["values"]["coverage"]["excluded_families"]:
            assert name in text
        assert "144" in text and "40" in text and "120" in text
        assert ("不等于实际换货/退款" if zh else "not executed transaction counts") in text
        assert ("不能把此子集" if zh else "Do not extrapolate") in text
        assert "event canceled" not in text
    assert "decreased" in write_summary.headline(40, 39)
    assert "did not increase" in write_summary.headline(39, 39)
    assert "increased" in write_summary.headline(38, 39)
