"""Self-contained CPU contracts; these tests do not replay any actual experiment row."""

import copy
import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest

from training.common import sha
from training.cuda_eval.evaluation import value_hash
from training.v8_eval2_partial import verify_results as verify


def synthetic_metadata():
    cases = [{"id": f"synthetic-{i:02d}"} for i in range(48)]
    rows = []
    for arm, count in zip(verify.original.ARMS, (48, 48, 40), strict=True):
        for case in cases[:count]:
            index = len(rows)
            calls = 6 if index < 85 else 5
            rows.append(
                {
                    "variant": arm,
                    "case_id": case["id"],
                    "model_calls": calls,
                    "trace": [{"raw": "synthetic fixture only", "timing": {}}] * calls,
                }
            )
    selected_ids = [c["id"] for c in cases[:40]]
    chosen = [(i, row) for i, row in enumerate(rows) if row["case_id"] in selected_ids]
    excluded = [(i, row) for i, row in enumerate(rows) if row["case_id"] not in selected_ids]

    def reference(index, row):
        return {
            "original_record_index": index,
            "variant": row["variant"],
            "case_id": row["case_id"],
            "canonical_sha256": value_hash(row),
        }

    selection = {
        "analysis_id": verify.ANALYSIS_ID,
        "run_id": verify.RUN_ID,
        "case_ids": selected_ids,
        "cases": [
            {"case_id": c["id"], "original_case_index": i, "canonical_sha256": value_hash(c)}
            for i, c in enumerate(cases[:40])
        ],
        "records": [reference(i, row) for i, row in chosen],
        "excluded_completed_records": [reference(i, row) for i, row in excluded],
        "canonical_selected_cases_sha256": value_hash(cases[:40]),
        "canonical_selected_records_sha256": value_hash([row for _, row in chosen]),
    }
    report = {
        "run_id": verify.RUN_ID,
        "round": "v8-eval2",
        "split": "test",
        "status": "partial",
        "resume_count": 0,
        "checkpoint_selection_performed": False,
        "additional_training_performed": False,
        "validation_model_calls": 0,
        "max_total_calls_per_task": 12,
        "model_loads_this_process": 1,
        "report_only": True,
        "deployment_or_default_provider_change_allowed": False,
        "cases_sha256": value_hash(cases),
        "records": rows,
        "pending_case": {"arm": verify.original.ARMS[-1], "case_id": cases[40]["id"]},
    }
    journal = [{"status": "completed"} for _ in range(765)] + [{"status": "started"}]
    old_plan = {
        "case_order": [c["id"] for c in cases],
        "dataset": {"canonical_cases_sha256": value_hash(cases)},
    }
    return cases, rows, selection, report, journal, old_plan


def test_unsealed_analysis_stops_before_tensor_token_or_business_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "REPORTS", tmp_path)
    (tmp_path / "analysis-plan.json").write_text(
        json.dumps(
            {
                "analysis_id": verify.ANALYSIS_ID,
                "run_id": verify.RUN_ID,
                "status": "draft_partial_analysis",
                "sealed": False,
            }
        )
    )
    monkeypatch.setattr(verify, "read_snapshot", lambda: pytest.fail("must not read raw for replay"))
    with pytest.raises(PermissionError, match="sealed"):
        verify.verify()


def test_sealed_scope_binds_source_human_selection_and_original_bytes(tmp_path, monkeypatch):
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    reports = root / "reports/v8-eval2-partial40"
    reports.mkdir(parents=True)
    snapshot = root / "snapshot"
    snapshot.mkdir()
    monkeypatch.setattr(verify, "ROOT", root)
    monkeypatch.setattr(verify, "REPORTS", reports)
    monkeypatch.setattr(verify, "SNAPSHOT", snapshot)
    source = root / "source.py"
    source.write_text("CPU contract fixture source only")
    approval = root / "approval.json"
    approval.write_text('{"fixture":"Not actual human authorization"}')
    for name in ("selection.json", "selection-audit.json"):
        (reports / name).write_text("{}")
    paths = {
        "analysis_plan": "reports/v8-eval2/analysis-plan.json",
        "artifact_identity": "reports/v8-eval2/artifact-identity.json",
        "raw_report": "reports/v8-eval2/evaluation-partial.json",
        "call_journal": "reports/v8-eval2/call-journal.json",
        "cases": "training/v8_eval2/datasets/cases.json",
        "snapshot_receipt": "snapshot-receipt.json",
        "import_verification": "import-verification.json",
        "resource_receipt": "reports/v8-eval2/resource-authorization.json",
        "deadline_marker": "reports/v8-eval2/deadline-stop.json",
    }
    experiment = {"expected_records": 144, "raw_records": 136, "original144_gate_complete": False}
    for label, name in paths.items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        experiment[label + "_path"] = str(path.relative_to(root))
        experiment[label + "_sha256"] = sha(path)
    archive = tmp_path / "exports/final.zip"
    archive.parent.mkdir()
    archive.write_bytes(b"CPU contract fixture only; not an experiment archive")
    monkeypatch.setattr(verify, "FINAL_ARCHIVE_SHA", sha(archive))
    experiment.update(final_archive_path="exports/final.zip", final_archive_sha256=sha(archive))
    plan = {
        "analysis_id": verify.ANALYSIS_ID,
        "run_id": verify.RUN_ID,
        "status": "frozen_partial_analysis",
        "sealed": True,
        "source_seal_pending": False,
        "human_scope_approval": {
            "explicit": True,
            "text": "CPU fixture only",
            "receipt_path": "approval.json",
            "receipt_sha256": sha(approval),
        },
        "pending_source_paths": [],
        "required_analysis_source_paths": ["source.py"],
        "source_inputs_sha256": {"source.py": sha(source)},
        "selection_sha256": sha(reports / "selection.json"),
        "selection_audit_sha256": sha(reports / "selection-audit.json"),
        "original_experiment": experiment,
        "original144_gate_complete": False,
        "mark_original144_complete_allowed": False,
        "post_hoc_partial_analysis": True,
        "quality_scores_computed_before_seal": False,
    }
    (reports / "analysis-plan.json").write_text(json.dumps(plan))
    assert verify.require_sealed_scope()[0] == plan
    source.write_text("changed source after sealing")
    with pytest.raises(ValueError, match="source changed"):
        verify.require_sealed_scope()


def test_derived_scoring_preserves_a_failed_raw_record_after_verification(monkeypatch):
    row = {
        "variant": verify.original.ARMS[-1],
        "case_id": "synthetic-failure",
        "raw_first": '{"arguments":{"reason":"wrong-enum"}}',
        **{field: False for field in verify.GRADE_FIELDS},
    }
    before = copy.deepcopy(row)
    seen = []

    def verified(*args):
        seen.append(args[0])
        return [[{"role": "user", "content": "CPU fixture prompt"}]]

    monkeypatch.setattr(verify.original, "verify_record", verified)
    monkeypatch.setattr(verify.original, "audit_consent", lambda record, case: {"fixture": True})
    audit, prompts = verify.verify_record_audit(7, row, {"id": row["case_id"]}, {}, {}, None, [], {}, set())
    assert seen == [row] and row == before
    assert audit["source_record_sha256"] == value_hash(before)
    assert audit["source_record_index"] == 7 and not audit["selected_for_paired40"]
    assert audit["derived_scoring"]["task_success"] is False
    assert audit["first_raw_before_repair_preserved"] is True and len(prompts) == 1


def test_fixed120_prefix_selection_preserves16_excluded_rows():
    cases, rows, selection, report, journal, plan = synthetic_metadata()
    verify.require_partial_record_prefix(report, journal, cases, plan)
    selected = verify.verify_selection(selection, rows, cases)
    assert len(selected) == 120
    assert len(rows) == 136 and len(selection["excluded_completed_records"]) == 16
    assert sum(r["model_calls"] for r in rows) == 765


def test_grade_fields_exist_in_real_frozen_label_aware_oracle_schema(tmp_path):
    # Teacher/schema contract only: these are generated CPU reference actions,
    # never retained candidate outputs or a replay of the actual136 model rows.
    from training.v8_eval2.fixtures import load_cases
    from training.v8_eval2.prepare import ReferenceProvider

    case = load_cases("test")[0]
    for label, Store, messages in (
        ("v7", verify.original.OldStore, verify.original.old_messages),
        ("v8", verify.original.NewStore, verify.original.new_messages),
    ):
        row = verify.original.run_case(
            Store(tmp_path / f"{label}-oracle.sqlite"), case, ReferenceProvider(case), messages, max_steps=12
        )
        assert set(verify.GRADE_FIELDS) <= set(row)
        assert "trace" in row and "events" in row


@pytest.mark.parametrize(
    "change", ["case_order", "duplicate_record", "wrong_record_hash", "drop_failure", "excluded_tail"]
)
def test_no_outcome_selected_prefix_or_missing_excluded_audit(change):
    cases, rows, selection, *_ = synthetic_metadata()
    if change == "case_order":
        selection["case_ids"][0], selection["case_ids"][1] = (
            selection["case_ids"][1],
            selection["case_ids"][0],
        )
    elif change == "duplicate_record":
        selection["records"][1] = copy.deepcopy(selection["records"][0])
    elif change == "wrong_record_hash":
        selection["records"][0]["canonical_sha256"] = "0" * 64
    elif change == "drop_failure":
        selection["records"].pop()
    else:
        selection["excluded_completed_records"].pop()
    with pytest.raises(AssertionError):
        verify.verify_selection(selection, rows, cases)


def test_complete_original_gate_cannot_be_substituted_for_partial_metadata():
    cases, _, _, report, journal, plan = synthetic_metadata()
    report["status"] = "complete"
    with pytest.raises(AssertionError):
        verify.require_partial_record_prefix(report, journal, cases, plan)


def test_excluded_tail_audit_failure_is_not_silently_dropped(monkeypatch):
    cases, rows, selection, *_ = synthetic_metadata()
    chosen = verify.verify_selection(selection, rows, cases)
    seen = []

    def fake_audit(index, row, *args):
        seen.append(index)
        if index == 40:
            assert (row["variant"], row["case_id"]) not in chosen
            raise AssertionError("Excluded complete tail failed independent audit")
        return {"selected_for_paired40": True}, []

    monkeypatch.setattr(verify, "verify_record_audit", fake_audit)
    with pytest.raises(AssertionError, match="Excluded complete tail"):
        verify.audit_records(rows, cases, {}, {}, None, [], {}, chosen)
    assert seen == list(range(41))


def unknown_fixture(monkeypatch):
    prompt = [{"role": "user", "content": "Synthetic unknown-input contract only"}]
    bindings = {
        "run_id": verify.RUN_ID,
        "analysis_plan_sha256": "a" * 64,
        "artifact_identity_sha256": "b" * 64,
        "resource_receipt_sha256": "c" * 64,
    }
    entry = {
        **bindings,
        "variant": verify.original.ARMS[-1],
        "case_id": "synthetic-pending",
        "call_index": 1,
        "status": "started",
        "started_at_utc": "2035-01-01T00:02:00+00:00",
        "request_messages": prompt,
        "request_sha256": value_hash(prompt),
    }
    pending = {"arm": entry["variant"], "case_id": entry["case_id"]}
    identity = {"sealed_at_utc": "2034-12-31T23:00:00+00:00"}
    resource = {
        "allocated_at_utc": "2035-01-01T00:00:00+00:00",
        "process_deadline_utc": "2035-01-01T00:03:00+00:00",
    }
    monkeypatch.setattr(verify, "initial_prompt", lambda case, variant: prompt)
    return entry, pending, {"id": entry["case_id"]}, identity, resource, bindings


def test_unknown_request_does_not_invent_input_tokens_response_or_apply(monkeypatch):
    fixture = unknown_fixture(monkeypatch)
    result = verify.verify_unknown_request(*fixture, datetime(2035, 1, 1, 0, 1, tzinfo=UTC))
    assert result["request_matches_frozen_initial_prompt"] is True
    assert result["actual_input_token_ids_retained"] is False
    assert result["actual_token_replay_performed"] is False
    assert result["actual_response_and_business_effects_unverifiable"] is True
    assert result["output_or_response_generated_or_reconstructed"] is False
    assert result["business_action_or_confirmation_applied"] is False


@pytest.mark.parametrize("change", ["raw", "timing", "completed_at_utc", "wrong_hash", "late", "wrong_case"])
def test_unknown_started_call_cannot_become_done_or_retried(monkeypatch, change):
    fixture = unknown_fixture(monkeypatch)
    entry = fixture[0]
    if change in {"raw", "timing", "completed_at_utc"}:
        entry[change] = "forbidden synthetic addition"
    elif change == "wrong_hash":
        entry["request_sha256"] = "0" * 64
    elif change == "late":
        entry["started_at_utc"] = "2035-01-01T00:03:00+00:00"
    else:
        entry["case_id"] = "different-case"
    with pytest.raises(AssertionError):
        verify.verify_unknown_request(*fixture, datetime(2035, 1, 1, 0, 1, tzinfo=UTC))


def test_no_model_framework_is_imported_for_contracts():
    code = (
        "import sys;import training.v8_eval2_partial.verify_results;"
        "assert 'torch' not in sys.modules;assert 'mlx.core' not in sys.modules"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, timeout=15)
