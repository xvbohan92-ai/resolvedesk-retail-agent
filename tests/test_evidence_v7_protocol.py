"""CPU-only contract tests for protocol integrity; fixtures are not model evidence."""

import copy
import json
from datetime import UTC, datetime

import pytest

from resolve_desk import evidence_v7 as evidence


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def protocol_bundle(tmp_path, monkeypatch, *, violation=False, blocked=False):
    root = tmp_path
    folder, data = root / "reports/v7", root / "training/v7/datasets"
    folder.mkdir(parents=True)
    data.mkdir(parents=True)
    source = root / "training/v7/protocol_audit.py"
    source.write_text("# Unit test source marker; never executed.\n")
    cases = [
        {
            "id": f"unit-contract-{n:02d}",
            "split": "test",
            "intent_initial": {
                "operation": "exchange" if n < 30 else "return",
            },
        }
        for n in range(48)
    ]
    write_json(data / "cases.json", cases)
    manifest = {"data": {"tasks": {"test": 48}}}
    manifest_path = data / "manifest.json"
    write_json(manifest_path, manifest)
    report = {
        "manifest_sha256": evidence.digest(manifest_path),
        "release_gate": {"promoted": True, "checks": {"unit_business_check": True}},
        "summary": {arm: {"tasks": 48} for arm in evidence.ARMS},
        "records": [
            {"variant": arm, "case_id": c["id"], "events": [{"created": 2}]}
            for arm in evidence.ARMS
            for c in cases
        ],
    }
    write_json(folder / "evaluation.json", report)
    plan = {
        "created_at_utc": datetime.fromtimestamp(1, UTC).isoformat(),
        "test_inference_started": False,
        "source_sha256": evidence.digest(source),
        "manifest_sha256": evidence.digest(manifest_path),
        "cases_sha256": evidence.digest(data / "cases.json"),
        "variants": list(evidence.ARMS),
        "expected_tasks_per_arm": 48,
        "business_scores_or_release_gate_modified": False,
        "selection_rule_modified": False,
    }
    write_json(folder / "protocol-audit-plan.json", plan)
    plan_sha = evidence.digest(folder / "protocol-audit-plan.json")
    monkeypatch.setattr(evidence, "PROTOCOL_PLAN_SHA256", plan_sha)
    rows = [
        {
            "variant": arm,
            "case_id": c["id"],
            "audited": c["intent_initial"]["operation"] == "exchange",
            **dict.fromkeys(evidence.PROTOCOL_COUNTS, 0),
            "attempts": [],
        }
        for arm in evidence.ARMS
        for c in cases
    ]
    target = next(r for r in rows if r["variant"] == "qlora_v7")
    if violation or blocked:
        target.update(raw_return_attempts=1, return_attempts_without_scoped_yes=1)
        target["attempts"] = [
            {
                "raw_return_attempt": True,
                "scoped_yes_verified": False,
                "return_proposal_created": violation,
                "backend_error": "policy_rejection" if blocked else None,
            }
        ]
    if violation:
        target.update(
            return_proposals=1,
            return_proposals_without_scoped_yes=1,
            committed_returns=1,
            committed_returns_without_scoped_yes=1,
            final_return_orders=1,
            final_returns_without_verified_commit=1,
        )
    summary = {
        arm: {
            "tasks": 48,
            "gold_exchange_tasks": 30,
            **{k: sum(r[k] for r in rows if r["variant"] == arm) for k in evidence.PROTOCOL_COUNTS},
        }
        for arm in evidence.ARMS
    }
    audit = {
        "scope": "CPU unit fixture only, not model evaluation evidence.",
        "limitations": "Synthetic audit integrity contract fixture.",
        "plan_created_at_utc": plan["created_at_utc"],
        "plan_precedes_heldout_events": True,
        "plan_sha256": plan_sha,
        "source_sha256": evidence.digest(source),
        "manifest_sha256": evidence.digest(manifest_path),
        "evaluation_sha256": evidence.digest(folder / "evaluation.json"),
        "records_audited": 192,
        "business_scores_or_release_gate_modified": False,
        "selection_rule_modified": False,
        "summary": summary,
        "records": rows,
    }
    values = {"primary_test_cases": 48}

    def refresh():
        write_json(folder / "protocol-audit.json", audit)
        values.update(
            protocol_audit_sha256=evidence.digest(folder / "protocol-audit.json"),
            protocol_audit_source_sha256=evidence.digest(source),
            protocol_audit_plan_sha256=plan_sha,
            protocol_audit=copy.deepcopy(audit),
        )

    refresh()
    return root, report, manifest_path, manifest, values, audit, refresh


def test_protocol_violation_is_visible_without_changing_a_passing_business_gate(tmp_path, monkeypatch):
    bundle = protocol_bundle(tmp_path, monkeypatch, violation=True)
    original_gate = copy.deepcopy(bundle[1]["release_gate"])
    public = evidence.protocol_evidence(*bundle[:5])
    assert public["candidate_safety_passed"] is False
    assert public["business_gate_unchanged"] is True
    assert bundle[1]["release_gate"] == original_gate
    assert public["summary"]["qlora_v7"]["return_proposals_without_scoped_yes"] == 1
    assert public["summary"]["qlora_v7"]["committed_returns_without_scoped_yes"] == 1
    assert public["summary"]["qlora_v7"]["final_returns_without_verified_commit"] == 1
    assert public["summary"]["qlora_v7"]["rejected_return_attempts_without_scoped_yes"] == 0


def test_blocked_raw_return_is_separate_from_executed_consent_violations(tmp_path, monkeypatch):
    bundle = protocol_bundle(tmp_path, monkeypatch, blocked=True)
    public = evidence.protocol_evidence(*bundle[:5])
    assert public["candidate_safety_passed"] is True
    assert public["summary"]["qlora_v7"]["return_attempts_without_scoped_yes"] == 1
    assert public["summary"]["qlora_v7"]["rejected_return_attempts_without_scoped_yes"] == 1
    assert public["summary"]["qlora_v7"]["return_proposals_without_scoped_yes"] == 0


@pytest.mark.parametrize(
    "tamper",
    ["source", "plan", "evaluation", "cases", "missing_record", "duplicate", "summary", "missing_audit"],
)
def test_protocol_evidence_rejects_tampered_or_incomplete_inputs(tmp_path, monkeypatch, tamper):
    bundle = protocol_bundle(tmp_path, monkeypatch)
    root, _, _, _, _, audit, refresh = bundle
    folder = root / "reports/v7"
    if tamper in {"source", "plan", "evaluation", "cases"}:
        path = {
            "source": root / "training/v7/protocol_audit.py",
            "plan": folder / "protocol-audit-plan.json",
            "evaluation": folder / "evaluation.json",
            "cases": root / "training/v7/datasets/cases.json",
        }[tamper]
        path.write_text(path.read_text() + " ")
    elif tamper == "missing_record":
        audit["records"].pop()
        audit["records_audited"] = 191
        refresh()
    elif tamper == "duplicate":
        audit["records"][1]["case_id"] = audit["records"][0]["case_id"]
        refresh()
    elif tamper == "summary":
        audit["summary"]["qlora_v7"]["return_proposals_without_scoped_yes"] = 1
        refresh()
    else:
        (folder / "protocol-audit.json").unlink()
    with pytest.raises((AssertionError, OSError)):
        evidence.protocol_evidence(*bundle[:5])
