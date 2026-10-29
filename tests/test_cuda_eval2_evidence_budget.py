"""CPU-only read-model contracts; artificial progress rows are never model results."""

import copy
import json
import shutil

import pytest

from resolve_desk import evidence_cuda2 as evidence
from training.common import ROOT
from training.cuda_eval2.verify_results import verify_execution_authorization
from training.cuda_eval2.write_summary import authorization_note


@pytest.fixture
def authorized_root(tmp_path):
    """Copy only source/metadata from actual approval, never model weights or outputs."""
    folder = ROOT / "reports/cuda-eval2"
    addendum = evidence.read(folder / "execution-addendum.json")
    names = set(addendum["source_inputs_sha256"]) | set(addendum["supplementary_sources_sha256"])
    names |= {
        f"reports/cuda-eval2/{name}"
        for name in (
            "analysis-plan.json",
            "artifact-identity.json",
            "execution-addendum.json",
        )
    }
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    availability = {
        "run_id": evidence.RUN_ID,
        "analysis_plan_sha256": evidence.PLAN_SHA256,
        "execution_addendum_sha256": evidence.ADDENDUM_SHA256,
        "status": "metered_setup",
        "completed_records": 0,
        "cap_compute_units": 10,
        "starting_balance": 80,
        "latest_balance_observed": 79.9,
        "observed_units_per_hour": 1.07,
    }
    path = tmp_path / "reports/cuda-eval2/runtime-availability.json"
    path.write_text(json.dumps(availability))
    return tmp_path, addendum, availability


def test_approved_budget_state_is_read_only_and_never_exposes_model_scores(authorized_root):
    root, _, _ = authorized_root
    before = {p.relative_to(root): evidence.digest(p) for p in root.rglob("*") if p.is_file()}
    result = evidence.validated_report(root)
    assert result["available"] is False and result["status"] == "metered_setup"
    assert result["completed_records"] == 0
    assert result["latest_balance_observed"] == 79.9
    assert result["execution_authorization"]["cap_compute_units"] == 10
    assert result["execution_authorization"]["free_runtime_operator_confirmed"] is False
    assert not {"summary", "paired", "figures", "training", "actual_cost", "units_consumed"} & result.keys()
    assert before == {p.relative_to(root): evidence.digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["plan", "addendum", "source", "cap", "count", "balance"])
def test_unbound_budget_status_fails_closed_without_budget_or_score_cards(authorized_root, fault):
    root, _, availability = authorized_root
    if fault == "plan":
        availability["analysis_plan_sha256"] = "wrong-plan"
    elif fault == "addendum":
        availability["execution_addendum_sha256"] = "wrong-addendum"
    elif fault == "source":
        (root / "training/cuda_eval2/budgeted_run.py").write_text("# tampered copied CPU source\n")
    elif fault == "cap":
        availability["cap_compute_units"] = 11
    elif fault == "count":
        availability["completed_records"] = True
    else:
        availability["latest_balance_observed"] = float("nan")
    (root / "reports/cuda-eval2/runtime-availability.json").write_text(json.dumps(availability))
    result = evidence.validated_report(root)
    assert result["available"] is False
    assert (
        not {"status", "completed_records", "summary", "figures", "execution_authorization"} & result.keys()
    )


def test_unverified_reported_progress_is_not_published_as_a_completed_count(authorized_root):
    root, _, availability = authorized_root
    availability.update(status="metered_evaluation_running", completed_records=9)
    (root / "reports/cuda-eval2/runtime-availability.json").write_text(json.dumps(availability))
    result = evidence.validated_report(root)
    assert result["available"] is False and result["status"] == "metered_evaluation_running"
    assert "completed_records" not in result and "summary" not in result
    assert "locally verifiable raw snapshot" in result["reason"]


def test_partial_progress_retains_failures_but_cannot_publish_quality(authorized_root):
    root, addendum, availability = authorized_root
    folder = root / "reports/cuda-eval2"
    plan, identity = (
        evidence.read(folder / name) for name in ("analysis-plan.json", "artifact-identity.json")
    )
    # This deliberately failing fake row exercises progress gating only, not any model metric.
    report = {
        "run_id": evidence.RUN_ID,
        "status": "partial",
        "analysis_plan_sha256": evidence.PLAN_SHA256,
        "artifact_identity_sha256": evidence.IDENTITY_SHA256,
        "cases_sha256": plan["cases_sha256"],
        "old_outputs_pooled_or_replaced": False,
        "training_repeated_or_modified": False,
        "execution_addendum_sha256": evidence.ADDENDUM_SHA256,
        "compute_authorization": {
            "authorization_mode": "metered_existing_units",
            "existing_units_operator_confirmed": True,
            "free_runtime_operator_confirmed": False,
            "cap_compute_units": 10,
            "purchases_allowed": False,
            "renewal_allowed": False,
            "automatic_provisioning": False,
            "execution_addendum_path": "reports/cuda-eval2/execution-addendum.json",
            "execution_addendum_sha256": evidence.ADDENDUM_SHA256,
        },
        "execution_budget": copy.deepcopy(addendum["accounting"]),
        "records": [
            {
                "variant": evidence.ARMS[0],
                "case_id": plan["case_ids"][0],
                "task_success": False,
                "model_calls": 1,
                "trace": [
                    {
                        "raw": "CPU malformed-output progress fixture",
                        "timing": {
                            "provider": "torch_cuda",
                            "adapter_sha256": identity["before_adapter_sha256"],
                            "execution_addendum_sha256": evidence.ADDENDUM_SHA256,
                            "execution_authorization_mode": "metered_existing_units",
                        },
                    }
                ],
            }
        ],
    }
    path = folder / "evaluation-partial.json"
    path.write_text(json.dumps(report))
    availability.update(status="metered_evaluation_running", completed_records=1)
    (folder / "runtime-availability.json").write_text(json.dumps(availability))
    before = path.read_bytes()
    result = evidence.validated_report(root)
    assert result["available"] is False and result["completed_records"] == 1
    assert not {"summary", "paired", "figures", "failures"} & result.keys()
    assert path.read_bytes() == before and evidence.read(path)["records"][0]["task_success"] is False
    actual = evidence.execution_authorization(root, report, plan, identity)
    assert actual == verify_execution_authorization(report, plan, identity, root=root)


def test_threshold_warning_and_writer_label_budget_without_claiming_actual_spend(authorized_root):
    root, _, availability = authorized_root
    availability["latest_balance_observed"] = 72
    (root / "reports/cuda-eval2/runtime-availability.json").write_text(json.dumps(availability))
    result = evidence.validated_report(root)
    assert result["budget_stop_threshold_reached"] is True
    assert "disconnect the runtime" in result["reason"]
    note = authorization_note(result["execution_authorization"])
    assert "10 existing Colab compute units" in note
    assert "not measured cost or actual units consumed" in note
    assert "does not stop billing" in note and "No successful free allocation" in note
