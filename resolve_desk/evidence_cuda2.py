"""Fail-closed, read-only evidence for the separate matched CUDA comparison."""

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

ARMS = ("predecessor_fp16", "cloud1344_fp16")
RUN_ID = "resolvedesk-cuda-eval-002"
PLAN_SHA256 = "a547fd0790ad2003fcd3dc0a990b581484bf5decd30222c87499237fd236f704"
IDENTITY_SHA256 = "f83a97af4775ac2c01bf82f5ff188de1743c010ec143c554b74a8309f7840d1e"
ADDENDUM_SHA256 = "e6e72a162af8c140a83ef9cf4c13102c24afb7428905ade797e98463d14c0e3e"
FIGURES = ("evaluation", "intent", "recovery", "families", "consent", "training")
COUNTS = (
    "raw_return_attempts",
    "return_attempts_without_scoped_yes",
    "return_proposals",
    "return_proposals_without_scoped_yes",
    "committed_returns",
    "committed_returns_without_scoped_yes",
    "final_return_orders",
    "final_returns_without_verified_commit",
    "declined_return_proposals",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local(root, name):
    relative = Path(name)
    assert not relative.is_absolute() and ".." not in relative.parts
    return root / relative


def resource_addendum(root, plan, identity):
    """Read the actually approved immutable authorization; never infer billing from GPU type."""
    folder = root / "reports/cuda-eval2"
    path = folder / "execution-addendum.json"
    assert digest(path) == ADDENDUM_SHA256
    addendum = read(path)
    assert addendum["schema_version"] == 1
    assert addendum["run_id"] == plan["run_id"] == identity["run_id"] == RUN_ID
    assert addendum["analysis_plan_sha256"] == digest(folder / "analysis-plan.json") == PLAN_SHA256
    assert (
        addendum["artifact_identity_sha256"] == digest(folder / "artifact-identity.json") == IDENTITY_SHA256
    )
    assert addendum["authorization_mode"] == "metered_existing_units"
    assert addendum["cases_sha256"] == plan["cases_sha256"]
    assert addendum["source_inputs_sha256"] == plan["source_inputs_sha256"]
    assert len(addendum["source_inputs_sha256"]) == 53
    for name, wanted in addendum["source_inputs_sha256"].items():
        assert digest(local(root, name)) == wanted, name
    assert set(addendum["supplementary_sources_sha256"]) == {
        "training/cuda_eval2/budgeted_run.py",
        "tests/test_cuda_eval2_budget.py",
    }
    for name, wanted in addendum["supplementary_sources_sha256"].items():
        assert digest(local(root, name)) == wanted, name
    for key in ("before_adapter_sha256", "after_adapter_sha256"):
        assert addendum[key] == identity[key]
    assert addendum["previous_interrupted_run"] == plan["previous_interrupted_run"]
    assert addendum["supersedes_only"]["field"] == "compute_authorization.free_gpu_only"
    assert addendum["supersedes_only"]["original_value"] is True
    assert addendum["supersedes_only"]["effective_authorization"] == "metered_existing_units"
    unchanged = addendum["unchanged_protocol"]
    assert unchanged["generation"] == plan["generation"]
    assert unchanged["inference_model_kwargs"] == plan["inference_model_kwargs"]
    assert unchanged["case_order"] == plan["case_ids"] and tuple(unchanged["arms"]) == ARMS
    assert unchanged["seed"] == 202610287 and unchanged["total_calls_per_task"] == 12
    assert unchanged["controller"] == unchanged["scoring"] == "frozen_v7"
    assert all(
        unchanged[k] is True
        for k in (
            "no_retraining",
            "no_checkpoint_selection",
            "no_old_run_replay",
            "no_old_outputs_pooled",
            "T4_only",
        )
    )
    human, accounting = addendum["human_authorization"], addendum["accounting"]
    assert human["explicit"] is human["T4_only"] is True
    assert isinstance(human["approval_text"], str) and human["approval_text"].strip()
    assert human["cap_compute_units"] == 10 and human["starting_balance_units"] == 80
    assert human["stop_balance_units"] == 72 and human["hard_floor_balance_units"] == 70
    assert human["purchase_allowed"] is human["renewal_allowed"] is human["free_allocation_claimed"] is False
    for key, value in {
        "starting_balance_units": 80,
        "hard_cap_units": 10,
        "hard_cap_balance": 70,
        "operator_stop_balance": 72,
        "reserved_units": 2,
    }.items():
        assert accounting[key] == value
    assert (
        accounting["automatic_compute_unit_metering"]
        is accounting["billing_stops_when_process_exits"]
        is False
    )
    assert accounting["projection_is_actual_consumption"] is False
    allocated, deadline, disconnect = (
        datetime.fromisoformat(accounting[key])
        for key in (
            "runtime_allocated_at_utc",
            "process_deadline_utc",
            "runtime_disconnect_deadline_utc",
        )
    )
    created = datetime.fromisoformat(addendum["created_at_utc"])
    assert datetime.fromisoformat(identity["sealed_at_utc"]) <= created
    assert allocated <= created < deadline < disconnect
    rate = accounting["observed_units_per_hour"]
    assert math.isfinite(rate) and rate > 0
    assert (
        accounting["projected_units_until_disconnect"]
        == ((disconnect - allocated).total_seconds() / 3600 * rate)
        <= 8
    )
    return addendum, {
        "authorization_mode": "metered_existing_units",
        "execution_addendum_required": True,
        "execution_addendum_path": "reports/cuda-eval2/execution-addendum.json",
        "execution_addendum_sha256": ADDENDUM_SHA256,
        "existing_units_operator_confirmed": True,
        "free_runtime_operator_confirmed": False,
        "cap_compute_units": 10,
        "starting_balance_units": 80,
        "stop_balance_units": 72,
        "hard_floor_balance_units": 70,
        "reserved_units": 2,
        "purchases_allowed": False,
        "renewal_allowed": False,
        "automatic_metering": False,
        "billing_stops_when_process_exits": False,
        "process_deadline_utc": accounting["process_deadline_utc"],
        "runtime_disconnect_deadline_utc": accounting["runtime_disconnect_deadline_utc"],
        "scope": (
            "Human-authorized existing-unit ceiling with external operator monitoring; not billing proof."
        ),
    }


def execution_authorization(root, report, plan, identity, journal=None):
    declared = report["compute_authorization"]
    assert plan["compute_authorization"]["free_gpu_only"] is True
    assert plan["compute_authorization"]["paid_fallback_allowed"] is False
    assert plan["compute_authorization"]["automatic_provisioning"] is False
    if declared.get("authorization_mode") != "metered_existing_units" and not report.get(
        "execution_addendum_sha256"
    ):
        assert declared.get("authorization_mode") in (None, "free_gpu")
        assert declared["free_runtime_operator_confirmed"] is True
        assert declared["paid_fallback_allowed"] is declared["automatic_provisioning"] is False
        return {
            "authorization_mode": "free_gpu",
            "free_runtime_operator_confirmed": True,
            "execution_addendum_required": False,
            "paid_fallback_allowed": False,
        }
    addendum, authorized = resource_addendum(root, plan, identity)
    assert declared["authorization_mode"] == authorized["authorization_mode"]
    assert report["execution_addendum_sha256"] == declared["execution_addendum_sha256"] == ADDENDUM_SHA256
    for key in (
        "existing_units_operator_confirmed",
        "free_runtime_operator_confirmed",
        "cap_compute_units",
        "purchases_allowed",
        "renewal_allowed",
        "execution_addendum_path",
    ):
        assert declared[key] == authorized[key]
    assert declared.get("automatic_provisioning", False) is False
    assert declared.get("paid_fallback_allowed", False) is False
    assert report["execution_budget"] == addendum["accounting"]
    for record in report["records"]:
        for observation in record["trace"]:
            assert observation["timing"]["execution_addendum_sha256"] == ADDENDUM_SHA256
            assert observation["timing"]["execution_authorization_mode"] == "metered_existing_units"
    if journal is not None:
        created = datetime.fromisoformat(addendum["created_at_utc"])
        deadline = datetime.fromisoformat(addendum["accounting"]["process_deadline_utc"])
        for entry in journal:
            assert entry["run_id"] == RUN_ID and entry["execution_addendum_sha256"] == ADDENDUM_SHA256
            assert created < datetime.fromisoformat(entry["started_at_utc"]) < deadline
            assert datetime.fromisoformat(entry["completed_at_utc"]) <= deadline
    return authorized


def pending(root=None):
    result = {
        "available": False,
        "experiment": "cuda_eval2",
        "reason": "Independent round 2 is pending a complete 96-record comparison, independent "
        "replay, training proof and verified figures. No round-1 partial scores are reused.",
    }
    try:
        root = Path(root)
        folder = root / "reports/cuda-eval2"
        availability = read(folder / "runtime-availability.json")
        assert availability["run_id"] == RUN_ID
        if availability["status"] in {"metered_setup", "metered_evaluation_running"}:
            plan, identity = (
                read(folder / name) for name in ("analysis-plan.json", "artifact-identity.json")
            )
            addendum, authorized = resource_addendum(root, plan, identity)
            assert availability["analysis_plan_sha256"] == PLAN_SHA256
            assert availability["execution_addendum_sha256"] == ADDENDUM_SHA256
            assert availability["cap_compute_units"] == 10 and availability["starting_balance"] == 80
            assert (
                availability["observed_units_per_hour"] == addendum["accounting"]["observed_units_per_hour"]
            )
            stated_count = availability["completed_records"]
            assert type(stated_count) is int and 0 <= stated_count <= 96
            balance = availability["latest_balance_observed"]
            assert balance is None or (
                type(balance) in (int, float) and math.isfinite(balance) and 0 <= balance <= 80
            )
            verified_count = 0 if stated_count == 0 else None
            for name in ("evaluation.json", "evaluation-partial.json"):
                if not (folder / name).exists():
                    continue
                partial = read(folder / name)
                assert partial["run_id"] == RUN_ID
                assert partial["analysis_plan_sha256"] == PLAN_SHA256
                assert partial["artifact_identity_sha256"] == IDENTITY_SHA256
                assert partial["cases_sha256"] == plan["cases_sha256"]
                assert partial["old_outputs_pooled_or_replaced"] is False
                assert partial["training_repeated_or_modified"] is False
                execution_authorization(root, partial, plan, identity)
                order = [(arm, cid) for arm in ARMS for cid in plan["case_ids"]]
                actual = [(row["variant"], row["case_id"]) for row in partial["records"]]
                assert actual == order[: len(actual)] and len(actual) <= 96
                assert all(0 < r["model_calls"] == len(r["trace"]) <= 12 for r in partial["records"])
                for record in partial["records"]:
                    adapter = identity[
                        "before_adapter_sha256" if record["variant"] == ARMS[0] else "after_adapter_sha256"
                    ]
                    assert all(
                        isinstance(o["raw"], str)
                        and o["timing"]["provider"] == "torch_cuda"
                        and o["timing"]["adapter_sha256"] == adapter
                        for o in record["trace"]
                    )
                verified_count = len(actual)
                break
            progress = (
                f"Last locally verified completed records: {verified_count}/96. "
                if verified_count is not None
                else "Current progress awaits a locally verifiable raw snapshot. "
            )
            stop = balance is not None and balance <= 72
            result.update(
                status=availability["status"],
                execution_authorization=authorized,
                latest_balance_observed=balance,
                observed_units_per_hour=availability["observed_units_per_hour"],
                budget_stop_threshold_reached=stop,
                runtime_receipt_sha256=digest(folder / "runtime-availability.json"),
                reason=(
                    "Existing-unit T4 execution is authorized up to 10 compute units; "
                    "no purchases or renewal. "
                    + progress
                    + (
                        "The operator stop threshold has been reached; disconnect the runtime. "
                        if stop
                        else ""
                    )
                    + "This is an externally observed budget/status snapshot, not actual cost measurement. "
                    "Complete independent quality verification and final figures are still pending."
                ),
            )
            if verified_count is not None:
                result["completed_records"] = verified_count
            return result
        assert availability["status"] == "free_gpu_quota_exhausted"
        assert availability["free_gpu_only"] is True
        result.update(
            status="waiting_free_gpu",
            reason=(
                "Independent round 2 has not started. Colab refused the free GPU because the "
                "account usage limit was reached. The 1,344-step final adapter is retained. "
                "No new comparison scores or figures are available."
            ),
        )
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        pass
    return result


def validated_report(root):
    try:
        root = Path(root)
        folder, figures = root / "reports/cuda-eval2", root / "docs/figures/cuda-eval2"
        plan_path, report_path, identity_path = (
            folder / n for n in ["analysis-plan.json", "evaluation.json", "artifact-identity.json"]
        )
        plan, report, identity = (read(path) for path in [plan_path, report_path, identity_path])
        assert digest(plan_path) == PLAN_SHA256 and digest(identity_path) == IDENTITY_SHA256
        proof, verification, figure_check = (
            read(folder / n) for n in ["training-proof.json", "verification.json", "figure-verification.json"]
        )
        protocol, values = read(folder / "protocol-audit.json"), read(figures / "values.json")
        assert all(row["run_id"] == "resolvedesk-cuda-eval-002" for row in [plan, report, identity])
        previous = plan["previous_interrupted_run"]
        assert previous == report["previous_interrupted_run"] == identity["previous_interrupted_run"]
        assert previous["analysis_plan_sha256"] == digest(root / "reports/cuda-eval/analysis-plan.json")
        assert previous["interruption_sha256"] == digest(root / previous["interruption_path"])
        assert previous["snapshot_audit_sha256"] == digest(root / previous["snapshot_audit_path"])
        assert report["old_outputs_pooled_or_replaced"] is False
        assert report["training_repeated_or_modified"] is False
        assert report["status"] == "complete" and report["provider"] == "torch_cuda"
        assert not report.get("pending_case")
        authorized = execution_authorization(root, report, plan, identity)
        assert report["analysis_plan_sha256"] == identity["analysis_plan_sha256"] == digest(plan_path)
        assert report["artifact_identity_sha256"] == digest(identity_path)
        assert tuple(plan["arms"]) == ARMS and plan["total_calls_per_task"] == 12
        assert plan["generation"]["max_new_tokens"] == 512 and plan["generation"]["do_sample"] is False
        assert (
            plan["after_checkpoint"]["completed_updates"]
            == identity["completed_true_optimizer_updates"]
            == 1344
        )
        assert identity["selection_performed"] is False
        assert (
            report["checkpoint_selection_performed"] is False
            and report["training_or_test_tuning_performed"] is False
        )
        assert not plan["test_inference_started"]
        assert report["source_inputs_sha256"] == plan["source_inputs_sha256"]
        for name, wanted in plan["source_inputs_sha256"].items():
            assert digest(local(root, name)) == wanted
        cases = {case["id"]: case for case in read(root / "training/cuda_eval2/datasets/cases.json")}
        assert len(cases) == 48 and list(cases) == plan["case_ids"]
        assert (
            report["cases_sha256"]
            == plan["cases_sha256"]
            == digest(root / "training/cuda_eval2/datasets/cases.json")
        )
        expected = Counter({(arm, cid): 1 for arm in ARMS for cid in cases})
        assert (
            len(report["records"]) == 96
            and Counter((r["variant"], r["case_id"]) for r in report["records"]) == expected
        )
        assert [(r["variant"], r["case_id"]) for r in report["records"]] == [
            (arm, cid) for arm in ARMS for cid in cases
        ]
        for record in report["records"]:
            assert 0 < record["model_calls"] == len(record["trace"]) <= 12
            wanted_sha = (
                identity["before_adapter_sha256"]
                if record["variant"] == ARMS[0]
                else identity["after_adapter_sha256"]
            )
            assert all(
                o["timing"]["provider"] == "torch_cuda" and o["timing"]["adapter_sha256"] == wanted_sha
                for o in record["trace"]
            )
        assert (
            proof["completed_updates"] == 1344
            and proof["parameters"] == 3670016
            and proof["tensor_count"] == 112
        )
        assert (
            proof["report_sha256"]
            == identity["training_report_sha256"]
            == digest(local(root, proof["report_path"]))
        )
        assert proof["after_adapter_sha256"] == identity["after_adapter_sha256"]
        assert (
            proof["before_adapter_sha256"]
            == identity["before_adapter_sha256"]
            == plan["before_initializer_sha256"]
        )
        assert proof["full_physical_rows_used"] and proof["final_fixed_before_test"]
        assert proof["numeric_initialization_and_final_delta_verified"] is True
        assert proof["verification_sha256"] == digest(local(root, proof["verification_path"]))
        training_verification = read(local(root, proof["verification_path"]))
        assert training_verification["report_sha256"] == proof["report_sha256"]
        assert training_verification["tensor_audit"]["numeric_values_independently_checked"] is True
        assert (
            training_verification["tensor_audit"]["files_sha256"]["final_adapter"]
            == proof["after_adapter_sha256"]
        )
        assert values["run_id"] == "resolvedesk-cuda-eval-002"
        assert values["previous_interrupted_run"] == previous
        for checked in [verification, figure_check]:
            assert checked["run_id"] == "resolvedesk-cuda-eval-002"
            assert checked["previous_interrupted_run"] == previous
            assert checked["execution_authorization"] == authorized
            assert checked["records_reconciled"] == checked["scoped_consent_rows_recomputed"] == 96
            assert checked["fresh_test_tasks"] == 48 and checked["max_total_calls_per_task"] == 12
            assert (
                checked["raw_actions_and_confirmations_replayed"]
                and checked["raw_first_output_metrics_reconciled"]
            )
            assert checked["prompt_and_output_tokens_replayed"] is True
            assert checked["evaluation_sha256"] == digest(report_path)
            assert checked["protocol_audit_sha256"] == digest(folder / "protocol-audit.json")
            assert checked["analysis_plan_sha256"] == digest(plan_path)
            assert checked["artifact_identity_sha256"] == digest(identity_path)
            assert checked["training_proof_sha256"] == digest(folder / "training-proof.json")
            assert checked["source_sha256"] == digest(root / "training/cuda_eval2/verify_results.py")
        assert figure_check["figures_reconciled"] is True
        assert figure_check["values_sha256"] == digest(figures / "values.json")
        assert figure_check["csv_sha256"] == values["csv_sha256"] == digest(figures / "per-case.csv")
        assert values["evaluation_sha256"] == digest(report_path)
        assert values["protocol_audit_sha256"] == digest(folder / "protocol-audit.json")
        assert values["analysis_plan_sha256"] == digest(plan_path)
        assert values["artifact_identity_sha256"] == digest(identity_path)
        assert values["training_report_sha256"] == proof["report_sha256"]
        assert values["training_proof_sha256"] == digest(folder / "training-proof.json")
        assert values["verification_sha256"] == digest(folder / "verification.json")
        assert values["analysis_source_sha256"] == digest(root / "training/cuda_eval2/visualize.py")
        assert values["metrics_source_sha256"] == digest(root / "training/cuda_eval2/metrics.py")
        expected_figures = {f"{name}.{extension}" for name in FIGURES for extension in ["png", "svg"]}
        assert set(values["figure_sha256"]) == expected_figures
        assert figure_check["figure_hashes_verified"] == values["figure_sha256"]
        for name, wanted in values["figure_sha256"].items():
            assert digest(figures / name) == wanted
        assert report["protocol_reference"] == {
            "path": "protocol-audit.json",
            "sha256": digest(folder / "protocol-audit.json"),
        }
        assert protocol["run_id"] == RUN_ID
        if authorized["execution_addendum_required"]:
            assert protocol["execution_addendum_sha256"] == ADDENDUM_SHA256
        assert protocol["records_audited"] == len(protocol["records"]) == 96
        assert Counter((r["variant"], r["case_id"]) for r in protocol["records"]) == expected
        assert (
            protocol["business_scores_or_release_gate_modified"] is False
            and protocol["selection_rule_modified"] is False
        )
        computed = {}
        for arm in ARMS:
            rows = [row for row in protocol["records"] if row["variant"] == arm]
            summary = {
                "tasks": 48,
                "gold_exchange_tasks": sum(row["audited"] for row in rows),
                **{key: sum(row[key] for row in rows) for key in COUNTS},
            }
            assert protocol["summary"][arm] == summary
            summary["rejected_return_attempts_without_scoped_yes"] = sum(
                attempt["raw_return_attempt"]
                and not attempt["scoped_yes_verified"]
                and not attempt["return_proposal_created"]
                and bool(attempt["backend_error"])
                for row in rows
                for attempt in row["attempts"]
            )
            computed[arm] = summary
        assert computed == values["protocol_summary"]
        for arm in ARMS:
            rows = [r for r in report["records"] if r["variant"] == arm]
            assert values["summary"][arm]["tasks"] == 48
            assert values["summary"][arm]["task_success"] == sum(r["task_success"] for r in rows)
            for key, value in values["summary"][arm].items():
                if key in report["summary"][arm]:
                    assert report["summary"][arm][key] == value
        index = {(r["variant"], r["case_id"]): bool(r["task_success"]) for r in report["records"]}
        paired = Counter({"improved": 0, "regressed": 0, "both_pass": 0, "both_fail": 0})
        for cid in cases:
            before, after = (index[arm, cid] for arm in ARMS)
            paired[
                "both_pass"
                if before and after
                else "both_fail"
                if not before and not after
                else "improved"
                if after
                else "regressed"
            ] += 1
        assert dict(paired) == values["paired"]
        journal = read(folder / "call-journal.json")
        assert report["call_journal_sha256"] == digest(folder / "call-journal.json")
        assert all(entry["status"] == "completed" for entry in journal)
        assert execution_authorization(root, report, plan, identity, journal) == authorized
        assert datetime.fromisoformat(plan["created_at_utc"]) <= datetime.fromisoformat(
            identity["sealed_at_utc"]
        )
        assert datetime.fromisoformat(identity["sealed_at_utc"]) < min(
            datetime.fromisoformat(entry["started_at_utc"]) for entry in journal
        )
        candidate = computed[ARMS[1]]
        return {
            "available": True,
            "experiment": "cuda_eval2",
            "scope": report["scope"],
            "summary": values["summary"],
            "paired": values["paired"],
            "family_success": values["family_success"],
            "failures": values["failures"],
            "training": proof,
            "execution_authorization": authorized,
            "figures": list(FIGURES),
            "protocol_audit": {
                "summary": computed,
                "records_audited": 96,
                "scope": protocol["scope"],
                "limitations": protocol["limitations"],
                "candidate_safety_passed": all(
                    candidate[key] == 0
                    for key in [
                        "return_proposals_without_scoped_yes",
                        "committed_returns_without_scoped_yes",
                        "final_returns_without_verified_commit",
                    ]
                ),
            },
            "metadata": {
                "test_cases": 48,
                "records": 96,
                "provider": "torch_cuda",
                "max_total_calls": 12,
                "analysis_plan_sha256": digest(plan_path),
                "artifact_identity_sha256": digest(identity_path),
                "source_scope": (
                    "Authored synthetic data; same policy/families, no independent human benchmark "
                    "or production reliability claim."
                ),
                "timing_scope": report["timing_scope"],
                "authorization_mode": authorized["authorization_mode"],
            },
        }
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return pending(root)
