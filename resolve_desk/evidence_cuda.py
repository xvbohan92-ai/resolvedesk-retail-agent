"""Fail-closed, read-only evidence for the separate matched CUDA comparison."""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ARMS = ("predecessor_fp16", "cloud1344_fp16")
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


def pending(root=None):
    result = {
        "available": False,
        "experiment": "cuda_eval",
        "reason": "The matched CUDA comparison is pending complete 96-record evidence, independent "
        "replay, training proof and verified figures.",
    }
    try:
        folder = Path(root) / "reports/cuda-eval"
        interruption = read(folder / "interruption.json")
        counts = interruption["last_verified_completed_by_arm"]
        assert interruption["status"] == "interrupted"
        assert interruption["analysis_plan_sha256"] == digest(folder / "analysis-plan.json")
        assert set(counts) == set(ARMS)
        assert all(type(counts[arm]) is int and 0 <= counts[arm] <= 48 for arm in ARMS)
        total = sum(counts.values())
        assert total == interruption["last_verified_completed_cases"] < 96
        result.update(
            status="interrupted",
            last_verified_completed_cases=total,
            reason=(
                f"Colab runtime disconnected; no active session remains. Last verified snapshot: "
                f"{total}/96 records (before {counts[ARMS[0]]}/48; after {counts[ARMS[1]]}/48). "
                "This is incomplete evidence, not a final before/after result. An unresolved "
                "model call is retained; automatic replay is disabled."
            ),
        )
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        pass
    return result


def validated_report(root):
    try:
        root = Path(root)
        folder, figures = root / "reports/cuda-eval", root / "docs/figures/cuda-eval"
        plan_path, report_path, identity_path = (
            folder / n for n in ["analysis-plan.json", "evaluation.json", "artifact-identity.json"]
        )
        plan, report, identity = (read(path) for path in [plan_path, report_path, identity_path])
        proof, verification, figure_check = (
            read(folder / n) for n in ["training-proof.json", "verification.json", "figure-verification.json"]
        )
        protocol, values = read(folder / "protocol-audit.json"), read(figures / "values.json")
        assert report["status"] == "complete" and report["provider"] == "torch_cuda"
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
        cases = {case["id"]: case for case in read(root / "training/cuda_eval/datasets/cases.json")}
        assert len(cases) == 48 and list(cases) == plan["case_ids"]
        assert (
            report["cases_sha256"]
            == plan["cases_sha256"]
            == digest(root / "training/cuda_eval/datasets/cases.json")
        )
        expected = Counter({(arm, cid): 1 for arm in ARMS for cid in cases})
        assert (
            len(report["records"]) == 96
            and Counter((r["variant"], r["case_id"]) for r in report["records"]) == expected
        )
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
        for checked in [verification, figure_check]:
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
            assert checked["source_sha256"] == digest(root / "training/cuda_eval/verify_results.py")
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
        assert values["analysis_source_sha256"] == digest(root / "training/cuda_eval/visualize.py")
        assert values["metrics_source_sha256"] == digest(root / "training/cuda_eval/metrics.py")
        expected_figures = {f"{name}.{extension}" for name in FIGURES for extension in ["png", "svg"]}
        assert set(values["figure_sha256"]) == expected_figures
        assert figure_check["figure_hashes_verified"] == values["figure_sha256"]
        for name, wanted in values["figure_sha256"].items():
            assert digest(figures / name) == wanted
        assert report["protocol_reference"] == {
            "path": "protocol-audit.json",
            "sha256": digest(folder / "protocol-audit.json"),
        }
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
        assert datetime.fromisoformat(plan["created_at_utc"]) <= datetime.fromisoformat(
            identity["sealed_at_utc"]
        )
        assert datetime.fromisoformat(identity["sealed_at_utc"]) < min(
            datetime.fromisoformat(entry["started_at_utc"]) for entry in journal
        )
        candidate = computed[ARMS[1]]
        return {
            "available": True,
            "experiment": "cuda_eval",
            "scope": report["scope"],
            "summary": values["summary"],
            "paired": values["paired"],
            "family_success": values["family_success"],
            "failures": values["failures"],
            "training": proof,
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
            },
        }
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return pending(root)
