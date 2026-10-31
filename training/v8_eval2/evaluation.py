"""CPU protocol for a new saved-weight-only V8 three-arm evaluation.

The old interrupted test is evidence, never an input queue or a pooled result.
Only explicit CLI commands write a new plan or identity. Importing loads no model.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT, sha
from training.cuda_eval.evaluation import value_hash
from training.cuda_eval.provider import MODEL_KWARGS, generation_settings
from training.cuda_torch.model import read_header
from training.v8.common import verify_frozen
from training.v8.evaluation import ARMS, MAX_CALLS

RUN_ID = "resolvedesk-v8-eval-002"
REPORTS = ROOT / "reports/v8-eval2"
DATA = ROOT / "training/v8_eval2/datasets"
BASE = ROOT / "data/models/qwen3-4b-mlx"
FINAL_SNAPSHOT = ROOT / "reports/v8/cloud-evidence/snapshots/20261029T201414293960Z"
FINAL_ARCHIVE_SHA = "e256d29c0bb85634cc9050fad652813c2eeee38a14eccc19d74dab301a707104"
BEFORE_SHA = "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053"
AFTER_SHA = "39dabb9b73ae6cb6230fc8a0bca9c914f6ba658c3c54a17f9775e3fcef60970f"
LEGACY_SHA = "041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08"
ORIGINAL_RESOURCE_SHA = "a495b3b036142993b6760ffc92913594e667b799c5d28d48602aeb1e85a35bb7"
PHYSICAL_PROOF_SHA = "51902c343ea1dd1a352a46a26dfa6f2c2f82fee81b527bb4f6c88a5385b79418"
WEIGHT_SOURCES = {
    "before": "reports/cuda-torch/probe-003/train-1344/adapter",
    "after": str((FINAL_SNAPSHOT / "reports/v8/cuda/train-1344/adapter").relative_to(ROOT)),
    "legacy": "data/finetuning/retail-qlora-v6_1/best",
}
ARTIFACT_SOURCES = {
    "training-report.json": FINAL_SNAPSHOT / "reports/v8/cuda/train-1344/report.json",
    "training-plan.json": FINAL_SNAPSHOT / "reports/v8/cuda/train-1344/plan.json",
    "training-proof.json": FINAL_SNAPSHOT / "reports/v8/training-proof.json",
    "v8-artifact-identity.json": FINAL_SNAPSHOT / "reports/v8/artifact-identity.json",
    "final-training-physical-verification.json": ROOT
    / "reports/v8/final-training-physical-verification.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def local_path(name):
    """Reject aliases on reads and writes, including an ancestor directory symlink."""
    path = Path(name)
    path = path if path.is_absolute() else ROOT / path
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("A protocol input/output must stay within this workspace")
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(ROOT)):
        raise ValueError("Protocol inputs/outputs cannot follow symlinks")
    return path


def weight_filename(arm):
    return "adapters.safetensors" if arm == "legacy" else "adapter_model.safetensors"


def weight_specs():
    expected = {"before": BEFORE_SHA, "after": AFTER_SHA, "legacy": LEGACY_SHA}
    result = {}
    for arm, folder in WEIGHT_SOURCES.items():
        source = local_path(folder)
        filename = weight_filename(arm)
        if sha(source / filename) != expected[arm]:
            raise ValueError("Saved adapter identity changed: " + arm)
        result[arm] = {
            "runtime_directory": f"data/v8-eval2/{arm}",
            "local_source_directory": folder,
            "filename": filename,
            "sha256": expected[arm],
            "config_sha256": sha(source / "adapter_config.json"),
        }
    return result


def source_names():
    """Inference code/data closure; mutable reporting and notebook transport stay separate."""
    from training.v8.seal import dependency_closure

    seeds = {
        "training/v8_eval2/evaluation.py",
        "training/v8_eval2/run.py",
        "training/v8_eval2/resource_guard.py",
        "training/v8_eval2/fixtures.py",
        "training/v8_eval2/prepare.py",
        "tests/test_v8_eval2_protocol.py",
        "tests/test_v8_eval2_resource.py",
        "tests/test_v8_eval2_data.py",
    }
    names = dependency_closure(seeds) | set(verify_frozen()["files"])
    names |= {str(p.relative_to(ROOT)) for p in DATA.glob("*.json")}
    summary = read(DATA / "data-summary.json")
    names.update(summary["quality"]["historical_datasets_sha256"])
    names |= {"training/v8/datasets/manifest.json", "training/base-model-spec.json"}
    return sorted(names)


def previous_evidence():
    imported = read(FINAL_SNAPSHOT / "import-verification.json")
    if imported["archive_sha256"] != FINAL_ARCHIVE_SHA or imported["binding_issues"]:
        raise ValueError("The actual final snapshot has not passed its byte-integrity import")
    receipt = read(FINAL_SNAPSHOT / "snapshot-receipt.json")
    if sha(FINAL_SNAPSHOT / "snapshot-receipt.json") != imported["member_sha256"]["snapshot-receipt.json"]:
        raise ValueError("The imported original snapshot receipt changed")
    for name in (
        "reports/v8/evaluation/evaluation-partial.json",
        "reports/v8/evaluation/call-journal.json",
        "reports/v8/validation/evaluation.json",
        "reports/v8/validation/call-journal.json",
        "reports/v8/validation/protocol-audit.json",
    ):
        if sha(FINAL_SNAPSHOT / name) != receipt["files"][name]:
            raise ValueError("Preserved old partial/validation bytes changed: " + name)
    old = FINAL_SNAPSHOT / "reports/v8/evaluation/evaluation-partial.json"
    journal = FINAL_SNAPSHOT / "reports/v8/evaluation/call-journal.json"
    old_report, entries = read(old), read(journal)
    if len(old_report["records"]) != 34 or old_report.get("status") == "complete":
        raise ValueError("Expected the preserved incomplete old run, not a claimed complete result")
    validation = FINAL_SNAPSHOT / "reports/v8/validation/evaluation.json"
    valid = read(validation)
    if valid.get("status") != "complete" or len(valid["records"]) != 48:
        raise ValueError("The existing48 validation must be complete; never issue it again")
    if valid["arms"] != [ARMS[-1]]:
        raise ValueError("Wrong previously completed validation arm")
    return {
        "archive_sha256": FINAL_ARCHIVE_SHA,
        "snapshot": str(FINAL_SNAPSHOT.relative_to(ROOT)),
        "old_partial_report_sha256": sha(old),
        "old_call_journal_sha256": sha(journal),
        "old_completed_records": 34,
        "old_issued_calls": len(entries),
        "old_unfinished_calls": sum(e["status"] != "completed" for e in entries),
        "old_pending_case": old_report.get("pending_case"),
        "old_completed_calls_outside_completed_task_rows": (
            sum(e["status"] == "completed" for e in entries)
            - sum(row["model_calls"] for row in old_report["records"])
        ),
        "old_pending_task_apply_or_finalization_not_reconstructed": True,
        "old_run_resume_allowed": False,
        "old_outputs_pooled": False,
        "old_calls_replayed": False,
        "existing_validation": {
            "records": 48,
            "evaluation_sha256": sha(validation),
            "journal_sha256": sha(validation.with_name("call-journal.json")),
            "protocol_sha256": sha(validation.with_name("protocol-audit.json")),
            "rerun_allowed": False,
            "quality_scores_reused_in_new_comparison": False,
        },
    }


def cpu_preflight():
    from training.v8_eval2.fixtures import load_cases
    from training.v8_eval2.prepare import denominators, quality_audit

    verify_frozen()
    cases = load_cases("test")
    if len(cases) != 48 or len({c["id"] for c in cases}) != 48:
        raise ValueError("Exactly48 independent new cases are required")
    if not all(c["id"].startswith("v8-eval2-") for c in cases):
        raise ValueError("This run must not reuse the interrupted old task queue")
    computed = {
        "cases": cases,
        "case_order": [c["id"] for c in cases],
        "cases_sha256": sha(DATA / "cases.json"),
        "canonical_cases_sha256": value_hash(cases),
        "public_sha256": sha(DATA / "public.json"),
        "denominators": denominators(cases),
        "quality": quality_audit(cases),
    }
    summary, reference = [read(DATA / name) for name in ("data-summary.json", "reference-audit.json")]
    if (
        summary["cases_sha256"] != computed["cases_sha256"]
        or summary["public_sha256"] != computed["public_sha256"]
        or summary["reference_audit_sha256"] != sha(DATA / "reference-audit.json")
        or reference["cases_sha256"] != computed["cases_sha256"]
        or reference["public_sha256"] != computed["public_sha256"]
        or summary["denominators"] != computed["denominators"]
        or summary["quality"] != computed["quality"]
    ):
        raise ValueError("Fresh data/reference receipt is stale or belongs to different cases")
    for receipt in (summary, reference):
        if receipt["frozen_v8_manifest_sha256"] != sha(ROOT / "training/v8/datasets/manifest.json"):
            raise ValueError("Data reference used a different frozen V8 manifest")
        for name, digest in receipt["source_inputs_sha256"].items():
            if sha(local_path(name)) != digest:
                raise ValueError("Data reference used an older source: " + name)
    expected = {(flow, case["id"]) for flow in ("v7", "v8") for case in cases}
    records = reference["records"]
    if (
        reference["tasks"] != 48
        or reference["reference_trajectories"] != 96
        or len(records) != 96
        or {(r["flow"], r["case_id"]) for r in records} != expected
        or any(r["task_success"] is not True or not 0 < r["model_calls"] <= 12 for r in records)
    ):
        raise ValueError("Both frozen flows require complete bounded CPU reference trajectories")
    if summary["cpu_reference_by_flow"] != reference["by_flow"]:
        raise ValueError("Data summary does not preserve the actual per-flow reference results")
    for flow in ("v7", "v8"):
        rows = [r for r in records if r["flow"] == flow]
        metrics = reference["by_flow"][flow]
        if (
            metrics["tasks"] != 48
            or metrics["all_business_references_pass"] is not True
            or metrics["max_reference_calls"] != max(r["model_calls"] for r in rows)
            or metrics["reference_total_calls"] != sum(r["model_calls"] for r in rows)
        ):
            raise ValueError("CPU reference call totals disagree with actual trajectories")
    computed["data_summary_sha256"] = sha(DATA / "data-summary.json")
    computed["reference_audit_sha256"] = sha(DATA / "reference-audit.json")
    computed["reference_by_flow"] = reference["by_flow"]
    return computed


def create_plan():
    """Seal only after independent CPU/data/source review; no model framework is imported."""
    if any(
        (REPORTS / n).exists()
        for n in ("analysis-plan.json", "artifact-identity.json", "evaluation-launch.json")
    ):
        raise FileExistsError("Never replace a sealed or issued evaluation-only run")
    preflight = cpu_preflight()
    old = previous_evidence()
    proof_path = ARTIFACT_SOURCES["final-training-physical-verification.json"]
    if sha(proof_path) != PHYSICAL_PROOF_SHA:
        raise ValueError("The independently verified saved final weight proof changed")
    proof = read(proof_path)
    if proof["training"]["final_adapter_sha256"] != AFTER_SHA:
        raise ValueError("No independently verified saved39dabb candidate")
    base_config = read(BASE / "config.json")
    hashes = {name: sha(local_path(name)) for name in source_names()}
    artifacts = {}
    for name, source in ARTIFACT_SOURCES.items():
        artifacts[name] = {"runtime_path": f"reports/v8-eval2/artifacts/{name}", "sha256": sha(source)}
    original_resource = FINAL_SNAPSHOT / "reports/v8/resource-authorization.json"
    resource = read(original_resource)
    if sha(original_resource) != ORIGINAL_RESOURCE_SHA:
        raise ValueError("Original round budget/evidence cannot be reset")
    plan = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "frozen_pre_inference",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_inference_started": False,
        "scope": (
            "New48 authored synthetic cases, three matched arms using already saved weights; no retraining."
        ),
        "arms": list(ARMS),
        "case_order": preflight["case_order"],
        "dataset": {k: v for k, v in preflight.items() if k != "cases"},
        "source_inputs_sha256": hashes,
        "frozen_v8_manifest_sha256": sha(ROOT / "training/v8/datasets/manifest.json"),
        "weights": weight_specs(),
        "artifact_bindings": artifacts,
        "old_run_provenance": old,
        "seed": read(ROOT / "training/v8/config.json")["seed"],
        "base_model_spec_sha256": sha(ROOT / "training/base-model-spec.json"),
        "base_config_sha256": sha(BASE / "config.json"),
        "generation": generation_settings(base_config),
        "model_kwargs": MODEL_KWARGS,
        "chat_template": {"enable_thinking": False, "truncation": False, "add_generation_prompt": True},
        "total_calls_per_task": MAX_CALLS,
        "maximum_total_issued_calls": 48 * len(ARMS) * MAX_CALLS,
        "first_output_scoring": "Frozen scorer reads first raw output before any enum/evidence repair.",
        "controller_and_scoring": "Exact frozen V7/V8 Store, run_case, summarize, scoped-consent audit.",
        "additional_training_allowed": False,
        "smoke_gradient_updates_allowed": False,
        "validation_inference_allowed": False,
        "checkpoint_selection_performed": False,
        "report_only": True,
        "deployment_or_default_provider_change_allowed": False,
        "resume_allowed": False,
        "ambiguous_call_replay_allowed": False,
        "resource_policy": {
            "new_explicit_human_approval_required": True,
            "maximum_new_window_compute_units": 3,
            "maximum_window_hours": 3,
            "transfer_release_buffer_compute_units": 0.25,
            "original_resource_receipt_sha256": ORIGINAL_RESOURCE_SHA,
            "original_round_initial_balance_units": resource["starting_balance_units"],
            "original_round_cap_compute_units": resource["cap_compute_units"],
            "original_operator_stop_balance_units": resource["operator_stop_balance_units"],
            "purchases_allowed": False,
            "renewals_allowed": False,
            "google_drive_mount_allowed": False,
        },
        "runtime": {
            "official_colab_image": "2026.07",
            "python_minor": "3.12",
            "torch": "2.11.0+cu128",
            "gpu": "T4",
        },
        "latency_scope": (
            "One model load; serial fixed arm/case order. No additional model warmup. "
            "Hardware timing is descriptive."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    with (REPORTS / "analysis-plan.json").open("x") as stream:
        stream.write(json.dumps(plan, indent=2) + "\n")
    return plan


def verify_plan():
    plan = read(REPORTS / "analysis-plan.json")
    if plan["run_id"] != RUN_ID or plan["arms"] != list(ARMS):
        raise ValueError("Different run or arm identity")
    for name, digest in plan["source_inputs_sha256"].items():
        if sha(local_path(name)) != digest:
            raise ValueError("Sealed inference source/data changed: " + name)
    if sha(ROOT / "training/v8/datasets/manifest.json") != plan["frozen_v8_manifest_sha256"]:
        raise ValueError("Original218 manifest changed")
    verify_frozen()
    if plan["weights"]["before"]["sha256"] != BEFORE_SHA or plan["weights"]["after"]["sha256"] != AFTER_SHA:
        raise ValueError("Only the already verified saved weights are permitted")
    if plan["total_calls_per_task"] != 12 or plan["maximum_total_issued_calls"] != 1728:
        raise ValueError("The original total-call budget changed")
    for forbidden in (
        "additional_training_allowed",
        "smoke_gradient_updates_allowed",
        "validation_inference_allowed",
        "checkpoint_selection_performed",
        "deployment_or_default_provider_change_allowed",
        "resume_allowed",
        "ambiguous_call_replay_allowed",
    ):
        if plan[forbidden] is not False:
            raise ValueError("This run is evaluation only without replay or selection")
    if plan.get("report_only") is not True:
        raise ValueError("This comparison is report only and cannot change the deployed provider")
    return plan


def seal_identity():
    from training.v8.verify_training import tensor_audit

    plan = verify_plan()
    path = REPORTS / "artifact-identity.json"
    if path.exists() or (REPORTS / "evaluation-launch.json").exists():
        raise FileExistsError("Never reseal an identity or an issued run")
    before = local_path(WEIGHT_SOURCES["before"]) / "adapter_model.safetensors"
    initial = FINAL_SNAPSHOT / "reports/v8/cuda/train-1344/initial-peft.safetensors"
    after = local_path(WEIGHT_SOURCES["after"]) / "adapter_model.safetensors"
    numeric = tensor_audit(before, initial, after)
    for spec in plan["weights"].values():
        source = local_path(spec["local_source_directory"])
        if (
            sha(source / spec["filename"]) != spec["sha256"]
            or sha(source / "adapter_config.json") != spec["config_sha256"]
        ):
            raise ValueError("Saved physical adapter changed before sealing")
    identity = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
        "before_adapter_sha256": BEFORE_SHA,
        "after_adapter_sha256": AFTER_SHA,
        "weights": plan["weights"],
        "artifact_bindings": plan["artifact_bindings"],
        "old_run_provenance": plan["old_run_provenance"],
        "independent_cpu_tensor_audit": numeric,
        "physical_proof_sha256": PHYSICAL_PROOF_SHA,
        "checkpoint_selection_performed": False,
        "no_training_performed_this_run": True,
    }
    with path.open("x") as stream:
        stream.write(json.dumps(identity, indent=2) + "\n")
    return identity


def verify_identity(*, physical=True):
    plan = verify_plan()
    identity = read(REPORTS / "artifact-identity.json")
    if identity["run_id"] != RUN_ID or identity["analysis_plan_sha256"] != sha(
        REPORTS / "analysis-plan.json"
    ):
        raise ValueError("The saved-weight evaluation identity changed")
    if identity["weights"] != plan["weights"] or identity["artifact_bindings"] != plan["artifact_bindings"]:
        raise ValueError("Model/proof identities differ from the sealed plan")
    if identity["old_run_provenance"] != plan["old_run_provenance"]:
        raise ValueError("Old run archival boundaries changed")
    numeric = identity["independent_cpu_tensor_audit"]
    if (
        identity.get("before_adapter_sha256") != BEFORE_SHA
        or identity.get("after_adapter_sha256") != AFTER_SHA
        or identity.get("physical_proof_sha256") != PHYSICAL_PROOF_SHA
        or identity.get("no_training_performed_this_run") is not True
        or identity.get("checkpoint_selection_performed") is not False
        or numeric.get("tensors_checked") != 112
        or numeric.get("parameters_checked") != 3_670_016
        or numeric.get("exact_prior_initialization") is not True
        or numeric.get("all_values_finite") is not True
        or numeric.get("cpu_only") is not True
    ):
        raise ValueError("The fixed saved-weight numeric proof/identity is incomplete")
    created = datetime.fromisoformat(plan["created_at_utc"])
    sealed = datetime.fromisoformat(identity["sealed_at_utc"])
    if created.tzinfo is None or sealed.tzinfo is None or not created < sealed < datetime.now(UTC):
        raise ValueError("The result-blind plan must precede identity sealing and inference")
    for spec in plan["artifact_bindings"].values():
        if sha(local_path(spec["runtime_path"])) != spec["sha256"]:
            raise ValueError("Staged existing training/physical proof changed")
    if physical:
        for spec in plan["weights"].values():
            folder = local_path(spec["runtime_directory"])
            if (
                sha(folder / spec["filename"]) != spec["sha256"]
                or sha(folder / "adapter_config.json") != spec["config_sha256"]
            ):
                raise ValueError("Runtime saved weights differ from the sealed identity")
        header = read_header(
            local_path(plan["weights"]["after"]["runtime_directory"]) / "adapter_model.safetensors"
        )
        if len(header) != 112 or any(t["dtype"] != "F32" for t in header.values()):
            raise ValueError("The saved candidate must remain112 F32 LoRA tensors")
    return plan, identity
