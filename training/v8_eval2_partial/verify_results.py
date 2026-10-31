"""CPU-only evidence audit for the separately approved partial40 analysis.

All136 complete records are replayed before selecting120 paired rows. The final
unknown request is checked without a response, apply, confirmation or model call.
The original144-task gate permanently remains incomplete.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT, sha
from training.cuda_eval.evaluation import value_hash
from training.cuda_eval.provider import MODEL_KWARGS, generation_settings
from training.v8_eval2 import verify_results as original

REPORTS = ROOT / "reports/v8-eval2-partial40"
ORIGINAL_REPORTS = ROOT / "reports/v8-eval2"
SNAPSHOT = ORIGINAL_REPORTS / "cloud-evidence/snapshots/20261030T150701917677Z"
RUN_ID = "resolvedesk-v8-eval-002"
ANALYSIS_ID = "resolvedesk-v8-eval-002-partial40"
FINAL_ARCHIVE_SHA = "381804949a84f3284b60074f6b7dd791c782ed8d90b2fec08192c5e59a5db183"
PLAN_SHA = "544521fbf83c63c1658af6605331f4e7d7b334523ea60fff7177d5cf9859ddac"
IDENTITY_SHA = "5e8ea77ec9076b31e7be1b9cf0e5e19c983fa5d87200a506150b08bd72b5b36b"
GRADE_FIELDS = (
    "proposal_correct",
    "task_success",
    "confirmation_preserved",
    "raw_task_success",
    "required_clarification",
    "clarification_obtained",
    "identity_and_order_checked",
    "catalog_inspected",
)


def load(path):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(), parse_constant=reject)


def require_sealed_scope():
    """Fail before tokenizer, tensor or business imports if the new scope is only draft."""
    plan = load(REPORTS / "analysis-plan.json")
    if plan.get("analysis_id") != ANALYSIS_ID or plan.get("run_id") != RUN_ID:
        raise ValueError("Different partial-analysis identity")
    if plan.get("sealed") is not True or not str(plan.get("status", "")).startswith("frozen"):
        raise PermissionError("The separately approved partial scope must be sealed before any replay")
    if plan.get("human_scope_approval", {}).get("explicit") is not True:
        raise PermissionError("No explicit human approval of the post-hoc40-case scope")
    approval = plan["human_scope_approval"]
    if not approval.get("text", "").strip():
        raise PermissionError("The actual separately approved partial scope must be recorded")
    if sha(original.regular(ROOT, approval["receipt_path"])) != approval["receipt_sha256"]:
        raise ValueError("The partial-scope human authorization receipt changed")
    if not plan.get("source_inputs_sha256"):
        raise ValueError("The new analysis source closure is not bound")
    if plan.get("pending_source_paths"):
        raise PermissionError("All required new analysis sources must be reviewed and sealed first")
    if plan.get("source_seal_pending") is not False:
        raise PermissionError("The partial analysis source seal is still pending")
    if (
        plan.get("original144_gate_complete") is not False
        or plan.get("mark_original144_complete_allowed") is not False
    ):
        raise ValueError("The new analysis cannot complete or replace the original144 gate")
    if (
        plan.get("post_hoc_partial_analysis") is not True
        or plan.get("quality_scores_computed_before_seal") is not False
    ):
        raise ValueError("This is a separately sealed post-hoc scope without pre-seal scoring")
    if not set(plan.get("required_analysis_source_paths", [])) <= set(plan["source_inputs_sha256"]):
        raise ValueError("Required partial-analysis helpers are absent from the source closure")
    for name, digest in plan["source_inputs_sha256"].items():
        if sha(original.regular(ROOT, name)) != digest:
            raise ValueError("New analysis source changed after sealing: " + name)
    selection = load(REPORTS / "selection.json")
    if plan.get("selection_sha256") != sha(REPORTS / "selection.json"):
        raise ValueError("The fixed paired-prefix selection changed")
    if plan.get("selection_audit_sha256") != sha(REPORTS / "selection-audit.json"):
        raise ValueError("The metadata-only selection audit changed")
    experiment = plan["original_experiment"]
    expected_paths = {
        "analysis_plan": SNAPSHOT / "reports/v8-eval2/analysis-plan.json",
        "artifact_identity": SNAPSHOT / "reports/v8-eval2/artifact-identity.json",
        "raw_report": SNAPSHOT / "reports/v8-eval2/evaluation-partial.json",
        "call_journal": SNAPSHOT / "reports/v8-eval2/call-journal.json",
        "cases": SNAPSHOT / "training/v8_eval2/datasets/cases.json",
        "snapshot_receipt": SNAPSHOT / "snapshot-receipt.json",
        "import_verification": SNAPSHOT / "import-verification.json",
        "resource_receipt": SNAPSHOT / "reports/v8-eval2/resource-authorization.json",
        "deadline_marker": SNAPSHOT / "reports/v8-eval2/deadline-stop.json",
    }
    for label, expected_path in expected_paths.items():
        path = original.regular(ROOT, experiment[label + "_path"])
        if path != expected_path or sha(path) != experiment[label + "_sha256"]:
            raise ValueError("The approved original immutable input changed: " + label)
    if experiment.get("expected_records") != 144 or experiment.get("raw_records") != 136:
        raise ValueError("The new analysis cannot replace the original144-task incomplete gate")
    if experiment.get("original144_gate_complete") is not False:
        raise ValueError("The original144 gate must remain incomplete")
    archive = original.regular(ROOT.parent, experiment["final_archive_path"])
    if sha(archive) != experiment["final_archive_sha256"] or sha(archive) != FINAL_ARCHIVE_SHA:
        raise ValueError("The original final archive changed")
    return plan, selection


def read_snapshot():
    receipt = load(SNAPSHOT / "snapshot-receipt.json")
    imported = load(SNAPSHOT / "import-verification.json")
    assert imported["archive_sha256"] == FINAL_ARCHIVE_SHA and not imported["binding_issues"]
    assert imported["member_sha256"]["snapshot-receipt.json"] == sha(SNAPSHOT / "snapshot-receipt.json")
    assert receipt["run_id"] == RUN_ID and receipt["progress"]["process_return_code"] == 124
    assert not receipt["complete_artifacts_present"]
    files = {}
    for name, digest in receipt["files"].items():
        path = original.regular(SNAPSHOT, name)
        assert sha(path) == digest, name
        files[str(path.relative_to(ROOT))] = digest
    report = load(SNAPSHOT / "reports/v8-eval2/evaluation-partial.json")
    journal = load(SNAPSHOT / "reports/v8-eval2/call-journal.json")
    assert not (SNAPSHOT / "reports/v8-eval2/evaluation.json").exists()
    return receipt, report, journal, files


def require_partial_record_prefix(report, journal, cases, old_plan):
    assert report["run_id"] == RUN_ID and report["status"] == "partial"
    assert report["round"] == "v8-eval2" and report["split"] == "test"
    assert report["resume_count"] == 0 and report["checkpoint_selection_performed"] is False
    assert report["additional_training_performed"] is False and report["validation_model_calls"] == 0
    assert report["max_total_calls_per_task"] == 12 and report["model_loads_this_process"] == 1
    assert report["report_only"] is True and report["deployment_or_default_provider_change_allowed"] is False
    assert len(cases) == 48 and [c["id"] for c in cases] == old_plan["case_order"]
    assert report["cases_sha256"] == old_plan["dataset"]["canonical_cases_sha256"] == value_hash(cases)
    expected = [(arm, case["id"]) for arm in original.ARMS for case in cases]
    rows = report["records"]
    assert len(rows) == 136 and [(r["variant"], r["case_id"]) for r in rows] == expected[:136]
    assert [sum(r["variant"] == arm for r in rows) for arm in original.ARMS] == [48, 48, 40]
    assert all(0 < r["model_calls"] == len(r["trace"]) <= 12 for r in rows)
    assert sum(r["model_calls"] for r in rows) == 765
    assert len(journal) == 766 and all(e["status"] == "completed" for e in journal[:765])
    assert journal[-1]["status"] == "started"
    assert (report["pending_case"]["arm"], report["pending_case"]["case_id"]) == expected[136]
    assert not report.get("summary") and not report.get("completed_at_utc")
    return rows


def verify_completed_journal(report, journal, prompts, identity, resource, bindings):
    """Exactly765 known responses; the started tail is not passed to the full-run verifier."""
    expected = [(row, index, obs) for row in report["records"] for index, obs in enumerate(row["trace"], 1)]
    assert len(expected) == 765 and len(journal) == 766
    last_completed = original.utc(report["started_at_utc"])
    for entry, (row, index, obs) in zip(journal[:765], expected, strict=True):
        assert entry["status"] == "completed"
        assert (entry["variant"], entry["case_id"], entry["call_index"]) == (
            row["variant"],
            row["case_id"],
            index,
        )
        assert entry["raw"] == obs["raw"] and entry["timing"] == obs["timing"]
        assert all(entry[k] == value for k, value in bindings.items())
        assert entry["request_messages"] == prompts[row["variant"], row["case_id"], index]
        assert entry["request_sha256"] == value_hash(entry["request_messages"])
        started, completed = original.utc(entry["started_at_utc"]), original.utc(entry["completed_at_utc"])
        assert original.utc(identity["sealed_at_utc"]) < started
        assert (
            original.utc(resource["allocated_at_utc"])
            <= started
            <= completed
            < original.utc(resource["process_deadline_utc"])
        )
        assert last_completed <= started
        last_completed = completed
    return last_completed


def initial_prompt(case, variant):
    old = variant == original.ARMS[0]
    Store, messages = (
        (original.OldStore, original.old_messages) if old else (original.NewStore, original.new_messages)
    )
    with tempfile.TemporaryDirectory(prefix="partial40-unknown-input-only-") as directory:
        state = Store(Path(directory) / "initial.sqlite").create(case)
        return messages(state)


def verify_unknown_request(entry, pending_case, case, identity, resource, bindings, last_completed):
    """Audit the retained first request only. No raw output, token output or Store.apply exists."""
    assert entry["status"] == "started" and entry["call_index"] == 1
    assert not any(name in entry for name in ("raw", "timing", "completed_at_utc"))
    assert (entry["variant"], entry["case_id"]) == (pending_case["arm"], pending_case["case_id"])
    assert entry["case_id"] == case["id"] and all(entry[k] == v for k, v in bindings.items())
    assert entry["request_sha256"] == value_hash(entry["request_messages"])
    started = original.utc(entry["started_at_utc"])
    assert original.utc(identity["sealed_at_utc"]) < started
    assert last_completed <= started < original.utc(resource["process_deadline_utc"])
    assert original.utc(resource["allocated_at_utc"]) <= started
    prompt = initial_prompt(case, entry["variant"])
    assert prompt == entry["request_messages"], "Unknown-call initial request does not match the frozen world"
    return {
        "variant": entry["variant"],
        "case_id": entry["case_id"],
        "call_index": 1,
        "original_entry_sha256": value_hash(entry),
        "request_sha256": entry["request_sha256"],
        "started_at_utc": entry["started_at_utc"],
        "status_remains": "started_response_unknown",
        "request_matches_frozen_initial_prompt": True,
        "actual_input_token_ids_retained": False,
        "actual_token_replay_performed": False,
        "actual_response_and_business_effects_unverifiable": True,
        "output_or_response_generated_or_reconstructed": False,
        "business_action_or_confirmation_applied": False,
        "model_calls": 0,
    }


def verify_record_audit(index, row, case, identity, generation, tokenizer, eos, bindings, selected_keys):
    prompts = original.verify_record(row, case, identity, generation, tokenizer, eos, bindings)
    consent = original.audit_consent(row, case)
    return {
        "source_record_index": index,
        "source_record_sha256": value_hash(row),
        "variant": row["variant"],
        "case_id": row["case_id"],
        "selected_for_paired40": (row["variant"], row["case_id"]) in selected_keys,
        "derived_scoring": {field: copy.deepcopy(row[field]) for field in GRADE_FIELDS},
        "derived_scoring_provenance": (
            "Frozen verify_record independently re-executed and compared every graded field; "
            "these equal values are preserved after that assertion, not a new grader."
        ),
        "business_events_and_confirmation_verified": True,
        "input_output_token_replay_verified": True,
        "first_raw_before_repair_preserved": True,
        "consent_audit": consent,
    }, prompts


def verify_selection(selection, rows, cases):
    selected_ids = [case["id"] for case in cases[:40]]
    assert selection["analysis_id"] == ANALYSIS_ID and selection["run_id"] == RUN_ID
    assert selection["case_ids"] == selected_ids and len(set(selected_ids)) == 40
    assert len(selection["cases"]) == 40
    for index, (ref, case) in enumerate(zip(selection["cases"], cases[:40], strict=True)):
        assert ref["case_id"] == case["id"] and ref["original_case_index"] == index
        assert ref["canonical_sha256"] == value_hash(case)
    selected_keys = {(arm, cid) for arm in original.ARMS for cid in selected_ids}
    indices = [i for i, row in enumerate(rows) if (row["variant"], row["case_id"]) in selected_keys]
    assert len(indices) == len(selection["records"]) == 120
    for ref, index in zip(selection["records"], indices, strict=True):
        row = rows[index]
        assert ref["original_record_index"] == index
        assert (ref["variant"], ref["case_id"]) == (row["variant"], row["case_id"])
        assert ref["canonical_sha256"] == value_hash(row)
    selected_rows = [rows[i] for i in indices]
    assert [(r["variant"], r["case_id"]) for r in selected_rows] == [
        (arm, cid) for arm in original.ARMS for cid in selected_ids
    ]
    assert selection["canonical_selected_cases_sha256"] == value_hash(cases[:40])
    assert selection["canonical_selected_records_sha256"] == value_hash(selected_rows)
    excluded = [i for i in range(len(rows)) if i not in indices]
    assert len(excluded) == len(selection["excluded_completed_records"]) == 16
    for ref, index in zip(selection["excluded_completed_records"], excluded, strict=True):
        row = rows[index]
        assert ref["original_record_index"] == index and ref["canonical_sha256"] == value_hash(row)
        assert (ref["variant"], ref["case_id"]) == (row["variant"], row["case_id"])
    return selected_keys


def audit_records(rows, cases, identity, generation, tokenizer, eos, bindings, selected_keys):
    """Audit all rows, including16 excluded tails; selection never suppresses a failed audit."""
    audits, prompts = [], {}
    by_id = {case["id"]: case for case in cases}
    for index, row in enumerate(rows):
        audit, actual_prompts = verify_record_audit(
            index, row, by_id[row["case_id"]], identity, generation, tokenizer, eos, bindings, selected_keys
        )
        audits.append(audit)
        for number, prompt in enumerate(actual_prompts, 1):
            prompts[row["variant"], row["case_id"], number] = prompt
    assert len(audits) == 136 and sum(a["selected_for_paired40"] for a in audits) == 120
    return audits, prompts


def verify_launch_and_load(report, old_plan, identity, resource, bindings, spec):
    loaded = load(SNAPSHOT / "reports/v8-eval2/evaluation-model-load.json")
    assert loaded == report["model_load_receipt"] and loaded["run_id"] == RUN_ID
    assert report["model_load_receipt_sha256"] == sha(
        SNAPSHOT / "reports/v8-eval2/evaluation-model-load.json"
    )
    assert (
        loaded["model_load_count"] == 1
        and loaded["resource_receipt_sha256"] == bindings["resource_receipt_sha256"]
    )
    assert (
        original.utc(identity["sealed_at_utc"])
        < original.utc(loaded["loaded_at_utc"])
        <= original.utc(report["started_at_utc"])
    )
    loader = loaded["loader_audit"]
    assert loader["initializer_sha256"] == original.LEGACY_SHA and loader["exact_tensor_equality"] is True
    assert loader["source_tensor_count"] == 112 and loader["source_parameter_count"] == 3_670_016
    assert loader["base_repo"] == spec["repo"] and loader["base_revision"] == spec["revision"]
    assert loader["base_model_sha256"] == next(
        f["sha256"] for f in spec["files"] if f["path"].endswith(".safetensors")
    )
    assert loader["adapter_master_dtype"] == "float32" and loader["base_storage_dtype"] == "float16"
    assert loader["not_qlora_or_nf4"] is True and loader["mlx_compute_bitwise_equivalence_claimed"] is False
    assert loaded["loader_source_sha256"] == sha(ROOT / "training/cuda_torch/model.py")
    for arm in original.ARMS:
        audit = report["model_audits"][arm]
        assert audit["source_sha256"] == (
            original.AFTER_SHA if arm == original.ARMS[-1] else original.BEFORE_SHA
        )
        assert audit["exact_tensor_equality"] is True and audit["source_tensor_count"] == 112
        assert audit["source_parameter_count"] == 3_670_016 and audit["base_reloaded"] is False
        assert audit["only_lora_tensors_overwritten"] is True
    launch = load(SNAPSHOT / "reports/v8-eval2/evaluation-launch.json")
    assert all(launch[k] == value for k, value in bindings.items()) and launch["no_automatic_retry"] is True
    assert launch["command"] == "python -u -B -m training.v8_eval2.run run --execute"
    assert original.utc(resource["allocated_at_utc"]) <= original.utc(launch["started_at_utc"])
    assert (
        old_plan["report_only"] is True and old_plan["deployment_or_default_provider_change_allowed"] is False
    )
    return {"model_loads": 1, "three_exact_saved_adapter_installations": True, "new_training": False}


def verify(*, tokenizer_root=None):
    if not __debug__:
        raise RuntimeError("Evidence verification requires Python assertions")
    if any(name == "torch" or name.startswith("mlx") for name in sys.modules):
        raise RuntimeError("Use a fresh CPU-only process without Torch/MLX imports")
    analysis_plan, selection = require_sealed_scope()
    receipt, report, journal, files = read_snapshot()
    old_plan = load(SNAPSHOT / "reports/v8-eval2/analysis-plan.json")
    identity = load(SNAPSHOT / "reports/v8-eval2/artifact-identity.json")
    assert sha(SNAPSHOT / "reports/v8-eval2/analysis-plan.json") == PLAN_SHA
    assert sha(SNAPSHOT / "reports/v8-eval2/artifact-identity.json") == IDENTITY_SHA
    assert all(
        analysis_plan["source_inputs_sha256"].get(name) == value
        for name, value in old_plan["source_inputs_sha256"].items()
    )
    from training.v8_eval2.fixtures import load_cases
    from training.v8_eval2.resource_guard import verify_resource_receipt

    cases = load_cases("test")
    rows = require_partial_record_prefix(report, journal, cases, old_plan)
    selected_keys = verify_selection(selection, rows, cases)
    artifacts = original.verify_protocol_inputs(old_plan, identity, report, cases)
    history = original.verify_historical_boundary(old_plan, cases)
    physical = original.verify_saved_adapters(old_plan, identity, artifacts)
    # The old resource window is expired. This is only a CPU evidence audit, not new execution.
    resource = verify_resource_receipt(active=False)
    assert resource == load(SNAPSHOT / "reports/v8-eval2/resource-authorization.json")
    bindings = {
        "run_id": RUN_ID,
        "analysis_plan_sha256": PLAN_SHA,
        "artifact_identity_sha256": IDENTITY_SHA,
        "resource_receipt_sha256": sha(SNAPSHOT / "reports/v8-eval2/resource-authorization.json"),
    }
    assert all(report[k] == value for k, value in bindings.items())
    marker = load(SNAPSHOT / "reports/v8-eval2/deadline-stop.json")
    assert marker["status"] == "deadline_exhausted" and marker["run_id"] == RUN_ID
    assert marker["resource_receipt_sha256"] == bindings["resource_receipt_sha256"]
    assert marker["automatic_retry_allowed"] is False and marker["billing_stopped"] is False
    assert original.utc(marker["stopped_at_utc"]) >= original.utc(resource["process_deadline_utc"])
    base = Path(tokenizer_root or ROOT / "data/models/qwen3-4b-mlx")
    config, spec = load(base / "config.json"), load(ROOT / "training/base-model-spec.json")
    assert sha(base / "config.json") == old_plan["base_config_sha256"]
    assert old_plan["generation"] == generation_settings(config) and old_plan["model_kwargs"] == MODEL_KWARGS
    for item in spec["files"]:
        if not item["path"].endswith(".safetensors"):
            assert sha(base / item["path"]) == item["sha256"]
    load_audit = verify_launch_and_load(report, old_plan, identity, resource, bindings, spec)
    os.environ["USE_TORCH"], os.environ["USE_TF"], os.environ["HF_HUB_OFFLINE"] = "0", "0", "1"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    eos = old_plan["generation"]["eos_token_id"]
    eos = [eos] if isinstance(eos, int) else eos
    audits, prompts = audit_records(
        rows, cases, identity, old_plan["generation"], tokenizer, eos, bindings, selected_keys
    )
    last_completed = verify_completed_journal(report, journal, prompts, identity, resource, bindings)
    pending = report["pending_case"]
    by_id = {case["id"]: case for case in cases}
    unknown = verify_unknown_request(
        journal[-1], pending, by_id[pending["case_id"]], identity, resource, bindings, last_completed
    )
    declared = selection["candidate_ambiguous_task"]
    assert (declared["variant"], declared["case_id"], declared["journal_index"], declared["call_index"]) == (
        journal[-1]["variant"],
        journal[-1]["case_id"],
        765,
        1,
    )
    assert (
        declared["request_sha256"] == journal[-1]["request_sha256"]
        and declared["response_available"] is False
    )
    assert declared["started_at_utc"] == journal[-1]["started_at_utc"]
    assert selection["candidate_unstarted_case_ids"] == old_plan["case_order"][41:]
    selected_audits = [audit for audit in audits if audit["selected_for_paired40"]]
    protocols = [audit["consent_audit"] for audit in audits]
    selected_protocols = [audit["consent_audit"] for audit in selected_audits]
    assert not any(name == "torch" or name.startswith("mlx") for name in sys.modules)
    names = {
        "reports/v8-eval2-partial40/analysis-plan.json",
        "reports/v8-eval2-partial40/selection.json",
        "reports/v8-eval2-partial40/selection-audit.json",
        "training/v8_eval2_partial/verify_results.py",
        "training/v8_eval2/verify_results.py",
        "training/base-model-spec.json",
        "reports/v8-eval2/analysis-plan.json",
        "reports/v8-eval2/artifact-identity.json",
        "reports/v8-eval2/resource-authorization.json",
        str((SNAPSHOT / "snapshot-receipt.json").relative_to(ROOT)),
        str((SNAPSHOT / "import-verification.json").relative_to(ROOT)),
        *analysis_plan["source_inputs_sha256"],
        *old_plan["source_inputs_sha256"],
        *(specification["runtime_path"] for specification in old_plan["artifact_bindings"].values()),
    }
    for specification in old_plan["weights"].values():
        names.add(specification["runtime_directory"] + "/adapter_config.json")
        names.add(specification["runtime_directory"] + "/" + specification["filename"])
    names.add(analysis_plan["human_scope_approval"]["receipt_path"])
    files.update({name: sha(original.regular(ROOT, name)) for name in names})
    result = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "run_id": RUN_ID,
        "status": "verified_partial_evidence",
        "verified_at_utc": datetime.now(UTC).isoformat(),
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
        "full_result_verified": False,
        "original144_gate_complete": False,
        "partial_evidence_verified": True,
        "original_gate_status": "terminal_incomplete_deadline",
        "original_run_modified": False,
        "replay": {
            "known_business_actions_events_confirmation": True,
            "known_actual_input_and_output_tokens": True,
            "known_scoped_consent": True,
            "unknown_request_only": True,
            "unknown_response_tokens_or_business_reconstructed": False,
        },
        "record_audits": audits,
        "selected_records": selected_audits,
        "protocol_records": protocols,
        "selected_protocol_records": selected_protocols,
        "selected_protocol_summary": original.consent_summary(selected_protocols),
        "unknown_request_audit": unknown,
        "saved_weight_audit": physical,
        "historical_boundary_audit": history,
        "original_model_load_audit": load_audit,
        "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
        "selection_sha256": sha(REPORTS / "selection.json"),
        "final_archive_sha256": FINAL_ARCHIVE_SHA,
        "files": files,
        "verifier_source_sha256": sha(__file__),
        "new_model_or_gpu_calls": 0,
        "training_updates": 0,
        "scope": (
            "All136 known rows audited; only120 rows on a post-hoc nonrandom common40-case prefix "
            "may be analyzed. Original144 remains incomplete. No old output pooling or unknown-call replay."
        ),
    }
    original.write_once(REPORTS / "partial-evidence-verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(tokenizer_root=args.tokenizer_root), indent=2))


if __name__ == "__main__":
    main()
