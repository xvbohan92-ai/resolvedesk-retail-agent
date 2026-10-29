"""CPU-only planning and sealing for a separate fresh cohort after an interrupted run.

The original run, its inputs and unresolved calls stay immutable. Pure generation,
journal, adapter and scoring helpers are reused without changing their globals.
"""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT
from training.cuda_eval.evaluation import (  # noqa: F401 -- Re-export frozen pure helpers.
    ARMS,
    MAX_CALLS,
    CallJournal,
    completed_summary,
    install_final_adapter,
    peft_mapping,
    protocol_report,
    reconcile_journal,
    value_hash,
    write_atomic,
)
from training.cuda_eval.evaluation import (
    source_names as previous_source_names,
)
from training.cuda_eval.provider import MODEL_KWARGS, generation_settings
from training.cuda_torch.model import INITIALIZER_SHA256, adapter_plan, read_header, sha
from training.v7.common import verify_frozen

RUN_ID = "resolvedesk-cuda-eval-002"
REPORTS = ROOT / "reports/cuda-eval2"
PREVIOUS_PLAN_SHA256 = "cb08373c73938335659c018b13bd2ca31e76dda43a96b73e34bc3b0e30ce5866"
PREVIOUS_ZIP_SHA256 = "5f94ca79b20997987d09d5849d645b9761fc000462f8de1a9e91801bcd6fc741"
FINAL_ADAPTER_SHA256 = "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053"
TRAINING_PROOF = "reports/cuda-torch/probe-003/train-1344"
HISTORY = (
    "reports/cuda-eval/analysis-plan.json",
    "reports/cuda-eval/artifact-identity.json",
    "reports/cuda-eval/interruption.json",
    "reports/cuda-eval/interrupted-snapshot-audit.json",
)
SCOPE = (
    "Independent new 48-case authored synthetic cohort after infrastructure interruption of run001. "
    "Exact V6.1 predecessor versus the same fixed final1344 continuation, SAME decoded-4bit-derived "
    "FP16 base/F32 PEFT layout/frozen V7 workflow; not base-versus-finetune or a production benchmark. "
    "No retraining, checkpoint selection, replay of old ambiguous calls or pooling with run001."
)


def source_names():
    """All old runtime dependencies plus new data/protocol and immutable provenance."""
    return sorted(
        set(previous_source_names())
        | {
            "training/cuda_eval2/__init__.py",
            "training/cuda_eval2/fixtures.py",
            "training/cuda_eval2/prepare.py",
            "training/cuda_eval2/evaluation.py",
            "training/cuda_eval2/run.py",
            "training/cuda_eval2/datasets/cases.json",
            "training/cuda_eval2/datasets/public.json",
            "tests/test_cuda_eval2_data.py",
            "tests/test_cuda_eval2_protocol.py",
            "reports/cuda-eval2/data-summary.json",
            "reports/cuda-eval2/reference-audit.json",
            *(f"{TRAINING_PROOF}/{leaf}" for leaf in ("plan.json", "report.json", "verification.json")),
            *HISTORY,
            "training/datasets/cases.json",
            *(f"training/v{i}/datasets/cases.json" for i in range(2, 7)),
            "training/v6_1/datasets/cases.json",
            "training/v7/datasets/cases.json",
        }
    )


def report_directory(destination):
    """Never redirect this run into the interrupted run, training or historical evidence."""
    base = ROOT.resolve()
    root = base / "reports/cuda-eval2"
    destination = Path(destination).absolute()
    for name in (base / "reports", root, destination, *destination.parents):
        if name.is_symlink():
            raise ValueError("Symlinks cannot redirect independent evaluation evidence")
    destination = destination.resolve()
    if not destination.is_relative_to(root):
        raise ValueError("Run002 outputs must remain below reports/cuda-eval2")
    if destination.exists() and any(child.is_symlink() for child in destination.rglob("*")):
        raise ValueError("Symlinks cannot redirect independent evaluation evidence")
    return destination


def previous_interrupted_run():
    """Bind a verified interruption receipt, never infer completion or authorize replay."""
    old_plan, old_identity, interruption, audit = (
        json.loads((ROOT / name).read_text()) for name in HISTORY
    )
    if sha(ROOT / HISTORY[0]) != PREVIOUS_PLAN_SHA256:
        raise ValueError("Original pre-inference plan identity changed")
    assert old_identity["analysis_plan_sha256"] == PREVIOUS_PLAN_SHA256
    assert old_plan["total_calls_per_task"] == 12 and len(old_plan["case_ids"]) == 48
    assert interruption["status"] == "interrupted"
    assert interruption["analysis_plan_sha256"] == PREVIOUS_PLAN_SHA256
    assert interruption["raw_snapshot"]["sha256"] == PREVIOUS_ZIP_SHA256
    assert interruption["last_verified_completed_cases"] == 84
    assert interruption["complete_record_calls"] == 744
    assert interruption["retained_journal_entries"] == 755
    assert interruption["retained_completed_journal_entries"] == 754
    assert interruption["retained_started_journal_entries"] == 1
    assert interruption["unresolved_call"]["status"] == "started"
    assert not interruption["automatic_model_replay_allowed"] and not interruption["model_replayed"]
    assert not interruption["complete_96_record_evidence_available"]
    assert not interruption["later_actual_progress_known"]
    assert audit["all_executed_checks_passed"]
    assert audit["latest_completed_arm_task_records"] == 84
    assert audit["latest_journal_counts"] == {"all_entries": 755, "completed": 754, "started": 1}
    return {
        "run_id": "resolvedesk-cuda-eval-001",
        "status": "interrupted_final_state_unverified",
        "analysis_plan_sha256": PREVIOUS_PLAN_SHA256,
        "archive_path": interruption["raw_snapshot"]["path"],
        "archive_sha256": PREVIOUS_ZIP_SHA256,
        "interruption_path": HISTORY[2],
        "interruption_sha256": sha(ROOT / HISTORY[2]),
        "snapshot_audit_path": HISTORY[3],
        "snapshot_audit_sha256": sha(ROOT / HISTORY[3]),
        "last_verified_completed_records": 84,
        "last_verified_completed_record_calls": 744,
        "last_verified_journal_entries": 755,
        "last_verified_completed_journal_entries": 754,
        "last_verified_started_journal_entries": 1,
        "unresolved_call": interruption["unresolved_call"],
        "automatic_resume_allowed": False,
        "old_outputs_pooled_or_replaced": False,
        "scope": (
            "Last snapshot only; later actual progress unknown. Original ambiguous calls never replayed."
        ),
    }


def cpu_preflight(base_path=None, destination=REPORTS):
    """Read-only source/data/provenance/config check; no Torch/tokenizer/model imports."""
    from training.cuda_eval2.fixtures import load_cases
    from training.cuda_eval2.prepare import denominators

    report_directory(destination)
    verify_frozen()
    cases = load_cases("test")
    if len(cases) != 48 or len({c["id"] for c in cases}) != 48:
        raise ValueError("Expected exactly48 unique new cases")
    assert len({c["initial_message"] for c in cases}) == 48
    assert Counter(c["family"] for c in cases) == {c["family"]: 2 for c in cases}
    old = json.loads((ROOT / "training/cuda_eval/datasets/cases.json").read_text())
    assert not ({c["id"] for c in cases} & {c["id"] for c in old})
    assert not ({c["initial_message"] for c in cases} & {c["initial_message"] for c in old})
    previous = previous_interrupted_run()
    base_path = Path(base_path or ROOT / "data/models/qwen3-4b-mlx")
    config = json.loads((base_path / "config.json").read_text())
    hashes = {name: sha(ROOT / name) for name in source_names()}
    summary = json.loads((ROOT / "reports/cuda-eval2/data-summary.json").read_text())
    reference = json.loads((ROOT / "reports/cuda-eval2/reference-audit.json").read_text())
    assert summary["model_inference_started"] is False
    assert summary["cases_sha256"] == hashes["training/cuda_eval2/datasets/cases.json"]
    assert summary["public_sha256"] == hashes["training/cuda_eval2/datasets/public.json"]
    assert summary["reference_audit_sha256"] == hashes["reports/cuda-eval2/reference-audit.json"]
    assert summary["all_48_cpu_reference_business_tasks_pass"]
    assert summary["all_reference_scoped_consent_audits_pass"]
    assert summary["denominators"] == denominators(cases)
    for name, wanted in summary["source_inputs_sha256"].items():
        assert sha(ROOT / name) == wanted, name
    assert summary["quality"]["historical_cases_checked"] == 1275
    for name, wanted in summary["quality"]["historical_datasets_sha256"].items():
        assert name in hashes and hashes[name] == wanted, name
    assert reference["cases_sha256"] == summary["cases_sha256"]
    assert reference["tasks"] == len(reference["records"]) == 48
    assert {row["case_id"] for row in reference["records"]} == {c["id"] for c in cases}
    assert reference["all_business_references_pass"] and reference["all_references_within_12_total_calls"]
    assert all(row["reference"]["task_success"] for row in reference["records"])
    assert all(0 < row["reference"]["model_calls"] <= MAX_CALLS for row in reference["records"])
    training = json.loads((ROOT / f"{TRAINING_PROOF}/report.json").read_text())
    proof = json.loads((ROOT / f"{TRAINING_PROOF}/verification.json").read_text())
    assert training["status"] == "completed" and training["completed_true_optimizer_updates"] == 1344
    assert training["final_adapter"]["sha256"] == FINAL_ADAPTER_SHA256
    assert proof["report_sha256"] == hashes[f"{TRAINING_PROOF}/report.json"]
    assert proof["tensor_audit"]["numeric_values_independently_checked"]
    assert proof["tensor_audit"]["files_sha256"]["final_adapter"] == FINAL_ADAPTER_SHA256
    return {
        "run_id": RUN_ID,
        "source_inputs_sha256": hashes,
        "case_ids": [c["id"] for c in cases],
        "previous_interrupted_run": previous,
        "base_config_sha256": sha(base_path / "config.json"),
        "generation": generation_settings(config),
        "model_imported_or_called": False,
        "files_written": False,
    }


def create_plan(destination=REPORTS, base_path=None):
    destination = report_directory(destination)
    path = destination / "analysis-plan.json"
    if any((destination / leaf).exists() for leaf in (
        "analysis-plan.json", "artifact-identity.json", "evaluation-partial.json",
        "evaluation.json", "call-journal.json",
    )):
        raise FileExistsError("Never overwrite any pre-inference plan, sealed identity or issued call")
    checked = cpu_preflight(base_path, destination)
    hashes = checked["source_inputs_sha256"]
    plan = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "test_inference_started": False,
        "arms": list(ARMS),
        "case_ids": checked["case_ids"],
        "case_order": "New frozen dataset order within predecessor48, then cloud1344 same48",
        "source_inputs_sha256": hashes,
        "cases_sha256": hashes["training/cuda_eval2/datasets/cases.json"],
        "frozen_v7_manifest_sha256": hashes["training/v7/datasets/manifest.json"],
        "base_config_sha256": checked["base_config_sha256"],
        "previous_interrupted_run": checked["previous_interrupted_run"],
        "old_outputs_pooled_or_replaced": False,
        "training_repeated_or_modified": False,
        "total_calls_per_task": MAX_CALLS,
        "generation": checked["generation"],
        "inference_model_kwargs": dict(MODEL_KWARGS),
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True, "truncation": False},
        "raw_output": "All token IDs retained; only one terminal EOS removed; no other special text repaired",
        "after_checkpoint": {
            "completed_updates": 1344,
            "adapter_sha256": FINAL_ADAPTER_SHA256,
            "selection": "Previously fixed final1344; no checkpoint selected using old or new test outputs",
        },
        "before_initializer_sha256": INITIALIZER_SHA256,
        "artifact_identity": "New run002 identity sealed before ANY new model generation",
        "model_lifecycle": "Load one base, before48, overwrite only112 LoRA tensors, after48; no warmups",
        "compute": "FP16 frozen base/F32 PEFT; no autocast; eval/inference_mode; checkpointing off; KV cache",
        "compute_authorization": {
            "free_gpu_only": True,
            "paid_fallback_allowed": False,
            "automatic_provisioning": False,
            "operator_must_confirm_free_runtime_at_launch": True,
        },
        "resume_policy": (
            "Run002 completed case boundaries only. Pending/ambiguous calls refuse automatic replay. "
            "Never resume run001, drop its started tail, or merge old and new records."
        ),
        "primary_metrics": [
            "Frozen V7 task_success and required answers",
            "First raw seven fields/parse/enums before repair",
            "Permission recall and unsupported yes",
            "Exchange/fallback/handoff/email cohorts",
            "Repeated field attempts/deliveries and identical action repeats separately",
            "All raw calls/proposals/commits/final orders in scoped-consent audit",
        ],
        "protocol_limitations": (
            "Frozen operation-rewrite bypass remains; audit exposes it without repairing the controller. "
            "Initial direct returns and unrestricted semantic operation changes outside audit."
        ),
        "quality_scope": "New authored24families, not human-annotated/external/random; no population CI",
        "timing_scope": (
            "Serial fixed-arm order; GPU-sync generation excludes load/swap; task time includes journaling. "
            "Later prompts diverge by actions; no isolated hardware or production speedup claim."
        ),
        "reporting_helpers": "Mutable, separately hash-bound; never change frozen inputs/scores/checkpoint",
    }
    write_atomic(path, plan)
    return plan


def verify_plan(destination=REPORTS):
    plan = json.loads((report_directory(destination) / "analysis-plan.json").read_text())
    verify_frozen()
    assert plan["run_id"] == RUN_ID and not plan["test_inference_started"]
    assert tuple(plan["arms"]) == ARMS and plan["total_calls_per_task"] == 12
    assert plan["generation"]["max_new_tokens"] == 512 and plan["generation"]["do_sample"] is False
    assert plan["previous_interrupted_run"] == previous_interrupted_run()
    assert plan["old_outputs_pooled_or_replaced"] is False and plan["training_repeated_or_modified"] is False
    assert plan["after_checkpoint"]["adapter_sha256"] == FINAL_ADAPTER_SHA256
    assert plan["after_checkpoint"]["completed_updates"] == 1344
    assert set(plan["source_inputs_sha256"]) == set(source_names())
    for name, wanted in plan["source_inputs_sha256"].items():
        if sha(ROOT / name) != wanted:
            raise ValueError(f"Run002 pre-inference source/data changed: {name}")
    return plan


def seal_artifacts(training_report, adapter_directory, base_path, initializer_path, destination=REPORTS):
    """CPU-read all final F32 values and bind the already completed immutable training proof."""
    destination = report_directory(destination)
    plan = verify_plan(destination)
    target = destination / "artifact-identity.json"
    if any((destination / leaf).exists() for leaf in (
        "artifact-identity.json", "evaluation-partial.json", "evaluation.json", "call-journal.json",
    )):
        raise FileExistsError("Never replace sealed weights or any started independent evaluation")
    training_report, adapter_directory = Path(training_report), Path(adapter_directory)
    report = json.loads(training_report.read_text())
    assert sha(training_report) == plan["source_inputs_sha256"][f"{TRAINING_PROOF}/report.json"]
    assert report["status"] == "completed" and report["kind"] == "train"
    assert report["completed_true_optimizer_updates"] == 1344 and len(report["metrics"]) == 1344
    assert [m["true_optimizer_updates"] for m in report["metrics"]] == list(range(1, 1345))
    assert report["all_gradients_finite"] and report["adapter_l2_change"] > 0
    assert report["smoke_weights_discarded"] and not report["smoke_weights_reused"]
    for name, wanted in report["source_inputs_sha256"].items():
        assert sha(ROOT / name) == wanted, name
    training_plan = training_report.with_name("plan.json")
    assert sha(training_plan) == report["plan_sha256"]
    after = adapter_directory / "adapter_model.safetensors"
    before = Path(initializer_path) / "adapters.safetensors"
    assert sha(after) == report["final_adapter"]["sha256"] == FINAL_ADAPTER_SHA256
    assert sha(before) == INITIALIZER_SHA256
    assert sha(Path(base_path) / "config.json") == plan["base_config_sha256"]
    raw_config = json.loads((Path(base_path) / "config.json").read_text())
    original = adapter_plan(read_header(before), raw_config)
    specs = {r["target"]: r["target_shape"] for r in original}
    final_mapping = peft_mapping(read_header(after), specs)
    adapter_config = json.loads((adapter_directory / "adapter_config.json").read_text())
    assert adapter_config["r"] == 8 and adapter_config["lora_alpha"] == 128
    assert adapter_config["lora_dropout"] == 0 and not adapter_config.get("use_rslora", False)
    assert set(adapter_config["target_modules"]) == {r["module"] for r in original}
    import numpy as np
    from safetensors.numpy import load_file

    final_values = load_file(str(after))
    assert set(final_values) == {r["source"] for r in final_mapping}
    assert all(v.dtype == np.float32 and bool(np.isfinite(v).all()) for v in final_values.values())
    del final_values
    identity = {
        "run_id": RUN_ID,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": sha(destination / "analysis-plan.json"),
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "training_report_sha256": sha(training_report),
        "training_plan_sha256": sha(training_plan),
        "training_verification_sha256": plan["source_inputs_sha256"][f"{TRAINING_PROOF}/verification.json"],
        "completed_true_optimizer_updates": 1344,
        "before_adapter_sha256": sha(before),
        "after_adapter_sha256": sha(after),
        "after_adapter_config_sha256": sha(adapter_directory / "adapter_config.json"),
        "base_spec_sha256": sha(ROOT / "training/base-model-spec.json"),
        "base_config_sha256": plan["base_config_sha256"],
        "model_source_sha256": sha(ROOT / "training/cuda_torch/model.py"),
        "parameter_count": 3_670_016,
        "tensor_count": len(final_mapping),
        "canonical_shapes_sha256": value_hash(specs),
        "all_final_adapter_values_finite_cpu_checked": True,
        "selection_performed": False,
        "scope": plan["scope"],
    }
    write_atomic(target, identity)
    return identity
