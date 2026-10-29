"""Independent run002: default read-only preflight, then explicit plan/seal/free-GPU run."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.common import ROOT
from training.cuda_eval.provider import RecordingProvider, TorchProvider
from training.cuda_eval2.evaluation import (
    ARMS,
    REPORTS,
    RUN_ID,
    CallJournal,
    completed_summary,
    cpu_preflight,
    create_plan,
    install_final_adapter,
    protocol_report,
    reconcile_journal,
    report_directory,
    seal_artifacts,
    value_hash,
    verify_plan,
    write_atomic,
)
from training.cuda_torch.model import INITIALIZER_SHA256, load_model, sha
from training.v7.evaluation import run_case


def verify_identity(destination, training_report, adapter_directory, initializer_path, base_path):
    destination = report_directory(destination)
    identity = json.loads((destination / "artifact-identity.json").read_text())
    assert identity["run_id"] == RUN_ID
    plan = verify_plan(destination)
    assert identity["previous_interrupted_run"] == plan["previous_interrupted_run"]
    assert identity["analysis_plan_sha256"] == sha(destination / "analysis-plan.json")
    assert identity["training_report_sha256"] == sha(training_report)
    assert identity["training_plan_sha256"] == sha(Path(training_report).with_name("plan.json"))
    assert identity["after_adapter_sha256"] == sha(Path(adapter_directory) / "adapter_model.safetensors")
    assert identity["after_adapter_config_sha256"] == sha(Path(adapter_directory) / "adapter_config.json")
    assert (
        identity["before_adapter_sha256"]
        == sha(Path(initializer_path) / "adapters.safetensors")
        == INITIALIZER_SHA256
    )
    assert identity["model_source_sha256"] == sha(ROOT / "training/cuda_torch/model.py")
    assert identity["base_config_sha256"] == sha(Path(base_path) / "config.json")
    return identity


def checkpoint_state(destination, plan, identity, resume=False):
    destination = report_directory(destination)
    if plan.get("run_id") != RUN_ID or identity.get("run_id") != RUN_ID:
        raise RuntimeError("Never initialize or resume with another run's plan/identity")
    path = destination / "evaluation-partial.json"
    if (destination / "evaluation.json").exists():
        raise FileExistsError("Never rerun or overwrite completed evaluation evidence")
    if path.exists():
        if not resume:
            raise FileExistsError("Use --resume only at a completed case boundary")
        report = json.loads(path.read_text())
        if report.get("run_id") != RUN_ID:
            raise RuntimeError("Never resume, import or relabel another run's records")
        assert report["analysis_plan_sha256"] == sha(destination / "analysis-plan.json")
        assert report["artifact_identity_sha256"] == sha(destination / "artifact-identity.json")
        assert report["previous_interrupted_run"] == plan["previous_interrupted_run"]
        if report.get("pending_case"):
            raise RuntimeError(
                "An interrupted/ambiguous case is retained; refusing to repeat its model calls"
            )
        return report
    if resume:
        raise FileNotFoundError("No prior partial evaluation to resume")
    assert datetime.fromisoformat(plan["created_at_utc"]) <= datetime.fromisoformat(identity["sealed_at_utc"])
    return {
        "run_id": RUN_ID,
        "status": "partial",
        "scope": plan["scope"],
        "analysis_plan_sha256": sha(destination / "analysis-plan.json"),
        "artifact_identity_sha256": sha(destination / "artifact-identity.json"),
        "cases_sha256": plan["cases_sha256"],
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "old_outputs_pooled_or_replaced": False,
        "training_repeated_or_modified": False,
        "provider": "torch_cuda",
        "records": [],
        "model_audits": {},
        "resume_count": 0,
        "checkpoint_selection_performed": False,
        "training_or_test_tuning_performed": False,
        "scope_limitations": plan["protocol_limitations"],
        "timing_scope": plan["timing_scope"],
    }


def validate_completed_prefix(plan, report, journal):
    """Reject ambiguous, orphan or reordered calls before importing any GPU runtime."""
    order = [(arm, cid) for arm in ARMS for cid in plan["case_ids"]]
    actual = [(r["variant"], r["case_id"]) for r in report["records"]]
    if actual != order[: len(actual)]:
        raise RuntimeError("Completed cases must form the fixed, unique order prefix")
    if any(e["status"] != "completed" for e in journal.entries):
        raise RuntimeError("An ambiguous model call is retained; do not retry it")
    reconcile_journal(journal.entries, report["records"])
    return order, actual


def evaluate(
    destination, base_path, initializer_path, training_report, adapter_directory,
    resume=False, free_gpu_confirmed=False,
):
    # Source and physical artifact identities are checked before importing a GPU runtime.
    if not free_gpu_confirmed:
        raise PermissionError(
            "Explicit --free-gpu-confirmed required; never provision or fall back to paid GPU"
        )
    destination = report_directory(destination)
    plan = verify_plan(destination)
    identity = verify_identity(destination, training_report, adapter_directory, initializer_path, base_path)
    report = checkpoint_state(destination, plan, identity, resume)
    journal = CallJournal(destination / "call-journal.json")
    order, actual = validate_completed_prefix(plan, report, journal)
    from training.cuda_eval2.fixtures import load_cases
    from training.cuda_torch.train import dependency_precheck

    environment = dependency_precheck()
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA; no CPU fallback")
    torch.manual_seed(202610287)
    cases = load_cases("test")
    assert [c["id"] for c in cases] == plan["case_ids"]
    if resume:
        report["resume_count"] += 1
    report["dependency_environment"] = environment
    report["compute_authorization"] = {
        "free_runtime_operator_confirmed": True,
        "paid_fallback_allowed": False,
        "automatic_provisioning": False,
        "scope": "Explicit operator attestation, not inferred from CUDA GPU type or billing introspection",
    }
    write_atomic(destination / "evaluation-partial.json", report)
    model, before_audit = load_model(base_path, initializer_path, device="cuda")
    report["model_audits"].setdefault(ARMS[0], before_audit)
    report.setdefault("model_load_events", []).append(
        {"loaded_at_utc": datetime.now(UTC).isoformat(), "audit": before_audit}
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True, trust_remote_code=False)
    config = json.loads((Path(base_path) / "config.json").read_text())
    for arm in ARMS:
        remaining = [c for c in cases if (arm, c["id"]) not in actual]
        if arm == ARMS[1]:
            report["model_audits"][arm] = install_final_adapter(model, adapter_directory, identity)
        if not remaining:
            continue
        adapter_sha = (
            identity["before_adapter_sha256"] if arm == ARMS[0] else identity["after_adapter_sha256"]
        )
        provider = TorchProvider(model, tokenizer, config, arm, adapter_sha)
        store = ResolutionStore(destination / "sandboxes" / f"{arm}.sqlite")
        for case in remaining:
            report["pending_case"] = {
                "variant": arm,
                "case_id": case["id"],
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            write_atomic(destination / "evaluation-partial.json", report)
            recorded = RecordingProvider(provider, journal, arm, case["id"])
            try:
                row = {"variant": arm, **run_case(store, case, recorded, messages, max_steps=12)}
                assert row["model_calls"] == recorded.calls <= 12
                assert all(o["timing"]["adapter_sha256"] == adapter_sha for o in row["trace"])
            except BaseException as error:
                report["interrupted_case_error"] = {"type": type(error).__name__, "error": str(error)}
                write_atomic(destination / "evaluation-partial.json", report)
                raise
            report["records"].append(row)
            actual.append((arm, case["id"]))
            report.pop("pending_case")
            write_atomic(destination / "evaluation-partial.json", report)
            print(
                json.dumps(
                    {
                        "variant": arm,
                        "case_id": case["id"],
                        "task_success": row["task_success"],
                        "model_calls": row["model_calls"],
                    }
                ),
                flush=True,
            )
    assert actual == order and len(report["records"]) == 96
    assert len(journal.entries) == sum(r["model_calls"] for r in report["records"])
    assert datetime.fromisoformat(identity["sealed_at_utc"]) < min(
        datetime.fromisoformat(e["started_at_utc"]) for e in journal.entries
    )
    verify_plan(destination)
    verify_identity(destination, training_report, adapter_directory, initializer_path, base_path)
    report["summary"] = completed_summary(report["records"])
    report["call_journal_sha256"] = sha(destination / "call-journal.json")
    protocol = protocol_report(report["records"], cases)
    protocol.update(
        run_id=RUN_ID,
        analysis_plan_sha256=report["analysis_plan_sha256"],
        artifact_identity_sha256=report["artifact_identity_sha256"],
        cases_sha256=plan["cases_sha256"],
        evaluation_records_sha256=value_hash(report["records"]),
        source_sha256=sha(ROOT / "training/v7/protocol_audit.py"),
    )
    write_atomic(destination / "protocol-audit.json", protocol)
    report["protocol_reference"] = {
        "path": "protocol-audit.json",
        "sha256": sha(destination / "protocol-audit.json"),
    }
    report.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat())
    write_atomic(destination / "evaluation.json", report)
    print(json.dumps({"status": "complete", "summary": report["summary"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="preflight", choices=["preflight", "plan", "seal", "run"])
    parser.add_argument("--output", type=Path, default=REPORTS)
    parser.add_argument("--base", type=Path, default=ROOT / "data/models/qwen3-4b-mlx")
    parser.add_argument("--initializer", type=Path, default=ROOT / "data/finetuning/retail-qlora-v6_1/best")
    parser.add_argument(
        "--training-report", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/report.json"
    )
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/adapter"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--free-gpu-confirmed", action="store_true")
    args = parser.parse_args()
    if args.mode == "preflight":
        print(json.dumps(cpu_preflight(args.base, args.output), indent=2))
    elif args.mode == "plan":
        result = create_plan(args.output, args.base)
        print(json.dumps({"plan_written": str(args.output / "analysis-plan.json"), "scope": result["scope"]}))
    elif args.mode == "seal":
        result = seal_artifacts(args.training_report, args.adapter, args.base, args.initializer, args.output)
        print(json.dumps(result, indent=2))
    else:
        evaluate(
            args.output, args.base, args.initializer, args.training_report, args.adapter,
            args.resume, args.free_gpu_confirmed,
        )


if __name__ == "__main__":
    main()
