"""Issue a new144-task comparison once, using only already verified saved weights."""

import argparse
import copy
import json
import os
import sys
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.cuda_eval.evaluation import CallJournal, install_final_adapter, value_hash, write_atomic
from training.cuda_eval.provider import TorchProvider
from training.v8.evaluation import ARMS, MAX_CALLS, run_case, summarize
from training.v8_eval2.evaluation import (
    BASE,
    REPORTS,
    RUN_ID,
    cpu_preflight,
    create_plan,
    local_path,
    read,
    seal_identity,
    verify_identity,
)
from training.v8_eval2.resource_guard import arm_process_deadline, verify_resource_receipt


class EvaluationJournal(CallJournal):
    """The complete immutable request identity is durable before the model is called."""

    def __init__(self, path, bindings):
        if path.exists():
            raise FileExistsError("Never reopen an issued journal, including a single ambiguous tail")
        super().__init__(path)
        self.bindings = copy.deepcopy(bindings)

    def start(self, variant, case_id, call_index, prompt):
        if self.entries and self.entries[-1]["status"] != "completed":
            raise RuntimeError("An earlier issued call is ambiguous; no replay or additional calls")
        if len(self.entries) >= 1728 or not 1 <= call_index <= 12:
            raise RuntimeError("The unchanged12-per-task/1728-overall issued-call ceiling was reached")
        entry = {
            **self.bindings,
            "variant": variant,
            "case_id": case_id,
            "call_index": call_index,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "status": "started",
            "request_messages": copy.deepcopy(prompt),
            "request_sha256": value_hash(prompt),
        }
        self.entries.append(entry)
        write_atomic(self.path, self.entries)
        return entry


class RecordedProvider:
    def __init__(self, provider, journal, arm, case_id, workflow, bindings, check=None):
        self.provider, self.journal, self.arm, self.case_id = provider, journal, arm, case_id
        self.workflow, self.bindings, self.calls = workflow, copy.deepcopy(bindings), 0
        self.check = check or verify_resource_receipt

    def generate(self, prompt):
        self.check()
        if sha(REPORTS / "resource-authorization.json") != self.bindings["resource_receipt_sha256"]:
            raise PermissionError("Resource authorization changed after this task started")
        self.calls += 1
        if self.calls > MAX_CALLS:
            raise RuntimeError("Rejected/corrective calls share the original total12-call budget")
        entry = self.journal.start(self.arm, self.case_id, self.calls, prompt)
        raw, timing = self.provider.generate(prompt)
        timing = {**timing, **self.bindings, "workflow": self.workflow, "round": "v8-eval2"}
        self.journal.finish(entry, raw, timing)
        return raw, timing


def runtime_precheck():
    """No install, allocation or fallback. Called only after explicit execution approval."""
    import torch

    from training.cuda_torch.train import dependency_precheck

    environment = dependency_precheck()
    if sys.version_info[:2] != (3, 12) or torch.__version__ != "2.11.0+cu128":
        raise RuntimeError("Use the reviewed official2026.07 Python3.12/Torch2.11cu128 image")
    if torch.version.cuda != "12.8" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one compatible CUDA12.8 T4 is required; no CPU fallback")
    if torch.cuda.get_device_name(0).strip().casefold() not in {"tesla t4", "nvidia t4", "t4"}:
        raise RuntimeError("This approval permits T4 only")
    return environment


def evaluate(model, tokenizer, base_config, plan, identity, load_receipt):
    from resolve_desk.resolution_v7 import ResolutionStore as OldStore
    from resolve_desk.resolution_v7 import messages as old_messages
    from resolve_desk.resolution_v8 import ResolutionStore as NewStore
    from resolve_desk.resolution_v8 import messages as new_messages
    from training.v8.protocol_audit import audit_record
    from training.v8.protocol_audit import summarize as consent_summary
    from training.v8_eval2.fixtures import load_cases

    cases = load_cases("test")
    if [c["id"] for c in cases] != plan["case_order"] or value_hash(cases) != plan["dataset"][
        "canonical_cases_sha256"
    ]:
        raise ValueError("Fresh task content/order changed after sealing")
    bindings = {
        "run_id": RUN_ID,
        "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
        "artifact_identity_sha256": sha(REPORTS / "artifact-identity.json"),
        "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
    }
    journal = EvaluationJournal(REPORTS / "call-journal.json", bindings)
    report = {
        **bindings,
        "round": "v8-eval2",
        "split": "test",
        "status": "partial",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "arms": list(ARMS),
        "records": [],
        "model_audits": {},
        "cases_sha256": plan["dataset"]["canonical_cases_sha256"],
        "max_total_calls_per_task": MAX_CALLS,
        "maximum_total_issued_calls": 1728,
        "model_loads_this_process": 1,
        "model_load_receipt_sha256": sha(REPORTS / "evaluation-model-load.json"),
        "model_load_receipt": load_receipt,
        "checkpoint_selection_performed": False,
        "report_only": True,
        "deployment_or_default_provider_change_allowed": False,
        "resume_count": 0,
        "additional_training_performed": False,
        "validation_model_calls": 0,
        "old_run_provenance": plan["old_run_provenance"],
        "scope": plan["scope"],
    }
    write_atomic(REPORTS / "evaluation-partial.json", report)
    try:
        for arm in ARMS:
            old, candidate = arm == ARMS[0], arm == ARMS[-1]
            spec = plan["weights"]["after" if candidate else "before"]
            report["model_audits"][arm] = install_final_adapter(
                model, local_path(spec["runtime_directory"]), {"after_adapter_sha256": spec["sha256"]}
            )
            provider = TorchProvider(model, tokenizer, base_config, arm, spec["sha256"])
            store = (OldStore if old else NewStore)(REPORTS / "sandboxes" / f"{arm}.sqlite")
            for case in cases:
                verify_resource_receipt()
                report["pending_case"] = {
                    "arm": arm,
                    "case_id": case["id"],
                    "started_at_utc": datetime.now(UTC).isoformat(),
                }
                write_atomic(REPORTS / "evaluation-partial.json", report)
                recorded = RecordedProvider(
                    provider, journal, arm, case["id"], "v7" if old else "v8", bindings
                )
                row = {
                    "variant": arm,
                    **run_case(
                        store, case, recorded, old_messages if old else new_messages, max_steps=MAX_CALLS
                    ),
                }
                if row["model_calls"] != recorded.calls or not 0 < recorded.calls <= MAX_CALLS:
                    raise RuntimeError("Hidden, missing or uncounted issued calls")
                report["records"].append(row)
                report.pop("pending_case")
                write_atomic(REPORTS / "evaluation-partial.json", report)
                print(
                    json.dumps(
                        {
                            "run_id": RUN_ID,
                            "arm": arm,
                            "case_id": case["id"],
                            "completed_tasks": len(report["records"]),
                            "calls": recorded.calls,
                        }
                    ),
                    flush=True,
                )
        expected = [(arm, c["id"]) for arm in ARMS for c in cases]
        if [(r["variant"], r["case_id"]) for r in report["records"]] != expected:
            raise RuntimeError("The144-task matched comparison is incomplete or duplicated")
        if len(journal.entries) != sum(r["model_calls"] for r in report["records"]):
            raise RuntimeError("The issued-call journal and raw traces differ")
        if any(e["status"] != "completed" for e in journal.entries):
            raise RuntimeError("An issued call remains ambiguous; preserve it without replay")
        verify_identity()
        verify_resource_receipt()
        report["summary"] = {
            arm: summarize([r for r in report["records"] if r["variant"] == arm]) for arm in ARMS
        }
        by_id = {c["id"]: c for c in cases}
        audit = [audit_record(row, by_id[row["case_id"]]) for row in report["records"]]
        protocol = {
            **bindings,
            "records": audit,
            "summary": consent_summary(audit),
            "source_sha256": sha(ROOT / "training/v8/protocol_audit.py"),
            "scope": "Frozen scoped-consent audit; direct-return semantics are not independently certified",
        }
        write_atomic(REPORTS / "protocol-audit.json", protocol)
        report["call_journal_sha256"] = sha(REPORTS / "call-journal.json")
        report["protocol_audit_sha256"] = sha(REPORTS / "protocol-audit.json")
        report.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat())
        write_atomic(REPORTS / "evaluation.json", report)
    except BaseException as error:
        report["interruption"] = {
            "type": type(error).__name__,
            "error": str(error),
            "automatic_replay_allowed": False,
        }
        write_atomic(REPORTS / "evaluation-partial.json", report)
        raise
    return report


def execute():
    plan, identity = verify_identity()
    receipt = verify_resource_receipt()
    issued = (
        "evaluation-launch.json",
        "evaluation-partial.json",
        "evaluation.json",
        "call-journal.json",
        "sandboxes",
    )
    if any((REPORTS / name).exists() for name in issued):
        raise FileExistsError("No restart/resume of an issued run, including a model-load-only failure")
    # Atomic exclusive claim precedes framework imports/model loading. No retries.
    with (REPORTS / "evaluation-launch.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "analysis_plan_sha256": sha(REPORTS / "analysis-plan.json"),
                    "artifact_identity_sha256": sha(REPORTS / "artifact-identity.json"),
                    "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
                    "no_automatic_retry": True,
                    "command": "python -u -B -m training.v8_eval2.run run --execute",
                },
                indent=2,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    timer = arm_process_deadline(receipt)
    try:
        environment = runtime_precheck()
        import torch
        from transformers import AutoTokenizer

        from training.cuda_torch.model import load_model

        torch.manual_seed(plan["seed"])
        model, loader_audit = load_model(
            BASE, local_path(plan["weights"]["legacy"]["runtime_directory"]) / "adapters.safetensors"
        )
        load_receipt = {
            "run_id": RUN_ID,
            "loaded_at_utc": datetime.now(UTC).isoformat(),
            "model_load_count": 1,
            "loader_audit": loader_audit,
            "dependency_environment": environment,
            "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
            "loader_source_sha256": sha(ROOT / "training/cuda_torch/model.py"),
            "adapter_installation": (
                "All112 sealed F32 tensors installed exactly for each fixed arm; "
                "no extra warmup or validation"
            ),
        }
        write_atomic(REPORTS / "evaluation-model-load.json", load_receipt)
        tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
        evaluate(model, tokenizer, read(BASE / "config.json"), plan, identity, load_receipt)
    finally:
        timer.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("inspect", "plan", "seal", "verify", "run"), default="inspect", nargs="?"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "run":
        if not args.execute:
            raise PermissionError(
                "Running144 tasks requires explicit execution approval; import/setup does not"
            )
        execute()
    elif args.execute:
        raise PermissionError("Execute flag is only meaningful for the separately approved run")
    elif args.mode == "plan":
        print(json.dumps(create_plan(), indent=2))
    elif args.mode == "seal":
        print(json.dumps(seal_identity(), indent=2))
    elif args.mode == "verify":
        plan, identity = verify_identity()
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "plan_sha256": sha(REPORTS / "analysis-plan.json"),
                    "identity_sha256": sha(REPORTS / "artifact-identity.json"),
                    "ready_for_separate_resource_approval": True,
                    "arms": plan["arms"],
                    "candidate": identity["after_adapter_sha256"],
                }
            )
        )
    else:
        preflight = cpu_preflight()
        print(json.dumps({k: v for k, v in preflight.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
