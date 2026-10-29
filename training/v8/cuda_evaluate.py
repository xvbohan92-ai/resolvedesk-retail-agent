"""One-shot V8 validation then three matched test arms; issued calls are never replayed."""

import argparse
import json
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.cuda_eval.evaluation import CallJournal, install_final_adapter, value_hash, write_atomic
from training.cuda_eval.provider import TorchProvider
from training.v8.common import CONFIG, verify_frozen
from training.v8.evaluation import ARMS, MAX_CALLS, run_case, summarize
from training.v8.resource_guard import arm_process_deadline, verify_resource_receipt

REPORTS = ROOT / "reports/v8"


class RecordedProvider:
    def __init__(self, provider, journal, arm, case_id, workflow, resource_sha=None):
        self.provider, self.journal, self.arm, self.case_id = provider, journal, arm, case_id
        self.workflow, self.calls = workflow, 0
        self.resource_sha256 = resource_sha or sha(REPORTS / "resource-authorization.json")

    def generate(self, prompt):
        self.calls += 1
        if self.calls > MAX_CALLS:
            raise RuntimeError("Every failed or corrective call consumes the same total12-call budget")
        verify_resource_receipt()
        if sha(REPORTS / "resource-authorization.json") != self.resource_sha256:
            raise PermissionError("Resource authorization changed after this task started")
        entry = self.journal.start(self.arm, self.case_id, self.calls, prompt)
        raw, timing = self.provider.generate(prompt)
        timing = {**timing, "workflow": self.workflow, "round": "v8"}
        self.journal.finish(entry, raw, timing)
        return raw, timing


def verify_identity():
    manifest = verify_frozen()
    identity = json.loads((REPORTS / "artifact-identity.json").read_text())
    config = json.loads(CONFIG.read_text())
    train = REPORTS / "cuda/train-1344/report.json"
    if identity["analysis_plan_sha256"] != sha(REPORTS / "pre-inference-plan.json"):
        raise ValueError("Evaluation plan changed after weight sealing")
    if identity["manifest_sha256"] != sha(ROOT / "training/v8/datasets/manifest.json"):
        raise ValueError("Source and data manifest changed")
    if identity["training_report_sha256"] != sha(train):
        raise ValueError("Actual training report changed")
    proof = REPORTS / "training-proof.json"
    if identity["training_proof_sha256"] != sha(proof):
        raise ValueError("Independent physical weight proof changed")
    checked = json.loads(proof.read_text())
    if checked.get("exact_prior_initialization") is not True or checked.get("true_optimizer_updates") != 1344:
        raise ValueError("No complete independent initializer and update proof")
    if datetime.fromisoformat(identity["sealed_at_utc"]) >= datetime.now(UTC):
        raise ValueError("Artifact identity must exist before inference starts")
    before = ROOT / config["initialization"]
    after = REPORTS / "cuda/train-1344/adapter"
    for prefix, directory in [("before", before), ("after", after)]:
        if identity[prefix + "_adapter_sha256"] != sha(directory / "adapter_model.safetensors"):
            raise ValueError("Physical weights do not match the sealed comparison")
        if identity[prefix + "_adapter_config_sha256"] != sha(directory / "adapter_config.json"):
            raise ValueError("Physical adapter configuration changed")
    if identity["before_adapter_sha256"] != config["initialization_sha256"]:
        raise ValueError("Baseline must be the last completed CUDA1344 candidate")
    if (
        checked.get("initializer_sha256") != identity["before_adapter_sha256"]
        or checked.get("final_adapter_sha256") != identity["after_adapter_sha256"]
        or checked.get("training_report_sha256") != identity["training_report_sha256"]
        or checked.get("training_plan_sha256") != identity["training_plan_sha256"]
    ):
        raise ValueError("Independent numeric proof is bound to different physical training artifacts")
    if identity["training_plan_sha256"] != sha(train.with_name("plan.json")):
        raise ValueError("Training plan differs from its sealed identity")
    return manifest, identity, config, before, after


def _evaluate_split(model, tokenizer, base_config, identity, before, after, split, load_receipt):
    from resolve_desk.resolution_v7 import ResolutionStore as OldStore
    from resolve_desk.resolution_v7 import messages as old_messages
    from resolve_desk.resolution_v8 import ResolutionStore as NewStore
    from resolve_desk.resolution_v8 import messages as new_messages
    from training.v8.eval_fixtures import load_cases as test_cases
    from training.v8.fixtures import load_cases as development_cases
    from training.v8.protocol_audit import audit_record
    from training.v8.protocol_audit import summarize as consent_summary

    cases = test_cases() if split == "test" else development_cases("valid")
    if len(cases) != 48 or len({c["id"] for c in cases}) != 48:
        raise ValueError("Each split must contain exactly48 distinct fixed tasks")
    folder = REPORTS / ("evaluation" if split == "test" else "validation")
    folder.mkdir(parents=True, exist_ok=False)
    arms = ARMS if split == "test" else (ARMS[-1],)
    journal = CallJournal(folder / "call-journal.json")
    report = {
        "round": "v8",
        "split": split,
        "status": "partial",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": sha(REPORTS / "pre-inference-plan.json"),
        "artifact_identity_sha256": sha(REPORTS / "artifact-identity.json"),
        "manifest_sha256": sha(ROOT / "training/v8/datasets/manifest.json"),
        "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
        "cases_sha256": value_hash(cases),
        "arms": list(arms),
        "records": [],
        "model_audits": {},
        "max_total_calls_per_task": MAX_CALLS,
        "model_loads_this_process": 1,
        "model_load_receipt_sha256": sha(REPORTS / "evaluation-model-load.json"),
        "model_load_receipt": load_receipt,
        "checkpoint_selection_performed": False,
        "resume_count": 0,
        "scope": "New authored synthetic cohort; same base and decoding. First raw output remains raw.",
    }
    write_atomic(folder / "evaluation-partial.json", report)
    try:
        for arm in arms:
            old = arm == ARMS[0]
            candidate = arm == ARMS[-1]
            adapter_sha = identity["after_adapter_sha256" if candidate else "before_adapter_sha256"]
            report["model_audits"][arm] = install_final_adapter(
                model, after if candidate else before, {"after_adapter_sha256": adapter_sha}
            )
            provider = TorchProvider(model, tokenizer, base_config, arm, adapter_sha)
            store = (OldStore if old else NewStore)(folder / "sandboxes" / f"{arm}.sqlite")
            for case in cases:
                report["pending_case"] = {
                    "arm": arm,
                    "case_id": case["id"],
                    "started_at_utc": datetime.now(UTC).isoformat(),
                }
                write_atomic(folder / "evaluation-partial.json", report)
                recorded = RecordedProvider(
                    provider,
                    journal,
                    arm,
                    case["id"],
                    "v7" if old else "v8",
                    resource_sha=report["resource_receipt_sha256"],
                )
                row = {
                    "variant": arm,
                    **run_case(
                        store, case, recorded, old_messages if old else new_messages, max_steps=MAX_CALLS
                    ),
                }
                if row["model_calls"] != recorded.calls or not row["model_calls"] <= MAX_CALLS:
                    raise RuntimeError("Hidden or uncounted model calls")
                report["records"].append(row)
                report.pop("pending_case")
                write_atomic(folder / "evaluation-partial.json", report)
                print(
                    json.dumps(
                        {
                            "split": split,
                            "arm": arm,
                            "case_id": case["id"],
                            "task_success": row["task_success"],
                            "calls": recorded.calls,
                        }
                    ),
                    flush=True,
                )
        expected = [(arm, c["id"]) for arm in arms for c in cases]
        if [(r["variant"], r["case_id"]) for r in report["records"]] != expected:
            raise RuntimeError("Incomplete or repeated arm/task results")
        if len(journal.entries) != sum(r["model_calls"] for r in report["records"]):
            raise RuntimeError("Call journal and recorded traces differ")
        if any(e["status"] != "completed" for e in journal.entries):
            raise RuntimeError("An issued call is ambiguous; preserve it and stop")
        verify_identity()
        if sha(REPORTS / "resource-authorization.json") != report["resource_receipt_sha256"]:
            raise PermissionError("Resource receipt drifted during the evaluation phase")
        report["summary"] = {
            arm: summarize([r for r in report["records"] if r["variant"] == arm]) for arm in arms
        }
        by_id = {c["id"]: c for c in cases}
        protocol = [audit_record(r, by_id[r["case_id"]]) for r in report["records"]]
        protocol_report = {
            "records": protocol,
            "summary": consent_summary(protocol),
            "source_sha256": sha(ROOT / "training/v8/protocol_audit.py"),
            "scope": "Exchange-origin scoped consent; direct-return semantics not certified",
        }
        write_atomic(folder / "protocol-audit.json", protocol_report)
        report["call_journal_sha256"] = sha(folder / "call-journal.json")
        report["protocol_audit_sha256"] = sha(folder / "protocol-audit.json")
        report.update(status="complete", completed_at_utc=datetime.now(UTC).isoformat())
        write_atomic(folder / "evaluation.json", report)
    except BaseException as error:
        report["interruption"] = {
            "type": type(error).__name__,
            "error": str(error),
            "automatic_replay_allowed": False,
        }
        write_atomic(folder / "evaluation-partial.json", report)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise PermissionError("CPU import does not authorize model inference; explicit execute is required")
    _, identity, config, before, after = verify_identity()
    receipt = verify_resource_receipt()
    if any((REPORTS / n).exists() for n in ("validation", "evaluation", "evaluation-launch.json")):
        raise FileExistsError("This new round has already been issued; do not replay calls")
    from training.cuda_torch.train import dependency_precheck

    dependency_precheck()
    timer = arm_process_deadline(receipt, "evaluation")
    write_atomic(
        REPORTS / "evaluation-launch.json",
        {
            "started_at_utc": datetime.now(UTC).isoformat(),
            "artifact_identity": identity,
            "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
            "command": "python -u -B -m training.v8.cuda_evaluate --execute",
            "no_automatic_retry": True,
        },
    )
    try:
        import torch
        from transformers import AutoTokenizer

        from training.cuda_torch.model import load_model

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("One CUDA T4 required; no fallback")
        if torch.cuda.get_device_name(0).strip().casefold() not in {"tesla t4", "nvidia t4", "t4"}:
            raise RuntimeError("T4-only resource approval")
        torch.manual_seed(config["seed"])
        model, loader_audit = load_model(
            ROOT / config["model"], ROOT / config["legacy_loader_initialization"]
        )
        load_receipt = {
            "loaded_at_utc": datetime.now(UTC).isoformat(),
            "model_load_count": 1,
            "loader_audit": loader_audit,
            "resource_receipt_sha256": sha(REPORTS / "resource-authorization.json"),
            "loader_source_sha256": sha(ROOT / "training/cuda_torch/model.py"),
            "adapter_installation": "Each subsequent arm installs all112 sealed F32 tensors exactly",
        }
        write_atomic(REPORTS / "evaluation-model-load.json", load_receipt)
        tokenizer = AutoTokenizer.from_pretrained(
            ROOT / config["model"], local_files_only=True, trust_remote_code=False
        )
        base_config = json.loads((ROOT / config["model"] / "config.json").read_text())
        _evaluate_split(model, tokenizer, base_config, identity, before, after, "valid", load_receipt)
        _evaluate_split(model, tokenizer, base_config, identity, before, after, "test", load_receipt)
    finally:
        timer.cancel()


if __name__ == "__main__":
    main()
