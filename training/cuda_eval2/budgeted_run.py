"""Explicit existing-unit authorization overlay; frozen model/test/scoring inputs stay unchanged.

This process can stop itself at an absolute deadline. Only the external Colab
operator can observe compute-unit usage and disconnect the runtime to stop billing.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import threading
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
    install_final_adapter,
    protocol_report,
    report_directory,
    value_hash,
    verify_plan,
    write_atomic,
)
from training.cuda_eval2.run import checkpoint_state, validate_completed_prefix, verify_identity
from training.cuda_torch.model import load_model, sha
from training.v7.evaluation import run_case

ADDENDUM = "execution-addendum.json"
MODE = "metered_existing_units"
ORIGINAL_PLAN_SHA = "a547fd0790ad2003fcd3dc0a990b581484bf5decd30222c87499237fd236f704"
ORIGINAL_IDENTITY_SHA = "f83a97af4775ac2c01bf82f5ff188de1743c010ec143c554b74a8309f7840d1e"
SUPPLEMENTARY_SOURCES = ("training/cuda_eval2/budgeted_run.py", "tests/test_cuda_eval2_budget.py")


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("An explicit UTC timestamp is required")
    return parsed


def is_t4(name):
    return name.strip().casefold() in {"tesla t4", "nvidia t4", "t4"}


def probe_t4():
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if len(result.stdout) > 1024 or not is_t4(result.stdout.strip()):
        raise RuntimeError("Metered execution is T4-only, with no accelerator fallback")
    return {"device_index": 0, "name": result.stdout.strip(), "source": "bounded nvidia-smi query"}


def compute_authorization(addendum_sha):
    return {
        "authorization_mode": MODE,
        "existing_units_operator_confirmed": True,
        "free_runtime_operator_confirmed": False,
        "cap_compute_units": 10,
        "purchases_allowed": False,
        "renewal_allowed": False,
        "automatic_provisioning": False,
        "execution_addendum_path": "reports/cuda-eval2/" + ADDENDUM,
        "execution_addendum_sha256": addendum_sha,
        "scope": (
            "Existing units only; external UI metering and runtime disconnect, not automatic CU billing proof"
        ),
    }


def authorize(
    destination,
    *,
    existing_units_confirmed,
    authorization_text,
    process_deadline_utc,
    runtime_allocated_at_utc,
    runtime_disconnect_deadline_utc,
    observed_units_per_hour,
):
    if not existing_units_confirmed or not authorization_text.strip():
        raise PermissionError("Explicit human authorization for <=10 existing units is required")
    destination = report_directory(destination)
    if any(
        (destination / n).exists()
        for n in [
            ADDENDUM,
            "evaluation-launch.json",
            "evaluation-partial.json",
            "evaluation.json",
            "call-journal.json",
            "budgeted-run-start.json",
        ]
    ):
        raise FileExistsError("Never replace authorization or any issued/attempted evaluation")
    plan = verify_plan(destination)
    assert sha(destination / "analysis-plan.json") == ORIGINAL_PLAN_SHA
    assert sha(destination / "artifact-identity.json") == ORIGINAL_IDENTITY_SHA
    identity = json.loads((destination / "artifact-identity.json").read_text())
    allocated, deadline, disconnect = map(
        utc, [runtime_allocated_at_utc, process_deadline_utc, runtime_disconnect_deadline_utc]
    )
    now = datetime.now(UTC)
    assert allocated <= now < deadline < disconnect
    assert math.isfinite(observed_units_per_hour) and observed_units_per_hour > 0
    projected = (disconnect - allocated).total_seconds() / 3600 * observed_units_per_hour
    if projected > 8:
        raise ValueError("Observed-rate projection exceeds8 units; reserve at least2 of the10-unit cap")
    addendum = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "created_at_utc": now.isoformat(),
        "authorization_mode": MODE,
        "human_authorization": {
            "explicit": True,
            "approval_text": authorization_text,
            "source": "Direct human authorization supplied by the parent operator",
            "cap_compute_units": 10,
            "starting_balance_units": 80,
            "stop_balance_units": 72,
            "hard_floor_balance_units": 70,
            "purchase_allowed": False,
            "renewal_allowed": False,
            "free_allocation_claimed": False,
            "T4_only": True,
        },
        "analysis_plan_sha256": ORIGINAL_PLAN_SHA,
        "artifact_identity_sha256": ORIGINAL_IDENTITY_SHA,
        "before_adapter_sha256": identity["before_adapter_sha256"],
        "after_adapter_sha256": identity["after_adapter_sha256"],
        "cases_sha256": plan["cases_sha256"],
        "source_inputs_sha256": copy.deepcopy(plan["source_inputs_sha256"]),
        "supplementary_sources_sha256": {n: sha(ROOT / n) for n in SUPPLEMENTARY_SOURCES},
        "previous_interrupted_run": plan["previous_interrupted_run"],
        "supersedes_only": {
            "field": "compute_authorization.free_gpu_only",
            "original_value": True,
            "effective_authorization": MODE,
            "reason": "Subsequent explicit human approval; original plan/identity/source53 remain immutable",
        },
        "unchanged_protocol": {
            "no_retraining": True,
            "no_checkpoint_selection": True,
            "no_old_run_replay": True,
            "no_old_outputs_pooled": True,
            "seed": 202610287,
            "total_calls_per_task": 12,
            "generation": plan["generation"],
            "inference_model_kwargs": plan["inference_model_kwargs"],
            "case_order": plan["case_ids"],
            "arms": list(ARMS),
            "controller": "frozen_v7",
            "scoring": "frozen_v7",
            "T4_only": True,
        },
        "accounting": {
            "owner": "External parent/human operator observing Colab UI and disconnecting runtime",
            "starting_balance_units": 80,
            "hard_cap_units": 10,
            "hard_cap_balance": 70,
            "operator_stop_balance": 72,
            "reserved_units": 2,
            "runtime_allocated_at_utc": allocated.isoformat(),
            "process_deadline_utc": deadline.isoformat(),
            "runtime_disconnect_deadline_utc": disconnect.isoformat(),
            "observed_units_per_hour": observed_units_per_hour,
            "observed_rate_source": "Operator observation, not an API-verified fixed billing rate",
            "projected_units_until_disconnect": projected,
            "projection_is_actual_consumption": False,
            "automatic_compute_unit_metering": False,
            "billing_stops_when_process_exits": False,
            "scope": (
                "10-unit human cap; external stop at balance<=72 or earlier deadline. No purchase/renewal."
            ),
        },
    }
    write_atomic(destination / ADDENDUM, addendum)
    return addendum


def verify_addendum(destination=REPORTS, *, require_not_expired=False):
    destination = report_directory(destination)
    plan = verify_plan(destination)
    addendum = json.loads((destination / ADDENDUM).read_text())
    identity = json.loads((destination / "artifact-identity.json").read_text())
    assert addendum["run_id"] == RUN_ID and addendum["authorization_mode"] == MODE
    assert addendum["analysis_plan_sha256"] == sha(destination / "analysis-plan.json") == ORIGINAL_PLAN_SHA
    assert (
        addendum["artifact_identity_sha256"]
        == sha(destination / "artifact-identity.json")
        == ORIGINAL_IDENTITY_SHA
    )
    assert addendum["source_inputs_sha256"] == plan["source_inputs_sha256"]
    assert addendum["cases_sha256"] == plan["cases_sha256"]
    for key in ["before_adapter_sha256", "after_adapter_sha256"]:
        assert addendum[key] == identity[key]
    assert set(addendum["supplementary_sources_sha256"]) == set(SUPPLEMENTARY_SOURCES)
    for name, wanted in addendum["supplementary_sources_sha256"].items():
        assert sha(ROOT / name) == wanted, name
    human, accounting = addendum["human_authorization"], addendum["accounting"]
    assert human["explicit"] and human["approval_text"].strip()
    assert human["cap_compute_units"] == 10 and human["starting_balance_units"] == 80
    assert human["stop_balance_units"] == 72 and human["hard_floor_balance_units"] == 70
    assert human["T4_only"] and not human["free_allocation_claimed"]
    assert not human["purchase_allowed"] and not human["renewal_allowed"]
    assert accounting["operator_stop_balance"] == 72 and accounting["reserved_units"] == 2
    assert accounting["hard_cap_units"] == 10 and accounting["hard_cap_balance"] == 70
    assert accounting["starting_balance_units"] == 80
    assert not accounting["automatic_compute_unit_metering"]
    assert not accounting["billing_stops_when_process_exits"]
    created, deadline = utc(addendum["created_at_utc"]), utc(accounting["process_deadline_utc"])
    assert utc(identity["sealed_at_utc"]) <= created < deadline
    assert utc(accounting["runtime_allocated_at_utc"]) <= created < deadline
    assert deadline < utc(accounting["runtime_disconnect_deadline_utc"])
    rate = accounting["observed_units_per_hour"]
    assert math.isfinite(rate) and rate > 0
    projected = (
        (
            utc(accounting["runtime_disconnect_deadline_utc"]) - utc(accounting["runtime_allocated_at_utc"])
        ).total_seconds()
        / 3600
        * rate
    )
    assert accounting["projected_units_until_disconnect"] == projected <= 8
    assert accounting["projection_is_actual_consumption"] is False
    assert addendum["supersedes_only"]["field"] == "compute_authorization.free_gpu_only"
    assert addendum["supersedes_only"]["original_value"] is True
    assert addendum["supersedes_only"]["effective_authorization"] == MODE
    if require_not_expired and datetime.now(UTC) >= deadline:
        raise TimeoutError("Process deadline expired; no model call or automatic extension permitted")
    unchanged = addendum["unchanged_protocol"]
    assert unchanged["generation"] == plan["generation"] and unchanged["seed"] == 202610287
    assert unchanged["inference_model_kwargs"] == plan["inference_model_kwargs"]
    assert unchanged["case_order"] == plan["case_ids"] and unchanged["arms"] == list(ARMS)
    assert unchanged["total_calls_per_task"] == 12
    assert all(
        unchanged[k]
        for k in [
            "no_retraining",
            "no_checkpoint_selection",
            "no_old_run_replay",
            "no_old_outputs_pooled",
            "T4_only",
        ]
    )
    assert addendum["previous_interrupted_run"] == plan["previous_interrupted_run"]
    return addendum


class BudgetJournal(CallJournal):
    def __init__(self, path, addendum_sha):
        super().__init__(path)
        self.addendum_sha = addendum_sha
        for entry in self.entries:
            if entry.get("execution_addendum_sha256") != addendum_sha or entry.get("run_id") != RUN_ID:
                raise RuntimeError("Never reuse calls from another authorization route")

    def start(self, variant, case_id, call_index, prompt):
        entry = {
            "variant": variant,
            "case_id": case_id,
            "call_index": call_index,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "status": "started",
            "request_messages": copy.deepcopy(prompt),
            "request_sha256": value_hash(prompt),
            "run_id": RUN_ID,
            "execution_addendum_sha256": self.addendum_sha,
        }
        self.entries.append(entry)
        write_atomic(self.path, self.entries)
        return entry


class BudgetProvider:
    def __init__(self, provider, addendum_sha, deadline):
        self.provider, self.addendum_sha, self.deadline = provider, addendum_sha, utc(deadline)

    def ensure_deadline(self):
        if datetime.now(UTC) >= self.deadline:
            raise TimeoutError("Process budget deadline reached; no further generation allowed")

    def generate(self, prompt):
        self.ensure_deadline()
        raw, timing = self.provider.generate(prompt)
        return raw, {
            **timing,
            "execution_addendum_sha256": self.addendum_sha,
            "execution_authorization_mode": MODE,
        }


class BudgetRecordingProvider(RecordingProvider):
    def generate(self, prompt):
        self.provider.ensure_deadline()
        return super().generate(prompt)


def deadline_callback(destination, addendum_sha, exit_process=os._exit):
    def stop():
        try:
            write_atomic(
                Path(destination) / "process-deadline-stop.json",
                {
                    "run_id": RUN_ID,
                    "status": "process_deadline_reached",
                    "stopped_at_utc": datetime.now(UTC).isoformat(),
                    "execution_addendum_sha256": addendum_sha,
                    "pending_calls_replay_allowed": False,
                    "billing_stopped": False,
                    "scope": "Only this process exits; parent must separately disconnect the Colab runtime",
                },
            )
        finally:
            exit_process(124)

    return stop


def _evaluate(
    destination,
    base_path,
    initializer_path,
    training_report,
    adapter_directory,
    resume=False,
    addendum=None,
):
    # Supplementary account authorization never changes model/scoring inputs.
    destination = report_directory(destination)
    plan = verify_plan(destination)
    identity = verify_identity(destination, training_report, adapter_directory, initializer_path, base_path)
    report = checkpoint_state(destination, plan, identity, resume)
    journal = BudgetJournal(destination / "call-journal.json", sha(destination / ADDENDUM))
    order, actual = validate_completed_prefix(plan, report, journal)
    from training.cuda_eval2.fixtures import load_cases
    from training.cuda_torch.train import dependency_precheck

    environment = dependency_precheck()
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA; no CPU fallback")
    if not is_t4(torch.cuda.get_device_name(0)) or torch.cuda.device_count() != 1:
        raise RuntimeError("Metered execution permits one T4 only; no accelerator fallback")
    torch.manual_seed(202610287)
    cases = load_cases("test")
    assert [c["id"] for c in cases] == plan["case_ids"]
    if resume:
        report["resume_count"] += 1
    report["dependency_environment"] = environment
    addendum_sha = sha(destination / ADDENDUM)
    if resume and report.get("execution_addendum_sha256") != addendum_sha:
        raise RuntimeError("Never import another authorization route or reset its budget")
    report["execution_addendum_sha256"] = addendum_sha
    report["compute_authorization"] = compute_authorization(addendum_sha)
    report["execution_budget"] = addendum["accounting"]
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
        provider = BudgetProvider(
            TorchProvider(model, tokenizer, config, arm, adapter_sha),
            addendum_sha,
            addendum["accounting"]["process_deadline_utc"],
        )
        store = ResolutionStore(destination / "sandboxes" / f"{arm}.sqlite")
        for case in remaining:
            report["pending_case"] = {
                "variant": arm,
                "case_id": case["id"],
                "started_at_utc": datetime.now(UTC).isoformat(),
            }
            write_atomic(destination / "evaluation-partial.json", report)
            recorded = BudgetRecordingProvider(provider, journal, arm, case["id"])
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
    verify_addendum(destination)
    report["summary"] = completed_summary(report["records"])
    report["call_journal_sha256"] = sha(destination / "call-journal.json")
    protocol = protocol_report(report["records"], cases)
    protocol.update(
        run_id=RUN_ID,
        execution_addendum_sha256=addendum_sha,
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


def execute(
    destination,
    base_path,
    initializer_path,
    training_report,
    adapter_directory,
    *,
    existing_units_confirmed=False,
    resume=False,
):
    if not existing_units_confirmed:
        raise PermissionError(
            "Explicit --existing-units-confirmed is required; never label metered GPU as free"
        )
    destination = report_directory(destination)
    addendum = verify_addendum(destination, require_not_expired=True)
    hardware = probe_t4()
    marker = destination / "budgeted-run-start.json"
    if not resume:
        with marker.open("x") as output:
            json.dump(
                {
                    "run_id": RUN_ID,
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "execution_addendum_sha256": sha(destination / ADDENDUM),
                    "hardware": hardware,
                },
                output,
            )
    else:
        assert json.loads(marker.read_text())["execution_addendum_sha256"] == sha(destination / ADDENDUM)
    seconds = (utc(addendum["accounting"]["process_deadline_utc"]) - datetime.now(UTC)).total_seconds()
    if seconds <= 0:
        raise TimeoutError("Deadline expired before process watchdog could start")
    watchdog = threading.Timer(seconds, deadline_callback(destination, sha(destination / ADDENDUM)))
    watchdog.daemon = True
    watchdog.start()
    try:
        return _evaluate(
            destination,
            base_path,
            initializer_path,
            training_report,
            adapter_directory,
            resume=resume,
            addendum=addendum,
        )
    finally:
        watchdog.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="audit", choices=["audit", "authorize", "run"])
    parser.add_argument("--output", type=Path, default=REPORTS)
    parser.add_argument("--base", type=Path, default=ROOT / "data/models/qwen3-4b-mlx")
    parser.add_argument("--initializer", type=Path, default=ROOT / "data/finetuning/retail-qlora-v6_1/best")
    parser.add_argument(
        "--training-report", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/report.json"
    )
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "reports/cuda-torch/probe-003/train-1344/adapter"
    )
    parser.add_argument("--existing-units-confirmed", action="store_true")
    parser.add_argument("--authorization-text")
    parser.add_argument("--process-deadline-utc")
    parser.add_argument("--runtime-allocated-at-utc")
    parser.add_argument("--runtime-disconnect-deadline-utc")
    parser.add_argument("--observed-units-per-hour", type=float)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "authorize":
        if any(
            v is None
            for v in [
                args.authorization_text,
                args.process_deadline_utc,
                args.runtime_allocated_at_utc,
                args.runtime_disconnect_deadline_utc,
                args.observed_units_per_hour,
            ]
        ):
            parser.error(
                "authorize requires explicit human text, allocation/deadline/disconnect UTC and observed rate"
            )
        result = authorize(
            args.output,
            existing_units_confirmed=args.existing_units_confirmed,
            authorization_text=args.authorization_text,
            process_deadline_utc=args.process_deadline_utc,
            runtime_allocated_at_utc=args.runtime_allocated_at_utc,
            runtime_disconnect_deadline_utc=args.runtime_disconnect_deadline_utc,
            observed_units_per_hour=args.observed_units_per_hour,
        )
    elif args.mode == "run":
        result = execute(
            args.output,
            args.base,
            args.initializer,
            args.training_report,
            args.adapter,
            existing_units_confirmed=args.existing_units_confirmed,
            resume=args.resume,
        )
    else:
        result = verify_addendum(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
