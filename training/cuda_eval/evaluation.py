"""Pre-inference protocol and CPU helpers for the new matched two-arm comparison."""

import copy
import hashlib
import json
import math
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from training.common import ROOT
from training.cuda_eval.provider import MODEL_KWARGS, generation_settings
from training.cuda_torch.model import INITIALIZER_SHA256, adapter_plan, read_header, sha
from training.v7.common import verify_frozen
from training.v7.evaluation import summarize
from training.v7.protocol_audit import COUNTS, audit_record

ARMS = ("predecessor_fp16", "cloud1344_fp16")
MAX_CALLS = 12
REPORTS = ROOT / "reports/cuda-eval"
SCOPE = (
    "New 48-case authored synthetic comparison: exact V6.1 predecessor versus fixed final cloud1344 "
    "continuation on the SAME decoded-4bit-derived FP16 base, F32 PEFT layout and frozen V7 workflow. "
    "Not base-versus-finetune, not a Metal-versus-CUDA comparison or official production benchmark. "
    "The final1344 checkpoint is predeclared, never selected using these outputs."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_names():
    return [
        "training/cuda_eval/__init__.py",
        "training/cuda_eval/fixtures.py",
        "training/cuda_eval/prepare.py",
        "training/cuda_eval/provider.py",
        "training/cuda_eval/evaluation.py",
        "training/cuda_eval/run.py",
        "tests/test_cuda_eval_data.py",
        "tests/test_cuda_eval_protocol.py",
        "training/cuda_eval/datasets/cases.json",
        "training/cuda_eval/datasets/public.json",
        "reports/cuda-eval/data-summary.json",
        "reports/cuda-eval/reference-audit.json",
        "training/cuda_torch/model.py",
        "training/cuda_torch/train.py",
        "training/base-model-spec.json",
        "training/v7/config.json",
        "training/v7/datasets/manifest.json",
        "training/v7/evaluation.py",
        "training/v7/scoring.py",
        "training/v7/protocol_audit.py",
        "training/v6/scoring.py",
        "training/v4/evaluation.py",
        "resolve_desk/resolution_v7.py",
        "resolve_desk/store.py",
        "resolve_desk/protocol.py",
        "resolve_desk/progress.py",
        "resolve_desk/workflow.py",
    ]


def report_directory(destination):
    root = (ROOT / "reports/cuda-eval").resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("CUDA comparison outputs must remain below reports/cuda-eval")
    return destination


def create_plan(destination=REPORTS):
    from training.cuda_eval.fixtures import load_cases

    destination = report_directory(destination)
    path = destination / "analysis-plan.json"
    if (
        path.exists()
        or (destination / "evaluation-partial.json").exists()
        or (destination / "evaluation.json").exists()
    ):
        raise FileExistsError("Never overwrite a pre-inference plan or evaluation")
    verify_frozen()
    cases = load_cases("test")
    if len(cases) != 48 or len({c["id"] for c in cases}) != 48:
        raise ValueError("Expected exactly48 unique cases")
    assert len(cases) == 48 and len({c["initial_message"] for c in cases}) == 48
    assert Counter(c["family"] for c in cases) == {c["family"]: 2 for c in cases}
    base_config = json.loads((ROOT / "data/models/qwen3-4b-mlx/config.json").read_text())
    source_inputs = {name: sha(ROOT / name) for name in source_names()}
    plan = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "test_inference_started": False,
        "arms": list(ARMS),
        "case_ids": [c["id"] for c in cases],
        "case_order": "Frozen dataset order within predecessor48, then cloud1344 same48",
        "source_inputs_sha256": source_inputs,
        "cases_sha256": source_inputs["training/cuda_eval/datasets/cases.json"],
        "frozen_v7_manifest_sha256": source_inputs["training/v7/datasets/manifest.json"],
        "total_calls_per_task": MAX_CALLS,
        "generation": generation_settings(base_config),
        "inference_model_kwargs": dict(MODEL_KWARGS),
        "chat_template": {"enable_thinking": False, "add_generation_prompt": True, "truncation": False},
        "raw_output": (
            "All token IDs retained; only one terminal EOS removed; no other special tokens/JSON repaired"
        ),
        "after_checkpoint": {
            "completed_updates": 1344,
            "selection": "Fixed final1344,672backup never evaluated/selected",
        },
        "before_initializer_sha256": INITIALIZER_SHA256,
        "artifact_identity": (
            "Separately sealed after completed1344 artifact exists, before ANY model generation"
        ),
        "model_lifecycle": "Load one base, before48, overwrite only112 LoRA tensors, after48; no warmups",
        "compute": (
            "FP16 frozen base, F32 PEFT adapter masters/default adapter compute; no autocast, "
            "model.eval/inference_mode, gradient checkpointing disabled, KV cache per call"
        ),
        "resume_policy": (
            "Completed case boundaries only. Any pending/ambiguous case refuses automatic replay."
        ),
        "primary_metrics": [
            "Frozen V7 task_success and required answers",
            "First raw seven fields/parse/enums",
            "Permission recall and unsupported yes",
            "Exchange/fallback/handoff/email cohorts",
            "Repeated field attempts/deliveries and identical action repeats separately",
            "All raw calls/proposals/commits/final orders in scoped-consent audit",
        ],
        "protocol_limitations": (
            "Frozen operation-rewrite bypass remains; audit detects exposure, never repairs it. "
            "Initial direct returns and unrestricted semantic operation changes are outside that audit."
        ),
        "quality_scope": (
            "Authored nonrandom24families; no population CI/significance/production "
            "or individual-training-choice attribution"
        ),
        "timing_scope": (
            "Serial fixed-arm order; actual GPU-sync model generation excludes load/swap, "
            "task time includes journaling; later prompts may diverge by actions; no hardware-only speedup"
        ),
        "reporting_helpers": (
            "Mutable reporting/replay/plot helpers do not alter bound model inputs, "
            "primary frozen scores or checkpoint choice"
        ),
    }
    write_atomic(path, plan)
    return plan


def verify_plan(destination=REPORTS):
    path = report_directory(destination) / "analysis-plan.json"
    plan = json.loads(path.read_text())
    verify_frozen()
    assert not plan["test_inference_started"] and tuple(plan["arms"]) == ARMS
    assert plan["total_calls_per_task"] == 12 and plan["generation"]["max_new_tokens"] == 512
    for name, wanted in plan["source_inputs_sha256"].items():
        if sha(ROOT / name) != wanted:
            raise ValueError(f"Pre-inference source/data changed: {name}")
    return plan


def peft_mapping(header, specs, expected_tensors=112, expected_parameters=3_670_016):
    """Saved PEFT strips only adapter-name '.default'; strict112 canonical F32 tensors."""
    result = []
    for target, shape in specs.items():
        if not target.endswith((".lora_A.default.weight", ".lora_B.default.weight")):
            raise ValueError("Only canonical default LoRA A/B tensors may be overwritten")
        source = target.replace(".default.weight", ".weight")
        if (
            source not in header
            or header[source]["dtype"] != "F32"
            or tuple(header[source]["shape"]) != tuple(shape)
        ):
            raise ValueError(f"Saved final PEFT tensor mismatch: {source}")
        result.append({"source": source, "target": target, "shape": tuple(shape)})
    if set(header) != {r["source"] for r in result}:
        raise ValueError("Unexpected missing/extra final adapter tensors")
    if len(result) != expected_tensors or sum(math.prod(r["shape"]) for r in result) != expected_parameters:
        raise ValueError("Final adapter must be exactly112 tensors /3,670,016 parameters")
    return result


def seal_artifacts(training_report, adapter_directory, base_path, initializer_path, destination=REPORTS):
    destination = report_directory(destination)
    plan = verify_plan(destination)
    target = destination / "artifact-identity.json"
    if target.exists() or (destination / "evaluation-partial.json").exists():
        raise FileExistsError("Never replace sealed weights or any started evaluation")
    training_report, adapter_directory = Path(training_report), Path(adapter_directory)
    report = json.loads(training_report.read_text())
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
    assert sha(after) == report["final_adapter"]["sha256"] and sha(before) == INITIALIZER_SHA256
    raw_config = json.loads((Path(base_path) / "config.json").read_text())
    original = adapter_plan(read_header(before), raw_config)
    specs = {r["target"]: r["target_shape"] for r in original}
    final_mapping = peft_mapping(read_header(after), specs)
    adapter_config = json.loads((adapter_directory / "adapter_config.json").read_text())
    assert adapter_config["r"] == 8 and adapter_config["lora_alpha"] == 128
    assert adapter_config["lora_dropout"] == 0 and not adapter_config.get("use_rslora", False)
    assert set(adapter_config["target_modules"]) == {r["module"] for r in original}
    # Check all final F32 values before spending any model calls on either arm.
    import numpy as np
    from safetensors.numpy import load_file

    final_values = load_file(str(after))
    assert set(final_values) == {r["source"] for r in final_mapping}
    assert all(
        value.dtype == np.float32 and bool(np.isfinite(value).all()) for value in final_values.values()
    )
    del final_values
    identity = {
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": sha(destination / "analysis-plan.json"),
        "training_report_sha256": sha(training_report),
        "training_plan_sha256": sha(training_plan),
        "completed_true_optimizer_updates": 1344,
        "before_adapter_sha256": sha(before),
        "after_adapter_sha256": sha(after),
        "after_adapter_config_sha256": sha(adapter_directory / "adapter_config.json"),
        "base_spec_sha256": sha(ROOT / "training/base-model-spec.json"),
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


def reconcile_journal(entries, records):
    """Before resuming, every issued call must belong to an already complete case."""
    expected = [(r["variant"], r["case_id"], i + 1, o) for r in records for i, o in enumerate(r["trace"])]
    if len(entries) != len(expected):
        raise RuntimeError("Orphan/ambiguous calls cannot be retried or dropped")
    for entry, (variant, case_id, index, observation) in zip(entries, expected, strict=True):
        if (
            entry["status"] != "completed"
            or (entry["variant"], entry["case_id"], entry["call_index"]) != (variant, case_id, index)
            or entry["raw"] != observation["raw"]
            or entry["timing"] != observation["timing"]
        ):
            raise RuntimeError("Call journal differs from retained completed-case evidence")


def install_final_adapter(model, directory, identity):
    import torch
    from safetensors import safe_open

    file = Path(directory) / "adapter_model.safetensors"
    assert sha(file) == identity["after_adapter_sha256"]
    parameters = dict(model.named_parameters())
    specs = {
        name: tuple(p.shape)
        for name, p in parameters.items()
        if name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
    }
    mapping = peft_mapping(read_header(file), specs)
    with torch.no_grad(), safe_open(str(file), framework="pt", device="cpu") as source:
        for item in mapping:
            before = source.get_tensor(item["source"])
            target = parameters[item["target"]]
            if target.dtype != torch.float32 or not bool(torch.isfinite(before).all()):
                raise ValueError("Final LoRA must remain finite F32")
            target.copy_(before.to(target.device))
            if not torch.equal(target.detach().cpu(), before):
                raise ValueError("Final adapter numeric equality failed")
    model.eval()
    return {
        "source_sha256": sha(file),
        "exact_tensor_equality": True,
        "source_tensor_count": len(mapping),
        "source_parameter_count": sum(math.prod(r["shape"]) for r in mapping),
        "base_reloaded": False,
        "only_lora_tensors_overwritten": True,
    }


class CallJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def start(self, variant, case_id, call_index, prompt):
        entry = {
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

    def finish(self, entry, raw, timing):
        entry.update(
            status="completed",
            raw=raw,
            timing=copy.deepcopy(timing),
            completed_at_utc=datetime.now(UTC).isoformat(),
        )
        write_atomic(self.path, self.entries)


def protocol_report(records, cases):
    by_id = {c["id"]: c for c in cases}
    rows = [audit_record(r, by_id[r["case_id"]]) for r in records]
    summaries = {}
    for arm in ARMS:
        selected = [r for r in rows if r["variant"] == arm]
        summaries[arm] = {
            "tasks": len(selected),
            "gold_exchange_tasks": sum(r["audited"] for r in selected),
            **{key: sum(r[key] for r in selected) for key in COUNTS},
        }
    return {
        "scope": "Frozen CPU audit of actual scoped questions/YES/proposals/commits on exchange-origin tasks",
        "limitations": (
            "Known operation-rewrite bypass remains; findings never repair frozen controller "
            "or change business metrics"
        ),
        "records_audited": len(rows),
        "business_scores_or_release_gate_modified": False,
        "selection_rule_modified": False,
        "summary": summaries,
        "records": rows,
    }


def completed_summary(records):
    return {arm: summarize([r for r in records if r["variant"] == arm]) for arm in ARMS}
