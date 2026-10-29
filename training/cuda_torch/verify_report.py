"""Independent CPU-only reconciliation of a separate CUDA training artifact; no Torch import."""

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from training.common import ROOT

PARAMETERS, TENSORS = 3_670_016, 112
INITIALIZER_SHA = "041cde85d1593f27f154da5b4afbec2f0635fb2bcfe4a34940339e092925ce08"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def finite(value, *, positive=False):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and (value > 0 if positive else value >= 0)


def expected_schedule(rows, seed, steps):
    buckets = {
        "intent_and_record_repair": [i for i, row in enumerate(rows) if row["origin"] == "record_request"],
        "execution_and_reply": [i for i, row in enumerate(rows) if row["origin"] != "record_request"],
    }
    assert all(buckets.values())
    rng, pending, sequence = random.Random(seed), {key: [] for key in buckets}, []
    for step in range(1, steps + 1):
        bucket = "intent_and_record_repair" if step % 2 else "execution_and_reply"
        if not pending[bucket]:
            pending[bucket] = list(buckets[bucket])
            rng.shuffle(pending[bucket])
        sequence.append({"iteration": step, "bucket": bucket, "sample_index": pending[bucket].pop()})
    return sequence


def sampling(rows, schedule):
    indices = [entry["sample_index"] for entry in schedule]
    return {
        "physical_rows": len(rows),
        "unique_rows_sampled": len(set(indices)),
        "unique_cases_sampled": len({rows[index]["case_id"] for index in indices}),
        "all_physical_rows_used": len(set(indices)) == len(rows),
        "bucket_rows": {
            "intent_and_record_repair": sum(row["origin"] == "record_request" for row in rows),
            "execution_and_reply": sum(row["origin"] != "record_request" for row in rows),
        },
        "bucket_updates": dict(Counter(entry["bucket"] for entry in schedule)),
    }


def reconcile(report, plan, rows):
    """Check actual completed-update evidence, not only the report's completion flag."""
    assert report["status"] == "completed"
    assert report["kind"] == plan["kind"] and report["source_inputs_sha256"] == plan["source_inputs_sha256"]
    assert report["initializer_sha256"] == plan["initializer_sha256"] == INITIALIZER_SHA
    assert report["test_used_for_selection"] is False and report["new_accuracy_claim"] is False
    assert report["cloud_compute_used"] is True and report["smoke_weights_reused"] is False
    assert report["smoke_weights_discarded"] is True and plan["fresh_predecessor_initialization"] is True
    config = plan["frozen_config"]
    count = plan["planned_true_optimizer_updates"]
    sequence = expected_schedule(rows, config["seed"], count)
    assert plan["planned_schedule"] == sequence
    assert plan["planned_sampling"] == report["sampling"] == sampling(rows, sequence)
    full = expected_schedule(rows, config["seed"], 1344)
    assert plan["full_1344_sampling"] == sampling(rows, full)
    assert report["completed_true_optimizer_updates"] == count == len(report["metrics"])
    assert report["all_gradients_finite"] is True
    assert not report.get("attempt_in_progress")
    assert report["dependency_environment"] == plan["dependency_environment"]
    dependencies = plan["dependency_environment"]
    assert dependencies["required_packages"] == plan["package_pins"] == report["environment"]["versions"]
    torchao = dependencies["torchao"]
    assert torchao["distribution_version"] is None and torchao["module_discoverable"] is False
    assert torchao["module_probe_error"] is None and torchao["in_sys_modules"] is False
    assert report["environment"]["python"] == dependencies["python"]
    audit = report["initializer_audit"]
    assert audit["exact_tensor_equality"] is True
    assert audit["source_tensor_count"] == TENSORS
    assert audit["source_parameter_count"] == audit["trainable_parameters"] == PARAMETERS
    assert audit["initializer_sha256"] == INITIALIZER_SHA
    assert audit["lora_rank"] == 8 and audit["mlx_direct_scale"] == 16 and audit["peft_lora_alpha"] == 128
    assert audit["adapter_master_dtype"] == "float32" and audit["base_storage_dtype"] == "float16"
    assert audit["tied_embedding_head"] is True and audit["not_qlora_or_nf4"] is True
    assert audit["attention_backend"] == "sdpa" and audit["all_parameters_and_buffers_materialized"] is True
    assert report["token_audit"]["all_physical_rows_checked"] == len(rows)
    assert report["token_audit"]["all_targets_covered"] is True
    assert report["token_audit"]["full_input_context_retained"] is True
    assert report["token_audit"]["projection_window"] == plan["compute_config"]["projection_window"] == 192
    assert report["token_audit"]["mlx_projection_window"] == 512
    assert report["token_audit"]["maximum_tail_including_padding"] <= 192
    for key, value in [
        ("ordinary_completion_weight", 1),
        ("permission_value_weight", 8),
        ("reason_value_weight", 4),
        ("prompt_and_padding_weight", 0),
    ]:
        assert report["token_audit"][key] == value
    previous_elapsed = 0
    for actual, planned in zip(report["metrics"], sequence, strict=True):
        index = planned["sample_index"]
        assert actual["iteration"] == actual["true_optimizer_updates"] == planned["iteration"]
        assert actual["sample_index"] == index and actual["sampling_bucket"] == planned["bucket"]
        assert actual["case_id"] == rows[index]["case_id"] and actual["tool"] == rows[index]["origin"]
        assert actual["gradients_finite"] is True
        for key in [
            "unscaled_gradient_norm",
            "grad_scaler_scale",
            "adapter_l2_change_from_initializer",
            "elapsed_gpu_sync_seconds",
            "elapsed_loop_gpu_sync_seconds",
        ]:
            finite(actual[key], positive=True)
        finite(actual["loss"])
        assert actual["elapsed_loop_gpu_sync_seconds"] > previous_elapsed
        previous_elapsed = actual["elapsed_loop_gpu_sync_seconds"]
        assert actual["peak_cuda_reserved_bytes"] >= actual["peak_cuda_allocated_bytes"] > 0
    finite(report["adapter_l2_change"], positive=True)
    assert math.isclose(
        report["metrics"][-1]["adapter_l2_change_from_initializer"],
        report["adapter_l2_change"],
        rel_tol=1e-4,
        abs_tol=1e-8,
    )
    assert report["elapsed_loop_gpu_sync_seconds"] >= previous_elapsed
    assert report["peak_cuda_allocated_bytes"] >= max(
        m["peak_cuda_allocated_bytes"] for m in report["metrics"]
    )
    assert report["peak_cuda_reserved_bytes"] >= max(m["peak_cuda_reserved_bytes"] for m in report["metrics"])
    assert report["peak_cuda_reserved_bytes"] >= report["peak_cuda_allocated_bytes"]
    assert [candidate["iteration"] for candidate in report["candidates"]] == plan["checkpoints"]
    if plan["kind"] == "smoke":
        assert 1 <= count <= 4 and not plan["checkpoints"]
    else:
        assert count == 1344 and report["sampling"]["all_physical_rows_used"]
        assert report["discarded_smoke_receipt"]["weights_reused"] is False
    later = [entry["elapsed_gpu_sync_seconds"] for entry in report["metrics"][1:]]
    return {
        "completed_true_optimizer_updates": count,
        "actual_sampling": report["sampling"],
        "recorded_initializer_exact_equality": True,
        "recorded_initializer_tensor_count": TENSORS,
        "recorded_trainable_parameters": PARAMETERS,
        "timing": {
            "first_update_seconds": report["metrics"][0]["elapsed_gpu_sync_seconds"],
            "steps_2_through_n": {
                "updates": len(later),
                "median_seconds": statistics.median(later) if later else None,
                "mean_seconds": statistics.mean(later) if later else None,
                "minimum_seconds": min(later) if later else None,
                "maximum_seconds": max(later) if later else None,
            },
            "synchronized_run_seconds": report["elapsed_loop_gpu_sync_seconds"],
            "scope": report["timing_scope"],
            "full_run_throughput_established_by_smoke": False,
        },
        "memory": {
            "peak_allocated_bytes": report["peak_cuda_allocated_bytes"],
            "peak_reserved_bytes": report["peak_cuda_reserved_bytes"],
            "peak_allocated_gib": report["peak_cuda_allocated_bytes"] / 1024**3,
            "peak_reserved_gib": report["peak_cuda_reserved_bytes"] / 1024**3,
            "scope": "CUDA allocator counters include the loaded model; distinct from total GPU/OS memory.",
        },
        "accuracy_validated": False,
        "scope": (
            "CPU artifact consistency verification; four smoke updates do not establish "
            "full-run stability, accuracy or isolated hardware speedup."
        ),
    }


def expected_peft_keys(source):
    expected = {}
    for name, value in source.items():
        if name.endswith(".lora_a"):
            stem, suffix = name.removesuffix(".lora_a"), "lora_A"
        elif name.endswith(".lora_b"):
            stem, suffix = name.removesuffix(".lora_b"), "lora_B"
        else:
            raise AssertionError(f"Unexpected predecessor tensor key: {name}")
        expected[f"base_model.model.{stem}.{suffix}.weight"] = value.T
    return expected


def audit_tensors(report_path, report, plan, *, require=False):
    folder = report_path.parent
    paths = {
        "predecessor": ROOT / plan["frozen_config"]["initialization"] / "adapters.safetensors",
        "saved_initial": folder / "initial-peft.safetensors",
        "final_adapter": folder / "adapter/adapter_model.safetensors",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        assert not require, ("Missing tensor evidence", missing)
        return {"available": False, "missing_files": missing, "numeric_values_independently_checked": False}
    import numpy as np
    from safetensors.numpy import load_file

    assert digest(paths["predecessor"]) == INITIALIZER_SHA
    assert digest(paths["saved_initial"]) == report["initial_peft_sha256"]
    assert digest(paths["final_adapter"]) == report["final_adapter"]["sha256"]
    expected = expected_peft_keys(load_file(str(paths["predecessor"])))
    raw_initial = load_file(str(paths["saved_initial"]))
    initial = {name.replace(".default.weight", ".weight"): value for name, value in raw_initial.items()}
    final = load_file(str(paths["final_adapter"]))
    assert len(raw_initial) == len(initial) == len(final) == len(expected) == TENSORS
    assert set(initial) == set(final) == set(expected)
    assert sum(value.size for value in expected.values()) == PARAMETERS
    squared = []
    for name, value in expected.items():
        assert initial[name].shape == final[name].shape == value.shape
        assert initial[name].dtype == final[name].dtype == value.dtype == np.dtype("float32")
        assert all(np.isfinite(array).all() for array in [initial[name], final[name], value])
        assert np.array_equal(initial[name], value), name
        squared.append(float(np.square(final[name].astype(np.float64) - value.astype(np.float64)).sum()))
    delta = math.sqrt(math.fsum(squared))
    assert delta > 0 and math.isclose(delta, report["adapter_l2_change"], rel_tol=1e-6, abs_tol=1e-10)
    return {
        "available": True,
        "numeric_values_independently_checked": True,
        "tensor_count": TENSORS,
        "parameter_count": PARAMETERS,
        "predecessor_to_saved_initial_exact_after_transpose": True,
        "all_values_finite": True,
        "independent_final_adapter_l2_change": delta,
        "files_sha256": {name: digest(path) for name, path in paths.items()},
        "scope": "CPU NumPy/safetensors tensor reads only; no model loaded and no tensor contents exported.",
    }


def verify(report_path, *, require_tensors=False):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    plan_path = report_path.with_name("plan.json")
    plan = json.loads(plan_path.read_text())
    assert report["plan_sha256"] == digest(plan_path)
    assert set(plan["source_inputs_sha256"]) == {
        "training/cuda_torch/train.py",
        "training/cuda_torch/model.py",
        "training/v7/config.json",
        "training/v7/tokenization.py",
        "training/v7/train.py",
        "training/v4/tokenization.py",
        "training/base-model-spec.json",
        "training/v7/datasets/manifest.json",
        "training/v7/datasets/train.jsonl",
        "training/v7/datasets/cases.json",
    }
    for name, wanted in plan["source_inputs_sha256"].items():
        source = Path(name)
        assert not source.is_absolute() and ".." not in source.parts
        assert digest(ROOT / source) == wanted, name
    assert plan["manifest_sha256"] == digest(ROOT / "training/v7/datasets/manifest.json")
    manifest = json.loads((ROOT / "training/v7/datasets/manifest.json").read_text())
    assert plan["frozen_manifest"] == manifest
    for name, wanted in manifest["files"].items():
        assert digest(ROOT / name) == wanted, name
    assert plan["frozen_config"] == json.loads((ROOT / "training/v7/config.json").read_text())
    assert plan["base_spec"] == json.loads((ROOT / "training/base-model-spec.json").read_text())
    initializer = ROOT / plan["frozen_config"]["initialization"] / "adapters.safetensors"
    if initializer.is_file():
        assert digest(initializer) == plan["initializer_sha256"] == INITIALIZER_SHA
    rows = [json.loads(line) for line in (ROOT / "training/v7/datasets/train.jsonl").read_text().splitlines()]
    assert len(rows) == len({json.dumps(row["messages"], sort_keys=True) for row in rows}) == 824
    result = reconcile(report, plan, rows)
    result["tensor_audit"] = audit_tensors(report_path, report, plan, require=require_tensors)
    result.update(
        report_sha256=digest(report_path),
        plan_sha256=digest(plan_path),
        verifier_sha256=digest(__file__),
        source_inputs_reconciled=len(plan["source_inputs_sha256"]),
        source_plan_and_training_files_modified=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-tensors", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.report, require_tensors=args.require_tensors)
    output = args.output or args.report.with_name("verification.json")
    if output.exists():
        raise FileExistsError("Never overwrite a verification receipt; choose a distinct --output")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
