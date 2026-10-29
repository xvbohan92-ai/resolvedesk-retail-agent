"""CPU-only numeric audit before any V8 model inference or candidate identity seal."""

import json
import math
import random
from collections import Counter
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.cuda_eval.evaluation import peft_mapping, write_atomic
from training.cuda_torch.model import read_header
from training.v8.common import CONFIG, DATA, read_rows, verify_frozen

REPORTS = ROOT / "reports/v8"


def verify_history(report, plan, config, expected_steps):
    """Validate the entire recorded optimization history without claiming to recompute gradients."""
    if report.get("kind") != plan.get("kind") or report["kind"] not in {"smoke", "train"}:
        raise ValueError("Wrong training phase identity")
    if (
        plan.get("planned_true_optimizer_updates") != expected_steps
        or report.get("completed_true_optimizer_updates") != expected_steps
    ):
        raise ValueError("Incomplete planned true optimizer updates")
    if (
        report.get("source_inputs_sha256") != plan.get("source_inputs_sha256")
        or plan.get("frozen_config") != config
    ):
        raise ValueError("Recorded source or optimizer configuration differs")
    if (
        report.get("smoke_weights_discarded") is not True
        or report.get("smoke_weights_reused") is not False
        or report.get("test_used_for_selection") is not False
        or plan.get("test_used_for_selection") is not False
        or plan.get("checkpoints") != []
        or report.get("candidates") != []
    ):
        raise ValueError("Smoke reuse, selection or alternative checkpoint found")
    metrics = report["metrics"]
    if len(metrics) != expected_steps or report.get("all_gradients_finite") is not True:
        raise ValueError("Incomplete or nonfinite recorded training history")
    previous_time, previous_scale = 0.0, 128.0
    for step, m in enumerate(metrics, 1):
        fields = (
            "loss",
            "unscaled_gradient_norm",
            "adapter_l2_change_from_initializer",
            "grad_scaler_scale",
            "elapsed_gpu_sync_seconds",
            "elapsed_loop_gpu_sync_seconds",
        )
        if any(not isinstance(m.get(k), (int, float)) or not math.isfinite(m[k]) for k in fields):
            raise ValueError("Nonfinite or missing numeric optimizer evidence")
        if (
            m["iteration"] != step
            or m["true_optimizer_updates"] != step
            or m["gradients_finite"] is not True
            or m["loss"] < 0
            or m["unscaled_gradient_norm"] <= 0
            or m["adapter_l2_change_from_initializer"] <= 0
            or m["elapsed_gpu_sync_seconds"] <= 0
            or m["elapsed_loop_gpu_sync_seconds"] < previous_time
            or m["grad_scaler_scale"] < previous_scale
        ):
            raise ValueError("Skipped, overflowed or invalid recorded optimizer update")
        if any(type(m.get(k)) is not int for k in ("peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes")):
            raise ValueError("CUDA memory counters must be finite integer byte counts")
        if not 0 < m["peak_cuda_allocated_bytes"] <= m["peak_cuda_reserved_bytes"]:
            raise ValueError("Impossible recorded CUDA memory usage")
        previous_time, previous_scale = m["elapsed_loop_gpu_sync_seconds"], m["grad_scaler_scale"]
    if [(m["iteration"], m["sample_index"], m["sampling_bucket"]) for m in metrics] != [
        (m["iteration"], m["sample_index"], m["bucket"]) for m in plan["planned_schedule"]
    ]:
        raise ValueError("Actual optimizer samples differ from the frozen schedule")
    return {
        "recorded_updates_checked": expected_steps,
        "all_recorded_numeric_values_finite": True,
        "gradient_recomputed_by_cpu_audit": False,
        "recorded_scaler_never_decreased": True,
    }


def tensor_audit(before, initial, after):
    import numpy as np
    from safetensors import safe_open

    start_header, end_header = read_header(initial), read_header(after)
    initial_shapes = {k: tuple(v["shape"]) for k, v in start_header.items()}
    mapping = peft_mapping(read_header(before), initial_shapes)
    final_mapping = peft_mapping(end_header, initial_shapes)
    if len(mapping) != 112 or len(final_mapping) != 112:
        raise ValueError("Only the112 rank8 last-eight-layer adapter tensors are permitted")
    if any(v["dtype"] != "F32" for v in [*start_header.values(), *end_header.values()]):
        raise ValueError("Initial and final LoRA masters must remain finite F32")
    squares, total = 0.0, 0
    final_by_target = {m["target"]: m["source"] for m in final_mapping}
    with (
        safe_open(str(before), framework="numpy") as source,
        safe_open(str(initial), framework="numpy") as start,
        safe_open(str(after), framework="numpy") as end,
    ):
        for item in mapping:
            actual = start.get_tensor(item["target"])
            expected = source.get_tensor(item["source"])
            final = end.get_tensor(final_by_target[item["target"]])
            if not np.array_equal(actual, expected):
                raise ValueError("Formal initialization does not exactly equal the actual prior adapter")
            if not all(np.isfinite(t).all() for t in (expected, actual, final)):
                raise ValueError("Nonfinite saved adapter value")
            total += actual.size
            squares += float(np.square(final.astype(np.float64) - actual.astype(np.float64)).sum())
    change = math.sqrt(squares)
    if total != 3_670_016 or not math.isfinite(change) or change <= 0:
        raise ValueError("The actual adapter must have3.67M parameters and a positive finite update")
    return {
        "tensors_checked": 112,
        "parameters_checked": total,
        "exact_prior_initialization": True,
        "all_values_finite": True,
        "adapter_l2_change": change,
        "cpu_only": True,
    }


def recompute_data_audit(config, report, plan):
    """Independently reconstruct schedule/coverage and check full-context weighted target tails."""
    from transformers import AutoTokenizer

    from training.v7.tokenization import encode_row

    rows = read_rows("train")
    pools = {
        "intent_and_record_repair": [i for i, r in enumerate(rows) if r["origin"] == "record_request"],
        "execution_and_reply": [i for i, r in enumerate(rows) if r["origin"] != "record_request"],
    }
    rng, cycles, schedule = random.Random(config["seed"]), {k: [] for k in pools}, []
    for step in range(1, 1345):
        bucket = "intent_and_record_repair" if step % 2 else "execution_and_reply"
        if not cycles[bucket]:
            cycles[bucket] = list(pools[bucket])
            rng.shuffle(cycles[bucket])
        schedule.append({"iteration": step, "bucket": bucket, "sample_index": cycles[bucket].pop()})
    if schedule != plan["planned_schedule"]:
        raise ValueError("Frozen optimizer schedule differs from independently reconstructed sampling")
    indices = [m["sample_index"] for m in schedule]
    coverage = {
        "physical_rows": len(rows),
        "unique_rows_sampled": len(set(indices)),
        "unique_cases_sampled": len({rows[i]["case_id"] for i in indices}),
        "all_physical_rows_used": len(set(indices)) == len(rows),
        "bucket_rows": {k: len(v) for k, v in pools.items()},
        "bucket_updates": dict(Counter(m["bucket"] for m in schedule)),
    }
    if coverage != plan["planned_sampling"] or coverage != report["sampling"]:
        raise ValueError("Reported training coverage differs from the actual fixed samples")
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / config["model"], local_files_only=True, trust_remote_code=False
    )
    totals = {
        "longest_sequence": 0,
        "longest_completion": 0,
        "maximum_tail_including_padding": 0,
        "maximum_padding_tokens": 0,
        "permission_value_tokens": 0,
        "reason_value_tokens": 0,
    }
    for row in rows:
        e = encode_row(row, tokenizer, 8.0, 4.0)
        length = len(e["tokens"])
        if length > config["max_seq_length"] or len(e["weights"]) != length - 1:
            raise ValueError("A full training input would be truncated or misaligned")
        if any(e["weights"][: e["prompt_length"] - 1]):
            raise ValueError("A customer/system prompt is supervised")
        padded = 1 + 32 * math.ceil(length / 32)
        first = next(i for i, weight in enumerate(e["weights"]) if weight)
        tail = padded - 1 - first
        if tail > 192:
            raise ValueError("A supervised token would fall outside the192-token logit projection")
        for key, value in [
            ("longest_sequence", length),
            ("longest_completion", length - e["prompt_length"]),
            ("maximum_tail_including_padding", tail),
            ("maximum_padding_tokens", padded - length),
        ]:
            totals[key] = max(totals[key], value)
        totals["permission_value_tokens"] += e["permission_tokens"]
        totals["reason_value_tokens"] += e["reason_tokens"]
    recorded = report["token_audit"]
    for key, value in totals.items():
        if recorded[key] != value:
            raise ValueError("Actual training token audit differs: " + key)
    if (
        recorded["all_physical_rows_checked"] != len(rows)
        or recorded["full_input_context_retained"] is not True
        or recorded["all_targets_covered"] is not True
        or recorded["projection_window"] != 192
        or recorded["prompt_and_padding_weight"] != 0
        or recorded["permission_value_weight"] != 8
        or recorded["reason_value_weight"] != 4
    ):
        raise ValueError("Full-context/weighted-target claims are not supported")
    return {"sampling": coverage, "tokens": totals, "all_inputs_and_targets_retained": True}


def verify_training():
    manifest = verify_frozen()
    config = json.loads(CONFIG.read_text())
    folder = REPORTS / "cuda/train-1344"
    report, plan = [json.loads((folder / n).read_text()) for n in ("report.json", "plan.json")]
    if report.get("status") != "completed" or report.get("completed_true_optimizer_updates") != 1344:
        raise ValueError("Full1344 true updates must finish before candidate inference")
    if report["plan_sha256"] != sha(folder / "plan.json"):
        raise ValueError("Saved training plan changed")
    if plan["source_inputs_sha256"] != manifest["files"]:
        raise ValueError("Training did not use the frozen V8 inputs")
    if report["initializer_sha256"] != config["initialization_sha256"]:
        raise ValueError("Wrong training predecessor")
    if plan["compute_config"]["learning_rate"] != 2e-6 or plan["checkpoints"]:
        raise ValueError("Fixed-final optimizer configuration changed")
    history = verify_history(report, plan, config, 1344)
    if report["kind"] != "train":
        raise ValueError("Formal proof requires the actual fresh training phase")
    resource_path = REPORTS / "resource-authorization.json"
    if report.get("resource_receipt_sha256") != sha(resource_path):
        raise ValueError("The training resource receipt drifted")
    smoke_folder = REPORTS / "cuda/smoke-4"
    smoke = json.loads((smoke_folder / "report.json").read_text())
    smoke_plan = json.loads((smoke_folder / "plan.json").read_text())
    if (
        smoke.get("status") != "completed"
        or smoke.get("kind") != "smoke"
        or smoke["plan_sha256"] != sha(smoke_folder / "plan.json")
        or report["discarded_smoke_receipt"]["report_sha256"] != sha(smoke_folder / "report.json")
        or smoke_plan["source_inputs_sha256"] != manifest["files"]
        or smoke.get("resource_receipt_sha256") != sha(resource_path)
    ):
        raise ValueError("Wrong or incomplete discarded smoke provenance")
    smoke_history = verify_history(smoke, smoke_plan, config, 4)
    if (
        sha(smoke_folder / "initial-peft.safetensors") != smoke["initial_peft_sha256"]
        or sha(smoke_folder / "adapter/adapter_model.safetensors") != smoke["final_adapter"]["sha256"]
    ):
        raise ValueError("Discarded smoke physical weights differ from its receipt")
    data_audit = recompute_data_audit(config, report, plan)
    metrics = report["metrics"]
    if [m["true_optimizer_updates"] for m in metrics] != list(range(1, 1345)):
        raise ValueError("Missing or skipped real optimizer updates")
    if not report["all_gradients_finite"] or not all(m["gradients_finite"] for m in metrics):
        raise ValueError("Nonfinite gradient observed")
    before = ROOT / config["initialization"] / "adapter_model.safetensors"
    initial = folder / "initial-peft.safetensors"
    after = folder / "adapter/adapter_model.safetensors"
    if sha(before) != config["initialization_sha256"]:
        raise ValueError("Physical predecessor changed")
    if sha(initial) != report["initial_peft_sha256"] or sha(after) != report["final_adapter"]["sha256"]:
        raise ValueError("Saved physical weights do not match the training receipt")
    audit = tensor_audit(before, initial, after)
    smoke_audit = tensor_audit(
        before, smoke_folder / "initial-peft.safetensors", smoke_folder / "adapter/adapter_model.safetensors"
    )
    if not math.isclose(audit["adapter_l2_change"], report["adapter_l2_change"], rel_tol=1e-7, abs_tol=1e-10):
        raise ValueError("Saved-weight delta differs from the reported actual update")
    sample = plan["planned_schedule"]
    if [(m["iteration"], m["sample_index"], m["sampling_bucket"]) for m in metrics] != [
        (m["iteration"], m["sample_index"], m["bucket"]) for m in sample
    ]:
        raise ValueError("Actual optimizer samples differ from the frozen schedule")
    proof = {
        "round": "v8",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "training_report_sha256": sha(folder / "report.json"),
        "training_plan_sha256": sha(folder / "plan.json"),
        "initializer_sha256": sha(before),
        "initial_snapshot_sha256": sha(initial),
        "final_adapter_sha256": sha(after),
        "true_optimizer_updates": 1344,
        "fixed_final_only": True,
        "test_used_for_selection": False,
        "formal_recorded_history": history,
        "discarded_smoke_recorded_history": smoke_history,
        "discarded_smoke_physical_weight_audit": smoke_audit,
        "resource_receipt_sha256": sha(resource_path),
        "independent_data_audit": data_audit,
        "source_sha256": sha(__file__),
        **audit,
    }
    path = REPORTS / "training-proof.json"
    if path.exists():
        raise FileExistsError("Never replace an issued weight proof")
    write_atomic(path, proof)
    identity = {
        "round": "v8",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "analysis_plan_sha256": sha(REPORTS / "pre-inference-plan.json"),
        "manifest_sha256": sha(DATA / "manifest.json"),
        "training_report_sha256": sha(folder / "report.json"),
        "training_plan_sha256": sha(folder / "plan.json"),
        "training_proof_sha256": sha(path),
        "before_adapter_sha256": sha(before),
        "after_adapter_sha256": sha(after),
        "before_adapter_config_sha256": sha(before.with_name("adapter_config.json")),
        "after_adapter_config_sha256": sha(after.with_name("adapter_config.json")),
        "checkpoint_selection_performed": False,
    }
    path = REPORTS / "artifact-identity.json"
    if path.exists():
        raise FileExistsError("Never replace sealed model identities")
    write_atomic(path, identity)
    return proof


if __name__ == "__main__":
    print(json.dumps(verify_training()))
