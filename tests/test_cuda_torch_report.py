"""Small CPU-only artifact fixtures; no CUDA, Torch, model weights or experiment evidence."""

import copy

import pytest

from training.cuda_torch import verify_report as verifier


def fixture():
    rows = [
        {"origin": "record_request", "case_id": "cpu-a"},
        {"origin": "get_order", "case_id": "cpu-b"},
        {"origin": "record_request", "case_id": "cpu-c"},
    ]
    sequence = verifier.expected_schedule(rows, 202610287, 4)
    dependencies = {
        "required_packages": {"cpu-fixture": "0"},
        "python": "CPU fixture only",
        "torchao": {
            "distribution_version": None,
            "module_discoverable": False,
            "module_probe_error": None,
            "in_sys_modules": False,
        },
    }
    plan = {
        "kind": "smoke",
        "source_inputs_sha256": {"CPU fixture marker": "not a source hash"},
        "initializer_sha256": verifier.INITIALIZER_SHA,
        "fresh_predecessor_initialization": True,
        "frozen_config": {"seed": 202610287},
        "planned_true_optimizer_updates": 4,
        "planned_schedule": sequence,
        "planned_sampling": verifier.sampling(rows, sequence),
        "full_1344_sampling": verifier.sampling(rows, verifier.expected_schedule(rows, 202610287, 1344)),
        "dependency_environment": dependencies,
        "package_pins": dependencies["required_packages"],
        "compute_config": {"projection_window": 192},
        "checkpoints": [],
    }
    metrics = []
    for i, scheduled in enumerate(sequence, 1):
        index = scheduled["sample_index"]
        metrics.append(
            {
                "iteration": i,
                "true_optimizer_updates": i,
                "sample_index": index,
                "sampling_bucket": scheduled["bucket"],
                "case_id": rows[index]["case_id"],
                "tool": rows[index]["origin"],
                "gradients_finite": True,
                "loss": 0.1 / i,
                "unscaled_gradient_norm": 0.1,
                "grad_scaler_scale": 128,
                "adapter_l2_change_from_initializer": i * 0.01,
                "elapsed_gpu_sync_seconds": [10, 3, 2, 4][i - 1],
                "elapsed_loop_gpu_sync_seconds": [10, 13, 15, 19][i - 1],
                "peak_cuda_allocated_bytes": 100 + i,
                "peak_cuda_reserved_bytes": 200 + i,
            }
        )
    report = {
        "status": "completed",
        "kind": "smoke",
        "source_inputs_sha256": plan["source_inputs_sha256"],
        "initializer_sha256": verifier.INITIALIZER_SHA,
        "test_used_for_selection": False,
        "new_accuracy_claim": False,
        "cloud_compute_used": True,
        "smoke_weights_reused": False,
        "smoke_weights_discarded": True,
        "sampling": plan["planned_sampling"],
        "completed_true_optimizer_updates": 4,
        "metrics": metrics,
        "all_gradients_finite": True,
        "dependency_environment": dependencies,
        "environment": {"versions": dependencies["required_packages"], "python": dependencies["python"]},
        "initializer_audit": {
            "exact_tensor_equality": True,
            "source_tensor_count": 112,
            "source_parameter_count": 3670016,
            "trainable_parameters": 3670016,
            "initializer_sha256": verifier.INITIALIZER_SHA,
            "lora_rank": 8,
            "mlx_direct_scale": 16,
            "peft_lora_alpha": 128,
            "adapter_master_dtype": "float32",
            "base_storage_dtype": "float16",
            "tied_embedding_head": True,
            "not_qlora_or_nf4": True,
            "attention_backend": "sdpa",
            "all_parameters_and_buffers_materialized": True,
        },
        "token_audit": {
            "all_physical_rows_checked": 3,
            "all_targets_covered": True,
            "full_input_context_retained": True,
            "projection_window": 192,
            "mlx_projection_window": 512,
            "maximum_tail_including_padding": 141,
            "permission_value_weight": 8,
            "reason_value_weight": 4,
            "prompt_and_padding_weight": 0,
            "ordinary_completion_weight": 1,
        },
        "adapter_l2_change": 0.04,
        "elapsed_loop_gpu_sync_seconds": 20,
        "peak_cuda_allocated_bytes": 104,
        "peak_cuda_reserved_bytes": 204,
        "candidates": [],
        "timing_scope": "CPU fixture only; not actual performance evidence",
    }
    return rows, plan, report


def test_report_reconciliation_excludes_first_step_from_short_timing_summary():
    rows, plan, report = fixture()
    original = copy.deepcopy(report)
    result = verifier.reconcile(report, plan, rows)
    assert result["completed_true_optimizer_updates"] == 4
    assert result["timing"]["first_update_seconds"] == 10
    assert result["timing"]["steps_2_through_n"] == {
        "updates": 3,
        "median_seconds": 3,
        "mean_seconds": 3,
        "minimum_seconds": 2,
        "maximum_seconds": 4,
    }
    assert result["timing"]["full_run_throughput_established_by_smoke"] is False
    assert result["accuracy_validated"] is False and report == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_index", 99),
        ("true_optimizer_updates", 2),
        ("gradients_finite", False),
        ("loss", float("nan")),
    ],
)
def test_tampered_update_schedule_count_or_nonfinite_gradient_is_rejected(field, value):
    rows, plan, report = fixture()
    report["metrics"][0][field] = value
    with pytest.raises(AssertionError):
        verifier.reconcile(report, plan, rows)


def test_initialization_or_dependency_evidence_cannot_be_replaced_with_completion_flag():
    rows, plan, report = fixture()
    report["initializer_audit"]["source_tensor_count"] = 111
    with pytest.raises(AssertionError):
        verifier.reconcile(report, plan, rows)
    rows, plan, report = fixture()
    report["dependency_environment"]["torchao"]["distribution_version"] = "0.10.0"
    with pytest.raises(AssertionError):
        verifier.reconcile(report, plan, rows)
