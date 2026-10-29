"""Corrupt optimization history and detached physical proofs must fail before inference."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from training.common import sha
from training.v8 import cuda_evaluate as evaluate
from training.v8.verify_training import verify_history


@pytest.fixture
def history():
    config = {"fixture": "no actual training"}
    steps = 4
    plan = {
        "kind": "smoke",
        "planned_true_optimizer_updates": steps,
        "source_inputs_sha256": {"fixture": "fixed"},
        "frozen_config": config,
        "test_used_for_selection": False,
        "checkpoints": [],
        "planned_schedule": [
            {"iteration": i, "sample_index": i - 1, "bucket": "fixture"} for i in range(1, steps + 1)
        ],
    }
    report = {
        "kind": "smoke",
        "completed_true_optimizer_updates": steps,
        "source_inputs_sha256": {"fixture": "fixed"},
        "smoke_weights_discarded": True,
        "smoke_weights_reused": False,
        "test_used_for_selection": False,
        "candidates": [],
        "all_gradients_finite": True,
        "metrics": [
            {
                "iteration": i,
                "true_optimizer_updates": i,
                "sample_index": i - 1,
                "sampling_bucket": "fixture",
                "gradients_finite": True,
                "loss": 0.2,
                "unscaled_gradient_norm": 1.0,
                "adapter_l2_change_from_initializer": 0.001 * i,
                "grad_scaler_scale": 128.0,
                "elapsed_gpu_sync_seconds": 1.0,
                "elapsed_loop_gpu_sync_seconds": float(i),
                "peak_cuda_allocated_bytes": 100,
                "peak_cuda_reserved_bytes": 200,
            }
            for i in range(1, steps + 1)
        ],
    }
    return report, plan, config


def test_full_recorded_history_is_checked(history):
    report, plan, config = history
    result = verify_history(report, plan, config, 4)
    assert result["recorded_updates_checked"] == 4
    assert result["gradient_recomputed_by_cpu_audit"] is False


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("loss", float("nan")),
        ("loss", -1),
        ("unscaled_gradient_norm", 0),
        ("adapter_l2_change_from_initializer", float("inf")),
        ("grad_scaler_scale", 64),
        ("elapsed_loop_gpu_sync_seconds", -1),
        ("elapsed_gpu_sync_seconds", 0),
        ("true_optimizer_updates", 99),
        ("gradients_finite", False),
        ("peak_cuda_reserved_bytes", 50),
        ("peak_cuda_reserved_bytes", float("inf")),
        ("sample_index", 999),
    ],
)
def test_a_corrupt_middle_update_is_never_hidden(history, key, bad):
    report, plan, config = history
    report["metrics"][1][key] = bad
    with pytest.raises(ValueError):
        verify_history(report, plan, config, 4)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("smoke_weights_reused", True),
        ("smoke_weights_discarded", False),
        ("test_used_for_selection", True),
        ("source_inputs_sha256", {"different": "source"}),
        ("completed_true_optimizer_updates", 3),
        ("candidates", [{"iteration": 2}]),
    ],
)
def test_phase_provenance_cannot_be_rewritten(history, key, bad):
    report, plan, config = history
    report[key] = bad
    with pytest.raises(ValueError):
        verify_history(report, plan, config, 4)


@pytest.fixture
def identity(tmp_path, monkeypatch):
    root = tmp_path
    reports = root / "reports/v8"
    train = reports / "cuda/train-1344"
    before, after = root / "before", train / "adapter"
    for directory in (reports, train, before, after):
        directory.mkdir(parents=True, exist_ok=True)
    for path in (
        before / "adapter_model.safetensors",
        after / "adapter_model.safetensors",
        before / "adapter_config.json",
        after / "adapter_config.json",
        train / "report.json",
        train / "plan.json",
        reports / "pre-inference-plan.json",
    ):
        path.write_text("fixture data: no actual weights or training")
    manifest = root / "training/v8/datasets/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    config_path = root / "config.json"
    config = {"initialization": "before", "initialization_sha256": sha(before / "adapter_model.safetensors")}
    config_path.write_text(json.dumps(config))
    proof = {
        "exact_prior_initialization": True,
        "true_optimizer_updates": 1344,
        "initializer_sha256": sha(before / "adapter_model.safetensors"),
        "final_adapter_sha256": sha(after / "adapter_model.safetensors"),
        "training_report_sha256": sha(train / "report.json"),
        "training_plan_sha256": sha(train / "plan.json"),
    }
    (reports / "training-proof.json").write_text(json.dumps(proof))
    value = {
        "sealed_at_utc": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
        "analysis_plan_sha256": sha(reports / "pre-inference-plan.json"),
        "manifest_sha256": sha(manifest),
        "training_proof_sha256": sha(reports / "training-proof.json"),
        "training_report_sha256": proof["training_report_sha256"],
        "training_plan_sha256": proof["training_plan_sha256"],
        "before_adapter_sha256": proof["initializer_sha256"],
        "after_adapter_sha256": proof["final_adapter_sha256"],
        "before_adapter_config_sha256": sha(before / "adapter_config.json"),
        "after_adapter_config_sha256": sha(after / "adapter_config.json"),
    }
    (reports / "artifact-identity.json").write_text(json.dumps(value))
    monkeypatch.setattr(evaluate, "ROOT", root)
    monkeypatch.setattr(evaluate, "REPORTS", reports)
    monkeypatch.setattr(evaluate, "CONFIG", config_path)
    monkeypatch.setattr(evaluate, "verify_frozen", lambda: {})
    return reports, after, value


def test_numeric_proof_and_actual_artifacts_are_cross_bound(identity):
    evaluate.verify_identity()


def test_replacing_weight_and_identity_cannot_reuse_an_old_numeric_proof(identity):
    reports, after, value = identity
    (after / "adapter_model.safetensors").write_text("fixture replacement weights")
    value["after_adapter_sha256"] = sha(after / "adapter_model.safetensors")
    (reports / "artifact-identity.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="different physical"):
        evaluate.verify_identity()
