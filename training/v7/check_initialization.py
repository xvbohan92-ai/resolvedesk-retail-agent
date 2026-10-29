"""CPU-only exact equality audit of the predecessor and saved pre-optimizer adapters."""

import argparse
import json
import sys
from datetime import UTC, datetime

from training.common import ROOT, sha
from training.v7.common import CONFIG, DATA, REPORTS, verify_frozen

AUDIT = REPORTS / "initialization-audit.json"


def paths(stage, config):
    source = ROOT / config["initialization"] / "adapters.safetensors"
    target = (
        ROOT
        / (config["adapter_path"] + ("-preflight" if stage == "preflight" else ""))
        / "initial.safetensors"
    )
    return source, target


def check(stage):
    # No mlx, torch, transformers, model loader or GPU runtime is imported.
    import numpy as np
    from safetensors.numpy import load_file

    verify_frozen()
    config = json.loads(CONFIG.read_text())
    source, target = paths(stage, config)
    if not target.is_file():
        raise FileNotFoundError(f"Saved {stage} initializer is not available: {target}")
    source_hash, target_hash = sha(source), sha(target)
    before, initialized = load_file(str(source)), load_file(str(target))
    source_keys, target_keys = set(before), set(initialized)
    common = sorted(source_keys & target_keys)
    shape_mismatches = [k for k in common if before[k].shape != initialized[k].shape]
    dtype_mismatches = [k for k in common if before[k].dtype != initialized[k].dtype]
    value_mismatches = [k for k in common if not np.array_equal(before[k], initialized[k])]
    source_finite = all(bool(np.isfinite(v).all()) for v in before.values())
    target_finite = all(bool(np.isfinite(v).all()) for v in initialized.values())
    gpu_modules = [
        name for name in ["mlx", "mlx.core", "torch", "jax", "tensorflow", "cupy"] if name in sys.modules
    ]
    identical = (
        source_hash == config["initialization_sha256"]
        and source_keys == target_keys
        and not shape_mismatches
        and not dtype_mismatches
        and not value_mismatches
        and source_finite
        and target_finite
        and not gpu_modules
    )
    entry = {
        "stage": stage,
        "status": "pass" if identical else "fail",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "source_path": str(source.relative_to(ROOT)),
        "target_path": str(target.relative_to(ROOT)),
        "source_sha256": source_hash,
        "target_sha256": target_hash,
        "expected_source_sha256": config["initialization_sha256"],
        "manifest_sha256": sha(DATA / "manifest.json"),
        "config_sha256": sha(CONFIG),
        "audit_source_sha256": sha(__file__),
        "source_tensor_count": len(before),
        "target_tensor_count": len(initialized),
        "source_parameter_count": sum(v.size for v in before.values()),
        "target_parameter_count": sum(v.size for v in initialized.values()),
        "exact_key_set": source_keys == target_keys,
        "source_only_keys": sorted(source_keys - target_keys),
        "target_only_keys": sorted(target_keys - source_keys),
        "shape_mismatch_keys": shape_mismatches,
        "dtype_mismatch_keys": dtype_mismatches,
        "value_mismatch_keys": value_mismatches,
        "all_source_values_finite": source_finite,
        "all_target_values_finite": target_finite,
        "exact_tensor_equality": identical,
        "gpu_modules_loaded": gpu_modules,
        "cpu_numpy_safetensors_only": True,
        "planned_formal_updates": config["iters"],
        "tensor_contents_exported": False,
    }
    assert sha(source) == source_hash and sha(target) == target_hash, "Initializer changed during CPU audit"
    report = (
        json.loads(AUDIT.read_text())
        if AUDIT.exists()
        else {
            "scope": (
                "Exact CPU NumPy/safetensors comparison; "
                "no GPU/model inference and no tensor values exported."
            ),
            "stages": {},
        }
    )
    report["stages"][stage] = entry
    AUDIT.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: entry[k]
                for k in [
                    "stage",
                    "status",
                    "source_tensor_count",
                    "target_tensor_count",
                    "source_parameter_count",
                    "target_parameter_count",
                    "exact_tensor_equality",
                    "gpu_modules_loaded",
                    "source_sha256",
                    "target_sha256",
                ]
            }
        ),
        flush=True,
    )
    if not identical:
        raise SystemExit(f"{stage} initializer differs from the predecessor. See {AUDIT}")
    return entry


def verify_audit(parameters):
    """Verify persisted evidence without importing array/GPU packages; remote CI has no weights."""
    config = json.loads(CONFIG.read_text())
    report = json.loads(AUDIT.read_text())
    assert set(report["stages"]) == {"preflight", "formal"}
    for stage, result in report["stages"].items():
        source, target = paths(stage, config)
        assert result["stage"] == stage and result["status"] == "pass"
        assert result["source_path"] == str(source.relative_to(ROOT))
        assert result["target_path"] == str(target.relative_to(ROOT))
        assert result["source_sha256"] == result["expected_source_sha256"] == config["initialization_sha256"]
        assert result["manifest_sha256"] == sha(DATA / "manifest.json")
        assert result["config_sha256"] == sha(CONFIG) and result["audit_source_sha256"] == sha(__file__)
        assert result["source_tensor_count"] == result["target_tensor_count"] > 0
        assert result["source_parameter_count"] == result["target_parameter_count"] == parameters
        assert result["exact_key_set"] and result["exact_tensor_equality"]
        assert result["all_source_values_finite"] and result["all_target_values_finite"]
        assert not any(
            result[k]
            for k in [
                "source_only_keys",
                "target_only_keys",
                "shape_mismatch_keys",
                "dtype_mismatch_keys",
                "value_mismatch_keys",
                "gpu_modules_loaded",
            ]
        )
        assert result["cpu_numpy_safetensors_only"] and not result["tensor_contents_exported"]
        assert result["planned_formal_updates"] == config["iters"]
        if source.exists():
            assert sha(source) == result["source_sha256"]
        if target.exists():
            assert sha(target) == result["target_sha256"]
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["preflight", "formal"])
    args = parser.parse_args()
    check(args.stage)


if __name__ == "__main__":
    main()
