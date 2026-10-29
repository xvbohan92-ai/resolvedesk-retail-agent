"""Explicit-mode CUDA reproduction wrapper around unchanged, hash-checked V7 modules."""

import argparse
import csv
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from training.colab.bundle import MANIFEST, import_initializer, sha, verify_source

PINS = {
    "mlx": "0.32.4",
    "mlx-cuda-12": "0.32.4",
    "mlx-lm": "0.32.2",
    "transformers": "5.21.0",
    "numpy": "2.5.4",
    "safetensors": "0.8.0",
}
MODES = ("inspect", "setup", "preflight", "train", "select", "evaluate")
STAGES = {"preflight": "train", "train": "train", "select": "select", "evaluate": "evaluate"}
DEFAULT_MINIMUM_FREE_MIB = 8192


def version_tuple(value):
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    if not match:
        raise ValueError(f"Unrecognized version: {value}")
    return tuple(int(n) for n in match[1].split("."))


def parse_gpu_rows(text, minimum_free_mib):
    rows = list(csv.reader(text.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("Expected one explicitly selected NVIDIA GPU")
    name, driver, total, free, compute = [v.strip() for v in rows[0]]
    if version_tuple(driver) < (550, 54, 14) or version_tuple(compute) < (7, 5):
        raise RuntimeError("MLX CUDA12 requires driver >=550.54.14 and compute capability >=7.5")
    result = {
        "name": name,
        "driver": driver,
        "memory_total_mib": int(total),
        "memory_free_mib": int(free),
        "compute_capability": compute,
        "device_index": 0,
    }
    if result["memory_free_mib"] < minimum_free_mib:
        raise MemoryError(f"GPU has less than {minimum_free_mib} MiB free; no cache is unloaded")
    return result


def probe_gpu(minimum_free_mib=DEFAULT_MINIMUM_FREE_MIB):
    # Fixed arguments, one device, bounded timeout/output. Never modify device state.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "No usable NVIDIA GPU detected. Choose a Colab GPU runtime; CPU fallback is disabled."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "NVIDIA GPU query timed out after 10 seconds; CPU fallback is disabled."
        ) from error
    if len(result.stdout) > 4096:
        raise ValueError("Unexpected NVIDIA query output size")
    return parse_gpu_rows(result.stdout, minimum_free_mib)


def validate_platform():
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use a Linux Python 3.12 kernel, matching the frozen project")
    libc, version = platform.libc_ver()
    if libc != "glibc" or version_tuple(version) < (2, 35):
        raise RuntimeError("MLX Linux wheels require glibc >=2.35")
    return {"python": platform.python_version(), "system": platform.system(), "glibc": version}


def probe_cuda_toolkit():
    result = subprocess.run(["nvcc", "--version"], check=True, text=True, capture_output=True, timeout=10)
    match = re.search(r"release (\d+\.\d+)", result.stdout[:4096])
    if not match or version_tuple(match[1]) < (12, 0):
        raise RuntimeError("Confirm a CUDA toolkit >=12.0 before this reproduction")
    return match[1]


def validate_mlx_gpu(mx):
    """Do not label an unavailable or CPU-fallback MLX runtime as a CUDA execution."""
    if mx.metal.is_available() or not mx.is_available(mx.gpu):
        raise RuntimeError("The NVIDIA CUDA MLX GPU backend must actually be available")
    mx.set_default_device(mx.gpu)
    if mx.default_device().type != mx.gpu:
        raise RuntimeError("MLX did not select its GPU backend")


def provision_base(root):
    """Download only the public pinned snapshot files, then verify every byte/hash."""
    root = Path(root)
    verify_source(root)
    from huggingface_hub import snapshot_download

    from training.v7.common import verify_local_base

    spec = json.loads((root / "training/base-model-spec.json").read_text())
    config = json.loads((root / "training/v7/config.json").read_text())
    snapshot_download(
        repo_id=spec["repo"],
        revision=spec["revision"],
        allow_patterns=[f["path"] for f in spec["files"]],
        local_dir=root / config["model"],
    )
    verify_local_base(config)


class ReportJSON:
    """Change reporting identity only. Model/config/data/metrics are never transformed."""

    def __init__(self, metadata):
        self.metadata = metadata

    def __getattr__(self, name):
        return getattr(json, name)

    def dumps(self, value, *args, **kwargs):
        if isinstance(value, dict) and (
            "cloud_compute_used" in value
            or "manifest_sha256" in value
            and any(k in value for k in ("dataset", "records", "selected_adapter_sha256"))
        ):
            value = {**value, "cloud_compute_used": True, "execution_metadata": self.metadata}
        return json.dumps(value, *args, **kwargs)


def metadata_for(root, mode, gpu, system):
    wrapper = Path(__file__).resolve()
    return {
        "run_identity": Path(root).name,
        "scope": "CUDA reproduction of the existing authored V7 test",
        "cloud_compute_used": True,
        "backend": "mlx_cuda12",
        "not_resume": True,
        "fresh_heldout_benchmark": False,
        "mode": mode,
        "wrapper_sha256": sha(wrapper),
        "source_manifest_sha256": sha(Path(root) / MANIFEST),
        "gpu": gpu,
        "platform": system,
        "package_versions": {name: importlib.metadata.version(name) for name in PINS},
        "adaptations": [
            "Skip Metal-only wired-memory policy on CUDA; no model/loss/data change",
            "Replace localhost Ollama wait with bounded NVIDIA free-memory checks; never unload models",
            "Correct hardcoded local reporting identity to cloud_compute_used=true and retain wrapper SHA",
        ],
    }


def check_stage_inputs(root, mode):
    report_root = Path(root) / "reports/v7"
    needed = {"train": "preflight.json", "select": "updates.json", "evaluate": "training.json"}
    if mode in needed:
        prior = {"train": "preflight", "select": "train", "evaluate": "select"}[mode]
        receipt = json.loads((Path(root) / f"colab-execution/{prior}.json").read_text())
        if receipt["status"] != "completed" or receipt["source_manifest_sha256"] != sha(
            Path(root) / MANIFEST
        ):
            raise ValueError("The prior CUDA stage did not complete with the same source identity")
        report = json.loads((report_root / needed[mode]).read_text())
        if not report.get("cloud_compute_used") or not report.get("execution_metadata", {}).get("not_resume"):
            raise ValueError("Stage input is not from this isolated CUDA reproduction")
        if report["execution_metadata"]["source_manifest_sha256"] != sha(Path(root) / MANIFEST):
            raise ValueError("Stage input has a different source identity")
    if mode == "train" and not json.loads((report_root / "preflight.json").read_text())["preflight_only"]:
        raise ValueError("A discarded one-step CUDA preflight must pass before formal training")
    if mode == "train":
        audit = json.loads((report_root / "initialization-audit.json").read_text())
        if not audit["stages"]["preflight"]["exact_tensor_equality"]:
            raise ValueError("The independent CPU initializer audit must pass first")


def run_stage(root, mode, gpu, system, minimum_free_mib):
    """Import frozen modules only after CUDA/platform/provision checks have succeeded."""
    verify_source(root)
    import mlx.core as mx

    validate_mlx_gpu(mx)
    mx.eval(mx.array([1.0, 2.0]) * 2)
    check_stage_inputs(root, mode)
    metadata = metadata_for(root, mode, gpu, system)
    journal = Path(root) / "colab-execution"
    journal.mkdir(exist_ok=True)
    receipt = journal / f"{mode}.json"
    if receipt.exists():
        raise FileExistsError("Never overwrite a CUDA stage receipt; use a new clean reproduction")
    record = {"status": "started", "started_at_utc": datetime.now(UTC).isoformat(), **metadata}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    # This backend policy adaptation is outside, and explicitly recorded alongside, frozen sources.
    original_wired = mx.set_wired_limit
    mx.set_wired_limit = lambda limit: original_wired(limit) if mx.metal.is_available() else 0
    resources = importlib.import_module("training.v7.resources")
    resources.wait_for_room = lambda: (probe_gpu(minimum_free_mib), 0.0)[1]
    target = importlib.import_module(f"training.v7.{STAGES[mode]}")
    target.json = ReportJSON(metadata)
    original_argv = sys.argv
    try:
        if mode == "evaluate":
            from training.v7.protocol_audit import write_plan

            write_plan()
        sys.argv = [target.__file__] + (["--preflight"] if mode == "preflight" else [])
        target.main()
        if mode == "evaluate":
            from training.v7.protocol_audit import run_audit

            run_audit()
        verify_source(root)
        if mode in {"preflight", "train"}:
            stage = "preflight" if mode == "preflight" else "formal"
            subprocess.run(
                [sys.executable, "-B", "-m", "training.v7.check_initialization", "--stage", stage],
                cwd=root,
                check=True,
                timeout=120,
            )
        record.update(status="completed", completed_at_utc=datetime.now(UTC).isoformat())
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise
    finally:
        sys.argv = original_argv
        receipt.write_text(json.dumps(record, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, default="inspect")
    parser.add_argument("--initializer-archive", type=Path)
    parser.add_argument("--provision-base", action="store_true")
    parser.add_argument("--minimum-free-mib", type=int, default=DEFAULT_MINIMUM_FREE_MIB)
    args = parser.parse_args()
    root = args.workspace.resolve()
    if root != Path(__file__).resolve().parents[2] or Path.cwd().resolve() != root:
        raise ValueError("Launch the exported wrapper from its own fresh reproduction directory")
    manifest = verify_source(root)
    if args.mode == "inspect":
        print(
            json.dumps(
                {
                    "mode": "inspect",
                    "source_files_verified": len(manifest["files"]),
                    "cuda_execution_validated": False,
                    "model_loaded": False,
                    "next": "Choose setup explicitly; upload the initializer yourself",
                },
                indent=2,
            )
        )
        return
    system = validate_platform()
    system["cuda_toolkit"] = probe_cuda_toolkit()
    gpu = probe_gpu(args.minimum_free_mib)
    for name, expected in PINS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Expected {name}=={expected}; do not silently upgrade the experiment")
    if args.initializer_archive:
        import_initializer(args.initializer_archive, root)
    if args.provision_base:
        provision_base(root)
    if args.mode == "setup":
        print(
            json.dumps(
                {
                    "platform": system,
                    "gpu": gpu,
                    "model_loaded": False,
                    "cuda_training_validated": False,
                    "source_verified": True,
                },
                indent=2,
            )
        )
        return
    from training.v7.common import verify_frozen, verify_local_base

    verify_frozen()
    config = json.loads((root / "training/v7/config.json").read_text())
    verify_local_base(config)
    for item in manifest["initializer"]["files"]:
        if sha(root / item["target"]) != item["sha256"]:
            raise ValueError("Missing or changed exact initializer")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_stage(root, args.mode, gpu, system, args.minimum_free_mib)


if __name__ == "__main__":
    main()
