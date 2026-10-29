"""Notebook-safe scaffold and CPU validation; inspect mode never performs model work."""

import argparse
import json
from pathlib import Path

NOTEBOOK = Path(__file__).with_name("resolvedesk_cuda.ipynb")

CELLS = [
    (
        "markdown",
        """# ResolveDesk: isolated CUDA reproduction

## Goal
Reproduce the frozen V7 training and authored test on a user-selected Colab GPU.
This is **not a new held-out benchmark or a resume** of local optimization.
**CUDA training has not been executed or timed here.** The existing local run is separate.
""",
    ),
    (
        "markdown",
        """## Setup
Upload the source ZIP and, separately, the exact predecessor initializer ZIP yourself.
Choose a Linux **Python 3.12** GPU kernel before opting into setup/compute.
Preparing this notebook performs no upload, cloud job, billing action or model call.
Default `inspect` installs nothing and loads no model. Copy the ZIP SHA from the local receipt.
""",
    ),
    (
        "code",
        """MODE = "inspect"  # inspect | setup | preflight | train | select | evaluate
BUNDLE_ARCHIVE = "/content/resolvedesk-colab-source.zip"
EXPECTED_SOURCE_SHA256 = ""  # Required for setup/compute: local export-receipt.json
INITIALIZER_ARCHIVE = "/content/resolvedesk-initializer.zip"
WORKSPACE = "/content/resolvedesk-cuda-repro-001"  # New directory, outside any original checkout
PROVISION_BASE = False  # Explicitly opt in to downloading the pinned PUBLIC base snapshot
MINIMUM_FREE_MIB = 8192  # Conservative 8 GiB free guard; fit still requires CUDA preflight
""",
    ),
    (
        "markdown",
        """## Steps
### 1. Inspect and verify the local export
Only an explicitly selected setup/compute mode extracts the archive. Files are not merged
into an existing checkout. Existing reproduction sources must retain their exact hashes.
""",
    ),
    (
        "code",
        """import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

allowed_modes = {"inspect", "setup", "preflight", "train", "select", "evaluate"}
assert MODE in allowed_modes
archive_path, workspace = Path(BUNDLE_ARCHIVE), Path(WORKSPACE)
source_ready = False
if not archive_path.is_file():
    if MODE != "inspect":
        raise FileNotFoundError("Upload the source ZIP yourself before selecting setup/compute")
    print("Inspect only: awaiting source ZIP. No installation, GPU access or model load.")
else:
    if archive_path.stat().st_size > 250 * 1024**2:
        raise ValueError("Source ZIP exceeds its bounded size limit")
    with archive_path.open("rb") as source:
        archive_digest = hashlib.file_digest(source, "sha256").hexdigest()
    if EXPECTED_SOURCE_SHA256 and archive_digest != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source ZIP SHA differs from the local export receipt")
    if MODE != "inspect" and not EXPECTED_SOURCE_SHA256:
        raise ValueError("Paste the source ZIP SHA from the local export receipt first")
    with zipfile.ZipFile(archive_path) as bundle:
        entries = bundle.infolist()
        if len(entries) > 500 or sum(e.file_size for e in entries) > 250 * 1024**2:
            raise ValueError("Unexpected source ZIP size")
        names = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or ".." in path.parts or "\\\\" in entry.filename
                    or entry.filename in names or entry.is_dir()
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("Unsafe source ZIP entry")
            names.add(entry.filename)
        manifest = json.loads(bundle.read("colab-source-manifest.json"))
        if names != set(manifest["files"]) | {"colab-source-manifest.json"}:
            raise ValueError("Source ZIP contains unlisted files")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != digest:
                raise ValueError(f"Changed source input: {name}")
        if MODE != "inspect":
            if workspace.is_symlink() or (workspace / ".git").exists():
                raise ValueError("Use a new reproduction directory, not an original checkout")
            if workspace.exists():
                marker = workspace / "colab-source-manifest.json"
                if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text()) != manifest:
                    raise ValueError("Never merge an export into an existing unrelated directory")
            else:
                workspace.mkdir(parents=True)
                bundle.extractall(workspace)
            # Do not import workspace code until stdlib checks every trusted source file.
            sizes = {entry.filename: entry.file_size for entry in entries}
            for name, digest in manifest["files"].items():
                current = workspace
                for part in PurePosixPath(name).parts:
                    current = current / part
                    if current.is_symlink():
                        raise ValueError(f"Source parent/link changed: {name}")
                if not current.resolve().is_relative_to(workspace.resolve()):
                    raise ValueError("Source escaped the reproduction directory")
                if not current.is_file() or current.stat().st_size != sizes[name]:
                    raise ValueError(f"Changed workspace source: {name}")
                with current.open("rb") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != digest:
                        raise ValueError(f"Changed workspace source: {name}")
    print({"source_zip_sha256": archive_digest, "verified_files": len(manifest["files"]),
           "mode": MODE, "cuda_execution_validated": False})
    source_ready = MODE != "inspect"
""",
    ),
    (
        "markdown",
        """### 2. Install the CUDA environment only after opting in
The existing Mac lock is provenance; this installs pinned CUDA core packages explicitly.
No version is silently upgraded if its matching Linux wheel is unavailable. All other
installed package versions are saved with the reproduction. No model is loaded here.
Python, free NVIDIA VRAM and the CUDA toolkit are checked **before any pip install**.
No GPU means stop: this notebook never falls back to CPU training or selects a paid runtime.
""",
    ),
    (
        "code",
        """import os
import subprocess

if source_ready:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Select a Python 3.12 Colab kernel")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run([sys.executable, "-B", "-c",
                    "import json, sys; from pathlib import Path; "
                    "from training.colab.bundle import verify_source; "
                    "from training.colab.runner import validate_platform, probe_gpu, probe_cuda_toolkit; "
                    "verify_source(Path.cwd()); system = validate_platform(); "
                    "gpu = probe_gpu(int(sys.argv[1])); toolkit = probe_cuda_toolkit(); "
                    "print(json.dumps({'platform': system, 'gpu': gpu, 'cuda_toolkit': toolkit}))",
                    str(MINIMUM_FREE_MIB)],
                   cwd=workspace, check=True, timeout=35)
    subprocess.run([sys.executable, "-m", "pip", "install",
                    "mlx-lm[train,cuda12]==0.32.2", "mlx==0.32.4",
                    "mlx-cuda-12==0.32.4", "transformers==5.21.0",
                    "numpy==2.5.4", "safetensors==0.8.0",
                    "-e", str(workspace)], check=True)
    versions = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                              text=True, capture_output=True, check=True)
    (workspace / "colab-installed-packages.txt").write_text(versions.stdout)
else:
    print("Inspect mode: dependency installation skipped.")
""",
    ),
    (
        "markdown",
        """### 3. Run one explicit stage
`setup` checks the GPU/toolkit and can provision uploaded initialization plus the pinned
public base. `preflight` performs one real update, then a separate CPU initializer audit;
its updated weights are discarded. `train` starts a separate 1,344-update run from V6.1.
`select` uses validation only. `evaluate` reproduces four arms × 48 existing authored tasks.
Each stage needs a fresh receipt; failed runs are retained rather than overwritten.
""",
    ),
    (
        "code",
        """if source_ready:
    command = [sys.executable, "-B", "-m", "training.colab.runner",
               "--workspace", str(workspace), "--mode", MODE,
               "--minimum-free-mib", str(MINIMUM_FREE_MIB)]
    if MODE == "setup":
        if Path(INITIALIZER_ARCHIVE).is_file():
            command.extend(["--initializer-archive", INITIALIZER_ARCHIVE])
        if PROVISION_BASE:
            command.append("--provision-base")
    elif PROVISION_BASE:
        raise ValueError("Provision the public base in setup mode before compute")
    subprocess.run(command, cwd=workspace, check=True)
else:
    print("Inspect mode: setup, model preflight, training, selection and evaluation skipped.")
""",
    ),
    (
        "markdown",
        """## Checks
The pending CUDA checks are: compatible Python/glibc/driver/toolkit and exact core versions;
free VRAM; pinned base hashes; exact predecessor initialization; finite one-step gradients
and lower sample loss; full-versus-tail projection agreement; completed optimizer updates;
validation-only selection; raw four-arm evaluation evidence and scoped-consent review.
Kernel differences may change outputs. No Metal/CUDA numerical equality or speedup is claimed.
""",
    ),
    (
        "code",
        """if source_ready:
    for filename in ("preflight.json", "updates.json", "training.json", "evaluation.json"):
        path = workspace / "reports/v7" / filename
        if path.is_file():
            report = json.loads(path.read_text())
            assert report.get("cloud_compute_used") is True
            identity = report["execution_metadata"]
            print({"report": filename, "backend": identity["backend"],
                   "wrapper_sha256": identity["wrapper_sha256"],
                   "completed_updates": report.get("completed_updates"),
                   "raw_records": len(report.get("records", [])),
                   "scope": "Existing V7 test reproduction; do not pool with local results"})
else:
    print("No CUDA results produced. Notebook remains an unexecuted CUDA handoff.")
""",
    ),
    (
        "markdown",
        """## Next Steps
Inspect actual stage receipts and all retained failures before using cloud results.
Download artifacts yourself, preserving their cloud reproduction identity. Do not copy
cloud reports over local V7 reports. Local serving of an adapter requires a separate
compatibility check. See `docs/colab.md` for source allowlists and backend adaptations.
""",
    ),
]


def document():
    cells = []
    for i, (kind, source) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"resolvedesk-{i:02d}", "metadata": {}, "source": source}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "resolvedesk": {"cuda_executed": False, "default_mode": "inspect"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate(path=NOTEBOOK):
    notebook = json.loads(Path(path).read_text())
    assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
    ids = set()
    for cell in notebook["cells"]:
        assert cell["id"] not in ids and isinstance(cell["metadata"], dict)
        ids.add(cell["id"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None or isinstance(cell["execution_count"], int)
            assert isinstance(cell["outputs"], list)
            compile("".join(cell["source"]), str(path) + ":" + cell["id"], "exec")
    try:
        import nbformat
    except ImportError:
        return {
            "format": "v4.5 structure checked; nbformat library unavailable",
            "all_code_cells_compiled": True,
            "cuda_executed": False,
        }
    nbformat.validate(nbformat.read(path, as_version=4))
    return {"format": "nbformat validated", "all_code_cells_compiled": True, "cuda_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if not args.validate:
        try:
            import nbformat
        except ImportError:
            NOTEBOOK.write_text(json.dumps(document(), indent=1) + "\n")
        else:
            nbformat.write(nbformat.from_dict(document()), NOTEBOOK)
    print(json.dumps(validate()))


if __name__ == "__main__":
    main()
