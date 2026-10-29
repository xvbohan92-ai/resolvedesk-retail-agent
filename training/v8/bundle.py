"""Prepare inspect-only Colab source/weight packages after V8 inputs are frozen."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from training.common import ROOT, sha
from training.v8.common import CONFIG, DATA, verify_frozen

PRIOR_EXPORT = ROOT.parent / "exports/resolvedesk-colab-eval2-ready-final/resolvedesk-colab-source.zip"
PRIOR_EXPORT_SHA = "65b9e93cd92fd3fb51d0bbbdde96df8cbacc54195c7b28c9c76fa90f1433eb4b"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def safe_member(name):
    path = PurePosixPath(name)
    return not (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or name.startswith(("data/", ".git/"))
        or any(p.startswith(".env") or p in {".git", ".aws", ".ssh"} for p in path.parts)
        or path.suffix in {".safetensors", ".gguf", ".sqlite", ".sqlite3", ".db", ".zip", ".pem", ".key"}
    )


def read_local(name):
    path = ROOT / name
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Export source escapes the project")
    if any(p.is_symlink() for p in (path, *path.parents) if p != ROOT.parent):
        raise ValueError("Export cannot follow a symlink")
    return path.read_bytes()


def zip_files(path, files, manifest_name):
    manifest = {"files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(files.items()):
            archive.writestr(name, raw)
        archive.writestr(manifest_name, json.dumps(manifest, indent=2) + "\n")
    return manifest


def cell(kind, source, identifier):
    result = {"cell_type": kind, "id": identifier, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(destination):
    manifest = verify_frozen()
    if sha(PRIOR_EXPORT) != PRIOR_EXPORT_SHA:
        raise ValueError("Known clean source skeleton changed")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(PRIOR_EXPORT) as previous:
        names = {n for n in previous.namelist() if n != "colab-source-manifest.json"}
    names.update(manifest["files"])
    names.add("training/v8/datasets/manifest.json")
    names.add("docs/round-8-plan.md")
    if any(not safe_member(name) for name in names):
        raise ValueError("Source package includes a forbidden path or weight/database/secret container")
    source = {name: read_local(name) for name in sorted(names)}
    source_path = destination / "resolvedesk-v8-source.zip"
    source_manifest = zip_files(source_path, source, "v8-source-manifest.json")
    config = json.loads(CONFIG.read_text())
    weights = {}
    for folder, basename in [
        (config["initialization"], "adapter_model.safetensors"),
        (config["legacy_loader_initialization"], "adapters.safetensors"),
    ]:
        for name in (basename, "adapter_config.json"):
            relative = str(PurePosixPath(folder) / name)
            weights[relative] = read_local(relative)
    if sha(ROOT / config["initialization"] / "adapter_model.safetensors") != config["initialization_sha256"]:
        raise ValueError("Actual prior adapter changed")
    weight_path = destination / "resolvedesk-v8-initializers.zip"
    weight_manifest = zip_files(weight_path, weights, "v8-initializer-manifest.json")
    hashes = {p.name: sha(p) for p in (source_path, weight_path)}
    notebook = make_notebook(hashes, sha(ROOT / "reports/v8/pre-inference-plan.json"))
    notebook_path = destination / "resolvedesk_v8.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    receipt = {
        "round": "v8",
        "mode": "inspect_only",
        "cloud_execution_authorized": False,
        "files": {**hashes, notebook_path.name: sha(notebook_path)},
        "source_members": len(source_manifest["files"]),
        "weight_members": len(weight_manifest["files"]),
        "analysis_plan_sha256": sha(ROOT / "reports/v8/pre-inference-plan.json"),
        "manifest_sha256": sha(DATA / "manifest.json"),
        "credentials_databases_customer_sessions_included": False,
        "only_weights_in_initializer_package": True,
    }
    (destination / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def make_notebook(hashes, plan_sha):
    from training.v8.colab_cells import CONFIG_CODE, LAUNCH_CODE, SETUP_CODE, STATUS_CODE

    setup = SETUP_CODE.replace("SOURCE_SHA_PLACEHOLDER", hashes["resolvedesk-v8-source.zip"])
    setup = setup.replace("WEIGHTS_SHA_PLACEHOLDER", hashes["resolvedesk-v8-initializers.zip"])
    setup = setup.replace("PLAN_SHA_PLACEHOLDER", plan_sha)
    cells = [
        cell(
            "markdown",
            """# ResolveDesk V8: repair, continue training, compare three arms

**Inspect only. No execution is authorized by opening this notebook.**
New direct approval is required for these two packages and this round's compute.
The completed run002 allowance is not reused. No purchase, renewal, Drive mount,
unrelated model-cache unload, checkpoint search or ambiguous-call replay.

Fixed work: four discarded smoke updates, a fresh 1,344-update continuation from
the actual prior CUDA adapter, one 48-case validation and three matched 48-case test
arms. All failed/corrective calls count toward twelve calls per task. New data is
authored synthetic; final model-quality results do not exist yet.

Only set the operator resource receipt after explicit human approval AND observing
the allocated T4's actual balance/rate. A Python process deadline does not stop VM
billing. Arm the official runtime-release timer before setup and release earlier
after preserving the actual final or failed evidence.
""",
            "v8-intro",
        ),
        cell("code", CONFIG_CODE, "v8-config"),
        cell("code", setup, "v8-setup"),
        cell("code", LAUNCH_CODE, "v8-launch"),
        cell("code", STATUS_CODE, "v8-status"),
    ]
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "resolvedesk_v8.ipynb"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
