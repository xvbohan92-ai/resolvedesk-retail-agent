"""Build a reviewed source-only export and an optional, separate initializer archive."""

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "colab-source-manifest.json"
FROZEN_MANIFESTS = tuple(f"training/{v}/datasets/manifest.json" for v in ("v6", "v6_1", "v7"))
REPORT_INPUTS = frozenset(
    {
        "reports/v6/updates.json",
        "reports/v6_1/analysis-plan.json",
        "reports/v6_1/training.json",
        "reports/v7/analysis-plan.json",
    }
)
ATTRIBUTION_FILES = frozenset(
    f"third_party/tau-bench/{name}" for name in ("LICENSE", "README.md", "source.json", "retail-policy.md")
)
SOURCE_DIRS = (
    "resolve_desk",
    "training",
    *(f"training/v{i}" for i in range(2, 8)),
    "training/v6_1",
    "training/colab",
    "training/cuda_torch",
    "training/cuda_eval",
    "training/cuda_eval2",
)
EVAL_INPUTS = frozenset(
    {
        "training/datasets/cases.json",
        *(f"training/v{version}/datasets/cases.json" for version in range(2, 6)),
        "training/cuda_eval/datasets/cases.json",
        "training/cuda_eval/datasets/public.json",
        "reports/cuda-eval/data-summary.json",
        "reports/cuda-eval/reference-audit.json",
        "reports/cuda-eval/analysis-plan.json",
        "reports/cuda-eval/artifact-identity.json",
        "reports/cuda-eval/interruption.json",
        "reports/cuda-eval/interrupted-snapshot-audit.json",
        "training/cuda_eval2/datasets/cases.json",
        "training/cuda_eval2/datasets/public.json",
        "reports/cuda-eval2/data-summary.json",
        "reports/cuda-eval2/reference-audit.json",
        "reports/cuda-eval2/analysis-plan.json",
        "reports/cuda-eval2/artifact-identity.json",
        "reports/cuda-torch/probe-003/train-1344/plan.json",
        "reports/cuda-torch/probe-003/train-1344/report.json",
        "reports/cuda-torch/probe-003/train-1344/verification.json",
    }
)
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "training/pyproject.toml",
    "training/uv.lock",
    "training/base-model-spec.json",
    "docs/colab.md",
    "docs/cuda-torch.md",
    "tests/test_cuda_torch_model.py",
    "tests/test_cuda_torch_training.py",
    "tests/test_cuda_eval_data.py",
    "tests/test_cuda_eval_protocol.py",
    "tests/test_cuda_eval_reporting.py",
    "tests/test_cuda_eval2_data.py",
    "tests/test_cuda_eval2_protocol.py",
    "tests/test_cuda_eval2_reporting.py",
    "docs/cuda-evaluation2.md",
    "training/v6/config.json",
    "training/v6_1/config.json",
    "training/v7/config.json",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def local_path(root, name):
    """All mutable artifacts must remain inside the isolated export, including parents."""
    root = Path(root).resolve()
    relative = safe_relative(name)
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink cannot redirect reproduction files: {name}")
    if not path.resolve().is_relative_to(root):
        raise ValueError("Reproduction path escapes its isolated root")
    return path


def guard_tree(root, name):
    """Reject links in existing output trees, including dangling receipt/file links."""
    path = local_path(root, name)
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"Symlink cannot redirect reproduction files: {child.relative_to(root)}")
    return path


def runtime_paths(root):
    """Validate every configured artifact root before download or frozen execution."""
    config = json.loads(local_path(root, "training/v7/config.json").read_text())
    for field in (
        "model",
        "data",
        "initialization",
        "adapter_path",
        "best_path",
        "training_updates_report",
        "training_manifest",
    ):
        if field in config:
            guard_tree(root, config[field])
    if "adapter_path" in config:
        guard_tree(root, config["adapter_path"] + "-preflight")
        for step in config.get("checkpoint_candidates_completed_updates", []):
            guard_tree(root, f"{config['adapter_path']}/candidates/{int(step)}")
    for name in ("data", "reports", "colab-execution", "colab-installed-packages.txt"):
        guard_tree(root, name)
    return config


def allowed_source(name):
    path = safe_relative(name)
    if any(p.startswith(".") for p in path.parts) or path.suffix in {".sqlite", ".gguf", ".safetensors"}:
        return False
    if (
        name in ROOT_FILES
        or name in REPORT_INPUTS
        or name in FROZEN_MANIFESTS
        or name in ATTRIBUTION_FILES
        or name in EVAL_INPUTS
    ):
        return True
    if str(path.parent) in SOURCE_DIRS and path.suffix == ".py":
        return True
    if str(path.parent) == "training/colab" and path.suffix == ".ipynb":
        return True
    return (
        len(path.parts) == 4
        and path.parts[:1] == ("training",)
        and path.parts[1] in {"v6", "v6_1", "v7"}
        and path.parts[2] == "datasets"
        and path.name in {"manifest.json", "cases.json", "train.jsonl", "valid.jsonl", "test.jsonl"}
    )


def source_paths(root):
    """Explicit directories/types only; no recursive tree, Git export or reports glob."""
    root = Path(root).resolve()
    names = (
        {n for n in ROOT_FILES if local_path(root, n).is_file()}
        | {n for n in EVAL_INPUTS if local_path(root, n).is_file()}
        | set(REPORT_INPUTS)
        | set(FROZEN_MANIFESTS)
        | set(ATTRIBUTION_FILES)
    )
    for directory in SOURCE_DIRS:
        names.update(str(p.relative_to(root)) for p in local_path(root, directory).glob("*.py"))
    names.update(str(p.relative_to(root)) for p in local_path(root, "training/colab").glob("*.ipynb"))
    for version in ("v6", "v6_1", "v7"):
        names.update(
            str(p.relative_to(root)) for p in local_path(root, f"training/{version}/datasets").glob("*")
        )
    for name in names:
        if not allowed_source(name) or not local_path(root, name).is_file():
            raise ValueError(f"File outside export allowlist or missing: {name}")
    # All three frozen verification functions can run without exporting historical sessions.
    for manifest in FROZEN_MANIFESTS:
        for name, digest in json.loads(local_path(root, manifest).read_text())["files"].items():
            if name not in names or sha(local_path(root, name)) != digest:
                raise ValueError(f"Missing/changed frozen verification input: {name}")
    return sorted(names)


def initializer_spec(root):
    config = json.loads(local_path(root, "training/v7/config.json").read_text())
    folder = safe_relative(config["initialization"])
    if str(folder) != "data/finetuning/retail-qlora-v6_1/best":
        raise ValueError("Unexpected predecessor initializer path")
    files = []
    for name in ("adapters.safetensors", "adapter_config.json"):
        source = local_path(root, f"{folder}/{name}")
        files.append(
            {
                "name": name,
                "target": f"{folder}/{name}",
                "sha256": sha(source),
                "bytes": source.stat().st_size,
            }
        )
    if files[0]["sha256"] != config["initialization_sha256"]:
        raise ValueError("Predecessor adapter does not match the frozen SHA")
    return {
        "purpose": "User-managed upload of the exact V6.1 initializer; no automatic upload",
        "files": files,
    }


def prepare(destination, root=ROOT, stage_initializer=False):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Export outside the source checkout to avoid exporting outputs again")
    names = source_paths(root)
    initializer = initializer_spec(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "purpose": "isolated_cuda_reproduction",
        "not_resume": True,
        "cuda_execution_validated": False,
        "raw_customer_sessions_exported": False,
        "files": {n: sha(local_path(root, n)) for n in names},
        "initializer": initializer,
        "frozen_v7_manifest_sha256": sha(local_path(root, FROZEN_MANIFESTS[-1])),
    }
    with zipfile.ZipFile(destination / "resolvedesk-colab-source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(local_path(root, name), name)
        archive.writestr(MANIFEST, json.dumps(manifest, indent=2) + "\n")
    (destination / "initializer-manifest.json").write_text(json.dumps(initializer, indent=2) + "\n")
    notebook = local_path(root, "training/colab/resolvedesk_cuda.ipynb")
    if notebook.exists():
        shutil.copyfile(notebook, destination / notebook.name)
    if stage_initializer:
        with zipfile.ZipFile(destination / "resolvedesk-initializer.zip", "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("initializer-manifest.json", json.dumps(initializer, indent=2) + "\n")
            for item in initializer["files"]:
                archive.write(local_path(root, item["target"]), item["name"])
    receipt = {
        "source_archive_sha256": sha(destination / "resolvedesk-colab-source.zip"),
        "source_file_count": len(names),
        "weights_in_source_archive": False,
        "initializer_staged_separately": stage_initializer,
        "uploaded": False,
        "cuda_execution_validated": False,
    }
    (destination / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def safe_extract(archive_path, destination):
    """Bounded ZIP extraction: no traversal, links, duplicate paths or overwrites."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a new reproduction directory; never merge source exports")
    with zipfile.ZipFile(archive_path) as archive:
        entries, names = archive.infolist(), set()
        if len(entries) > 500 or sum(i.file_size for i in entries) > 250 * 1024**2:
            raise ValueError("Source archive exceeds its bounded size/file limit")
        for item in entries:
            safe_relative(item.filename)
            if item.filename in names or item.is_dir() or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("Duplicate, directory or link archive entry")
            names.add(item.filename)
        if MANIFEST not in names:
            raise ValueError("Missing source manifest")
        manifest = json.loads(archive.read(MANIFEST))
        if names != set(manifest["files"]) | {MANIFEST}:
            raise ValueError("Archive differs from the reviewed source allowlist")
        if any(not allowed_source(name) for name in manifest["files"]):
            raise ValueError("Archive contains a file outside the source allowlist")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Source digest mismatch: {name}")
        destination.mkdir(parents=True)
        archive.extractall(destination)
    return verify_source(destination)


def verify_source(root):
    root = Path(root).resolve()
    manifest = json.loads(local_path(root, MANIFEST).read_text())
    if manifest["purpose"] != "isolated_cuda_reproduction" or not manifest["not_resume"]:
        raise ValueError("Not a clean Colab reproduction export")
    if (root / ".git").exists():
        raise ValueError("Refusing to run inside an original Git checkout")
    for name, digest in manifest["files"].items():
        path = local_path(root, name)
        if not allowed_source(name) or path.is_symlink() or sha(path) != digest:
            raise ValueError(f"Source export changed: {name}")
    if sha(local_path(root, FROZEN_MANIFESTS[-1])) != manifest["frozen_v7_manifest_sha256"]:
        raise ValueError("V7 manifest changed")
    runtime_paths(root)
    return manifest


def import_initializer(archive_path, root):
    root = Path(root)
    expected = verify_source(root)["initializer"]
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory() as tmp:
        wanted = {i["name"] for i in expected["files"]} | {"initializer-manifest.json"}
        if len(archive.infolist()) != len(wanted) or set(archive.namelist()) != wanted:
            raise ValueError("Initializer archive must contain exactly the two declared files and manifest")
        if json.loads(archive.read("initializer-manifest.json")) != expected:
            raise ValueError("Initializer manifest differs from the source binding")
        for item in expected["files"]:
            info = archive.getinfo(item["name"])
            if info.file_size != item["bytes"] or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Initializer size/link mismatch")
            staged = Path(tmp) / item["name"]
            with archive.open(item["name"]) as source, staged.open("wb") as target:
                shutil.copyfileobj(source, target)
            if sha(staged) != item["sha256"]:
                raise ValueError("Initializer SHA mismatch")
            target = local_path(root, item["target"])
            if target.exists():
                if target.is_symlink() or sha(target) != item["sha256"]:
                    raise ValueError("Never overwrite a different initializer")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--stage-initializer",
        action="store_true",
        help="Stage weights in a SEPARATE local archive for manual upload",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, stage_initializer=args.stage_initializer), indent=2))


if __name__ == "__main__":
    main()
