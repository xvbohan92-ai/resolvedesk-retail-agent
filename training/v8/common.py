"""Round-eight paths and immutable-input checks; the parent owns final sealing."""

import json

from training.common import ROOT, sha

DATA = ROOT / "training/v8/datasets"
REPORTS = ROOT / "reports/v8"
CONFIG = ROOT / "training/v8/config.json"
INITIAL_ADAPTER_SHA256 = "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053"


def read_rows(split):
    normalized = {"train": "v8-train", "valid": "v8-valid"}.get(split, split)
    if normalized not in {"v8-train", "v8-valid"}:
        raise ValueError("Development loader accepts only train/valid, never held-out test")
    return [json.loads(line) for line in (DATA / f"{normalized}.jsonl").read_text().splitlines()]


def verify_frozen():
    from training.v7.common import verify_frozen as verify_previous

    verify_previous()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Frozen V8 input changed: {name}")
    if manifest["initial_adapter_sha256"] != INITIAL_ADAPTER_SHA256:
        raise ValueError("V8 must continue from the fixed final1344 adapter")
    if manifest.get("test_inference_started_at_freeze") is not False:
        raise ValueError("V8 inputs must be sealed before test inference")
    return manifest
