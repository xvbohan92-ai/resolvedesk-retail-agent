import json

from training.common import ROOT, sha

REPORTS = ROOT / "reports/v4"
DATA = ROOT / "training/v4/datasets"


def verify_frozen():
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"V4 frozen experiment input changed: {name}")
    return manifest


def read_rows(split):
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").read_text().splitlines()]
