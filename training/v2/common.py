import json

from training.common import ROOT, sha

REPORTS = ROOT / "reports/v2"
DATA = ROOT / "training/v2/datasets"


def verify_frozen():
    m = json.loads((DATA / "manifest.json").read_text())
    for name, digest in m["files"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"V2 frozen input changed: {name}")
    return m


def read_rows(split):
    return [json.loads(s) for s in (DATA / f"{split}.jsonl").read_text().splitlines()]
