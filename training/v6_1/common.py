import json

from training.common import ROOT, sha

REPORTS = ROOT / "reports/v6_1"
DATA = ROOT / "training/v6_1/datasets"
CONFIG = ROOT / "training/v6_1/config.json"


def verify_frozen():
    from training.v6.common import verify_frozen as verify_training

    verify_training()
    manifest = json.loads((DATA / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert sha(ROOT / path) == digest, path
    assert not manifest["test_inference_started_at_freeze"]
    return manifest


def read_rows(split):
    return [json.loads(s) for s in (DATA / f"{split}.jsonl").read_text().splitlines()]


def verify_local_base(config):
    spec = json.loads((ROOT / "training/base-model-spec.json").read_text())
    assert (config["model_repo"], config["model_revision"]) == (spec["repo"], spec["revision"])
    for item in spec["files"]:
        path = ROOT / config["model"] / item["path"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"], item["path"]
    return len(spec["files"])
