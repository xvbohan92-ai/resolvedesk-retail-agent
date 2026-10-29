import json

from training.common import ROOT, sha

REPORTS = ROOT / "reports/v7"
DATA = ROOT / "training/v7/datasets"
CONFIG = ROOT / "training/v7/config.json"


def verify_frozen():
    from training.v6_1.common import verify_frozen as verify_previous

    verify_previous()
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


def freeze():
    from training.v6_1.common import verify_frozen as verify_previous

    verify_previous()
    assert not (DATA / "manifest.json").exists(), "Never overwrite a frozen experiment"
    assert not (REPORTS / "evaluation-partial.json").exists()
    names = [
        "common.py",
        "fixtures.py",
        "scoring.py",
        "prepare.py",
        "evaluation.py",
        "evaluate.py",
        "select.py",
        "provider.py",
        "resources.py",
        "train.py",
        "tokenization.py",
    ]
    files = [ROOT / "training/v7" / n for n in names]
    files += [
        CONFIG,
        ROOT / "resolve_desk/resolution_v7.py",
        ROOT / "resolve_desk/agent_v7.py",
        ROOT / "resolve_desk/resolution_v6_1.py",
        ROOT / "training/runtime.py",
        ROOT / "training/v4/tokenization.py",
        ROOT / "training/v4/evaluation.py",
        ROOT / "training/v6/scoring.py",
        ROOT / "training/v6_1/datasets/manifest.json",
        ROOT / "reports/v6_1/training.json",
        ROOT / "training/base-model-spec.json",
        REPORTS / "analysis-plan.json",
        *sorted(DATA.glob("*.jsonl")),
        DATA / "cases.json",
    ]
    config = json.loads(CONFIG.read_text())
    manifest = {
        "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "data": json.loads((REPORTS / "data-summary.json").read_text()),
        "initial_adapter_sha256": config["initialization_sha256"],
        "previous_manifest_sha256": sha(ROOT / "training/v6_1/datasets/manifest.json"),
        "previous_training_sha256": sha(ROOT / "reports/v6_1/training.json"),
        "test_inference_started_at_freeze": False,
        "scope": "One new V7 SFT run and separately held-out four-arm evaluation; no test-based selection.",
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
