"""Read only evidence whose source, model and figure fingerprints match."""

import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_report(root):
    try:
        folder, figures = root / "reports/v5", root / "docs/figures/v5"
        report = json.loads((folder / "evaluation.json").read_text())
        training = json.loads((folder / "training.json").read_text())
        values = json.loads((figures / "values.json").read_text())
        manifest_path = root / "training/v5/datasets/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert report["manifest_sha256"] == digest(manifest_path) == training["manifest_sha256"]
        assert report["training_sha256"] == digest(folder / "training.json") == values["training_sha256"]
        assert values["evaluation_sha256"] == digest(folder / "evaluation.json")
        assert values["analysis_source_sha256"] == digest(root / "training/v5/visualize.py")
        assert values["csv_sha256"] == digest(figures / "per-case.csv")
        assert values["summary"] == report["summary"]
        for name, sha in manifest["files"].items():
            assert digest(root / name) == sha, name
        for name, sha in values["figure_sha256"].items():
            assert digest(figures / name) == sha, name
        return {
            "available": True,
            "experiment": "v5",
            "summary": report["summary"],
            "scope": report["scope"],
            "release_gate": report["release_gate"],
            "paired": values["paired"],
            "failures": values["failures"],
            "family_success": values["family_success"],
            "confusion": values["confusion"],
            "permission_diagnostics": values["permission_diagnostics"],
            "training": {
                "parameters": training["trainable_parameters"],
                "selected": training["best_checkpoint"],
            },
        }
    except (OSError, ValueError, KeyError, AssertionError, TypeError):
        return {
            "available": False,
            "experiment": "v5",
            "reason": "Round 5 verified evidence is pending. Earlier rounds remain archived.",
        }
