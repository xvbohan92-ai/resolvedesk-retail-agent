"""Read only evidence whose source, model and figure fingerprints match."""

import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_report(root):
    try:
        folder, figures = root / "reports/v6", root / "docs/figures/v6"
        if (folder / "retirement.json").exists():
            return {
                "available": False,
                "experiment": "v6",
                "reason": (
                    "Development plan retired before held-out inference. See the final round-six workflow."
                ),
            }
        report = json.loads((folder / "evaluation.json").read_text())
        training = json.loads((folder / "training.json").read_text())
        values = json.loads((figures / "values.json").read_text())
        manifest_path = root / "training/v6/datasets/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert report["manifest_sha256"] == digest(manifest_path) == training["manifest_sha256"]
        assert report["training_sha256"] == digest(folder / "training.json") == values["training_sha256"]
        assert values["evaluation_sha256"] == digest(folder / "evaluation.json")
        assert values["analysis_source_sha256"] == digest(root / "training/v6/visualize.py")
        assert values["csv_sha256"] == digest(figures / "per-case.csv")
        assert values["summary"] == report["summary"]
        assert values["analysis_plan_sha256"] == digest(folder / "analysis-plan.json")
        for name, sha in manifest["files"].items():
            assert digest(root / name) == sha, name
        for name, sha in values["figure_sha256"].items():
            assert digest(figures / name) == sha, name
        return {
            "available": True,
            "experiment": "v6",
            "summary": report["summary"],
            "primary_summary": values["primary_summary"],
            "primary_test_cases": values["primary_test_cases"],
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
            "experiment": "v6",
            "reason": "Round 6 verified evidence is pending. Earlier rounds remain archived.",
        }
