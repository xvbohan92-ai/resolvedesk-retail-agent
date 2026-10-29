import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated_report(root):
    try:
        report_path, training_path = root / "reports/evaluation-graded.json", root / "reports/training.json"
        report = json.loads(report_path.read_text())
        training = json.loads(training_path.read_text())
        values = json.loads((root / "docs/figures/values.json").read_text())
        checks = {
            report["training_sha256"]: digest(training_path),
            values["evaluation_sha256"]: digest(report_path),
            values["training_sha256"]: digest(training_path),
            report["store_sha256"]: digest(root / "resolve_desk/store.py"),
            report["protocol_sha256"]: digest(root / "resolve_desk/protocol.py"),
            report["raw_evaluation_sha256"]: digest(root / "reports/evaluation.json"),
            report["grading_source_sha256"]: digest(root / "training/grade.py"),
        }
        if any(a != b for a, b in checks.items()):
            return {"available": False, "reason": "Evidence or runtime identity changed."}
        for name, sha in report["dataset"]["files"].items():
            if digest(root / name) != sha:
                return {"available": False, "reason": "Frozen experiment inputs changed."}
        for name, sha in values["figure_sha256"].items():
            if digest(root / "docs/figures" / name) != sha:
                return {"available": False, "reason": "Figure does not match its saved evidence."}
        if values["csv_sha256"] != digest(root / "docs/figures/per-case.csv"):
            return {"available": False, "reason": "Per-case CSV does not match its evidence."}
        gate = json.loads((root / "reports/release-gate.json").read_text())
        if gate["evaluation_sha256"] != digest(report_path):
            return {"available": False, "reason": "Promotion check belongs to another evaluation."}
        return {
            "available": True,
            "summary": report["summary"],
            "scope": report["scope"],
            "training": {
                "parameters": training["trainable_parameters"],
                "selected": training["best_checkpoint"],
                "dataset": training["dataset"]["cases"],
                "adapter_sha256": training["selected_adapter_sha256"],
            },
            "paired": values["paired"],
            "failures": values["failures"],
            "release_gate": gate,
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "available": False,
            "reason": "The fixed training and evaluation run is still being prepared.",
        }


def validated_v2_report(root):
    try:
        folder = root / "reports/v2"
        report_path, training_path = folder / "evaluation.json", folder / "training.json"
        report = json.loads(report_path.read_text())
        training = json.loads(training_path.read_text())
        figures = root / "docs/figures/v2"
        values = json.loads((figures / "values.json").read_text())
        if (
            report["training_sha256"] != digest(training_path)
            or values["training_sha256"] != digest(training_path)
            or values["evaluation_sha256"] != digest(report_path)
        ):
            raise ValueError("V2 evidence identity changed")
        for name, sha in report["dataset"]["files"].items():
            if digest(root / name) != sha:
                raise ValueError("V2 frozen experiment inputs changed")
        for name, sha in values["figure_sha256"].items():
            if digest(figures / name) != sha:
                raise ValueError("V2 figure identity changed")
        if values["csv_sha256"] != digest(figures / "per-case.csv") or values["summary"] != report["summary"]:
            raise ValueError("V2 quantitative source identity changed")
        if values["analysis_source_sha256"] != digest(root / "training/v2/visualize.py"):
            raise ValueError("V2 analysis source identity changed")
        return {
            "available": True,
            "experiment": "v2",
            "summary": report["summary"],
            "scope": report["scope"],
            "release_gate": report["release_gate"],
            "training": {
                "parameters": training["trainable_parameters"],
                "selected": training["best_checkpoint"],
            },
            "paired": values["paired"],
            "failures": values["failures"],
            "identical_action_repeats": values["identical_action_repeats"],
            "family_success": values["family_success"],
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "available": False,
            "experiment": "v2",
            "reason": "Round 2 evidence is not ready. Round 1 remains in the archive.",
        }


def validated_v3_report(root):
    try:
        folder, figures = root / "reports/v3", root / "docs/figures/v3"
        report = json.loads((folder / "evaluation.json").read_text())
        training = json.loads((folder / "training.json").read_text())
        values = json.loads((figures / "values.json").read_text())
        checks = [
            (report["training_sha256"], digest(folder / "training.json")),
            (report["v2_training_sha256"], digest(root / "reports/v2/training.json")),
            (values["evaluation_sha256"], digest(folder / "evaluation.json")),
            (values["training_sha256"], digest(folder / "training.json")),
            (values["analysis_source_sha256"], digest(root / "training/v3/visualize.py")),
            (values["shared_visualization_sha256"], digest(root / "training/v2/visualize.py")),
            (values["csv_sha256"], digest(figures / "per-case.csv")),
        ]
        if any(a != b for a, b in checks) or values["summary"] != report["summary"]:
            raise ValueError("V3 report identity changed")
        if any(digest(root / name) != sha for name, sha in report["dataset"]["files"].items()):
            raise ValueError("V3 frozen inputs changed")
        if any(digest(figures / name) != sha for name, sha in values["figure_sha256"].items()):
            raise ValueError("V3 figure identity changed")
        return {
            "available": True,
            "experiment": "v3",
            "summary": report["summary"],
            "scope": report["scope"],
            "training": {
                "parameters": training["trainable_parameters"],
                "selected": training["best_checkpoint"],
            },
            "paired": values["paired"],
            "family_success": values["family_success"],
            "confusion": values["confusion"],
            "failures": values["failures"],
            "release_gate": report["release_gate"],
            "identical_action_repeats": values["identical_action_repeats"],
        }
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return {
            "available": False,
            "experiment": "v3",
            "reason": "Round 3 verified evidence is pending. Earlier experiments remain archived.",
        }


def validated_v4_report(root):
    try:
        folder, figures = root / "reports/v4", root / "docs/figures/v4"
        report = json.loads((folder / "evaluation.json").read_text())
        training = json.loads((folder / "training.json").read_text())
        values = json.loads((figures / "values.json").read_text())
        checks = [
            (report["training_sha256"], digest(folder / "training.json")),
            (report["previous_training_sha256"], digest(root / "reports/v3/training.json")),
            (values["evaluation_sha256"], digest(folder / "evaluation.json")),
            (values["training_sha256"], digest(folder / "training.json")),
            (values["analysis_source_sha256"], digest(root / "training/v4/visualize.py")),
            (values["shared_visualization_sha256"], digest(root / "training/v2/visualize.py")),
            (values["csv_sha256"], digest(figures / "per-case.csv")),
        ]
        if any(a != b for a, b in checks) or values["summary"] != report["summary"]:
            raise ValueError("V4 report identity changed")
        if any(digest(root / name) != sha for name, sha in report["dataset"]["files"].items()):
            raise ValueError("V4 frozen inputs changed")
        if any(digest(figures / name) != sha for name, sha in values["figure_sha256"].items()):
            raise ValueError("V4 figure identity changed")
        pref_path = folder / "preference-manifest.json"
        pref = json.loads(pref_path.read_text())
        if training["preference_manifest_sha256"] != digest(pref_path):
            raise ValueError("V4 preference protocol identity changed")
        for name, sha in pref["files"].items():
            if digest(root / name) != sha:
                raise ValueError("V4 preference inputs changed")
        action_path = folder / "action-manifest.json"
        action = json.loads(action_path.read_text())
        if training["action_manifest_sha256"] != digest(action_path):
            raise ValueError("V4 action-preference protocol identity changed")
        if action["original_preference_manifest_sha256"] != digest(pref_path):
            raise ValueError("V4 protocol revision ancestry changed")
        if action["original_sft_manifest_sha256"] != digest(root / "training/v4/datasets/manifest.json"):
            raise ValueError("V4 SFT ancestry changed")
        if training["preference"]["manifest"] != action:
            raise ValueError("V4 final training phase belongs to another protocol")
        if action["failed_validation_sha256"] != digest(folder / "validation-604.json"):
            raise ValueError("V4 failed validation evidence changed")
        for name, sha in action["files"].items():
            if digest(root / name) != sha:
                raise ValueError("V4 action-preference inputs changed")
        recovery_path = folder / "recovery-manifest.json"
        recovery = json.loads(recovery_path.read_text())
        if training["recovery_manifest_sha256"] != digest(recovery_path):
            raise ValueError("V4 recovery protocol identity changed")
        if report["recovery_manifest_sha256"] != digest(recovery_path):
            raise ValueError("V4 evaluation used another recovery protocol")
        if report["facts_source_sha256"] != digest(root / "resolve_desk/decision_facts.py"):
            raise ValueError("V4 runtime facts changed")
        if recovery["original_action_manifest_sha256"] != digest(action_path):
            raise ValueError("V4 recovery ancestry changed")
        if recovery["previous_validation_sha256"] != digest(folder / "validation-616.json"):
            raise ValueError("V4 action-only validation evidence changed")
        if training["recovery"]["manifest"] != recovery:
            raise ValueError("V4 recovery training belongs to another protocol")
        for name, sha in recovery["files"].items():
            if digest(root / name) != sha:
                raise ValueError("V4 recovery inputs changed")
        calibration_path = folder / "calibration-manifest.json"
        calibration = json.loads(calibration_path.read_text())
        if training["calibration_manifest_sha256"] != digest(calibration_path):
            raise ValueError("V4 calibration identity changed")
        if calibration["previous_manifest_sha256"] != digest(recovery_path):
            raise ValueError("V4 calibration ancestry changed")
        if training["calibration"]["manifest"] != calibration:
            raise ValueError("V4 calibration training belongs to another protocol")
        for name, sha in calibration["files"].items():
            if digest(root / name) != sha:
                raise ValueError("V4 calibration inputs changed")
        return {
            "available": True,
            "experiment": "v4",
            "summary": report["summary"],
            "scope": report["scope"],
            "training": {
                "parameters": training["trainable_parameters"],
                "selected": training["best_checkpoint"],
            },
            "paired": values["paired"],
            "family_success": values["family_success"],
            "confusion": values["confusion"],
            "handoff_diagnostics": values["handoff_diagnostics"],
            "failures": values["failures"],
            "release_gate": report["release_gate"],
            "identical_action_repeats": values["identical_action_repeats"],
        }
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return {
            "available": False,
            "experiment": "v4",
            "reason": "Round 4 verified evidence is pending. Earlier experiments remain archived.",
        }
