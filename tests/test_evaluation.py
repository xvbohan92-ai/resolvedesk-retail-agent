import json

from training.common import ROOT, sha, verify_frozen
from training.fixtures import load_cases
from training.grade import prerequisites


def test_frozen_cases_and_demonstrations_remain_unchanged():
    manifest = verify_frozen()
    assert manifest["cases"] == {"train": 72, "valid": 18, "test": 27}
    assert manifest["rows"] == {"train": 256, "valid": 64, "test": 96}
    sets = [{c["orders"][0]["id"] for c in load_cases(split)} for split in ("train", "valid", "test")]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))


def test_correct_guesses_do_not_replace_required_customer_clarification():
    case = next(c for c in load_cases("test") if c["family"] == "missing_reason")
    trace = [
        {
            "action": {"tool": "identify_customer", "arguments": {}},
            "result": {"user_id": case["customer"]["id"]},
        },
        {
            "action": {"tool": "get_order", "arguments": {"order_id": case["expected"]["order_id"]}},
            "result": {"id": case["expected"]["order_id"]},
        },
    ]
    assert not prerequisites(case, {"questions": [], "trace": trace})["clarification_obtained"]
    assert prerequisites(case, {"questions": ["reason"], "trace": trace})["clarification_obtained"]


def test_training_provenance_has_real_updates_and_validation_only_selection():
    report = json.loads((ROOT / "reports/training.json").read_text())
    protocol = json.loads((ROOT / "reports/training-protocol.json").read_text())
    assert report["trainable_parameters"] == 3670016
    assert report["adapter_l2_change"] > 0
    assert report["initial_adapter_sha256"] != report["selected_adapter_sha256"]
    assert not report["test_used_for_selection"] and not report["cloud_compute_used"]
    candidates = [r for r in report["metrics"] if r["kind"] == "valid" and r["iteration"] > 0]
    assert report["best_checkpoint"]["val_loss"] == min(r["val_loss"] for r in candidates)
    assert protocol["runner_sha256"] == sha(ROOT / "training/train.py")
    assert protocol["common_sha256"] == sha(ROOT / "training/common.py")


def test_primary_scores_recompute_from_raw_outcomes_and_required_information():
    report = json.loads((ROOT / "reports/evaluation-graded.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    for row in report["records"]:
        checks = prerequisites(cases[row["case_id"]], row)
        expected = bool(
            row["raw_task_success"]
            and checks["clarification_obtained"]
            and checks["identity_and_order_checked"]
        )
        assert row["task_success"] == expected
    for variant, summary in report["summary"].items():
        rows = [r for r in report["records"] if r["variant"] == variant]
        assert len(rows) == 27
        assert summary["task_success"] == sum(r["task_success"] for r in rows)
        assert summary["model_calls"] == sum(len(r["trace"]) for r in rows)
        assert summary["schema_valid"] == sum(o["action"] is not None for r in rows for o in r["trace"])
    assert report["raw_evaluation_sha256"] == sha(ROOT / "reports/evaluation.json")
    assert report["grading_source_sha256"] == sha(ROOT / "training/grade.py")


def test_figures_csv_and_paired_counts_match_the_corrected_evaluation():
    import csv

    from resolve_desk.evidence import validated_report
    from training.release_gate import decision

    report = json.loads((ROOT / "reports/evaluation-graded.json").read_text())
    values = json.loads((ROOT / "docs/figures/values.json").read_text())
    assert values["summary"] == report["summary"]
    assert sum(values["paired"].values()) == 27
    assert values["csv_sha256"] == sha(ROOT / "docs/figures/per-case.csv")
    with (ROOT / "docs/figures/per-case.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 54
    for variant, summary in report["summary"].items():
        assert (
            sum(r["task_success"] == "True" for r in rows if r["variant"] == variant)
            == summary["task_success"]
        )
    assert validated_report(ROOT)["available"]
    assert not decision(report["summary"])["promoted"]


def test_packaged_demos_have_no_test_tasks_gold_answers_or_simulated_reply_script():
    from resolve_desk.fixtures import load_demo_cases

    rows = load_demo_cases()
    assert len(rows) == 18
    assert all(r["id"].startswith("valid-") for r in rows)
    assert all(not ({"answers", "expected", "confirmation"} & set(r)) for r in rows)
