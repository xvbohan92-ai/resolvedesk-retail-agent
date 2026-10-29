"""Independently reconcile actor, business, repeated-question and artifact counts."""

import csv
import json
import math
from collections import Counter

from resolve_desk.evidence import validated_v4_report
from training.common import ROOT, sha
from training.common import verify_frozen as verify_v1
from training.v2.common import verify_frozen as verify_v2
from training.v3.common import verify_frozen as verify_v3
from training.v4.action_preference import MANIFEST, verify_protocol
from training.v4.calibration import MANIFEST as CALIBRATION_MANIFEST
from training.v4.calibration import verify_protocol as verify_calibration
from training.v4.calibration_choice import prefer_calibration
from training.v4.common import REPORTS, verify_frozen
from training.v4.evaluation import ARMS
from training.v4.fixtures import load_cases
from training.v4.preference import MANIFEST as PREVIOUS_MANIFEST
from training.v4.preference import verify_protocol as verify_previous_protocol
from training.v4.recovery import MANIFEST as RECOVERY_MANIFEST
from training.v4.recovery import verify_protocol as verify_recovery


def verify():
    for verifier in [verify_v1, verify_v2, verify_v3]:
        verifier()
    manifest = verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    resource_repair = json.loads((REPORTS / "inference-resource-repair.json").read_text())
    assert resource_repair["launcher_source_sha256"] == sha(ROOT / "training/v4/resource_runner.py")
    assert resource_repair["cache_limit_bytes"] == 256 * 1024**2
    assert resource_repair["wire_limit_bytes"] == 6 * 1024**3
    assert resource_repair["applied_to_all_four_test_arms"]
    assert not resource_repair["other_model_caches_unloaded_by_agent"]
    assert not resource_repair["test_inference_started_at_repair"]
    values = json.loads((ROOT / "docs/figures/v4/values.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    assert len(cases) == 32 and len(report["records"]) == 128
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == Counter(
        {(v, i): 1 for v in ARMS for i in cases}
    )
    assert training["dataset"] == report["dataset"] == manifest
    assert training["completed_updates"] in {696, 744} and not training["test_used_for_selection"]
    assert training["sft_completed_updates"] == 480
    assert training["preference_completed_updates"] == 136
    pref = verify_protocol()
    assert training["preference_manifest_sha256"] == sha(PREVIOUS_MANIFEST)
    assert training["action_manifest_sha256"] == sha(MANIFEST)
    verify_previous_protocol()
    assert training["preference"]["manifest"] == pref
    assert pref["pair_count"] == 136 and not pref["test_inference_started_at_freeze"]
    updates = json.loads((REPORTS / "updates.json").read_text())
    preference = json.loads((REPORTS / "action-updates.json").read_text())
    assert training["preference"] == preference
    assert updates["completed_updates"] == 480 and updates["dataset"] == manifest
    assert updates["metrics"][-1]["iteration"] == 480
    origin = next(c for c in updates["candidates"] if c["completed_updates"] == 480)
    assert origin["adapter_sha256"] == pref["sft_480_sha256"]
    assert preference["completed_updates"] == 136 and not preference["preflight_only"]
    assert [m["iteration"] for m in preference["metrics"]] == list(range(1, 137))
    assert all(math.isfinite(m["loss"]) for m in preference["metrics"])
    preflight = json.loads((REPORTS / "action-preflight.json").read_text())
    assert preflight["manifest"] == pref and preflight["preflight_only"]
    assert preflight["gradient_audit"]["chain_rule_max_absolute_error"] < 1e-6
    assert preflight["gradient_audit"]["finite_difference_error"] < 1e-3
    assert (
        preflight["metrics"][0]["post_update_mean_logp_gap"]
        > preflight["metrics"][0]["pre_update_mean_logp_gap"]
    )
    assert preflight["projection_audit"]["all_train_targets_covered"]
    assert preflight["projection_audit"]["mean_logp_max_error_vs_full_projection"] < 0.01
    assert (
        preflight["metrics"][0]["post_update_chosen_completion_nll"]
        < preflight["metrics"][0]["chosen_completion_nll"]
    )
    failed = training["unregularized_attempt"]
    assert not failed["weights_reused"]
    assert failed["updates_sha256"] == sha(REPORTS / "preference-updates.json")
    assert failed["validation_sha256"] == sha(REPORTS / "validation-604.json")
    assert failed["training_sha256"] == sha(REPORTS / "training-unregularized.json")
    original_config = updates["config"]
    for key, value in original_config.items():
        if key != "best_path":
            assert training["config"][key] == value
    assert (
        training["config"]["best_path"]
        == (training["calibration"] if training["calibration_adopted"] else training["recovery"])["protocol"][
            "best_path"
        ]
    )
    assert training["trainable_parameters"] == preference["trainable_parameters"]
    recovery_manifest = verify_recovery()
    recovery = json.loads((REPORTS / "recovery-updates.json").read_text())
    assert training["recovery"] == recovery
    assert training["recovery_completed_updates"] == recovery["completed_updates"] == 80
    assert recovery_manifest == recovery["manifest"]
    assert (
        training["recovery_manifest_sha256"] == report["recovery_manifest_sha256"] == sha(RECOVERY_MANIFEST)
    )
    assert report["facts_source_sha256"] == sha(ROOT / "resolve_desk/decision_facts.py")
    assert recovery_manifest["initial_weights_sha256"] == preference["adapter_sha256"]
    assert not recovery_manifest["test_inference_started_at_freeze"]
    assert [m["iteration"] for m in recovery["metrics"]] == list(range(1, 81))
    assert all(
        math.isfinite(m["loss"]) and math.isfinite(m["chosen_completion_nll"]) for m in recovery["metrics"]
    )
    assert recovery["adapter_l2_change_from_action136"] > 0
    rp = json.loads((REPORTS / "recovery-preflight.json").read_text())
    assert rp["manifest"] == recovery_manifest and rp["preflight_only"]
    assert rp["gradient_audit"]["chain_rule_max_absolute_error"] < 1e-6
    assert rp["projection_audit"]["all_train_targets_covered"]
    assert rp["projection_audit"]["mean_logp_max_error_vs_full_projection"] < 0.01
    assert rp["metrics"][0]["post_update_chosen_completion_nll"] < rp["metrics"][0]["chosen_completion_nll"]
    assert training["action_only_training_sha256"] == sha(REPORTS / "training-action-only.json")
    calibration_manifest = verify_calibration()
    trial = json.loads((REPORTS / "calibration-updates.json").read_text())
    assert training["calibration"] == trial and trial["manifest"] == calibration_manifest
    assert training["calibration_manifest_sha256"] == sha(CALIBRATION_MANIFEST)
    assert trial["completed_updates"] == 48 and not trial["preflight_only"]
    assert [m["iteration"] for m in trial["metrics"]] == list(range(1, 49))
    assert calibration_manifest["initial_weights_sha256"] == recovery["adapter_sha256"]
    base_validation = json.loads((REPORTS / "validation-696.json").read_text())["summary"]
    trial_validation = json.loads((REPORTS / "validation-744.json").read_text())["summary"]
    adopted = prefer_calibration(trial_validation, base_validation)
    assert training["calibration_adopted"] == adopted
    assert training["completed_updates"] == (744 if adopted else 696)
    assert training["calibration_completed_updates"] == (48 if adopted else 0)
    assert training["recovery_only_training_sha256"] == sha(REPORTS / "training-recovery-only.json")
    token_audit = json.loads((REPORTS / "calibration-token-audit.json").read_text())
    assert token_audit["unquoted_value_ranges_only"] and len(token_audit["records"]) == 8
    assert training["adapter_l2_change"] > 0
    assert training["selected_adapter_sha256"] != training["initial_adapter_sha256"]
    candidates = training["validation_candidates"]
    assert {r["completed_updates"] for r in candidates} == {240, 604, 616, 696, 744}
    chosen = None
    for candidate in candidates:
        full = json.loads((REPORTS / f"validation-{candidate['completed_updates']}.json").read_text())
        records = full["records"]
        s = full["summary"]
        assert len(records) == 32 and all(r["case_id"].startswith("v4-valid-") for r in records)
        assert s["task_success"] == sum(r["task_success"] for r in records)
        ex = [r for r in records if r["expected"]["kind"] == "exchange"]
        fallback = [r for r in records if r["family"] == "exchange_fallback"]
        handoff = [r for r in records if r["expected"]["kind"] == "handoff"]
        ex_n = sum(r["task_success"] for r in ex)
        fa_n = sum(r["task_success"] for r in fallback)
        ha_n = sum(r["task_success"] and r["model_handoff"] for r in handoff)
        assert (ex_n, fa_n, ha_n) == (
            s["fulfillable_exchange_success"],
            s["fallback_success"],
            s["handoff_success"],
        )
        assert full["adapter_sha256"] == candidate["adapter_sha256"]
        assert s == candidate["summary"]
        if candidate["completed_updates"] == training["best_checkpoint"]["iteration"]:
            assert full["selection_key"] == []
            chosen = candidate
    assert chosen is not None
    assert chosen["completed_updates"] == training["best_checkpoint"]["iteration"]
    assert chosen["adapter_sha256"] == training["selected_adapter_sha256"]
    assert (
        chosen["adapter_sha256"]
        == (training["calibration"] if training["calibration_adopted"] else training["recovery"])[
            "adapter_sha256"
        ]
    )
    assert training["preference"]["adapter_l2_change_from_sft480"] > 0
    with (ROOT / "docs/figures/v4/per-case.csv").open() as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 128
    csv_index = {(r["variant"], r["case_id"]): r for r in csv_rows}
    for variant in ARMS:
        rows = [r for r in report["records"] if r["variant"] == variant]
        s = report["summary"][variant]
        assert s["tasks"] == 32 and s["task_success"] == sum(r["task_success"] for r in rows)
        for r in rows:
            case = cases[r["case_id"]]
            assert csv_index[variant, r["case_id"]]["expected_outcome"] == case["expected"]["kind"]
            asks = []
            sent = []
            for o in r["trace"]:
                if o["action"]:
                    assert json.loads(o["raw"]) == o["action"]
                    if o["action"]["tool"] == "ask_customer":
                        asks.append(o["action"]["arguments"]["field"])
                if "question" in o["result"] and "error" not in o["result"]:
                    sent.append(o["result"]["field"])
            repeats = sum(n - 1 for n in Counter(asks).values())
            delivered = sum(n - 1 for n in Counter(sent).values())
            assert (
                repeats == r["repeated_question_attempts"] and delivered == r["repeated_questions_delivered"]
            )
            assert int(csv_index[variant, r["case_id"]]["repeated_question_attempts"]) == repeats
            assert (csv_index[variant, r["case_id"]]["task_success"] == "True") == r["task_success"]
            error = case["expected"]["kind"] == "exchange" and any(
                o["action"] and o["action"]["tool"] == "propose_return" for o in r["trace"]
            )
            assert r["exchange_as_return"] == error
            assert r["order_before_confirmation"] == case["orders"]
            committed = [e for e in r["events"] if e["kind"] == "operation_committed"]
            if not case["confirmation"]:
                assert not committed and r["order_after_confirmation"] == case["orders"]
            if r["task_success"]:
                required = case["required_clarification"]
                assert required is None or required in r["questions"]
                if case["expected"]["kind"] == "handoff":
                    assert r["status_final"] == "handoff" and r["order_after_confirmation"] == case["orders"]
                else:
                    p = next(
                        o["result"]["proposal"] for o in reversed(r["trace"]) if "proposal" in o["result"]
                    )
                    assert (
                        p["kind"] == case["expected"]["kind"]
                        and p["amount_cents"] == case["expected"]["amount_cents"]
                    )
                    if p["kind"] == "exchange":
                        assert p["new_item_id"] == case["expected"]["new_item_id"]
                    if case["confirmation"]:
                        assert len(committed) == 1 and r["status_final"] == "completed"
        for key in [
            "repeated_question_attempts",
            "repeated_questions_delivered",
            "exchange_as_return",
            "model_calls",
        ]:
            assert s[key] == sum(r[key] for r in rows)
        assert sum(sum(row) for row in values["confusion"][variant]) == 32
        attempted = [r for r in rows if r["model_handoff"]]
        diagnostic = values["handoff_diagnostics"][variant]
        assert diagnostic["model_handoffs"] == len(attempted)
        assert diagnostic["verified_handoffs"] == sum(
            r["expected"]["kind"] == "handoff" and r["task_success"] for r in attempted
        )
        assert diagnostic["unnecessary_handoffs"] == sum(
            r["expected"]["kind"] != "handoff" for r in attempted
        )
        assert (
            diagnostic["model_handoffs"]
            == diagnostic["verified_handoffs"]
            + diagnostic["unnecessary_handoffs"]
            + diagnostic["unverified_policy_handoffs"]
        )
        for i, e in enumerate(["cancel", "return", "exchange", "handoff"]):
            for j, p in enumerate(["cancel", "return", "exchange", "handoff", "unfinished"]):
                assert values["confusion"][variant][i][j] == sum(
                    r["expected"]["kind"] == e and r["predicted_outcome"] == p for r in rows
                )
    assert validated_v4_report(ROOT)["available"]
    return {
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "source_sha256": sha(__file__),
        "records_reconciled": 128,
        "new_test_tasks": 32,
        "no_test_based_selection": True,
        "final_checkpoint_fixed_before_test": True,
        "question_attempts_and_delivery_separately_reconciled": True,
        "scope": "Source/trace/export reconciliation, not independent human adjudication.",
    }


if __name__ == "__main__":
    result = verify()
    (REPORTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
