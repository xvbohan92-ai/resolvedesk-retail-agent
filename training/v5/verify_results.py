"""Independent raw-trace, business-score, intent-label and export reconciliation."""

import csv
import json
import math
from collections import Counter

from resolve_desk.evidence_v5 import validated_report
from training.common import ROOT, sha
from training.v2.scoring import score
from training.v4.verify_results import verify as verify_previous
from training.v5.common import DATA, REPORTS, verify_frozen
from training.v5.evaluation import ARMS, promotion, selection_key
from training.v5.fixtures import load_cases


def verify():
    verify_previous()
    manifest = verify_frozen()
    report = json.loads((REPORTS / "evaluation.json").read_text())
    training = json.loads((REPORTS / "training.json").read_text())
    updates = json.loads((REPORTS / "updates.json").read_text())
    preflight = json.loads((REPORTS / "preflight.json").read_text())
    values = json.loads((ROOT / "docs/figures/v5/values.json").read_text())
    cases = {c["id"]: c for c in load_cases("test")}
    assert len(cases) == 32 and len(report["records"]) == 128
    assert Counter((r["variant"], r["case_id"]) for r in report["records"]) == Counter(
        {(v, cid): 1 for v in ARMS for cid in cases}
    )
    assert training["dataset"] == updates["dataset"] == manifest
    assert training["completed_updates"] == updates["completed_updates"] == 512
    assert len(updates["metrics"]) == 512
    assert [r["iteration"] for r in updates["metrics"]] == list(range(1, 513))
    assert all(math.isfinite(r["loss"]) and r["loss"] >= 0 for r in updates["metrics"])
    assert updates["adapter_l2_change"] > 0 and not updates["preflight_only"]
    assert preflight["preflight_only"] and not updates["preflight_weights_reused"]
    assert preflight["metrics"][0]["post_update_same_sample_loss"] < preflight["metrics"][0]["loss"]
    assert preflight["token_audit"]["loss_error_vs_full_output_projection"] < 0.001
    assert updates["token_audit"]["all_targets_covered"]
    assert updates["token_audit"]["uniform_completion_weight"] == 1
    assert training["initialization_sha256"] == manifest["initial_adapter_sha256"]
    assert report["training_sha256"] == sha(REPORTS / "training.json")
    assert report["manifest_sha256"] == sha(DATA / "manifest.json")
    assert report["evaluator_sha256"] == sha(ROOT / "training/v5/evaluate.py")
    assert not training["test_used_for_selection"]
    candidates = [json.loads((REPORTS / f"validation-{n}.json").read_text()) for n in [256, 512]]
    best = max(candidates, key=lambda c: selection_key(c["summary"]))
    assert best["iteration"] == training["best_checkpoint"]["iteration"]
    assert best["adapter_sha256"] == training["selected_adapter_sha256"]
    for candidate in candidates:
        assert len(candidate["records"]) == 32
        assert all(r["case_id"].startswith("v5-valid-") for r in candidate["records"])
        assert candidate["summary"]["task_success"] == sum(r["task_success"] for r in candidate["records"])
    with (ROOT / "docs/figures/v5/per-case.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 128
    exported = {(r["variant"], r["case_id"]): r for r in csv_rows}
    for r in report["records"]:
        case, trace = cases[r["case_id"]], r["trace"]
        assert len(trace) == r["model_calls"] <= 12
        assert all(o["timing"]["provider"] == "mlx" for o in trace)
        assert [e["detail"] for e in r["events"] if e["kind"] == "model_tool"] == trace
        questions, pending, turns = [], None, [case["initial_message"]]
        for event in r["events"]:
            if event["kind"] == "model_tool":
                result = event["detail"]["result"]
                if "question" in result and "error" not in result:
                    pending = result["field"]
            elif event["kind"] == "customer_message":
                assert pending is not None
                assert event["detail"]["content"] == case["answers"][pending]
                turns.append(event["detail"]["content"])
                questions.append(pending)
                pending = None
        assert questions == r["questions"]
        for o in trace:
            if o["action"] is not None:
                assert json.loads(o["raw"]) == o["action"]
            if "resolution_request" in o["result"]:
                evidence = o["result"]["evidence"]
                latest = turns[o["result"]["source_message_count"] - 1]
                assert evidence in latest and len(evidence) >= min(8, len(latest))
                assert evidence == o["action"]["arguments"]["evidence"]
                assert o["result"]["resolution_request"] == {
                    k: v for k, v in o["action"]["arguments"].items() if k != "evidence"
                }
        initial = next(
            (
                o["result"]["resolution_request"]
                for o in trace
                if "resolution_request" in o["result"] and o["result"]["source_message_count"] == 1
            ),
            None,
        )
        assert r["intent_initial"] == initial
        proposals = [o["result"]["proposal"] for o in trace if "proposal" in o["result"]]
        proposal = proposals[-1] if proposals else None
        assert r["predicted_outcome"] == (
            proposal["kind"] if proposal else "handoff" if r["status_final"] == "handoff" else "unfinished"
        )
        judged = score(
            case,
            {"status": r["status_final"], "proposal": proposal, "observations": trace},
            r["order_after_confirmation"],
            r["order_before_confirmation"],
            questions,
        )
        assert all(r[k] == v for k, v in judged.items()), r["case_id"]
        assert r["order_before_confirmation"] == case["orders"]
        fields = [
            o["action"]["arguments"]["field"]
            for o in trace
            if o["action"] and o["action"]["tool"] == "ask_customer"
        ]
        assert r["repeated_question_attempts"] == sum(n - 1 for n in Counter(fields).values())
        unnecessary = case["expected"]["kind"] != "handoff"
        assert r["unnecessary_handoff_attempts"] == sum(
            bool(o["action"] and o["action"]["tool"] == "handoff" and unnecessary) for o in trace
        )
        assert r["unnecessary_handoff_executed"] == int(
            unnecessary and any(o["result"].get("handoff") for o in trace)
        )
        for key, val in r["intent_field_correct"].items():
            assert val == bool(
                r["intent_initial"] and r["intent_initial"][key] == case["intent_initial"][key]
            )
        row = exported[r["variant"], r["case_id"]]
        assert row["task_success"] == str(r["task_success"])
        for key in [
            "model_calls",
            "repeated_question_attempts",
            "unnecessary_handoff_attempts",
            "unnecessary_handoff_executed",
            "quote_rejections",
            "policy_gate_rejections",
        ]:
            assert int(row[key]) == r[key]
    for variant in ARMS:
        rows = [r for r in report["records"] if r["variant"] == variant]
        s = report["summary"][variant]
        assert s["task_success"] == sum(r["task_success"] for r in rows)
        assert s["fallback_success"] == sum(
            r["task_success"] for r in rows if r["family"] == "exchange_fallback"
        )
        assert s["fulfillable_exchange_success"] == sum(
            r["task_success"] for r in rows if r["expected"]["kind"] == "exchange"
        )
        assert s["handoff_success"] == sum(
            r["task_success"] and r["model_handoff"] for r in rows if r["expected"]["kind"] == "handoff"
        )
        for key in [
            "unnecessary_handoff_attempts",
            "unnecessary_handoff_executed",
            "repeated_question_attempts",
            "model_calls",
            "exchange_as_return",
        ]:
            assert s[key] == sum(r[key] for r in rows)
        for family in values["family_success"]:
            assert values["family_success"][family][variant] == sum(
                r["task_success"] for r in rows if r["family"] == family
            )
    for variant, metric in values["permission_diagnostics"].items():
        rows = [r for r in report["records"] if r["variant"] == variant]

        def yes(r):
            return cases[r["case_id"]]["intent_initial"]["fallback_return"] == "yes"

        def predicted(r):
            return bool(r["intent_initial"] and r["intent_initial"]["fallback_return"] == "yes")

        assert metric["true_affirmative"] == sum(yes(r) and predicted(r) for r in rows)
        assert metric["unsupported_affirmative"] == sum(not yes(r) and predicted(r) for r in rows)
        assert metric["authorized_tasks"] == sum(yes(r) for r in rows) == 4
        assert metric["non_authorized_tasks"] == sum(not yes(r) for r in rows) == 28
    assert report["release_gate"] == promotion(report["records"])
    assert validated_report(ROOT)["available"]
    # Entity separation is checked across splits and every prior round.
    split_ids = {s: {o["id"] for c in load_cases(s) for o in c["orders"]} for s in ["train", "valid", "test"]}
    assert [len(v) for v in split_ids.values()] == [56, 28, 28]
    assert not (
        split_ids["train"] & split_ids["valid"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["valid"] & split_ids["test"]
    )
    for source in [
        "training/datasets/cases.json",
        "training/v2/datasets/cases.json",
        "training/v3/datasets/cases.json",
        "training/v4/datasets/cases.json",
    ]:
        old = json.loads((ROOT / source).read_text())
        assert not {o["id"] for c in old for o in c["orders"]} & set.union(*split_ids.values())
    result = {
        "records_reconciled": 128,
        "fresh_test_tasks": 32,
        "no_test_based_selection": True,
        "business_scores_recomputed": True,
        "intent_and_attempt_metrics_reconciled": True,
        "evaluation_sha256": sha(REPORTS / "evaluation.json"),
        "source_sha256": sha(__file__),
        "scope": "Independent code/trace/export reconciliation; not independent human annotation.",
    }
    (REPORTS / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify()))
