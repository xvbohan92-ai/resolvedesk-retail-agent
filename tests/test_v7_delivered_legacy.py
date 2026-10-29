"""Reconcile delivered JSON artifacts on CPU, without weights or running model services."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx

from training.v7 import write_summary

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "reports/v7"
ENUMS = {
    "operation": {"cancel", "return", "exchange", "handoff", "unknown"},
    "payment_method": {"original", "gift_card", "unknown"},
    "reason": {"no longer needed", "ordered by mistake", "unknown"},
    "fallback_return": {"yes", "no", "unknown"},
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def independent_metrics(raw, case):
    try:
        action = json.loads(raw)
    except (ValueError, TypeError):
        action = None
    action = action if isinstance(action, dict) else None
    args = action.get("arguments") if action else None
    fields = set(case["intent_initial"]) | {"evidence"}
    intent = (
        {k: v for k, v in args.items() if k != "evidence"}
        if action
        and action.get("tool") == "record_request"
        and isinstance(args, dict)
        and set(args) == fields
        and all(isinstance(v, str) for v in args.values())
        else None
    )
    errors = (
        {
            k: args[k]
            for k, choices in ENUMS.items()
            if k in args and (not isinstance(args[k], str) or args[k] not in choices)
        }
        if action and action.get("tool") == "record_request" and isinstance(args, dict)
        else {}
    )
    return {
        "intent_initial": intent,
        "initial_raw_parse_failure": intent is None,
        "initial_enum_errors": errors,
        "intent_enum_valid": intent is not None and not errors,
        "permission_authorized": case["intent_initial"]["fallback_return"] == "yes",
        "permission_yes": bool(intent and intent["fallback_return"] == "yes"),
        "intent_field_correct": {
            k: bool(intent and intent[k] == v) for k, v in case["intent_initial"].items()
        },
    }


def independent_summary(rows, cases):
    available = [r for r in rows if r["response_valid"]]
    ids = {r["case_id"] for r in rows}
    return {
        "cohort_tasks": len(cases),
        "attempted_tasks": len(rows),
        "raw_responses_available": len(available),
        "unavailable_raw_responses": len(rows) - len(available),
        "authorized_tasks": sum(c["intent_initial"]["fallback_return"] == "yes" for c in cases),
        "non_authorized_tasks": sum(c["intent_initial"]["fallback_return"] != "yes" for c in cases),
        "true_permission_yes": sum(
            r["metrics"]["permission_authorized"] and r["metrics"]["permission_yes"] for r in available
        ),
        "unsupported_permission_yes": sum(
            not r["metrics"]["permission_authorized"] and r["metrics"]["permission_yes"] for r in available
        ),
        "initial_raw_parse_failures": sum(r["metrics"]["initial_raw_parse_failure"] for r in available),
        "initial_enum_error_tasks": sum(bool(r["metrics"]["initial_enum_errors"]) for r in available),
        "initial_reason_enum_error_tasks": sum(
            "reason" in r["metrics"]["initial_enum_errors"] for r in available
        ),
        "intent_field_correct": {
            k: sum(r["metrics"]["intent_field_correct"][k] for r in available)
            for k in cases[0]["intent_initial"]
        },
        "intent_joint_correct": sum(all(r["metrics"]["intent_field_correct"].values()) for r in available),
        "missing_case_ids": [c["id"] for c in cases if c["id"] not in ids],
    }


def test_delivered_retention_inputs_and_duplicate_prompt_scope():
    plan = read(FOLDER / "legacy-intent-regression-plan.json")
    report = read(FOLDER / "legacy-intent-regression.json")
    cases = [c for c in read(ROOT / "training/v6_1/datasets/cases.json") if c["split"] == "test"]
    assert report["status"] == "complete" and report["plan_sha256"] == digest(
        FOLDER / "legacy-intent-regression-plan.json"
    )
    assert report["source_inputs_sha256"] == plan["source_inputs_sha256"]
    for name, wanted in plan["source_inputs_sha256"].items():
        assert digest(ROOT / name) == wanted
    assert [r["case_id"] for r in report["records"]] == plan["case_ids"] == [c["id"] for c in cases]
    prompts = Counter(c["initial_message"] for c in cases)
    assert len(cases) == 48 and len(prompts) == 46 and sorted(prompts.values()) == [1] * 44 + [2, 2]
    assert len({r["request_sha256"] for r in report["records"]}) == 46
    assert Counter(c["intent_initial"]["fallback_return"] for c in cases) == {
        "yes": 6,
        "no": 12,
        "unknown": 30,
    }
    assert not report["checkpoint_selection_modified"] and not report["business_gate_modified"]
    assert (
        report["selected_adapter_sha256"]
        == report["identity"]["adapter_sha256"]
        == plan["selected_adapter_sha256"]
    )


def test_delivered_first_raw_responses_recompute_without_model_repair():
    plan = read(FOLDER / "legacy-intent-regression-plan.json")
    report = read(FOLDER / "legacy-intent-regression.json")
    cases = [c for c in read(ROOT / "training/v6_1/datasets/cases.json") if c["split"] == "test"]
    by_id = {c["id"]: c for c in cases}
    for row in report["records"]:
        assert (
            row["attempted_generate_requests"] == 1
            and row["response_valid"]
            and row["response_status"] == 200
        )
        body = json.loads(row["response_body"])
        assert body == row["response_json"] and body["text"] == row["raw"]
        assert (
            row["timing"] == body["timing"]
            and row["timing"]["adapter_sha256"] == plan["selected_adapter_sha256"]
        )
        assert (
            hashlib.sha256(canonical(row["request_messages"]).encode()).hexdigest() == row["request_sha256"]
        )
        assert row["request_sha256"] == plan["new_prompt_sha256"][row["case_id"]]
        visible = json.loads(row["request_messages"][1]["content"])
        assert visible["messages"] == [
            {"role": "customer", "content": by_id[row["case_id"]]["initial_message"]}
        ]
        assert (
            visible["completed_reads"] == []
            and visible["recent_events"] == []
            and visible["current_request"] is None
        )
        assert not ({"family", "answers", "intent_initial", "expected"} & set(visible))
        assert row["metrics"] == independent_metrics(row["raw"], by_id[row["case_id"]])
    summary = independent_summary(report["records"], cases)
    assert summary == report["summary"]["current_v7"]
    assert summary["true_permission_yes"] == summary["authorized_tasks"] == 6
    assert summary["unsupported_permission_yes"] == summary["initial_reason_enum_error_tasks"] == 0
    assert summary["intent_field_correct"]["requested_option"] == 45 and summary["intent_joint_correct"] == 44


def test_archived_baseline_is_the_first_raw_event_not_a_repaired_record():
    plan = read(FOLDER / "legacy-intent-regression-plan.json")
    report = read(FOLDER / "legacy-intent-regression.json")
    cases = [c for c in read(ROOT / "training/v6_1/datasets/cases.json") if c["split"] == "test"]
    by_id = {c["id"]: c for c in cases}
    archive = {
        r["case_id"]: r
        for r in read(ROOT / "reports/v6_1/evaluation.json")["records"]
        if r["variant"] == "qlora_v6_1"
    }
    assert len(archive) == len(report["archived_records"]) == 48
    for row in report["archived_records"]:
        original = archive[row["case_id"]]
        first = next(e["detail"] for e in original["events"] if e["kind"] == "model_tool")
        assert first == original["trace"][0]
        assert row["raw"] == first["raw"] and row["timing"] == first["timing"]
        assert row["metrics"] == independent_metrics(first["raw"], by_id[row["case_id"]])
    summary = independent_summary(report["archived_records"], cases)
    assert summary == plan["archived_first_output_summary"] == report["summary"]["archived_v6_1"]
    assert summary["true_permission_yes"] == 3 and summary["authorized_tasks"] == 6
    assert summary["unsupported_permission_yes"] == 0 and summary["non_authorized_tasks"] == 42
    assert summary["initial_reason_enum_error_tasks"] == 2 and summary["intent_joint_correct"] == 36


def test_summary_fragments_read_delivered_evidence_without_weights_or_services(monkeypatch):
    original_open = Path.open

    def no_weights(path, *args, **kwargs):
        assert path.suffix not in {".safetensors", ".gguf"} and not path.is_relative_to(ROOT / "data")
        return original_open(path, *args, **kwargs)

    def no_service(*args, **kwargs):
        raise AssertionError("Delivered-artifact checks must not access model services")

    monkeypatch.setattr(Path, "open", no_weights)
    monkeypatch.setattr(httpx, "Client", no_service)
    legacy = write_summary.legacy_retention_section(ROOT)
    assert "46 unique initial prompts and two duplicate pairs" in legacy
    assert "Both **workflow and weights changed**" in legacy and "development retention" in legacy
    assert "| Explicit conditional grants recognized | 3/6 | 6/6 |" in legacy
    assert "| First-output reason-enum errors ↓ | 2/48 | 0/48 |" in legacy
    assert "| Raw replacement option correct | 47/48 | 45/48 |" in legacy
    cloud = write_summary.colab_evidence_paragraph(ROOT)
    assert "one-update MLX CUDA preflight" in cloud and "Tesla T4" in cloud
    assert "184.94 seconds" in cloud and "0.00384618 → 0.00191919" in cloud
    assert "PyTorch/PEFT FP16 smoke completed four true optimizer updates" in cloud
    assert "19.692511 seconds" in cloud and "0.0283294" in cloud
    assert "formal training completed 1,344 true optimizer updates" in cloud
    assert "6589.486070 seconds" in cloud and "824/824 unique physical rows" in cloud
    assert (
        "0.6083016" in cloud and "33fe5c505ddf08cd8cb4032dbd9d350fa205134417ec0f1f90648f4af658e053" in cloud
    )
    assert "48-case, two-arm before/after evaluation was interrupted" in cloud
    assert "last verified snapshot contains 84/96 records" in cloud
    assert "no new model-quality score is reported here" in cloud
    assert "original V7 scores and failed business gate" in cloud
    assert "formal training is in progress" not in cloud
    assert "not an isolated GPU hardware speed comparison" in cloud
