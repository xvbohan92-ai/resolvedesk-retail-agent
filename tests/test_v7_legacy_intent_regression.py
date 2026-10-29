"""CPU-only retention harness contracts; these mock responses are not model evidence."""

import copy
import json
from collections import Counter

import httpx
import pytest

from training.v7 import legacy_intent_regression as probe


def raw_request(case, **changes):
    args = {**case["intent_initial"], "evidence": case["initial_message"]}
    args.update(changes)
    return json.dumps({"tool": "record_request", "arguments": args})


def fixture_plan(tmp_path, monkeypatch, cases):
    path = tmp_path / "cpu-plan.json"
    path.write_text('{"scope":"CPU mock fixture only"}\n')
    monkeypatch.setattr(probe, "PLAN", path)
    plan = {
        "case_ids": [c["id"] for c in cases],
        "selected_adapter_sha256": "cpu-adapter-sha",
        "source_inputs_sha256": {
            "reports/v7/training.json": "cpu-training-sha",
            "training/v7/datasets/manifest.json": "cpu-manifest-sha",
        },
        "base": "cpu-base",
        "base_revision": "cpu-revision",
        "new_prompt_sha256": {c["id"]: probe.prompt_hash(probe.make_prompt(c)) for c in cases},
    }
    identity = {
        "provider": "mlx",
        "variant": "qlora_v7",
        "workflow": "v7",
        "model_interface": "v7",
        "adapter_sha256": "cpu-adapter-sha",
        "training_report_sha256": "cpu-training-sha",
        "manifest_sha256": "cpu-manifest-sha",
        "base": "cpu-base",
        "base_revision": "cpu-revision",
        "cloud_api": False,
    }
    baseline = [
        {
            "case_id": c["id"],
            "response_valid": True,
            "raw": raw_request(c),
            "metrics": probe.first_metrics(raw_request(c), c),
        }
        for c in cases
    ]
    return plan, identity, baseline


def test_original_archived_cohort_uses_first_raw_not_later_accepted_intent():
    cases = probe.legacy_cases()
    rows = probe.archived_rows(cases)
    summary = probe.summarize(rows, cases)
    assert Counter(c["intent_initial"]["fallback_return"] for c in cases) == {
        "yes": 6,
        "no": 12,
        "unknown": 30,
    }
    assert summary["cohort_tasks"] == summary["attempted_tasks"] == summary["raw_responses_available"] == 48
    assert summary["true_permission_yes"] == 3 and summary["unsupported_permission_yes"] == 0
    assert summary["authorized_tasks"] == 6 and summary["non_authorized_tasks"] == 42
    assert summary["initial_raw_parse_failures"] == 0 and summary["initial_enum_error_tasks"] == 2


def test_first_raw_parse_keeps_invalid_enum_and_never_substitutes_backend_values():
    case = probe.legacy_cases()[0]
    raw = raw_request(case, reason="non-enum raw reason", fallback_return="yes")
    metrics = probe.first_metrics(raw, case)
    assert metrics["initial_enum_errors"] == {"reason": "non-enum raw reason"}
    assert metrics["intent_initial"]["reason"] == "non-enum raw reason"
    assert not metrics["initial_raw_parse_failure"] and not metrics["intent_enum_valid"]
    assert metrics["permission_yes"] and not metrics["permission_authorized"]
    malformed = probe.first_metrics('{"tool":"record_request","arguments":{"reason":"invalid"}}', case)
    assert malformed["initial_raw_parse_failure"] and malformed["initial_enum_errors"] == {
        "reason": "invalid"
    }
    assert probe.first_metrics("not JSON", case)["initial_raw_parse_failure"]


def test_prompt_projects_only_public_world_and_actual_initial_message():
    case = copy.deepcopy(probe.legacy_cases()[0])
    case.update(
        expected={"poison": "SECRET_EXPECTED"}, answers={"poison": "SECRET_REPLY"}, family="SECRET_FAMILY"
    )
    prompt = probe.make_prompt(case)
    visible = json.loads(prompt[1]["content"])
    assert visible["messages"] == [{"role": "customer", "content": case["initial_message"]}]
    assert visible["workflow_version"] == "v7" and visible["completed_reads"] == []
    assert all(
        marker not in json.dumps(prompt) for marker in ["SECRET_EXPECTED", "SECRET_REPLY", "SECRET_FAMILY"]
    )
    assert probe.prompt_hash(prompt) == probe.prompt_hash(probe.make_prompt(case))


def test_transport_failure_is_retained_and_resume_only_calls_untouched_case(tmp_path, monkeypatch):
    cases = probe.legacy_cases()[:2]
    plan, identity, baseline = fixture_plan(tmp_path, monkeypatch, cases)
    requests = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=identity)
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(503, text="CPU mock transient failure: retain and never retry")
        return httpx.Response(
            200,
            json={
                "text": raw_request(cases[1]),
                "timing": {"provider": "mlx", "variant": "qlora_v7", "adapter_sha256": "cpu-adapter-sha"},
            },
        )

    destination = tmp_path / "cpu-results.json"
    with httpx.Client(base_url=probe.SERVICE, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            probe.execute_calls(client, plan, cases, baseline, destination)
        partial = json.loads(destination.read_text())
        assert partial["status"] == "partial" and len(partial["records"]) == 1
        assert partial["records"][0]["response_body"].startswith("CPU mock transient")
        assert partial["records"][0]["response_valid"] is False
        result = probe.execute_calls(client, plan, cases, baseline, destination, resume=True)
    assert len(requests) == 2 and requests[0] != requests[1]
    assert result["records"][0] == partial["records"][0]
    assert (
        result["status"] == "complete" and result["summary"]["current_v7"]["unavailable_raw_responses"] == 1
    )
    assert (tmp_path / "legacy-intent-before-resume-1.json").exists()


def test_malformed_model_output_is_a_retained_prediction_not_a_transport_retry(tmp_path, monkeypatch):
    cases = probe.legacy_cases()[:2]
    plan, identity, baseline = fixture_plan(tmp_path, monkeypatch, cases)
    calls = []

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json=identity)
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "text": "{malformed raw output",
                "timing": {"provider": "mlx", "variant": "qlora_v7", "adapter_sha256": "cpu-adapter-sha"},
            },
        )

    with httpx.Client(base_url=probe.SERVICE, transport=httpx.MockTransport(handler)) as client:
        result = probe.execute_calls(client, plan, cases, baseline, tmp_path / "cpu-results.json")
    assert len(calls) == 2 and result["summary"]["current_v7"]["initial_raw_parse_failures"] == 2
    assert result["summary"]["current_v7"]["unavailable_raw_responses"] == 0
    assert all(r["raw"] == "{malformed raw output" for r in result["records"])


def test_wrong_service_identity_stops_before_any_generate_call(tmp_path, monkeypatch):
    cases = probe.legacy_cases()[:1]
    plan, identity, baseline = fixture_plan(tmp_path, monkeypatch, cases)
    identity["adapter_sha256"] = "wrong-cpu-adapter"
    paths = []

    def handler(request):
        paths.append(request.url.path)
        return httpx.Response(200, json=identity)

    destination = tmp_path / "cpu-results.json"
    with httpx.Client(base_url=probe.SERVICE, transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AssertionError):
            probe.execute_calls(client, plan, cases, baseline, destination)
    assert paths == ["/health"] and not destination.exists()
