"""CPU-only report consistency fixtures; no model inference or experiment artifacts."""

import copy
import json
import random
from pathlib import Path

import httpx
import pytest

from resolve_desk.resolution_v7 import ResolutionStore, messages
from training.v7 import integration, protocol_audit
from training.v7.evaluation import ARMS, run_case, summarize
from training.v7.fixtures import make_case
from training.v7.prepare import reference
from training.v7.verify_results import audit_record, reconcile_protocol_rows
from training.v7.write_summary import protocol_observation, protocol_table


class CapturedStore(ResolutionStore):
    def create(self, case):
        state = super().create(case)
        self.latest_sid = state["id"]
        return state


class CPUReference:
    def __init__(self, case, store, fault):
        self.case, self.store, self.fault = case, store, fault
        self.injected = False
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        state = self.store.get(self.store.latest_sid)
        action = reference(self.case, state)
        if self.fault == "first_enum" and self.calls == 1:
            action["arguments"]["reason"] = "Keep the purchased item"
        elif self.fault == "late_enum" and len(state["messages"]) > 1 and not self.injected:
            assert action["tool"] == "record_request"
            action["arguments"]["reason"] = "This is not a cancellation reason"
            self.injected = True
        elif self.fault == "policy_loop" and len(state["messages"]) > 1:
            action = {"tool": "get_order", "arguments": {"order_id": self.case["orders"][0]["id"]}}
        return json.dumps(action), {"provider": "cpu_reference", "output_tokens": 1}


def record_fixture(tmp_path, family, fault):
    case = make_case("valid", family, 0)
    if fault == "late_enum":
        # Force actual model interpretation of this noncanonical field reply.
        case["answers"]["email"] = (
            "The only address that should be used for this example is " + case["customer"]["email"]
        )
    store = CapturedStore(tmp_path / "cpu-reference.sqlite")
    provider = CPUReference(case, store, fault)
    record = {"variant": "qlora_v7", **run_case(store, case, provider, messages)}
    return case, record, provider


def test_first_reason_error_is_not_replaced_by_later_accepted_repair(tmp_path):
    case, record, _ = record_fixture(tmp_path, "exchange", "first_enum")
    assert record["task_success"]
    assert record["initial_enum_errors"] == {"reason": "Keep the purchased item"}
    assert not record["intent_field_correct"]["reason"]
    assert record["reason_enum_errors"] == 1 and record["reason_enum_recovered"]
    original = copy.deepcopy(record)
    audit_record(record, case, expected_provider="cpu_reference")
    assert record == original


def test_later_turn_reason_error_is_not_reported_as_initial_reason_error(tmp_path):
    case, record, _ = record_fixture(tmp_path, "missing_email", "late_enum")
    assert record["task_success"]
    assert record["initial_enum_errors"] == {}
    assert record["intent_field_correct"]["reason"]
    assert record["reason_enum_errors"] == 1 and record["reason_enum_recovered"]
    aggregate = summarize([record])
    assert aggregate["initial_enum_error_tasks"] == 0
    assert aggregate["reason_enum_error_tasks"] == aggregate["reason_enum_recovered_tasks"] == 1
    audit_record(record, case, expected_provider="cpu_reference")


def test_twelve_calls_are_total_across_customer_reply_and_rejected_tool_loops(tmp_path):
    case, record, provider = record_fixture(tmp_path, "missing_email", "policy_loop")
    assert record["questions"] == ["email"]
    assert provider.calls == record["model_calls"] == 12
    assert record["status_final"] == "running" and not record["task_success"]
    assert record["order_before_confirmation"] == record["order_after_confirmation"] == case["orders"]
    audit_record(record, case, expected_provider="cpu_reference")


def test_selected_672_coverage_is_494_and_not_the_full_824_rows():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "training/v7/config.json").read_text())
    rows = [json.loads(line) for line in (root / "training/v7/datasets/train.jsonl").read_text().splitlines()]
    pools = {
        "record": [i for i, row in enumerate(rows) if row["origin"] == "record_request"],
        "other": [i for i, row in enumerate(rows) if row["origin"] != "record_request"],
    }
    rng, pending, indices = random.Random(config["seed"]), {key: [] for key in pools}, []
    for step in range(1, config["iters"] + 1):
        key = "record" if step % 2 else "other"
        if not pending[key]:
            pending[key] = rng.sample(pools[key], len(pools[key]))
        indices.append(pending[key].pop())
    # Counts are order-independent: 336 execution rows + every 158 record row.
    assert len(rows) == 824 and {key: len(value) for key, value in pools.items()} == {
        "record": 158,
        "other": 666,
    }
    assert len(set(indices[:672])) == 494
    assert len(set(indices[:1344])) == 824
    assert len({rows[i]["case_id"] for i in indices[:672]}) == 96


def test_supplemental_reconciliation_rejects_tampered_summary_and_missing_arm(tmp_path):
    case, record, _ = record_fixture(tmp_path, "exchange", "first_enum")
    report = {"records": [{**copy.deepcopy(record), "variant": arm} for arm in ARMS]}
    rows = [protocol_audit.audit_record(r, case) for r in report["records"]]
    protocol = {"records": rows, "summary": protocol_audit.summarize(rows), "records_audited": len(rows)}
    original = copy.deepcopy(protocol)
    reconcile_protocol_rows(report, protocol, {case["id"]: case})
    assert protocol == original
    protocol["summary"]["qlora_v7"]["committed_returns_without_scoped_yes"] += 1
    with pytest.raises(AssertionError):
        reconcile_protocol_rows(report, protocol, {case["id"]: case})
    with pytest.raises(AssertionError):
        reconcile_protocol_rows({"records": report["records"][:-1]}, original, {case["id"]: case})


def test_protocol_reporting_keeps_event_units_and_uncertainty_separate():
    counts = dict.fromkeys(protocol_audit.COUNTS, 0)
    counts.update(
        tasks=48,
        gold_exchange_tasks=30,
        raw_return_attempts=7,
        return_attempts_without_scoped_yes=5,
        return_proposals=3,
        return_proposals_without_scoped_yes=2,
        committed_returns=2,
        committed_returns_without_scoped_yes=1,
        final_return_orders=1,
        final_returns_without_verified_commit=1,
    )
    protocol = {"summary": {arm: counts for arm in ARMS}, "records": []}
    table = protocol_table(protocol)
    assert "5/7" in table and "2/3" in table and "1/2" in table and "1/1" in table
    assert "Rejected raw return calls" in table
    assert "even if the frozen business gate passes" in protocol_observation(counts)
    counts.update(
        return_proposals_without_scoped_yes=0,
        committed_returns_without_scoped_yes=0,
        final_returns_without_verified_commit=0,
    )
    assert "known operation-rewrite bypass" in protocol_observation(counts)
    assert "zero observations do not establish execution safety" in protocol_observation(counts)


class ProbeClient:
    def __init__(self, responses, fail_messages=False):
        self.responses, self.fail_messages = responses, fail_messages
        self.started, self.confirmed = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def get(self, path):
        return httpx.Response(
            200,
            request=httpx.Request("GET", "http://cpu-mock" + path),
            json={"models": {"finetuned": {"variant": "qlora_v7", "adapter_sha256": "cpu-unit-adapter"}}},
        )

    def post(self, path, json):
        request = httpx.Request("POST", "http://cpu-mock" + path)
        if path == "/api/sessions":
            self.started.append(json["case_id"])
            return httpx.Response(200, request=request, json=copy.deepcopy(self.responses[json["case_id"]]))
        if path.endswith("/messages") and self.fail_messages:
            return httpx.Response(503, request=request, json={"detail": "CPU mock transport failure"})
        self.confirmed.append(path)
        raise AssertionError("Unexpected confirmation/retry during CPU reporting test")


def probe_environment(tmp_path, monkeypatch, *, infrastructure=False):
    cases = [
        {
            "id": "cpu-first",
            "family": "exchange_fallback",
            "orders": [{"status": "delivered"}],
            "intent_initial": {"operation": "exchange"},
            "answers": {"email": "a@example.com"},
            "confirmation": True,
        },
        {
            "id": "cpu-next",
            "family": "handoff",
            "orders": [{"status": "delivered"}],
            "intent_initial": {"operation": "handoff"},
            "answers": {},
            "confirmation": True,
        },
    ]
    observations = [{"timing": {"adapter_sha256": "cpu-unit-adapter"}, "result": {"field": "email"}}]
    first = {
        "session": {
            "id": "cpu-sid",
            "revision": 1,
            "status": "awaiting_customer" if infrastructure else "awaiting_confirmation",
            "observations": observations,
            "proposal": None if infrastructure else {"kind": "return"},
        },
        "orders": cases[0]["orders"],
        "timeline": [],
    }
    second = {
        "session": {
            "id": "cpu-next-sid",
            "status": "handoff",
            "observations": observations,
            "proposal": None,
        },
        "orders": cases[1]["orders"],
        "timeline": [],
    }
    client = ProbeClient({"cpu-first": first, "cpu-next": second}, fail_messages=infrastructure)
    (tmp_path / "training.json").write_text('{"selected_adapter_sha256":"cpu-unit-adapter"}')
    (tmp_path / "analysis-plan.json").write_text(
        json.dumps({"splits": {"valid": {"representatives": [c["id"] for c in cases]}}})
    )
    monkeypatch.setattr(integration, "REPORTS", tmp_path)
    monkeypatch.setattr(integration, "verify_frozen", lambda: {})
    monkeypatch.setattr(integration, "load_cases", lambda split: cases)
    monkeypatch.setattr(integration, "score", lambda *args: {"task_success": True})
    monkeypatch.setattr(integration.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr("sys.argv", ["cpu-integration-test"])
    return client, first


def test_semantic_probe_failure_retains_raw_response_and_continues_without_retry(tmp_path, monkeypatch):
    client, first = probe_environment(tmp_path, monkeypatch)
    integration.main()
    result = json.loads((tmp_path / "live-api-verification.json").read_text())
    assert result["status"] == "complete"
    assert client.started == ["cpu-first", "cpu-next"] and not client.confirmed
    bad, good = result["records"]
    assert bad["outcome_correct"] is False and good["outcome_correct"] is True
    assert bad["initial_response"] == bad["before_confirmation"] == first
    assert bad["integration_error_type"] == "semantic_or_safety_invariant"
    assert "scoped YES" in bad["integration_error"]


def test_infrastructure_failure_retains_initial_and_attempted_reply_then_stops(tmp_path, monkeypatch):
    client, first = probe_environment(tmp_path, monkeypatch, infrastructure=True)
    with pytest.raises(httpx.HTTPStatusError):
        integration.main()
    result = json.loads((tmp_path / "live-api-verification.json").read_text())
    assert result["status"] == "partial" and client.started == ["cpu-first"]
    bad = result["records"][0]
    assert bad["initial_response"] == bad["before_confirmation"] == first
    assert bad["scripted_replies"] == [{"requested_field": "email", "scripted_reply": "a@example.com"}]
    assert bad["infrastructure_status"] == 503
    assert bad["infrastructure_response"] == {"detail": "CPU mock transport failure"}
    assert bad["integration_error_type"] == "infrastructure_or_invalid_structure"


def test_invalid_json_response_is_preserved_before_stopping(tmp_path, monkeypatch):
    client, _ = probe_environment(tmp_path, monkeypatch)

    def invalid_response(path, json):
        client.started.append(json["case_id"])
        return httpx.Response(
            200, request=httpx.Request("POST", "http://cpu-mock" + path), text="CPU mock truncated JSON {"
        )

    monkeypatch.setattr(client, "post", invalid_response)
    with pytest.raises(json.JSONDecodeError):
        integration.main()
    result = json.loads((tmp_path / "live-api-verification.json").read_text())
    assert result["status"] == "partial" and client.started == ["cpu-first"]
    assert result["records"][0]["infrastructure_response"] == "CPU mock truncated JSON {"
    assert result["records"][0]["outcome_correct"] is False
