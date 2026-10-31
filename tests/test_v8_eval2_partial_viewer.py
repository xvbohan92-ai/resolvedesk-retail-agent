"""Read-only partial-report delivery fails closed and never serves private evidence."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from training.v8_eval2.metrics import ARMS
from training.v8_eval2_partial import viewer


def fixture_bundle():
    return {
        "values": {
            "analysis_id": "cpu-fixture-partial40",
            "paired_cases": 40,
            "paired_records": 120,
            "summary": {arm: {"tasks": 40} for arm in ARMS},
            "paired": {},
            "protocol_summary": {},
            "failed_records": [],
            "family_success": {},
            "training": {"new_optimizer_updates": 0},
            "saved_weights": {"before_sha256": "a" * 64},
            "coverage": {"included_families": [], "excluded_families": [], "per_family": {}},
            "scope": {"nonrandom_truncation": True},
            "private_raw": "must never be served",
            "resource_receipt": {"private_account": "must never be served"},
        },
        "private_source_path": "/not-a-public-artifact",
    }


def test_missing_or_rejected_evidence_has_no_scores_or_assets(tmp_path, monkeypatch):
    def missing(_root):
        raise ValueError("No independent verified partial evidence")

    monkeypatch.setattr(viewer, "verified_bundle", missing)
    client = TestClient(viewer.create_app(tmp_path))
    response = client.get("/api/report")
    assert response.status_code == 200
    data = response.json()
    assert data["available"] is False and data["original144_complete"] is False
    assert "summary" not in data
    assert client.get("/api/asset/evaluation.png").status_code == 404


def test_valid_payload_only_exposes_reviewed_aggregates(tmp_path, monkeypatch):
    monkeypatch.setattr(viewer, "verified_bundle", lambda _: fixture_bundle())
    data = TestClient(viewer.create_app(tmp_path)).get("/api/report").json()
    assert data["available"] is True and data["partial_analysis"] is True
    assert data["paired_cases"] == 40 and data["paired_records"] == 120
    assert data["original144_complete"] is data["verification_is_quality_pass"] is False
    assert "private_raw" not in data and "resource_receipt" not in data
    assert "private_source_path" not in data
    assert "must never be served" not in json.dumps(data)


@pytest.mark.parametrize(
    "path",
    (
        "/api/asset/resource-authorization.json",
        "/api/asset/call-journal.json",
        "/api/asset/adapter_model.safetensors",
        "/api/asset/analysis-plan.json",
        "/api/asset/..%2Fresource-authorization.json",
        "/docs",
        "/openapi.json",
    ),
)
def test_private_and_unlisted_paths_cannot_be_served(path, tmp_path, monkeypatch):
    def must_not_enter(_):
        pytest.fail("Private paths must be rejected before any report read")

    monkeypatch.setattr(viewer, "verified_bundle", must_not_enter)
    assert TestClient(viewer.create_app(tmp_path)).get(path).status_code == 404


def test_asset_requires_gate_and_uses_exact_whitelist(tmp_path, monkeypatch):
    path = tmp_path / viewer.OUTPUT / "per-case.csv"
    path.parent.mkdir(parents=True)
    path.write_text("fixture_header\nfixture_row\n")
    calls = []
    monkeypatch.setattr(viewer, "verified_bundle", lambda root: calls.append(root) or fixture_bundle())
    response = TestClient(viewer.create_app(tmp_path)).get("/api/asset/per-case.csv")
    assert response.status_code == 200 and response.text == path.read_text()
    assert len(calls) == 1
    assert response.headers["x-content-type-options"] == "nosniff"


def test_no_model_or_mutation_routes(tmp_path):
    client = TestClient(viewer.create_app(tmp_path))
    for path in ("/", "/api/report", "/api/asset/per-case.csv"):
        assert client.post(path, json={"execute": True}).status_code == 405
    assert client.get("/api/generate").status_code == 404
    assert client.get("/api/health").json()["model_routes"] is False


def test_persisted_export_gate_cannot_be_skipped(tmp_path, monkeypatch):
    from training.v8_eval2_partial import report_results

    monkeypatch.setattr(report_results, "load_verified", lambda _: fixture_bundle())
    expected = {"fixture": "verified_partial_exports"}
    monkeypatch.setattr(report_results, "verify_delivery", lambda *a, **k: expected)
    path = tmp_path / viewer.REPORTS / "figure-verification.json"
    path.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="persisted"):
        viewer.verified_bundle(tmp_path)
    path.write_text(json.dumps({"fixture": "tampered"}))
    with pytest.raises(ValueError, match="persisted"):
        viewer.verified_bundle(tmp_path)
    path.write_text(json.dumps(expected))
    assert viewer.verified_bundle(tmp_path)["values"]["paired_cases"] == 40
    bad = copy.deepcopy(fixture_bundle())
    bad["values"]["paired_cases"] = 39
    monkeypatch.setattr(report_results, "load_verified", lambda _: bad)
    with pytest.raises(ValueError, match="cohort"):
        viewer.verified_bundle(tmp_path)
