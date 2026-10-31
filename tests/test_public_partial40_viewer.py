"""Public result delivery needs no private logs, model weights or authorization."""

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from resolve_desk import public_partial40 as viewer


def test_actual_public_projection_is_read_only_and_not_raw_replay():
    client = TestClient(viewer.create_app())
    data = client.get("/api/report").json()
    assert data["available"] and data["read_only"] and data["partial_analysis"]
    assert data["original144_complete"] is data["verification_is_quality_pass"] is False
    assert data["private_raw_or_tokens_verified_by_this_page"] is False
    assert data["paired_cases"] == 40 and data["paired_records"] == 120
    assert [data["summary"][a]["task_success"] for a in data["arms"]] == [30, 30, 33]
    assert len(data["failed_records"]) == 27
    assert "request_messages" not in data and "resource_receipt" not in data


def test_missing_public_data_fails_closed(tmp_path):
    client = TestClient(viewer.create_app(tmp_path))
    data = client.get("/api/report").json()
    assert data["available"] is False and "summary" not in data
    assert client.get("/api/asset/evaluation.png").status_code == 404


def test_unlisted_assets_are_rejected_before_checks(tmp_path, monkeypatch):
    def forbidden(_):
        raise AssertionError("Private requests must not reach the evidence reader")

    monkeypatch.setattr(viewer, "checked_projection", forbidden)
    client = TestClient(viewer.create_app(tmp_path))
    for name in ("resource-authorization.json", "adapter_model.safetensors", "call-journal.json"):
        assert client.get("/api/asset/" + name).status_code == 404
    assert client.get("/api/asset/..%2Fpublic-summary.json").status_code == 404


def test_public_asset_is_the_checked_file():
    response = TestClient(viewer.create_app()).get("/api/asset/per-case.csv")
    assert response.status_code == 200
    assert response.content == (viewer.ROOT / viewer.FIG / "per-case.csv").read_bytes()
    assert response.headers["x-content-type-options"] == "nosniff"


def test_http_has_no_write_or_model_entrypoint():
    client = TestClient(viewer.create_app())
    for path in ("/", "/api/report", "/api/asset/per-case.csv"):
        assert client.post(path, json={"execute": True}).status_code == 405
    assert client.get("/api/generate").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/api/health").json()["model_routes"] is False


def test_template_explains_public_projection_boundary():
    html = (Path(viewer.__file__).parent / "static/public_partial40.html").read_text()
    assert "公开数值投影" in html and "不重放" in html


def test_inline_javascript_has_valid_syntax():
    if not shutil.which("node"):
        pytest.skip("Node is needed only for the viewer JavaScript syntax check")
    html = (Path(viewer.__file__).parent / "static/public_partial40.html").read_text()
    script = html.split("<script>", 1)[1].split("</script>", 1)[0]
    result = subprocess.run(
        ["node", "--check", "--input-type=module"],
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
