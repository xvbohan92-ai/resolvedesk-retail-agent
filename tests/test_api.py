import importlib

from fastapi.testclient import TestClient

from tests.test_workflow import walk
from training.fixtures import make_case


def client(monkeypatch, tmp_path):
    monkeypatch.setenv("RESOLVEDESK_DATA", str(tmp_path))
    from resolve_desk import api

    api = importlib.reload(api)
    return api, TestClient(api.app)


def test_api_confirmation_requires_boolean_revision_and_session_bound_proposal(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    state = walk(api.store, make_case("valid", "return", 0))
    sid, pid = state["id"], state["proposal"]["id"]
    path = f"/api/sessions/{sid}/confirm"
    assert (
        c.post(path, json={"proposal_id": pid, "revision": state["revision"], "accepted": "yes"}).status_code
        == 422
    )
    assert c.post(path, json={"proposal_id": pid, "revision": 0, "accepted": True}).status_code == 409
    result = c.post(path, json={"proposal_id": pid, "revision": state["revision"], "accepted": True})
    assert result.status_code == 200
    assert result.json()["session"]["status"] == "completed"
    assert (
        c.post(path, json={"proposal_id": pid, "revision": state["revision"], "accepted": True}).json()[
            "confirmation_result"
        ]
        == result.json()["confirmation_result"]
    )


def test_public_case_metadata_never_exposes_test_cases_or_expected_outcomes(monkeypatch, tmp_path):
    _, c = client(monkeypatch, tmp_path)
    cases = c.get("/api/cases").json()
    assert len(cases) == 270
    assert sum(row["id"].startswith("v2-valid-") for row in cases) == 28
    assert sum(row["id"].startswith("v3-valid-") for row in cases) == 16
    assert sum(row["id"].startswith("v4-valid-") for row in cases) == 32
    assert sum(row["id"].startswith("v5-valid-") for row in cases) == 32
    assert sum(row["id"].startswith("v6-valid-") for row in cases) == 48
    assert sum(row["id"].startswith("v6_1-valid-") for row in cases) == 48
    assert sum(row["id"].startswith("v7-valid-") for row in cases) == 48
    assert all(
        row["id"].startswith(
            (
                "valid-",
                "v2-valid-",
                "v3-valid-",
                "v4-valid-",
                "v5-valid-",
                "v6-valid-",
                "v6_1-valid-",
                "v7-valid-",
            )
        )
        and set(row) == {"id", "family", "message"}
        for row in cases
    )
    assert (
        c.post("/api/sessions", json={"case_id": "test-return-00", "provider": "ollama"}).status_code == 400
    )
    assert c.post("/api/sessions", json={"case_id": "v2-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v3-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v4-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v5-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v6-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v6_1-test-exchange-00"}).status_code == 400
    assert c.post("/api/sessions", json={"case_id": "v7-test-exchange-00"}).status_code == 400


def test_unknown_provider_and_missing_session_return_clear_errors(monkeypatch, tmp_path):
    _, c = client(monkeypatch, tmp_path)
    assert (
        c.post("/api/sessions", json={"case_id": "valid-return-00", "provider": "cloud"}).status_code == 400
    )
    assert c.get("/api/sessions/unknown").status_code == 404


def test_cuda_comparison_is_pending_without_artifacts_and_never_exposes_test_worlds(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    evidence = c.get("/api/evidence", params={"experiment": "cuda_eval"})
    assert evidence.status_code == 200 and evidence.json()["available"] is False
    assert evidence.json()["experiment"] == "cuda_eval"
    assert c.get("/api/figure/evaluation.png", params={"experiment": "cuda_eval"}).status_code == 404
    assert c.post("/api/sessions", json={"case_id": "cuda-fresh-test-return-00"}).status_code == 400
    assert not any(row["id"].startswith("cuda-fresh-") for row in c.get("/api/cases").json())


def test_editable_customer_request_reaches_the_model_without_fixture_answers(monkeypatch, tmp_path):
    import json

    api, c = client(monkeypatch, tmp_path)
    request = "My request has changed. Please transfer me to a human."

    class Model:
        def generate(self, prompt):
            visible = json.loads(prompt[1]["content"])
            assert visible["messages"][0]["content"] == request
            assert set(visible) == {
                "messages",
                "completed_reads",
                "recent_events",
                "clarification_memory",
                "current_request",
                "previous_request",
                "decision_gate",
                "workflow_version",
                "validation_feedback",
                "clarification_state",
            }
            if visible["decision_gate"]["next_action"]["tool"] == "record_request":
                return json.dumps(
                    {
                        "tool": "record_request",
                        "arguments": {
                            "operation": "handoff",
                            "email": "unknown",
                            "order_id": "unknown",
                            "requested_option": "unknown",
                            "payment_method": "unknown",
                            "reason": "unknown",
                            "fallback_return": "unknown",
                            "evidence": request,
                        },
                    }
                ), {}
            return '{"tool":"handoff","arguments":{"reason":"Customer requested a human."}}', {}

    api.providers["ollama"] = Model()
    response = c.post(
        "/api/sessions",
        json={"case_id": "valid-return-00", "message": request, "experimental_flow": True},
    )
    assert response.status_code == 200
    assert response.json()["session"]["messages"][0]["content"] == request
    assert response.json()["session"]["status"] == "handoff"
    assert c.post("/api/sessions", json={"case_id": "valid-return-00", "message": "   "}).status_code == 400


def test_rejected_candidate_requires_explicit_experimental_session(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "infer", lambda sid, name: api.snapshot(sid))
    # Even a newly added demo world must not switch the default controller.
    default = c.post("/api/sessions", json={"case_id": "v5-valid-exchange-00"}).json()
    assert default["session"].get("workflow_version") != "v7"
    for options in [{"experimental_flow": True}, {"provider": "finetuned"}]:
        explicit = c.post("/api/sessions", json={"case_id": "valid-return-00", **options}).json()
        assert explicit["session"]["workflow_version"] == "v7"
    restored = c.get(f"/api/sessions/{default['session']['id']}").json()
    assert restored["session"].get("workflow_version") != "v7"


def test_existing_v5_session_retains_its_original_controller(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    from training.v5.fixtures import make_case as previous_case

    old = api.previous_typed_store.create(previous_case("valid", "exchange", 0))
    controllers = []
    monkeypatch.setattr(api, "advance", lambda controller, sid, provider: controllers.append(controller))
    api.infer(old["id"], "ollama")
    assert controllers == [api.previous_typed_store]
    assert c.get(f"/api/sessions/{old['id']}").json()["session"]["workflow_version"] == "v5"


def test_existing_v61_session_retains_controller_and_pending_state(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    from training.v6_1.fixtures import make_case as previous_case

    case = previous_case("valid", "exchange", 0)
    old = api.previous_final_store.create(case)
    import json

    from training.v6_1.prepare import reference

    for _ in range(5):
        old = api.previous_final_store.apply(
            old["id"], json.dumps(reference(case, old)), {"provider": "unit"}, old["revision"]
        )
    assert old["status"] == "awaiting_confirmation"
    seen = []
    monkeypatch.setattr(api, "advance", lambda controller, sid, provider: seen.append(controller))
    api.infer(old["id"], "ollama")
    assert seen == [api.previous_final_store]
    restored = c.get(f"/api/sessions/{old['id']}").json()["session"]
    assert restored == old


def test_finetuned_profiles_require_matching_session_model(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    from training.v6_1.fixtures import make_case as previous_case

    class LatestModel:
        def identity(self):
            return {"variant": "qlora_v7"}

    api.providers["finetuned"] = LatestModel()
    case = previous_case("valid", "exchange", 0)
    old = api.previous_final_store.create(case)
    import json

    from training.v6_1.prepare import reference

    for _ in range(5):
        old = api.previous_final_store.apply(
            old["id"], json.dumps(reference(case, old)), {"provider": "unit"}, old["revision"]
        )
    assert old["status"] == "awaiting_confirmation"
    monkeypatch.setattr(api, "advance", lambda *args, **kwargs: None)
    response = c.post(
        f"/api/sessions/{old['id']}/messages",
        json={"revision": old["revision"], "message": "Please continue.", "provider": "finetuned"},
    )
    assert response.status_code == 503
    assert "matching QLoRA v6_1" in response.json()["detail"]
    assert c.get(f"/api/sessions/{old['id']}").json()["session"] == old


def test_new_evidence_pending_never_substitutes_historical_results(monkeypatch, tmp_path):
    api, c = client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    pending = c.get("/api/evidence").json()
    assert pending["experiment"] == "v7" and pending["available"] is False
    assert c.get("/api/figure/recovery.png", params={"experiment": "v7"}).status_code == 404
    archived = c.get("/api/evidence", params={"experiment": "v6_1"}).json()
    assert archived["experiment"] == "v6_1" and archived["available"] is False


def test_independent_cuda_round_never_reuses_old_figures_or_public_test_worlds(monkeypatch, tmp_path):
    import json

    api, c = client(monkeypatch, tmp_path)
    monkeypatch.setattr(api, "ROOT", tmp_path)
    folder = tmp_path / "reports/cuda-eval2"
    folder.mkdir(parents=True)
    (folder / "runtime-availability.json").write_text(json.dumps({
        "run_id": "resolvedesk-cuda-eval-002",
        "status": "free_gpu_quota_exhausted",
        "free_gpu_only": True,
    }))
    old_figure = tmp_path / "docs/figures/cuda-eval/evaluation.png"
    old_figure.parent.mkdir(parents=True)
    old_figure.write_bytes(b"stale old-round figure")
    evidence = c.get("/api/evidence", params={"experiment": "cuda_eval2"}).json()
    assert evidence["available"] is False and evidence["status"] == "waiting_free_gpu"
    assert evidence["experiment"] == "cuda_eval2" and "summary" not in evidence
    assert c.get("/api/figure/evaluation.png", params={"experiment": "cuda_eval2"}).status_code == 404
    assert c.post("/api/sessions", json={"case_id": "cuda-fresh2-test-return-00"}).status_code == 400
    assert not any(row["id"].startswith("cuda-fresh2-") for row in c.get("/api/cases").json())
